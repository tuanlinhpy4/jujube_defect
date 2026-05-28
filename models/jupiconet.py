"""JuPicoNet — < 300K params, jujube classifier optimised for imbalanced data.

Manh mối thiết kế (đối thủ chính = reptakunet, 0.28M, F1 = 83.72%):
  - Reparameterizable DW branch (RepBlock idea, FastViT/RepViT/MobileOne):
    train với 3x3 + 1x1 + identity, deploy fuse về 1 conv 3x3 → giảm latency
    và FLOPS ở runtime mà vẫn giữ đa nhánh khi train.
  - GhostConv expansion (Han et al., 2020): cắt một nửa chi phí 1x1 expand
    — vốn là phần nặng nhất của MBConv inverted residual.
  - ECA (channel) + CS_STAM (shape/texture spatial) lấy từ reptakunet baseline,
    đã chứng minh tốt cho ảnh quả táo có texture hư hại nhỏ.
  - Multi-scale GAP fusion cuối: concat pooled-stage3 ⊕ pooled-stage4 trước
    classifier — giữ texture detail (cho mold, insect_damaged) song song với
    semantic stage cuối.
  - Cosine classifier (weight-normalized linear): trên data lệch nặng (dry 1469
    vs mold 346), classifier chuẩn hoá tránh việc các vector logits của lớp
    đông đè hẳn lớp ít, kết hợp tốt với class_weights cross-entropy ở train.py.
  - DropPath stochastic depth: chỉ bật trên block cuối mỗi stage, regularization
    không tốn param.

Ngân sách param: ~ 270K.
Channels [24, 48, 96, 144], depths [2, 3, 5, 2].
"""

from __future__ import annotations

import math
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F


ARCH_NAME = "jupiconet"
ARCH_ALIASES = ["JuPicoNet", "jupico"]


# ---------------------------------------------------------------------------
# Stochastic depth
# ---------------------------------------------------------------------------

class DropPath(nn.Module):
    def __init__(self, p: float = 0.0):
        super().__init__()
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.p <= 0.0 or not self.training:
            return x
        keep = 1.0 - self.p
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = torch.bernoulli(torch.full(shape, keep,
                                           device=x.device, dtype=x.dtype))
        return x / keep * mask


# ---------------------------------------------------------------------------
# ECA — channel attention, kernel 1x1d ≈ 3-5 params, gần như miễn phí
# ---------------------------------------------------------------------------

class ECA(nn.Module):
    def __init__(self, channels: int, gamma: float = 2.0, b: float = 1.0):
        super().__init__()
        t = int(abs(math.log2(channels) / gamma) + b / gamma)
        k = t if t % 2 else t + 1
        k = max(k, 3)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv1d = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.pool(x).squeeze(-1).transpose(-1, -2)
        y = self.conv1d(y).transpose(-1, -2).unsqueeze(-1)
        return x * torch.sigmoid(y)


# ---------------------------------------------------------------------------
# CS-STAM — shape (vertical/horizontal stripes) + texture spatial attention
# ---------------------------------------------------------------------------

class CS_STAM(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.shape_v = nn.Conv2d(channels, channels, (5, 1), padding=(2, 0),
                                  groups=channels, bias=False)
        self.shape_h = nn.Conv2d(channels, channels, (1, 5), padding=(0, 2),
                                  groups=channels, bias=False)
        self.tex = nn.Conv2d(channels, channels, 3, padding=1,
                             groups=channels, bias=False)
        self.fusion = nn.Conv2d(2 * channels, 1, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ms = self.shape_v(x) + self.shape_h(x)
        mt = self.tex(x)
        m = torch.sigmoid(self.fusion(torch.cat([ms, mt], dim=1)))
        return x + x * m


# ---------------------------------------------------------------------------
# GhostConv — cắt một nửa chi phí expansion 1x1
# ---------------------------------------------------------------------------

class GhostConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, ratio: int = 2):
        super().__init__()
        self.out_ch = out_ch
        init_ch = math.ceil(out_ch / ratio)
        new_ch = out_ch - init_ch
        self.primary = nn.Sequential(
            nn.Conv2d(in_ch, init_ch, 1, bias=False),
            nn.BatchNorm2d(init_ch),
            nn.ReLU6(inplace=True),
        )
        self.cheap = nn.Sequential(
            nn.Conv2d(init_ch, new_ch, 3, padding=1, groups=init_ch, bias=False),
            nn.BatchNorm2d(new_ch),
            nn.ReLU6(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        p = self.primary(x)
        c = self.cheap(p)
        return torch.cat([p, c], dim=1)[:, :self.out_ch]


# ---------------------------------------------------------------------------
# RepGhostBlock — train với rep multi-branch DW, deploy fuse về 1 conv 3x3
# ---------------------------------------------------------------------------

class RepGhostBlock(nn.Module):
    """Reparameterizable Ghost Inverted-Residual block.

    Cấu trúc lúc train:
      - DW branch song song: dw3x3+BN ⊕ dw1x1+BN ⊕ identity-BN (RepVGG/MobileOne)
      - Ghost expansion: in_ch → hidden (1x1 chi phí một nửa)
      - Project: hidden → in_ch (1x1)
      - Residual: x + DropPath(out)
    Deploy: fuse 3 nhánh DW thành 1 conv 3x3 duy nhất.
    """

    def __init__(self, channels: int, expand_ratio: float = 1.5,
                 drop_path: float = 0.0, use_attention: bool = True,
                 use_cs_stam: bool = False):
        super().__init__()
        self.channels = channels
        hidden = max(int(round(channels * expand_ratio)), channels)

        # Multi-branch DW (rep)
        self.dw_3x3 = nn.Conv2d(channels, channels, 3, padding=1,
                                 groups=channels, bias=False)
        self.dw_1x1 = nn.Conv2d(channels, channels, 1,
                                 groups=channels, bias=False)
        self.bn_3x3 = nn.BatchNorm2d(channels)
        self.bn_1x1 = nn.BatchNorm2d(channels)
        self.bn_id = nn.BatchNorm2d(channels)
        self.act_dw = nn.ReLU6(inplace=True)

        # Ghost expansion 1x1
        self.expand = GhostConv(channels, hidden, ratio=2)

        # Project 1x1
        self.project = nn.Sequential(
            nn.Conv2d(hidden, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
        )

        # Attention
        self.eca = ECA(channels) if use_attention else nn.Identity()
        self.cs_stam = CS_STAM(channels) if use_cs_stam else nn.Identity()

        self.drop_path = DropPath(drop_path)
        self.deploy = False
        self.dw_reparam = None

    def _dw_forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.deploy:
            return self.act_dw(self.dw_reparam(x))
        y = (self.bn_3x3(self.dw_3x3(x))
             + self.bn_1x1(self.dw_1x1(x))
             + self.bn_id(x))
        return self.act_dw(y)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self._dw_forward(x)
        h = self.expand(h)
        h = self.project(h)
        h = self.eca(h)
        h = self.cs_stam(h)
        return x + self.drop_path(h)

    # -------------------- reparameterization (deploy) --------------------
    @torch.no_grad()
    def switch_to_deploy(self) -> None:
        if self.deploy:
            return
        kernel, bias = self._fused_dw()
        self.dw_reparam = nn.Conv2d(self.channels, self.channels, 3,
                                     padding=1, groups=self.channels, bias=True)
        self.dw_reparam.weight.copy_(kernel)
        self.dw_reparam.bias.copy_(bias)
        for attr in ("dw_3x3", "dw_1x1", "bn_3x3", "bn_1x1", "bn_id"):
            delattr(self, attr)
        self.deploy = True

    def _fused_dw(self):
        def fuse(conv_w, bn):
            gamma = bn.weight
            std = (bn.running_var + bn.eps).sqrt()
            t = (gamma / std).reshape(-1, 1, 1, 1)
            return conv_w * t, bn.bias - bn.running_mean * gamma / std

        k3, b3 = fuse(self.dw_3x3.weight, self.bn_3x3)
        k1, b1 = fuse(self.dw_1x1.weight, self.bn_1x1)
        k1 = F.pad(k1, [1, 1, 1, 1])
        device = self.bn_id.weight.device
        k_id = torch.zeros(self.channels, 1, 3, 3, device=device)
        for i in range(self.channels):
            k_id[i, 0, 1, 1] = 1.0
        k_id, b_id = fuse(k_id, self.bn_id)
        return k3 + k1 + k_id, b3 + b1 + b_id


# ---------------------------------------------------------------------------
# Downsampler — concat (input bypass) ⊕ (block output) → 1x1 + maxpool
# ---------------------------------------------------------------------------

class Downsampler(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_ch * 2, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU6(inplace=True),
        )
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x_in: torch.Tensor, x_blocks: torch.Tensor) -> torch.Tensor:
        return self.pool(self.proj(torch.cat([x_in, x_blocks], dim=1)))


# ---------------------------------------------------------------------------
# Stage
# ---------------------------------------------------------------------------

class Stage(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, depth: int,
                 expand_ratio: float = 1.5,
                 drop_path: float = 0.0,
                 use_cs_stam: bool = False):
        super().__init__()
        self.blocks = nn.ModuleList([
            RepGhostBlock(
                in_ch,
                expand_ratio=expand_ratio,
                drop_path=drop_path if i == depth - 1 else 0.0,
                use_attention=True,
                use_cs_stam=use_cs_stam and (i == depth - 1),
            )
            for i in range(depth)
        ])
        self.down = Downsampler(in_ch, out_ch) if out_ch != in_ch or True else nn.Identity()
        self._needs_down = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_in = x
        for blk in self.blocks:
            x = blk(x)
        return self.down(x_in, x)


class FinalStage(nn.Module):
    """Stage cuối: không downsample, đưa thẳng feature ra cho head fusion."""

    def __init__(self, in_ch: int, depth: int,
                 expand_ratio: float = 1.5,
                 drop_path: float = 0.0):
        super().__init__()
        self.blocks = nn.ModuleList([
            RepGhostBlock(
                in_ch,
                expand_ratio=expand_ratio,
                drop_path=drop_path if i == depth - 1 else 0.0,
                use_attention=True,
                use_cs_stam=(i == depth - 1),
            )
            for i in range(depth)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            x = blk(x)
        return x


# ---------------------------------------------------------------------------
# Cosine classifier — chống bias lớp đông trên data lệch
# ---------------------------------------------------------------------------

class CosineClassifier(nn.Module):
    """Weight-normalized linear; logits = scale * cos(feat, weight).

    Trên data lệch (dry 1469 vs mold 346), classifier chuẩn hoá khử bias do
    norm vector lớp đông tự nhiên lớn hơn. Kết hợp với class_weights CE đã được
    train.py tự sinh từ inverse-frequency.
    """

    def __init__(self, in_features: int, num_classes: int, scale: float = 16.0):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_classes, in_features))
        nn.init.kaiming_normal_(self.weight, nonlinearity="linear")
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.normalize(x, dim=1)
        w = F.normalize(self.weight, dim=1)
        return self.scale * F.linear(x, w)


# ---------------------------------------------------------------------------
# Head — multi-scale GAP fusion (s3 ⊕ s4) → cosine classifier
# ---------------------------------------------------------------------------

class FusionHead(nn.Module):
    def __init__(self, ch_s3: int, ch_s4: int, num_classes: int,
                 fc_dim: int = 192, dropout: float = 0.1):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Sequential(
            nn.Linear(ch_s3 + ch_s4, fc_dim, bias=False),
            nn.BatchNorm1d(fc_dim),
            nn.ReLU6(inplace=True),
            nn.Dropout(dropout),
        )
        self.classifier = CosineClassifier(fc_dim, num_classes)

    def forward(self, x_s3: torch.Tensor, x_s4: torch.Tensor) -> torch.Tensor:
        f = torch.cat([self.pool(x_s3).flatten(1),
                        self.pool(x_s4).flatten(1)], dim=1)
        return self.classifier(self.proj(f))


# ---------------------------------------------------------------------------
# Stem
# ---------------------------------------------------------------------------

class Stem(nn.Module):
    def __init__(self, in_channels: int = 3, out_ch: int = 16):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_ch, 3,
                               stride=2, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU6(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


# ---------------------------------------------------------------------------
# JuPicoNet
# ---------------------------------------------------------------------------

class JuPicoNet(nn.Module):
    """JuPicoNet — < 300K params, multi-scale fusion + cosine classifier.

    224x224x3
      → Stem 3→16            (112x112x16)
      → Stage1 16→24, d=2     (56x56x24)
      → Stage2 24→48, d=3     (28x28x48)
      → Stage3 48→96, d=5     (14x14x96)   ← s3 features
      → Stage4 96→144, d=2    (14x14x144)  ← s4 features (no downsample)
      → FusionHead(s3, s4)    → num_classes
    """

    def __init__(self,
                  in_channels: int = 3,
                  num_classes: int = 5,
                  channels: List[int] = (16, 24, 48, 96, 144),
                  depths: List[int] = (2, 3, 5, 2),
                  drop_path: float = 0.05,
                  expand_ratio: float = 1.5,
                  fc_dim: int = 192,
                  dropout: float = 0.1):
        super().__init__()
        c0, c1, c2, c3, c4 = channels

        self.stem = Stem(in_channels, c0)
        self.stage1 = Stage(c0, c1, depths[0], expand_ratio=expand_ratio,
                             drop_path=drop_path, use_cs_stam=False)
        self.stage2 = Stage(c1, c2, depths[1], expand_ratio=expand_ratio,
                             drop_path=drop_path, use_cs_stam=True)
        self.stage3 = Stage(c2, c3, depths[2], expand_ratio=expand_ratio,
                             drop_path=drop_path, use_cs_stam=True)

        # Stage 4 không downsample — giữ resolution 14x14 với features ngữ nghĩa cao
        self.stage4_proj = nn.Sequential(
            nn.Conv2d(c3, c4, 1, bias=False),
            nn.BatchNorm2d(c4),
            nn.ReLU6(inplace=True),
        )
        self.stage4 = FinalStage(c4, depths[3], expand_ratio=expand_ratio,
                                  drop_path=drop_path)

        self.head = FusionHead(c3, c4, num_classes,
                                fc_dim=fc_dim, dropout=dropout)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                         nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x_s3 = self.stage3(x)
        x = self.stage4_proj(x_s3)
        x_s4 = self.stage4(x)
        return self.head(x_s3, x_s4)

    def switch_to_deploy(self) -> None:
        for m in self.modules():
            if hasattr(m, "switch_to_deploy") and m is not self:
                m.switch_to_deploy()


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

MODEL_CONFIGS: Dict[str, Dict] = {
    # ~ 271K params, 166M MACs — primary target (< 300K, F1 tối đa)
    "jupiconet": dict(
        channels=(16, 28, 56, 112, 160),
        depths=(2, 3, 5, 2),
        drop_path=0.05,
        expand_ratio=1.5,
        fc_dim=192,
        dropout=0.1,
        input_size=224,
        params_M=0.271,
    ),
    # ~ 221K params, 138M MACs — bản gọn hơn nếu cần latency thấp
    "jupiconet_m": dict(
        channels=(16, 24, 48, 96, 144),
        depths=(2, 3, 5, 2),
        drop_path=0.05,
        expand_ratio=1.5,
        fc_dim=192,
        dropout=0.1,
        input_size=224,
        params_M=0.221,
    ),
    # ~ 141K params, 113M MACs — bản nhỏ nhất, tham chiếu
    "jupiconet_s": dict(
        channels=(16, 24, 40, 72, 112),
        depths=(2, 3, 4, 2),
        drop_path=0.03,
        expand_ratio=1.5,
        fc_dim=160,
        dropout=0.1,
        input_size=224,
        params_M=0.141,
    ),
}

DEFAULT_VARIANT = "jupiconet"


def build_model(name: str = "jupiconet", num_classes: int = 5,
                in_channels: int = 3, **overrides) -> JuPicoNet:
    if name not in MODEL_CONFIGS:
        raise ValueError(
            f"Unknown variant: {name}. Choose from {list(MODEL_CONFIGS)}"
        )
    cfg = {k: v for k, v in MODEL_CONFIGS[name].items()
           if k not in ("input_size", "params_M")}
    cfg.update(overrides)
    return JuPicoNet(in_channels=in_channels, num_classes=num_classes, **cfg)
