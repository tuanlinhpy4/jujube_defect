"""
EdgeJuFormer v4 — SE + RepVGG DW (clean, no ablation)
------------------------------------------------------

Kiến trúc:

    1. SE block (Squeeze-and-Excitation)
       - Chỉ dùng ở stage3
       - GAP → FC(r=4) → ReLU → FC → Sigmoid
       - Channel recalibration phù hợp jujube: phân biệt qua màu/texture

    2. RepVGG-style DW (RepDWConv)
       - Train: 3×3 + 1×1 DWConv song song (multi-branch)
       - Inference: fuse thành 1 conv duy nhất → zero overhead
       - Gọi model.reparameterize() trước khi deploy/benchmark

Chạy:
    python train.py --dataset jujube_ours --model juformer --seed 1
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Registry metadata
# ============================================================

ARCH_NAME = "juformer"

MODEL_CONFIGS = {
    "juformer":    {"input_size": 224, "width_mult": 1.00},
    "juformer_s":  {"input_size": 224, "width_mult": 0.75},
    "juformer_xs": {"input_size": 224, "width_mult": 0.50},
}

DEFAULT_VARIANT = "juformer"


# ============================================================
# Utility
# ============================================================

def _round_ch(c: float, divisor: int = 8):
    return int(math.ceil(c / divisor) * divisor)


def _cbr(in_ch, out_ch, k=1, s=1, g=1):
    return nn.Sequential(
        nn.Conv2d(
            in_ch,
            out_ch,
            kernel_size=k,
            stride=s,
            padding=k // 2,
            groups=g,
            bias=False
        ),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True)
    )


# ============================================================
# Deterministic GAP helper
# ============================================================

class _GAPFull(nn.Module):
    """Global Average Pool → (B, C, 1, 1) — deterministic."""
    def forward(self, x):
        return x.mean(dim=[2, 3], keepdim=True)


# ============================================================
# DropPath
# ============================================================

class DropPath(nn.Module):

    def __init__(self, p=0.0):
        super().__init__()
        self.p = p

    def forward(self, x):

        if self.p == 0.0 or not self.training:
            return x

        keep = 1 - self.p
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = torch.bernoulli(
            torch.full(shape, keep, device=x.device)
        )
        return x * mask / keep


# ============================================================
# SE Block
# ============================================================

class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation channel recalibration.

    - SE dùng GAP toàn cục → channel-only, đủ cho jujube vì
      phân biệt class chủ yếu qua màu sắc và texture (channel)
      không phải vị trí không gian
    - reduction=4 (không phải 16 mặc định) vì channels nhỏ
      (64ch ở stage3) → hidden=16, đủ capacity mà không thêm
      nhiều params.
    """

    def __init__(self, channels, reduction=4):
        super().__init__()

        hidden = max(channels // reduction, 8)

        self.gap = _GAPFull()

        self.fc1 = nn.Conv2d(channels, hidden, 1, bias=True)
        self.act = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(hidden, channels, 1, bias=True)

    def forward(self, x):
        w = self.gap(x)
        w = self.act(self.fc1(w))
        w = torch.sigmoid(self.fc2(w))
        return x * w


# ============================================================
# RepDWConv — RepVGG-style depthwise conv
# ============================================================

class RepDWConv(nn.Module):
    """
    RepVGG-style Depthwise Conv cho edge deployment.

    Train mode: 3 nhánh song song
        - DWConv 3×3  (spatial mixing chính)
        - DWConv 1×1  (channel scaling, receptive field = 1)
        - Identity    (residual nếu stride=1)
    Mỗi nhánh có BN riêng → gradient flow tốt.

    Inference mode: gọi reparameterize() để fuse 3 nhánh
    thành 1 DWConv 3×3 duy nhất → zero overhead so với
    plain 3×3 DWConv, nhưng đã học được đa dạng feature hơn.
    """

    def __init__(self, channels, stride=1):
        super().__init__()

        self.channels = channels
        self.stride   = stride

        # ── nhánh 3×3 ─────────────────────────────────────────
        self.dw3 = nn.Conv2d(
            channels, channels, 3,
            stride=stride, padding=1,
            groups=channels, bias=False
        )
        self.bn3 = nn.BatchNorm2d(channels)

        # ── nhánh 1×1 ─────────────────────────────────────────
        self.dw1 = nn.Conv2d(
            channels, channels, 1,
            stride=stride, padding=0,
            groups=channels, bias=False
        )
        self.bn1 = nn.BatchNorm2d(channels)

        # ── nhánh identity (chỉ khi stride=1) ─────────────────
        self.bn_id = (
            nn.BatchNorm2d(channels)
            if stride == 1 else None
        )

        self.act = nn.ReLU(inplace=True)

        # fused conv (None khi đang train)
        self._fused: Optional[nn.Conv2d] = None

    # ── forward ───────────────────────────────────────────────

    def forward(self, x):

        if self._fused is not None:
            # inference: 1 conv duy nhất
            return self.act(self._fused(x))

        # train: multi-branch
        out = self.bn3(self.dw3(x)) + self.bn1(self.dw1(x))

        if self.bn_id is not None:
            out = out + self.bn_id(x)

        return self.act(out)

    # ── reparameterize (gọi trước khi deploy) ─────────────────

    def reparameterize(self):
        """Fuse tất cả nhánh thành 1 DWConv 3×3 + bias."""

        if self._fused is not None:
            return  # đã fuse rồi

        k3, b3 = self._fuse_branch(self.dw3, self.bn3)
        k1, b1 = self._fuse_1x1(self.dw1, self.bn1)
        kid, bid = self._fuse_identity()

        fused_k = k3 + k1 + kid
        fused_b = b3 + b1 + bid

        fused = nn.Conv2d(
            self.channels, self.channels, 3,
            stride=self.stride, padding=1,
            groups=self.channels, bias=True
        )
        fused.weight.data = fused_k
        fused.bias.data   = fused_b

        self._fused = fused

        # giải phóng bộ nhớ train branches
        del self.dw3, self.bn3, self.dw1, self.bn1
        if self.bn_id is not None:
            del self.bn_id

    def _fuse_branch(self, conv, bn):
        """Fuse Conv + BN → weight + bias."""
        k = conv.weight                         # (C, 1, kH, kW)
        mean, var, gamma, beta, eps = (
            bn.running_mean, bn.running_var,
            bn.weight, bn.bias, bn.eps
        )
        std  = (var + eps).sqrt()
        t    = (gamma / std).reshape(-1, 1, 1, 1)
        return k * t, beta - mean * gamma / std

    def _fuse_1x1(self, conv, bn):
        """Pad 1×1 kernel → 3×3 rồi fuse BN."""
        k1 = conv.weight                        # (C, 1, 1, 1)
        k3 = F.pad(k1, [1, 1, 1, 1])           # (C, 1, 3, 3)
        return self._fuse_branch(
            _KernelProxy(k3, conv.stride,
                         conv.padding, conv.groups),
            bn
        )

    def _fuse_identity(self):
        """Identity nhánh: kernel = delta tại tâm."""
        if self.bn_id is None:
            zeros = torch.zeros(
                self.channels, 1, 3, 3,
                device=self.dw3.weight.device
                if hasattr(self, 'dw3') else
                self._fused.weight.device
            )
            return zeros, zeros.squeeze()

        # delta kernel: 1 ở vị trí trung tâm (1,1)
        k = torch.zeros(
            self.channels, 1, 3, 3,
            device=self.bn_id.weight.device
        )
        k[:, 0, 1, 1] = 1.0

        mean, var, gamma, beta, eps = (
            self.bn_id.running_mean,
            self.bn_id.running_var,
            self.bn_id.weight,
            self.bn_id.bias,
            self.bn_id.eps
        )
        std = (var + eps).sqrt()
        t   = (gamma / std).reshape(-1, 1, 1, 1)
        return k * t, beta - mean * gamma / std


class _KernelProxy:
    """Proxy giả lập nn.Conv2d interface cho _fuse_branch."""
    def __init__(self, weight, stride, padding, groups):
        self.weight = weight
        self.stride = stride
        self.padding = padding
        self.groups = groups


# ============================================================
# TinyTokenMixer
# ============================================================

class TinyTokenMixer(nn.Module):
    """
    MetaFormer-style lightweight token mixer.
    """

    def __init__(self, channels, num_heads=4):
        super().__init__()

        hidden = max(channels // 2, 8)

        self.norm = nn.GroupNorm(1, channels)

        self.dwconv = nn.Conv2d(
            channels, channels,
            kernel_size=5, stride=1, padding=2,
            groups=channels, bias=False
        )
        self.dwbn = nn.BatchNorm2d(channels)

        self.pw1 = nn.Conv2d(channels, hidden, 1, bias=False)
        self.pw1bn = nn.BatchNorm2d(hidden)

        self.act = nn.GELU()

        self.pw2 = nn.Conv2d(hidden, channels, 1, bias=False)
        self.pw2bn = nn.BatchNorm2d(channels)

        self.gate = nn.Sequential(
            _GAPFull(),
            nn.Conv2d(channels, hidden, 1, bias=True),
            nn.GELU(),
            nn.Conv2d(hidden, channels, 1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):

        x = self.norm(x)

        x = self.dwbn(self.dwconv(x))
        x = self.act(x)

        x = self.pw1bn(self.pw1(x))
        x = self.act(x)

        x = self.pw2bn(self.pw2(x))

        return x * self.gate(x)


# ============================================================
# Hybrid Block
# ============================================================

class EdgeJuFormerBlock(nn.Module):

    def __init__(
        self,
        in_ch,
        out_ch,
        stride=1,
        expand_ratio=4,
        use_token=True,
        use_se=False,           # SE chỉ bật ở stage3
        drop_path=0.0,
    ):
        super().__init__()

        hidden = in_ch * expand_ratio
        self.use_res = (stride == 1 and in_ch == out_ch)

        # expansion
        self.expand = _cbr(in_ch, hidden, k=1)

        # RepDWConv
        self.dynamic_dw = RepDWConv(hidden, stride=1)

        # downsample (stride conv — không fuse RepVGG)
        if stride == 2:
            self.downsample = _cbr(hidden, hidden, k=3, s=2, g=hidden)
        else:
            self.downsample = nn.Identity()

        # TinyTokenMixer vs Identity
        self.token = (
            TinyTokenMixer(hidden) if use_token else nn.Identity()
        )

        # projection
        self.project = nn.Sequential(
            nn.Conv2d(hidden, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch)
        )

        # SE block (chỉ dùng ở stage3, use_se=True)
        self.se = SEBlock(out_ch, reduction=4) if use_se else nn.Identity()

        self.drop = (
            DropPath(drop_path) if drop_path > 0 else nn.Identity()
        )

    def forward(self, x):

        identity = x

        x = self.expand(x)
        x = self.dynamic_dw(x)
        x = self.downsample(x)
        x = self.token(x)
        x = self.project(x)
        x = self.se(x)

        if self.use_res:
            x = self.drop(x) + identity

        return x


# ============================================================
# Model
# ============================================================

class EdgeJuFormer(nn.Module):

    def __init__(
        self,
        num_classes=4,
        width_mult=1.0,
        drop_path=0.1,
    ):
        super().__init__()

        c = lambda x: _round_ch(x * width_mult)

        c1 = c(16)
        c2 = c(24)
        c3 = c(40)
        c4 = c(64)
        c5 = c(96)

        # stem
        self.stem = _cbr(3, c1, k=3, s=2)

        # stage1/2: CNN-only, no SE (giữ nhẹ)
        self.stage1 = EdgeJuFormerBlock(
            c1, c2, stride=2,
            use_token=False, use_se=False,
        )

        self.stage2 = EdgeJuFormerBlock(
            c2, c3, stride=2,
            use_token=False, use_se=False,
        )

        # stage3: TokenMixer + SE (chỉ ở đây mới dùng SE)
        self.stage3 = nn.Sequential(
            EdgeJuFormerBlock(
                c3, c4, stride=2,
                use_token=True, use_se=True,
            ),
            EdgeJuFormerBlock(
                c4, c4, stride=1,
                use_token=True, use_se=True,
                drop_path=drop_path,
            )
        )

        # head
        self.head = _cbr(c4, c5, k=1)

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Linear(c5, num_classes)

        self._init_weights()

    # --------------------------------------------------------

    def _init_weights(self):

        for m in self.modules():

            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu'
                )
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # --------------------------------------------------------

    def forward(self, x):

        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.head(x)
        x = self.pool(x).flatten(1)
        x = self.classifier(x)

        return x

    # --------------------------------------------------------

    def reparameterize(self):
        """
        Fuse tất cả RepDWConv branches → 1 conv duy nhất.
        Gọi trước khi benchmark latency hoặc deploy.

        Ví dụ:
            model.eval()
            model.reparameterize()
            torch.save(model.state_dict(), 'deploy.pth')
        """
        for m in self.modules():
            if isinstance(m, RepDWConv):
                m.reparameterize()

    # --------------------------------------------------------

    def count_parameters(self):
        return sum(
            p.numel() for p in self.parameters() if p.requires_grad
        )

    # --------------------------------------------------------

    def forward_features(self, x):

        feats = {}

        x = self.stem(x);      feats["stem"]   = x
        x = self.stage1(x);    feats["stage1"] = x
        x = self.stage2(x);    feats["stage2"] = x
        x = self.stage3(x);    feats["stage3"] = x
        x = self.head(x);      feats["head"]   = x

        return feats

    # --------------------------------------------------------

    def __repr__(self):
        p = self.count_parameters()
        return (
            f"EdgeJuFormer("
            f"params={p:,}, "
            f"size={p * 4 / 1024:.1f} KB)"
        )


# ============================================================
# Factory
# ============================================================

def edgejuformer(num_classes=4):
    return EdgeJuFormer(num_classes=num_classes, width_mult=1.0)


def edgejuformer_s(num_classes=4):
    return EdgeJuFormer(num_classes=num_classes, width_mult=0.75)


def edgejuformer_xs(num_classes=4):
    return EdgeJuFormer(num_classes=num_classes, width_mult=0.5)


# ============================================================
# build_model  (required by model registry)
# ============================================================

def build_model(name: str = "juformer",
                num_classes: int = 4,
                in_channels: int = 3,
                **kwargs):

    cfg = MODEL_CONFIGS.get(name)
    if cfg is None:
        raise ValueError(
            f"Unknown variant '{name}'. "
            f"Choose from: {list(MODEL_CONFIGS)}"
        )

    model = EdgeJuFormer(
        num_classes=num_classes,
        width_mult=cfg["width_mult"],
    )

    if in_channels != 3:
        old = model.stem[0]
        model.stem[0] = nn.Conv2d(
            in_channels,
            old.out_channels,
            kernel_size=old.kernel_size,
            stride=old.stride,
            padding=old.padding,
            bias=False
        )

    model.num_classes = num_classes

    return model


# ============================================================
# Quick self-test
# ============================================================

if __name__ == "__main__":

    x = torch.randn(1, 3, 224, 224)

    print(f"{'Variant':<16} {'Params (train)':>16}  {'Output'}")
    print("-" * 60)

    for name in MODEL_CONFIGS:
        m = build_model(name, num_classes=5)
        m.eval()
        with torch.no_grad():
            y = m(x)
        p_train = m.count_parameters()
        print(f"{name:<16} {p_train:>16,}  {tuple(y.shape)}")

    print()
    print("After reparameterize():")
    m = build_model("juformer", num_classes=5)
    m.eval()
    m.reparameterize()
    with torch.no_grad():
        y = m(x)
    p_deploy = m.count_parameters()
    print(f"  params (deploy): {p_deploy:,}  output: {tuple(y.shape)}")
