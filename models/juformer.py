"""
EdgeJuFormer v5 — CBAM + RepVGG DW
------------------------------------

Cải tiến từ v4 (SE + RepDW):
    SE → CBAM (Channel + Spatial Attention)

Lý do thay SE bằng CBAM:
    - SE chỉ hỏi "channel nào quan trọng?" → channel-only
    - CBAM thêm spatial attention "vùng nào quan trọng?" → fix được:
        * bianxing (biến dạng): spatial attention locate vùng deform
        * meibian (mốc): spatial attention focus patch nhỏ low-contrast
    - Params tăng rất ít: thêm ~2k (Conv7×7 spatial branch)
    - Vẫn chỉ dùng ở stage3 — giữ stage1/2 nhẹ

v4 kết quả: jujube_ours F1=84.44%, Jujube12000 F1=93.5%
v5 target : jujube_ours F1>85%,   Jujube12000 F1≥95%

Ablation variants (giữ nguyên):
    juformer_a1 → juformer_a5

Chạy:
    python train.py --dataset jujube_ours --model juformer --seed 1
    python train.py --dataset jujube      --model juformer --seed 1
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
    "juformer":    {"input_size": 224, "width_mult": 1.00, "ablation": None},
    "juformer_s":  {"input_size": 224, "width_mult": 0.75, "ablation": None},
    "juformer_xs": {"input_size": 224, "width_mult": 0.50, "ablation": None},

    "juformer_a1": {"input_size": 224, "width_mult": 1.00, "ablation": "no_dyndw"},
    "juformer_a2": {"input_size": 224, "width_mult": 1.00, "ablation": "no_token"},
    "juformer_a3": {"input_size": 224, "width_mult": 1.00, "ablation": "no_ca"},
    "juformer_a4": {"input_size": 224, "width_mult": 1.00, "ablation": "cnn_only"},
    "juformer_a5": {"input_size": 224, "width_mult": 1.00, "ablation": "baseline"},
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
            in_ch, out_ch,
            kernel_size=k, stride=s,
            padding=k // 2, groups=g,
            bias=False
        ),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True)
    )


# ============================================================
# Deterministic GAP helpers
# ============================================================

class _GAPFull(nn.Module):
    """Global Average Pool → (B, C, 1, 1)  — deterministic."""
    def forward(self, x):
        return x.mean(dim=[2, 3], keepdim=True)


class _GAPH(nn.Module):
    """Pool along W only → (B, C, H, 1)  — deterministic."""
    def forward(self, x):
        return x.mean(dim=3, keepdim=True)


class _GAPW(nn.Module):
    """Pool along H only → (B, C, 1, W)  — deterministic."""
    def forward(self, x):
        return x.mean(dim=2, keepdim=True)


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
        keep  = 1 - self.p
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask  = torch.bernoulli(
            torch.full(shape, keep, device=x.device)
        )
        return x * mask / keep


# ============================================================
# [v5] CBAM — thay SEBlock
# ============================================================

class CBAMBlock(nn.Module):
    """
    Convolutional Block Attention Module (Woo et al. 2018).

    2 giai đoạn tuần tự:
        1. Channel Attention (giống SE):
           AvgPool + MaxPool → FC → ReLU → FC → Sigmoid → scale
        2. Spatial Attention (mới so với SE):
           AvgPool + MaxPool theo channel dim → concat →
           Conv7×7 → Sigmoid → scale

    Tại sao CBAM tốt hơn SE cho jujube:
        - bianxing (biến dạng): spatial attention locate được
          vùng deformation trên surface, không chỉ recalibrate channel
        - meibian (mốc): spatial map focus đúng patch mốc nhỏ
          low-contrast mà SE bỏ qua
        - Params overhead: chỉ +2k (Conv7×7 spatial, depthwise=False
          nhưng in_ch=2 rất nhỏ)

    reduction=4 cho channel attention (giống SE trước):
        channels=64 → hidden=16

    kernel_size=7 cho spatial attention:
        receptive field đủ rộng để capture cả patch nhỏ lẫn
        global shape ở feature map 14×14 (stage3, input 224)
    """

    def __init__(self, channels, reduction=4, spatial_k=7):
        super().__init__()

        hidden = max(channels // reduction, 8)

        # ── Channel Attention ──────────────────────────────────
        # avg_pool: deterministic mean — dùng _GAPFull
        # max_pool: KHÔNG dùng AdaptiveMaxPool2d (non-deterministic
        #   backward trên CUDA khi use_deterministic_algorithms=True)
        #   → dùng x.flatten(2).max(dim=2) thay thế
        self.avg_pool = _GAPFull()

        self.ch_fc1 = nn.Conv2d(channels, hidden, 1, bias=False)
        self.ch_act = nn.ReLU(inplace=True)
        self.ch_fc2 = nn.Conv2d(hidden, channels, 1, bias=False)

        # ── Spatial Attention ──────────────────────────────────
        # input: concat(avg_c, max_c) → 2 channels
        self.sp_conv = nn.Conv2d(
            2, 1,
            kernel_size=spatial_k,
            padding=spatial_k // 2,
            bias=False
        )

    @staticmethod
    def _max_pool_det(x: torch.Tensor) -> torch.Tensor:
        """Deterministic global max pool → (B, C, 1, 1).

        AdaptiveMaxPool2d backward không deterministic trên CUDA.
        flatten + max trên dim=2 hoạt động đúng với
        use_deterministic_algorithms(True).
        """
        return x.flatten(2).max(dim=2).values.unsqueeze(-1).unsqueeze(-1)

    def forward(self, x):

        # 1. Channel attention (avg + max, chia sẻ FC)
        avg  = self.ch_fc2(self.ch_act(self.ch_fc1(self.avg_pool(x))))
        mx   = self.ch_fc2(self.ch_act(self.ch_fc1(self._max_pool_det(x))))
        ch_w = torch.sigmoid(avg + mx)
        x    = x * ch_w

        # 2. Spatial attention
        avg_sp = x.mean(dim=1, keepdim=True)           # (B,1,H,W)
        max_sp = x.max(dim=1, keepdim=True).values      # (B,1,H,W) — deterministic
        sp_w   = torch.sigmoid(
            self.sp_conv(torch.cat([avg_sp, max_sp], dim=1))
        )
        x = x * sp_w

        return x


class _IdentityAttn(nn.Module):
    """Ablation: attention replaced with Identity."""
    def __init__(self, channels, reduction=4, spatial_k=7):
        super().__init__()

    def forward(self, x):
        return x


# ============================================================
# Ghost Module  (giữ nguyên từ v4, dùng trong ablation nếu cần)
# ============================================================

class GhostModule(nn.Module):

    def __init__(self, in_ch, out_ch, k=1, s=1,
                 ratio=2, dw_k=3, act=True):
        super().__init__()

        self.out_ch   = out_ch
        primary_ch    = math.ceil(out_ch / ratio)
        ghost_ch      = out_ch - primary_ch

        self.primary = nn.Sequential(
            nn.Conv2d(in_ch, primary_ch, k, s,
                      padding=k // 2, bias=False),
            nn.BatchNorm2d(primary_ch),
        )
        self.ghost = nn.Sequential(
            nn.Conv2d(primary_ch, ghost_ch, dw_k, 1,
                      padding=dw_k // 2,
                      groups=primary_ch, bias=False),
            nn.BatchNorm2d(ghost_ch),
        )
        self.act = nn.ReLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        p   = self.primary(x)
        g   = self.ghost(p)
        out = torch.cat([p, g], dim=1)[:, :self.out_ch]
        return self.act(out)


# ============================================================
# RepDWConv  (giữ nguyên từ v4)
# ============================================================

class RepDWConv(nn.Module):
    """
    RepVGG-style Depthwise Conv.

    Train : 3×3 + 1×1 + Identity (BN riêng mỗi nhánh)
    Deploy: fuse → 1 DWConv 3×3 duy nhất sau reparameterize()
    """

    def __init__(self, channels, stride=1):
        super().__init__()

        self.channels = channels
        self.stride   = stride

        self.dw3  = nn.Conv2d(channels, channels, 3,
                              stride=stride, padding=1,
                              groups=channels, bias=False)
        self.bn3  = nn.BatchNorm2d(channels)

        self.dw1  = nn.Conv2d(channels, channels, 1,
                              stride=stride, padding=0,
                              groups=channels, bias=False)
        self.bn1  = nn.BatchNorm2d(channels)

        self.bn_id = (
            nn.BatchNorm2d(channels) if stride == 1 else None
        )

        self.act    = nn.ReLU(inplace=True)
        self._fused: Optional[nn.Conv2d] = None

    def forward(self, x):
        if self._fused is not None:
            return self.act(self._fused(x))

        out = self.bn3(self.dw3(x)) + self.bn1(self.dw1(x))
        if self.bn_id is not None:
            out = out + self.bn_id(x)
        return self.act(out)

    def reparameterize(self):
        if self._fused is not None:
            return

        k3, b3   = self._fuse_branch(self.dw3, self.bn3)
        k1, b1   = self._fuse_1x1(self.dw1, self.bn1)
        kid, bid = self._fuse_identity()

        fused = nn.Conv2d(
            self.channels, self.channels, 3,
            stride=self.stride, padding=1,
            groups=self.channels, bias=True
        )
        fused.weight.data = k3 + k1 + kid
        fused.bias.data   = b3 + b1 + bid
        self._fused = fused

        del self.dw3, self.bn3, self.dw1, self.bn1
        if self.bn_id is not None:
            del self.bn_id

    def _fuse_branch(self, conv, bn):
        k    = conv.weight
        std  = (bn.running_var + bn.eps).sqrt()
        t    = (bn.weight / std).reshape(-1, 1, 1, 1)
        return k * t, bn.bias - bn.running_mean * bn.weight / std

    def _fuse_1x1(self, conv, bn):
        k3 = F.pad(conv.weight, [1, 1, 1, 1])
        return self._fuse_branch(
            _KernelProxy(k3, conv.stride, conv.padding, conv.groups), bn
        )

    def _fuse_identity(self):
        if self.bn_id is None:
            dev   = self.dw3.weight.device if hasattr(self, 'dw3') \
                    else self._fused.weight.device
            zeros = torch.zeros(self.channels, 1, 3, 3, device=dev)
            return zeros, zeros.squeeze()

        k        = torch.zeros(self.channels, 1, 3, 3,
                               device=self.bn_id.weight.device)
        k[:, 0, 1, 1] = 1.0
        bn       = self.bn_id
        std      = (bn.running_var + bn.eps).sqrt()
        t        = (bn.weight / std).reshape(-1, 1, 1, 1)
        return k * t, bn.bias - bn.running_mean * bn.weight / std


class _KernelProxy:
    def __init__(self, weight, stride, padding, groups):
        self.weight  = weight
        self.stride  = stride
        self.padding = padding
        self.groups  = groups


class _StaticDWConv(nn.Module):
    """Ablation A1/A5: fixed 3×3 DWConv."""
    def __init__(self, channels):
        super().__init__()
        self.dw = _cbr(channels, channels, k=3, g=channels)

    def forward(self, x):
        return self.dw(x)


# ============================================================
# TinyTokenMixer  (giữ nguyên từ v4)
# ============================================================

class TinyTokenMixer(nn.Module):
    """MetaFormer-style lightweight token mixer."""

    def __init__(self, channels, num_heads=4):
        super().__init__()

        hidden = max(channels // 2, 8)

        self.norm   = nn.GroupNorm(1, channels)

        self.dwconv = nn.Conv2d(
            channels, channels, kernel_size=5,
            stride=1, padding=2, groups=channels, bias=False
        )
        self.dwbn   = nn.BatchNorm2d(channels)

        self.pw1    = nn.Conv2d(channels, hidden, 1, bias=False)
        self.pw1bn  = nn.BatchNorm2d(hidden)

        self.act    = nn.GELU()

        self.pw2    = nn.Conv2d(hidden, channels, 1, bias=False)
        self.pw2bn  = nn.BatchNorm2d(channels)

        self.gate   = nn.Sequential(
            _GAPFull(),
            nn.Conv2d(channels, hidden, 1, bias=True),
            nn.GELU(),
            nn.Conv2d(hidden, channels, 1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.norm(x)
        x = self.act(self.dwbn(self.dwconv(x)))
        x = self.act(self.pw1bn(self.pw1(x)))
        x = self.pw2bn(self.pw2(x))
        return x * self.gate(x)


# ============================================================
# Hybrid Block  [v5: CBAM thay SE, RepDW giữ nguyên]
# ============================================================

class EdgeJuFormerBlock(nn.Module):

    def __init__(
        self,
        in_ch,
        out_ch,
        stride=1,
        expand_ratio=4,
        use_token=True,
        use_attn=False,         # CBAM chỉ bật ở stage3
        drop_path=0.0,
        ablation_mode: str | None = None,
    ):
        super().__init__()

        hidden        = in_ch * expand_ratio
        self.use_res  = (stride == 1 and in_ch == out_ch)

        # expansion
        self.expand   = _cbr(in_ch, hidden, k=1)

        # RepDWConv / StaticDWConv
        use_static_dw = ablation_mode in ("no_dyndw", "baseline")
        self.dynamic_dw = (
            _StaticDWConv(hidden) if use_static_dw
            else RepDWConv(hidden, stride=1)
        )

        # downsample
        self.downsample = (
            _cbr(hidden, hidden, k=3, s=2, g=hidden)
            if stride == 2 else nn.Identity()
        )

        # TinyTokenMixer
        disable_token = ablation_mode in ("no_token", "cnn_only", "baseline")
        self.token = (
            nn.Identity()
            if (not use_token or disable_token)
            else TinyTokenMixer(hidden)
        )

        # projection
        self.project = nn.Sequential(
            nn.Conv2d(hidden, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch)
        )

        # [v5] CBAM thay SE — chỉ stage3, tắt khi ablation no_ca
        disable_attn = ablation_mode in ("no_ca", "cnn_only", "baseline")
        if disable_attn or not use_attn:
            self.attn = _IdentityAttn(out_ch)
        else:
            self.attn = CBAMBlock(out_ch, reduction=4, spatial_k=7)

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
        x = self.attn(x)           # CBAM (hoặc Identity)

        if self.use_res:
            x = self.drop(x) + identity

        return x


# ============================================================
# Model  [v5]
# ============================================================

class EdgeJuFormer(nn.Module):

    def __init__(
        self,
        num_classes=4,
        width_mult=1.0,
        drop_path=0.1,
        ablation_mode: str | None = None,
    ):
        super().__init__()

        c  = lambda x: _round_ch(x * width_mult)
        c1 = c(16); c2 = c(24); c3 = c(40)
        c4 = c(64); c5 = c(96)

        # stem
        self.stem = _cbr(3, c1, k=3, s=2)

        # stage1/2: CNN-only, no attention
        self.stage1 = EdgeJuFormerBlock(
            c1, c2, stride=2,
            use_token=False, use_attn=False,
            ablation_mode=ablation_mode
        )
        self.stage2 = EdgeJuFormerBlock(
            c2, c3, stride=2,
            use_token=False, use_attn=False,
            ablation_mode=ablation_mode
        )

        # stage3: TokenMixer + CBAM
        self.stage3 = nn.Sequential(
            EdgeJuFormerBlock(
                c3, c4, stride=2,
                use_token=True, use_attn=True,
                ablation_mode=ablation_mode
            ),
            EdgeJuFormerBlock(
                c4, c4, stride=1,
                use_token=True, use_attn=True,
                drop_path=drop_path,
                ablation_mode=ablation_mode
            )
        )

        # head
        self.head       = _cbr(c4, c5, k=1)
        self.pool       = nn.AdaptiveAvgPool2d(1)
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
        """Fuse RepDWConv branches → 1 conv. Gọi trước benchmark/deploy."""
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
        x = self.stem(x);   feats["stem"]   = x
        x = self.stage1(x); feats["stage1"] = x
        x = self.stage2(x); feats["stage2"] = x
        x = self.stage3(x); feats["stage3"] = x
        x = self.head(x);   feats["head"]   = x
        return feats

    # --------------------------------------------------------

    def __repr__(self):
        p = self.count_parameters()
        return f"EdgeJuFormer(params={p:,}, size={p * 4 / 1024:.1f} KB)"


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

_ABLATION_MODE_MAP = {
    "no_dyndw": "no_dyndw",
    "no_token": "no_token",
    "no_ca":    "no_ca",
    "cnn_only": "cnn_only",
    "baseline": "baseline",
    None:       None,
}


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
        ablation_mode=_ABLATION_MODE_MAP[cfg.get("ablation")],
    )

    if in_channels != 3:
        old = model.stem[0]
        model.stem[0] = nn.Conv2d(
            in_channels, old.out_channels,
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

    print(f"{'Variant':<16} {'Params':>10}  Output")
    print("-" * 45)

    for name in MODEL_CONFIGS:
        m = build_model(name, num_classes=6)
        m.eval()
        with torch.no_grad():
            y = m(x)
        print(f"{name:<16} {m.count_parameters():>10,}  {tuple(y.shape)}")

    print()
    print("After reparameterize():")
    m = build_model("juformer", num_classes=6)
    m.eval()
    m.reparameterize()
    with torch.no_grad():
        y = m(x)
    print(f"  params: {m.count_parameters():,}  output: {tuple(y.shape)}")
