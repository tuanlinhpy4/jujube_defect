"""
EdgeJuFormer v4 — SE + RepVGG DW
----------------------------------

Cải tiến kiến trúc từ A3 (best ablation, F1=84.44%, lat=6.88ms):

    1. SE block (Squeeze-and-Excitation) thay CoordAttention
       - Chỉ dùng ở stage3 (giống A3 bỏ CA hoàn toàn)
       - GAP → FC(r=4) → ReLU → FC → Sigmoid — nhẹ hơn CA ~0.2ms
       - Channel recalibration phù hợp jujube: phân biệt qua màu/texture

    2. RepVGG-style DW (RepDWConv) thay DynamicDWConv
       - Train: 3×3 + 1×1 DWConv song song (multi-branch)
       - Inference: fuse thành 1 conv duy nhất → zero overhead
       - Gọi model.reparameterize() trước khi deploy/benchmark
sa
    Ghost Module đã bỏ: giảm params 320k→254k làm F1 drop trên dataset nhỏ.

Ablation variants (giữ nguyên):
    juformer_a1 → juformer_a5

Chạy:
    python train.py --dataset jujube_ours --model juformer --seed 1
"""

from __future__ import annotations

import math
import copy
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Registry metadata
# ============================================================

ARCH_NAME = "juformer"

MODEL_CONFIGS = {
    # ── v4 full model ────────────────────────────────────────
    "juformer":    {"input_size": 224, "width_mult": 1.00,
                    "ablation": None},
    "juformer_s":  {"input_size": 224, "width_mult": 0.75,
                    "ablation": None},
    "juformer_xs": {"input_size": 224, "width_mult": 0.50,
                    "ablation": None},
 # ── giữ lại bản full (có SE) để so sánh, đổi tên tránh trùng ──
    "juformer_full_se": {"input_size": 224, "width_mult": 1.00, "ablation": None},

    # ── sub-ablation của main model (ablation của ablation) ──
    "juformer_sub_no_token": {"input_size": 224, "width_mult": 1.00, "ablation": "cnn_only"},        # bỏ Token, giữ RepDW
    "juformer_sub_no_dyndw": {"input_size": 224, "width_mult": 1.00, "ablation": "no_ca_no_dyndw"},  # bỏ RepDW, giữ Token  ← mới
    "juformer_sub_bare":     {"input_size": 224, "width_mult": 1.00, "ablation": "baseline"},         # bỏ cả hai

    # ── ablation variants (giữ nguyên để so sánh) ────────────
    "juformer_a1": {"input_size": 224, "width_mult": 1.00,
                    "ablation": "no_dyndw"},
    "juformer_a2": {"input_size": 224, "width_mult": 1.00,
                    "ablation": "no_token"},
    "juformer_a3": {"input_size": 224, "width_mult": 1.00,
                    "ablation": "no_ca"},
    "juformer_a4": {"input_size": 224, "width_mult": 1.00,
                    "ablation": "cnn_only"},
    "juformer_a5": {"input_size": 224, "width_mult": 1.00,
                    "ablation": "baseline"},
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

        keep = 1 - self.p
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = torch.bernoulli(
            torch.full(shape, keep, device=x.device)
        )
        return x * mask / keep


# ============================================================
# [NEW v4] SE Block — thay CoordAttention
# ============================================================

class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation channel recalibration.

    Lý do dùng thay CoordAttention:
        - CA dùng pool theo H và W riêng → spatial-aware nhưng nặng hơn
        - SE dùng GAP toàn cục → channel-only, đủ cho jujube vì
          phân biệt class chủ yếu qua màu sắc và texture (channel)
          không phải vị trí không gian
        - Ít params hơn CA: 2×FC thay vì 3×Conv + pool_h + pool_w
        - GPU-friendly: 2 Conv1×1 nhỏ fuse tốt với torch.compile

    reduction=4 thay vì 16 (default) vì channels nhỏ (64ch ở stage3)
    → hidden=16, đủ capacity mà không thêm nhiều params.
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


class _IdentitySE(nn.Module):
    """Ablation: SE replaced with Identity."""
    def __init__(self, channels, reduction=4):
        super().__init__()

    def forward(self, x):
        return x


# ============================================================
# [NEW v4] Ghost Module — thay Conv1×1 trong expand + project
# ============================================================

class GhostModule(nn.Module):
    """
    Ghost Module (Han et al. 2020) thay thế Conv1×1 đắt tiền.

    Ý tưởng:
        - Tạo out_ch//2 channel "thật" bằng Conv1×1 (primary)
        - Tạo out_ch//2 channel "ghost" bằng DWConv3×3 rẻ từ primary
        - Concat → out_ch channels đầy đủ
        - FLOPs giảm ~50% so với Conv1×1 thông thường

    Phù hợp jujube: texture pattern đơn giản (màu, độ ẩm bề mặt)
    → ghost feature (linear transform của primary) vẫn capture đủ.

    ratio=2: mỗi primary channel tạo ra 1 ghost → tổng = 2× primary
    """

    def __init__(self, in_ch, out_ch, k=1, s=1,
                 ratio=2, dw_k=3, act=True):
        super().__init__()

        self.out_ch = out_ch
        primary_ch = math.ceil(out_ch / ratio)
        ghost_ch   = out_ch - primary_ch

        # primary: Conv1×1 (hoặc Conv k×k) thật
        self.primary = nn.Sequential(
            nn.Conv2d(in_ch, primary_ch, k, s,
                      padding=k // 2, bias=False),
            nn.BatchNorm2d(primary_ch),
        )

        # ghost: DWConv rẻ trên primary features
        self.ghost = nn.Sequential(
            nn.Conv2d(primary_ch, ghost_ch, dw_k, 1,
                      padding=dw_k // 2,
                      groups=primary_ch, bias=False),
            nn.BatchNorm2d(ghost_ch),
        )

        self.act = nn.ReLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        p = self.primary(x)
        g = self.ghost(p)
        out = torch.cat([p, g], dim=1)
        # trim nếu lẻ (do ceil)
        out = out[:, :self.out_ch]
        return self.act(out)


def _ghost_cbr(in_ch, out_ch, k=1, s=1):
    """Ghost module với BN + ReLU — drop-in thay _cbr(k=1)."""
    if k == 1:
        return GhostModule(in_ch, out_ch, k=k, s=s, act=True)
    # k>1 (ví dụ stride conv): dùng cbr thường
    return _cbr(in_ch, out_ch, k=k, s=s)


# ============================================================
# [NEW v4] RepDWConv — thay DynamicDWConv
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

    So với DynamicDWConv (softmax gate):
        - DynamicDWConv: tốn 2 DWConv + GAP + 2 Conv gate mỗi forward
        - RepDWConv train: 3 DWConv nhẹ hơn (1×1 rất rẻ)
        - RepDWConv infer: 1 DWConv duy nhất → nhanh hơn đáng kể
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


class _StaticDWConv(nn.Module):
    """Ablation A1/A5: fixed 3×3 DWConv, no dynamic gate."""

    def __init__(self, channels):
        super().__init__()
        self.dw = _cbr(channels, channels, k=3, g=channels)

    def forward(self, x):
        return self.dw(x)


# ============================================================
# TinyTokenMixer  [giữ nguyên từ bản gốc]
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
# Hybrid Block  [v4: Ghost expand/project, RepDW, SE]
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
        ablation_mode: str | None = None,
    ):
        super().__init__()

        hidden = in_ch * expand_ratio
        self.use_res = (stride == 1 and in_ch == out_ch)

        # expansion — plain _cbr (Ghost bị bỏ: giảm capacity trên dataset nhỏ)
        self.expand = _cbr(in_ch, hidden, k=1)

        # ── [v4] RepDWConv / StaticDWConv / DynamicDWConv ─────
        use_static_dw = ablation_mode in ("no_dyndw", "baseline","no_ca_no_dyndw")
        if use_static_dw:
            self.dynamic_dw = _StaticDWConv(hidden)
        else:
            self.dynamic_dw = RepDWConv(hidden, stride=1)

        # downsample (stride conv — không fuse RepVGG)
        if stride == 2:
            self.downsample = _cbr(hidden, hidden, k=3, s=2, g=hidden)
        else:
            self.downsample = nn.Identity()

        # ── TinyTokenMixer vs Identity ─────────────────────────
        disable_token = ablation_mode in ("no_token", "cnn_only", "baseline","no_ca_no_dyndw")
        self.token = (
            nn.Identity()
            if (not use_token or disable_token) else
            TinyTokenMixer(hidden)
        )

        # projection — plain Conv1×1 (Ghost bị bỏ)
        self.project = nn.Sequential(
            nn.Conv2d(hidden, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch)
        )

        # ── [v4] SE block thay CoordAttention ─────────────────
        # SE chỉ dùng ở stage3 (use_se=True), bỏ stage1/2
        disable_ca = ablation_mode in ("no_ca", "cnn_only", "baseline")
        if disable_ca or not use_se:
            self.se = _IdentitySE(out_ch)
        else:
            self.se = SEBlock(out_ch, reduction=4)

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
# Model  [v4]
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
            ablation_mode=ablation_mode
        )

        self.stage2 = EdgeJuFormerBlock(
            c2, c3, stride=2,
            use_token=False, use_se=False,
            ablation_mode=ablation_mode
        )

        # stage3: TokenMixer + SE (chỉ ở đây mới dùng SE)
        self.stage3 = nn.Sequential(
            EdgeJuFormerBlock(
                c3, c4, stride=2,
                use_token=True, use_se=True,
                ablation_mode=ablation_mode
            ),
            EdgeJuFormerBlock(
                c4, c4, stride=1,
                use_token=True, use_se=True,
                drop_path=drop_path,
                ablation_mode=ablation_mode
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

_ABLATION_MODE_MAP = {
    "no_dyndw":  "no_dyndw",
    "no_token":  "no_token",
    "no_ca":     "no_ca",
    "cnn_only":  "cnn_only",
    "baseline":  "baseline",
    None:        None,
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

    ablation_mode = _ABLATION_MODE_MAP[cfg.get("ablation")]

    model = EdgeJuFormer(
        num_classes=num_classes,
        width_mult=cfg["width_mult"],
        ablation_mode=ablation_mode,
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

    print(f"{'Variant':<16} {'Params (train)':>16}  {'Output'}  {'Note'}")
    print("-" * 70)

    for name in MODEL_CONFIGS:
        m = build_model(name, num_classes=5)
        m.eval()
        with torch.no_grad():
            y = m(x)
        p_train = m.count_parameters()
        print(f"{name:<16} {p_train:>16,}  {tuple(y.shape)}  train mode")

    print()
    print("After reparameterize():")
    m = build_model("juformer", num_classes=5)
    m.eval()
    m.reparameterize()
    with torch.no_grad():
        y = m(x)
    p_deploy = m.count_parameters()
    print(f"  params (deploy): {p_deploy:,}  output: {tuple(y.shape)}")
