"""
Jujube-YOLO v1 — DCSE + C3k2_RCM + MBCA (clean, ablation-ready)
------------------------------------------------------------------

Dựa trên: Wang, L. et al. (2025). "Jujube-YOLO: a precise jujube fruit
recognition model in unstructured environments." Expert Systems With
Applications, 291, 128530. https://doi.org/10.1016/j.eswa.2025.128530

Kiến trúc (backbone -> neck -> head, theo Fig. 8 của bài báo):

    1. DCSE (Double Convolution Squeeze-and-Excitation)
       - Thay thế 2 conv stem đầu (layer 0, 1)
       - Conv3x3 -> BN -> SiLU  (x2)  ->  SE gate (GAP -> FC(r=16) -> ReLU -> FC -> Sigmoid)
       - Tăng khả năng bắt các vết nứt nông / cạnh mờ trên vỏ táo tàu

    2. C3k2_RCM (Rectangular self-Calibration Module trong khối C3k2)
       - Thay Bottleneck chuẩn của C3k2 bằng RCMBottleneck (2x RCM residual)
       - RCM: axial pooling (H, W) + large-kernel conv "self-calibrate"
         hình chữ nhật, fuse Hadamard với nhánh local 3x3 DW, rồi BN+MLP+residual
       - Dùng ở backbone layer 2, 4, 6, 8

    3. MBCA (Multi-Branch Channel Attention)
       - 4 nhánh song song (1x1, DW+PW, 3x3, global-pool) -> concat -> shuffle
         -> channel attention (SE-style) -> residual skip
       - Thay conv chuẩn trong neck (layer 17, 20)

Ablation: mọi module trên đều có thể bật/tắt độc lập qua MODEL_CONFIGS,
tái hiện đúng dàn thí nghiệm Table 2 của bài báo (baseline / +DCSE /
+C3k2_RCM / +MBCA / các cặp / full model).

Chạy:
    python jujube_yolo.py
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn


# ============================================================
# Registry metadata
# ============================================================

ARCH_NAME = "jujube_yolo"

MODEL_CONFIGS = {
    # ── kích thước (giống scaling n/s của YOLO) ────────────────
    "jujube_yolo":    {"width_mult": 1.00, "depth_mult": 1.00,
                        "use_dcse": True,  "use_rcm": True,  "use_mbca": True},
    "jujube_yolo_s":  {"width_mult": 0.75, "depth_mult": 0.67,
                        "use_dcse": True,  "use_rcm": True,  "use_mbca": True},
    "jujube_yolo_n":  {"width_mult": 0.50, "depth_mult": 0.34,
                        "use_dcse": True,  "use_rcm": True,  "use_mbca": True},

    # ── ablation, tái hiện Table 2 của bài báo (baseline = YOLOv11n) ─
    "baseline":            {"width_mult": 1.0, "depth_mult": 1.0,
                             "use_dcse": False, "use_rcm": False, "use_mbca": False},
    "only_dcse":           {"width_mult": 1.0, "depth_mult": 1.0,
                             "use_dcse": True,  "use_rcm": False, "use_mbca": False},
    "only_rcm":            {"width_mult": 1.0, "depth_mult": 1.0,
                             "use_dcse": False, "use_rcm": True,  "use_mbca": False},
    "only_mbca":           {"width_mult": 1.0, "depth_mult": 1.0,
                             "use_dcse": False, "use_rcm": False, "use_mbca": True},
    "dcse_rcm":            {"width_mult": 1.0, "depth_mult": 1.0,
                             "use_dcse": True,  "use_rcm": True,  "use_mbca": False},
    "dcse_mbca":           {"width_mult": 1.0, "depth_mult": 1.0,
                             "use_dcse": True,  "use_rcm": False, "use_mbca": True},
    "rcm_mbca":            {"width_mult": 1.0, "depth_mult": 1.0,
                             "use_dcse": False, "use_rcm": True,  "use_mbca": True},
    "full":                {"width_mult": 1.0, "depth_mult": 1.0,
                             "use_dcse": True,  "use_rcm": True,  "use_mbca": True},
}

DEFAULT_VARIANT = "jujube_yolo"


# ============================================================
# Utility
# ============================================================

def _round_ch(c: float, divisor: int = 8):
    return int(math.ceil(c / divisor) * divisor)


def _round_depth(n: int, depth_mult: float):
    return max(1, round(n * depth_mult))


def autopad(k, p=None, d=1):
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class Conv(nn.Module):
    """Conv2d -> BatchNorm2d -> SiLU (khối conv chuẩn của YOLO)."""

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


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
        mask = torch.bernoulli(torch.full(shape, keep, device=x.device))
        return x * mask / keep


# ============================================================
# 1. DCSE — Double Convolution Squeeze-and-Excitation (Table 1 / Eq.1)
# ============================================================

class DCSE(nn.Module):
    """
    Double Convolution Squeeze-and-Excitation module (thay thế stem conv).

    Conv3x3-BN-SiLU -> Conv3x3-BN-SiLU -> SE gate (GAP->FC(r)->ReLU->FC->Sigmoid)
    Output = features * gate                      (Eq. 1 của bài báo)

    reduction=16 mặc định theo bài báo (Table 1: C2 -> C2/16 -> C2).
    """

    def __init__(self, c1, c2, s=2, reduction=16):
        super().__init__()
        self.conv1 = nn.Conv2d(c1, c2, kernel_size=3, stride=s, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(c2)
        self.act1 = nn.SiLU()

        self.conv2 = nn.Conv2d(c2, c2, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(c2)
        self.act2 = nn.SiLU()

        hidden = max(1, c2 // reduction)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(c2, hidden, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(hidden, c2, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.act1(self.bn1(self.conv1(x)))
        x = self.act2(self.bn2(self.conv2(x)))
        gate = self.sigmoid(self.fc2(self.relu(self.fc1(self.gap(x)))))
        return x * gate


# ============================================================
# 2. RCM — Rectangular self-Calibration Module (Fig.5 / Eq.2)
# ============================================================

class RCA(nn.Module):
    """
    Rectangular self-Calibration Attention:
    DWConv -> axial pooling (H, W) -> large-kernel rescale -> broadcasting add
    -> Hadamard fuse với nhánh local 3x3 DW -> Conv-BN-ReLU-Conv-Sigmoid.
    """

    def __init__(self, c, k_large=11):
        super().__init__()
        self.dwconv = nn.Conv2d(c, c, kernel_size=3, padding=1, groups=c, bias=False)

        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        pad = k_large // 2
        self.large_h = nn.Conv2d(c, c, kernel_size=(k_large, 1), padding=(pad, 0), groups=c, bias=False)
        self.large_w = nn.Conv2d(c, c, kernel_size=(1, k_large), padding=(0, pad), groups=c, bias=False)

        self.local = nn.Conv2d(c, c, kernel_size=3, padding=1, groups=c, bias=False)

        self.conv1 = nn.Conv2d(c, c, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(c)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(c, c, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        feat = self.dwconv(x)

        p_h = self.large_h(self.pool_h(feat))
        p_w = self.large_w(self.pool_w(feat))
        rect = p_h + p_w                       # broadcasting addition -> (B,C,H,W)

        fused = rect * self.local(feat)        # Hadamard fuse (local detail)

        attn = self.relu(self.bn(self.conv1(fused)))
        return self.sigmoid(self.conv2(attn))


class RCM(nn.Module):
    """Fout = rho( xiF(xiC(P_H(x) (+) P_V(x))) (x) x ) + x   -- Eq. (2)."""

    def __init__(self, c, mlp_ratio=2):
        super().__init__()
        self.rca = RCA(c)
        self.bn = nn.BatchNorm2d(c)
        hidden = c * mlp_ratio
        self.mlp = nn.Sequential(
            nn.Conv2d(c, hidden, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.GELU(),
            nn.Conv2d(hidden, c, kernel_size=1, bias=False),
        )

    def forward(self, x):
        gated = self.bn(x * self.rca(x))
        return self.mlp(gated) + x


class RCMBottleneck(nn.Module):
    """2 khối RCM residual liên tiếp (Fig. 6, phải)."""

    def __init__(self, c):
        super().__init__()
        self.rcm1 = RCM(c)
        self.rcm2 = RCM(c)

    def forward(self, x):
        return x + self.rcm2(self.rcm1(x))


class PlainBottleneck(nn.Module):
    """Bottleneck chuẩn (Conv3x3-Conv3x3 + residual) — dùng khi ablation use_rcm=False."""

    def __init__(self, c):
        super().__init__()
        self.cv1 = Conv(c, c, k=3)
        self.cv2 = Conv(c, c, k=3)

    def forward(self, x):
        return x + self.cv2(self.cv1(x))


class C3k2(nn.Module):
    """
    Khối CSP-style C3k2 của YOLOv11, với inner-bottleneck có thể chọn
    RCMBottleneck (use_rcm=True, -> C3k2_RCM) hoặc PlainBottleneck (ablation).
    """

    def __init__(self, c1, c2, n=1, e=0.5, use_rcm=True):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, 2 * c_, k=1)
        self.cv2 = Conv((2 + n) * c_, c2, k=1)
        block = (lambda: RCMBottleneck(c_)) if use_rcm else (lambda: PlainBottleneck(c_))
        self.m = nn.ModuleList(block() for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        for m in self.m:
            y.append(m(y[-1]))
        return self.cv2(torch.cat(y, 1))


# ============================================================
# 3. MBCA — Multi-Branch Channel Attention (Fig.7 / Eq.3-5)
# ============================================================

class MBCA(nn.Module):
    """
    4 nhánh song song (1x1 / DW+PW / 3x3 / global-pool) -> concat -> channel
    shuffle -> channel attention (SE-style) -> residual skip connection.
    """

    def __init__(self, c1, c2, k=3, reduction=8, groups_shuffle=4):
        super().__init__()
        assert c2 % 4 == 0, "MBCA: c2 phải chia hết cho 4 (1 nhánh / phần tư)"
        c_b = c2 // 4

        self.branch_1x1 = Conv(c1, c_b, k=1)

        self.branch_dw = nn.Sequential(
            nn.Conv2d(c1, c1, kernel_size=k, padding=k // 2, groups=c1, bias=False),
            nn.BatchNorm2d(c1),
            nn.SiLU(),
            Conv(c1, c_b, k=1),
        )

        self.branch_3x3 = Conv(c1, c_b, k=3)

        self.branch_global = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c1, c_b, kernel_size=1, bias=False),
        )

        self.groups_shuffle = groups_shuffle if c2 % groups_shuffle == 0 else 1

        hidden = max(1, c2 // reduction)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(c2, hidden, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(hidden, c2, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

        self.skip = Conv(c1, c2, k=1) if c1 != c2 else nn.Identity()
        self.act_out = nn.SiLU()

    @staticmethod
    def _channel_shuffle(x, groups):
        b, c, h, w = x.shape
        if c % groups != 0:
            return x
        x = x.view(b, groups, c // groups, h, w)
        x = x.transpose(1, 2).contiguous()
        return x.view(b, c, h, w)

    def forward(self, x):
        b1 = self.branch_1x1(x)
        b2 = self.branch_dw(x)
        b3 = self.branch_3x3(x)
        b4 = self.branch_global(x).expand(-1, -1, x.shape[2], x.shape[3])

        feat = torch.cat([b1, b2, b3, b4], dim=1)
        feat = self._channel_shuffle(feat, self.groups_shuffle)

        gate = self.sigmoid(self.fc2(self.relu(self.fc1(self.gap(feat)))))
        feat = feat * gate

        return self.act_out(feat + self.skip(x))


# ============================================================
# SPPF (khối chuẩn YOLO, không đổi so với YOLOv11)
# ============================================================

class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast."""

    def __init__(self, c1, c2, k=5):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, k=1)
        self.cv2 = Conv(c_ * 4, c2, k=1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat([x, y1, y2, self.m(y2)], dim=1))


# ============================================================
# Backbone + Neck (theo sơ đồ Fig. 8 của bài báo, rút gọn)
# ============================================================

class JujubeYOLOBackboneNeck(nn.Module):
    """
    Backbone (stem DCSE x2 -> C3k2_RCM x4 -> SPPF) + Neck FPN/PAN
    (upsample/concat -> C3k2 -> MBCA), trả về 3 feature map P3/P4/P5
    cho detection head.
    """

    def __init__(self, width_mult=1.0, depth_mult=1.0,
                 use_dcse=True, use_rcm=True, use_mbca=True):
        super().__init__()

        c = lambda x: _round_ch(x * width_mult)
        n = lambda x: _round_depth(x, depth_mult)

        c1, c2, c3, c4, c5 = c(64), c(128), c(256), c(512), c(1024)

        stem_block = DCSE if use_dcse else (lambda c1, c2, s: Conv(c1, c2, k=3, s=s))
        self.stem1 = stem_block(3, c1, s=2)     # 0  P1/2
        self.stem2 = stem_block(c1, c2, s=2)    # 1  P2/4

        self.stage_p3 = C3k2(c2, c3, n=n(2), e=0.25, use_rcm=use_rcm)   # 2
        self.down_p3 = Conv(c3, c3, k=3, s=2)                            # 3  P3/8
        self.stage_p4 = C3k2(c3, c4, n=n(2), e=0.25, use_rcm=use_rcm)   # 4
        self.down_p4 = Conv(c4, c4, k=3, s=2)                            # 5  P4/16
        self.stage_p4b = C3k2(c4, c4, n=n(2), e=0.5, use_rcm=use_rcm)   # 6
        self.down_p5 = Conv(c4, c5, k=3, s=2)                            # 7  P5/32
        self.stage_p5 = C3k2(c5, c5, n=n(2), e=0.5, use_rcm=use_rcm)    # 8
        self.sppf = SPPF(c5, c5)                                         # 9  (thay cho C2PSA, giữ đơn giản)

        # ---- neck (FPN top-down + PAN bottom-up) ----
        self.up1 = nn.Upsample(scale_factor=2, mode="nearest")
        self.c3k2_n1 = C3k2(c5 + c4, c4, n=n(2), e=0.5, use_rcm=False)   # 13
        self.up2 = nn.Upsample(scale_factor=2, mode="nearest")
        self.c3k2_n2 = C3k2(c4 + c3, c3, n=n(2), e=0.5, use_rcm=False)   # 16 -> P3 out

        fuse_block = MBCA if use_mbca else (lambda c1, c2: Conv(c1, c2, k=3))
        self.fuse_p3 = fuse_block(c3, c3)                                 # 17

        self.down_n1 = Conv(c3, c3, k=3, s=2)
        self.c3k2_n3 = C3k2(c3 + c4, c4, n=n(2), e=0.5, use_rcm=False)   # 19 -> P4 out
        self.fuse_p4 = fuse_block(c4, c4)                                 # 20

        self.down_n2 = Conv(c4, c4, k=3, s=2)
        self.c3k2_n4 = C3k2(c4 + c5, c5, n=n(2), e=0.5, use_rcm=use_rcm)  # 22 -> P5 out

        self.out_channels = (c3, c4, c5)

    def forward(self, x):
        x = self.stem1(x)
        x = self.stem2(x)

        p3 = self.stage_p3(x)
        x = self.down_p3(p3)
        p4 = self.stage_p4(x)
        x = self.down_p4(p4)
        p4 = self.stage_p4b(x)
        x = self.down_p5(p4)
        p5 = self.stage_p5(x)
        p5 = self.sppf(p5)

        # top-down (FPN)
        u1 = self.up1(p5)
        n1 = self.c3k2_n1(torch.cat([u1, p4], dim=1))
        u2 = self.up2(n1)
        n2 = self.c3k2_n2(torch.cat([u2, p3], dim=1))     # P3-out (before/without extra fuse)
        p3_out = self.fuse_p3(n2)

        # bottom-up (PAN)
        d1 = self.down_n1(n2)
        n3 = self.c3k2_n3(torch.cat([d1, n1], dim=1))
        p4_out = self.fuse_p4(n3)

        d2 = self.down_n2(n3)
        p5_out = self.c3k2_n4(torch.cat([d2, p5], dim=1))

        return p3_out, p4_out, p5_out


# ============================================================
# Detection Head (decoupled, đơn giản hoá — cls + box per scale)
# ============================================================

class DetectHead(nn.Module):
    """Head tách nhánh cls/box tối giản, áp dụng cho cả 3 scale P3/P4/P5."""

    def __init__(self, channels, nc=2, reg_max=16):
        super().__init__()
        self.nc = nc
        self.reg_max = reg_max
        self.cls_branches = nn.ModuleList()
        self.box_branches = nn.ModuleList()
        for ch in channels:
            self.cls_branches.append(nn.Sequential(
                Conv(ch, ch, k=3), Conv(ch, ch, k=3), nn.Conv2d(ch, nc, 1)
            ))
            self.box_branches.append(nn.Sequential(
                Conv(ch, ch, k=3), Conv(ch, ch, k=3), nn.Conv2d(ch, 4 * reg_max, 1)
            ))

    def forward(self, feats):
        outs = []
        for feat, cls_b, box_b in zip(feats, self.cls_branches, self.box_branches):
            outs.append(torch.cat([box_b(feat), cls_b(feat)], dim=1))
        return outs  # list các tensor (B, 4*reg_max + nc, H, W) cho P3/P4/P5


# ============================================================
# Model
# ============================================================

class JujubeYOLO(nn.Module):

    def __init__(
        self,
        nc: int = 2,
        width_mult: float = 1.0,
        depth_mult: float = 1.0,
        use_dcse: bool = True,
        use_rcm: bool = True,
        use_mbca: bool = True,
    ):
        super().__init__()

        self.backbone_neck = JujubeYOLOBackboneNeck(
            width_mult=width_mult, depth_mult=depth_mult,
            use_dcse=use_dcse, use_rcm=use_rcm, use_mbca=use_mbca,
        )
        self.head = DetectHead(self.backbone_neck.out_channels, nc=nc)

        self.nc = nc
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        feats = self.backbone_neck(x)
        return self.head(feats)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self):
        p = self.count_parameters()
        return f"JujubeYOLO(params={p:,}, size={p * 4 / 1024 / 1024:.2f} MB)"


# ============================================================
# Factory
# ============================================================

def jujube_yolo(nc=2):
    return JujubeYOLO(nc=nc, width_mult=1.0, depth_mult=1.0)


def jujube_yolo_s(nc=2):
    return JujubeYOLO(nc=nc, width_mult=0.75, depth_mult=0.67)


def jujube_yolo_n(nc=2):
    return JujubeYOLO(nc=nc, width_mult=0.50, depth_mult=0.34)


# ============================================================
# build_model  (bắt buộc bởi model registry)
# ============================================================

def build_model(name: str = "jujube_yolo", nc: int = 2, **kwargs):

    cfg = MODEL_CONFIGS.get(name)
    if cfg is None:
        raise ValueError(f"Unknown variant '{name}'. Choose from: {list(MODEL_CONFIGS)}")

    model = JujubeYOLO(
        nc=nc,
        width_mult=cfg["width_mult"],
        depth_mult=cfg["depth_mult"],
        use_dcse=cfg["use_dcse"],
        use_rcm=cfg["use_rcm"],
        use_mbca=cfg["use_mbca"],
    )
    model.variant_name = name
    return model


# ============================================================
# Quick self-test
# ============================================================

if __name__ == "__main__":

    x = torch.randn(1, 3, 640, 640)

    print(f"{'Variant':<14} {'Params':>12}  {'Size(MB)':>9}  {'P3':<16}{'P4':<16}{'P5':<16}")
    print("-" * 90)

    for name in MODEL_CONFIGS:
        m = build_model(name, nc=2)
        m.eval()
        with torch.no_grad():
            outs = m(x)
        p = m.count_parameters()
        shapes = "".join(f"{str(tuple(o.shape)):<16}" for o in outs)
        print(f"{name:<14} {p:>12,}  {p * 4 / 1024 / 1024:>9.2f}  {shapes}")
