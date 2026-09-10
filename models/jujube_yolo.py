"""
Jujube-YOLO v2 — DCSE + C3k2_RCM + MBCA (classification variant, fixed)
------------------------------------------------------------------------

Dựa trên: Wang, L. et al. (2025). "Jujube-YOLO: a precise jujube fruit
recognition model in unstructured environments." Expert Systems With
Applications, 291, 128530. https://doi.org/10.1016/j.eswa.2025.128530

Thay đổi so với v1
-------------------
1. FIX BUG: v1 dùng nhầm feature map ở P2/4 (stride 4) làm skip-connection
   cho nhánh upsample thứ 2 trong neck, trong khi nhánh đó đang ở P3/8
   (stride 8) -> lệch kích thước -> RuntimeError khi concat. v2 đặt lại tên
   biến rõ ràng theo đúng 5 tầng backbone của Fig. 8 (P2/4, P3/8, P4/16,
   P5/32) và nối đúng tầng.
2. CHUYỂN THÀNH CLASSIFIER: bản gốc trong paper là detector (trả về 3
   feature map box+cls). Để dùng trực tiếp với harness classification
   (train.py: cross-entropy, ảnh 224x224, num_classes=N) giống format
   EdgeJuFormer, forward() ở đây trả về logits (B, num_classes) bằng cách
   GAP từng scale P3/P4/P5 rồi concat -> Linear. Việc dùng cả 3 scale (thay
   vì chỉ P5) giúp giữ lại thông tin nứt/mốc/sâu ở mức chi tiết nông (P3)
   lẫn ngữ nghĩa sâu (P5) — phù hợp bài toán phân loại chất lượng quả.
   DetectHead (đầu dò object detection gốc) vẫn được giữ lại trong file,
   không dùng mặc định, cho ai cần bài toán detection thật sự.

Kiến trúc module (không đổi so với v1):
    1. DCSE  — stem, thay 2 conv đầu (layer 0, 1)
    2. C3k2_RCM — thay C3k2 ở backbone (layer 2, 4, 6, 8)
    3. MBCA  — thay conv fusion ở neck (layer 17, 20)

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

    # ── speed-oriented (chap nhan giam accuracy de doi lay FPS) ─
    # Tat RCM (module nang nhat: DW-conv kernel 11x1/1x11 x2, MLP x2 trong
    # moi bottleneck) + giam width/depth. Giu DCSE (re, van co loi).
    "jujube_yolo_fast":    {"width_mult": 0.75, "depth_mult": 0.67,
                             "use_dcse": True,  "use_rcm": False, "use_mbca": True},
    # Tat ca RCM + MBCA, width/depth nho nhat -> nhanh nhat trong cac
    # option co san, chap nhan hy sinh accuracy nhieu nhat.
    "jujube_yolo_lite":    {"width_mult": 0.50, "depth_mult": 0.34,
                             "use_dcse": True,  "use_rcm": False, "use_mbca": False},
    # Toi da toc do, khong quan tam accuracy: tat CA 3 module dac thu
    # (DCSE -> plain conv, RCM -> plain bottleneck, MBCA -> plain conv),
    # width/depth toi thieu. Nen ket hop them: giam input resolution
    # (vd 96-128px), fp16, torch.compile khi inference (xem README).
    "jujube_yolo_ultra":   {"width_mult": 0.25, "depth_mult": 0.25,
                             "use_dcse": False, "use_rcm": False, "use_mbca": False},
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
# Deterministic global-average-pool helpers
# ------------------------------------------------------------
# nn.AdaptiveAvgPool2d has no deterministic CUDA backward, which crashes
# training when torch.use_deterministic_algorithms(True) is set (as
# train.py does for seed reproducibility). x.mean(dim=..., keepdim=True)
# computes the exact same values but has a deterministic backward, so we
# use it everywhere instead of nn.AdaptiveAvgPool2d.
# ============================================================

class _GAPFull(nn.Module):
    """Global average pool -> (B, C, 1, 1)."""
    def forward(self, x):
        return x.mean(dim=[2, 3], keepdim=True)


class _GAPh(nn.Module):
    """Average over width only -> (B, C, H, 1)  (replaces AdaptiveAvgPool2d((None,1)))."""
    def forward(self, x):
        return x.mean(dim=3, keepdim=True)


class _GAPw(nn.Module):
    """Average over height only -> (B, C, 1, W)  (replaces AdaptiveAvgPool2d((1,None)))."""
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
        mask = torch.bernoulli(torch.full(shape, keep, device=x.device))
        return x * mask / keep


# ============================================================
# 1. DCSE — Double Convolution Squeeze-and-Excitation (Table 1 / Eq.1)
# ============================================================

class DCSE(nn.Module):
    """Conv3x3-BN-SiLU (x2) -> SE gate (GAP->FC(r=16)->ReLU->FC->Sigmoid)."""

    def __init__(self, c1, c2, s=2, reduction=16):
        super().__init__()
        self.conv1 = nn.Conv2d(c1, c2, kernel_size=3, stride=s, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(c2)
        self.act1 = nn.SiLU()

        self.conv2 = nn.Conv2d(c2, c2, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(c2)
        self.act2 = nn.SiLU()

        hidden = max(1, c2 // reduction)
        self.gap = _GAPFull()
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
    """DWConv -> axial pooling (H,W) -> large-kernel rescale -> broadcasting add
    -> Hadamard fuse với nhánh local 3x3 DW -> Conv-BN-ReLU-Conv-Sigmoid."""

    def __init__(self, c, k_large=11):
        super().__init__()
        self.dwconv = nn.Conv2d(c, c, kernel_size=3, padding=1, groups=c, bias=False)

        self.pool_h = _GAPh()
        self.pool_w = _GAPw()

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
        rect = p_h + p_w

        fused = rect * self.local(feat)

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
    """Khối CSP-style C3k2 của YOLOv11, inner-bottleneck chọn RCMBottleneck
    (use_rcm=True, -> C3k2_RCM) hoặc PlainBottleneck (ablation / neck)."""

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
    """4 nhánh song song (1x1 / DW+PW / 3x3 / global-pool) -> concat -> channel
    shuffle -> channel attention (SE-style) -> residual skip connection."""

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
            _GAPFull(),
            nn.Conv2d(c1, c_b, kernel_size=1, bias=False),
        )

        self.groups_shuffle = groups_shuffle if c2 % groups_shuffle == 0 else 1

        hidden = max(1, c2 // reduction)
        self.gap = _GAPFull()
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
# Backbone + Neck (đúng theo 5 tầng của Fig. 8: stem->P2/4->P3/8->P4/16->P5/32)
# ============================================================

class JujubeYOLOBackboneNeck(nn.Module):
    """
    Backbone (stem DCSE x2 -> C3k2_RCM x4 -> SPPF) + Neck FPN/PAN
    (upsample/concat -> C3k2 -> MBCA), trả về 3 feature map P3/P4/P5.

    QUAN TRỌNG (fix so với v1): mỗi tầng backbone chỉ được lưu MỘT LẦN vào
    một biến duy nhất (feat_p2/feat_p3/feat_p4/feat_p5), không tái sử dụng
    tên biến giữa các tầng khác resolution — đây chính là nguyên nhân lỗi
    "Sizes of tensors must match" ở bản v1 (nối nhầm P2/4 với nhánh đang ở
    P3/8).
    """

    def __init__(self, in_channels=3, width_mult=1.0, depth_mult=1.0,
                 use_dcse=True, use_rcm=True, use_mbca=True):
        super().__init__()

        c = lambda x: _round_ch(x * width_mult)
        n = lambda x: _round_depth(x, depth_mult)

        ch1, ch2, ch3, ch4, ch5 = c(64), c(128), c(256), c(512), c(1024)

        stem_block = DCSE if use_dcse else (lambda c1, c2, s: Conv(c1, c2, k=3, s=s))
        self.stem1 = stem_block(in_channels, ch1, s=2)   # 0            -> P1/2
        self.stem2 = stem_block(ch1, ch2, s=2)            # 1            -> P2/4

        # ---- backbone (4 stage C3k2_RCM, theo layer 2/4/6/8 của Fig.8) ----
        self.stage_p2 = C3k2(ch2, ch3, n=n(2), e=0.25, use_rcm=use_rcm)   # 2   @ P2/4
        self.down_p3 = Conv(ch3, ch3, k=3, s=2)                            # 3   -> P3/8
        self.stage_p3 = C3k2(ch3, ch4, n=n(2), e=0.25, use_rcm=use_rcm)   # 4   @ P3/8  (skip nông)
        self.down_p4 = Conv(ch4, ch4, k=3, s=2)                            # 5   -> P4/16
        self.stage_p4 = C3k2(ch4, ch4, n=n(2), e=0.5, use_rcm=use_rcm)    # 6   @ P4/16 (skip sâu)
        self.down_p5 = Conv(ch4, ch5, k=3, s=2)                            # 7   -> P5/32
        self.stage_p5 = C3k2(ch5, ch5, n=n(2), e=0.5, use_rcm=use_rcm)    # 8   @ P5/32
        self.sppf = SPPF(ch5, ch5)                                         # 9

        # ---- neck (FPN top-down + PAN bottom-up); C3k2 ở neck là plain (use_rcm=False), giống Fig.8 ----
        self.up1 = nn.Upsample(scale_factor=2, mode="nearest")
        self.c3k2_n1 = C3k2(ch5 + ch4, ch4, n=n(2), e=0.5, use_rcm=False)   # 12/13 @ P4/16
        self.up2 = nn.Upsample(scale_factor=2, mode="nearest")
        self.c3k2_n2 = C3k2(ch4 + ch4, ch3, n=n(2), e=0.5, use_rcm=False)   # 15/16 @ P3/8 -> P3 out

        fuse_block = MBCA if use_mbca else (lambda c1, c2: Conv(c1, c2, k=3))
        self.fuse_p3 = fuse_block(ch3, ch3)                                  # 17

        self.down_n1 = Conv(ch3, ch3, k=3, s=2)
        self.c3k2_n3 = C3k2(ch3 + ch4, ch4, n=n(2), e=0.5, use_rcm=False)   # 18/19 @ P4/16 -> P4 out
        self.fuse_p4 = fuse_block(ch4, ch4)                                  # 20

        self.down_n2 = Conv(ch4, ch4, k=3, s=2)
        self.c3k2_n4 = C3k2(ch4 + ch5, ch5, n=n(2), e=0.5, use_rcm=False)   # 21/22 @ P5/32 -> P5 out

        self.out_channels = (ch3, ch4, ch5)

    def forward(self, x):
        x = self.stem1(x)
        x = self.stem2(x)                    # P2/4

        feat_p2 = self.stage_p2(x)           # P2/4
        x = self.down_p3(feat_p2)            # -> P3/8
        feat_p3 = self.stage_p3(x)           # P3/8   (skip nông, dùng ở neck)
        x = self.down_p4(feat_p3)            # -> P4/16
        feat_p4 = self.stage_p4(x)           # P4/16  (skip sâu, dùng ở neck)
        x = self.down_p5(feat_p4)            # -> P5/32
        feat_p5 = self.sppf(self.stage_p5(x))  # P5/32

        # top-down (FPN)
        u1 = self.up1(feat_p5)                                    # -> P4/16
        n1 = self.c3k2_n1(torch.cat([u1, feat_p4], dim=1))        # concat @ P4/16 (khớp kích thước)
        u2 = self.up2(n1)                                         # -> P3/8
        n2 = self.c3k2_n2(torch.cat([u2, feat_p3], dim=1))        # concat @ P3/8  (khớp kích thước)
        p3_out = self.fuse_p3(n2)

        # bottom-up (PAN)
        d1 = self.down_n1(p3_out)                                 # -> P4/16
        n3 = self.c3k2_n3(torch.cat([d1, n1], dim=1))
        p4_out = self.fuse_p4(n3)

        d2 = self.down_n2(p4_out)                                 # -> P5/32
        p5_out = self.c3k2_n4(torch.cat([d2, feat_p5], dim=1))

        return p3_out, p4_out, p5_out


# ============================================================
# Detection Head (giữ lại cho bài toán detection gốc — KHÔNG dùng mặc định
# trong JujubeYOLO classifier bên dưới)
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
# Model — bản CLASSIFICATION (GAP đa scale + Linear), tương thích train.py
# ============================================================

class JujubeYOLO(nn.Module):
    """
    Backbone/neck Jujube-YOLO dùng làm feature extractor cho bài toán phân
    loại ảnh: GAP riêng từng scale P3/P4/P5 rồi concat -> Linear(num_classes).
    forward() trả về logits (B, num_classes), tương thích trực tiếp với
    nn.CrossEntropyLoss trong train.py.
    """

    def __init__(
        self,
        num_classes: int = 4,
        in_channels: int = 3,
        width_mult: float = 1.0,
        depth_mult: float = 1.0,
        use_dcse: bool = True,
        use_rcm: bool = True,
        use_mbca: bool = True,
        drop_rate: float = 0.0,
    ):
        super().__init__()

        self.backbone_neck = JujubeYOLOBackboneNeck(
            in_channels=in_channels,
            width_mult=width_mult, depth_mult=depth_mult,
            use_dcse=use_dcse, use_rcm=use_rcm, use_mbca=use_mbca,
        )

        ch3, ch4, ch5 = self.backbone_neck.out_channels
        self.gap = _GAPFull()
        self.dropout = nn.Dropout(drop_rate) if drop_rate > 0 else nn.Identity()
        self.classifier = nn.Linear(ch3 + ch4 + ch5, num_classes)

        self.num_classes = num_classes
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        p3, p4, p5 = self.backbone_neck(x)
        feat = torch.cat([self.gap(p3).flatten(1),
                           self.gap(p4).flatten(1),
                           self.gap(p5).flatten(1)], dim=1)
        return self.classifier(self.dropout(feat))

    def forward_features(self, x):
        """Trả về 3 feature map thô (P3, P4, P5), hữu ích cho detection/Grad-CAM."""
        return self.backbone_neck(x)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self):
        p = self.count_parameters()
        return f"JujubeYOLO(params={p:,}, size={p * 4 / 1024 / 1024:.2f} MB)"


# ============================================================
# Factory
# ============================================================

def jujube_yolo(num_classes=4):
    return JujubeYOLO(num_classes=num_classes, width_mult=1.0, depth_mult=1.0)


def jujube_yolo_s(num_classes=4):
    return JujubeYOLO(num_classes=num_classes, width_mult=0.75, depth_mult=0.67)


def jujube_yolo_n(num_classes=4):
    return JujubeYOLO(num_classes=num_classes, width_mult=0.50, depth_mult=0.34)


# ============================================================
# build_model  (bắt buộc bởi model registry — cùng chữ ký với EdgeJuFormer)
# ============================================================

def build_model(name: str = "jujube_yolo",
                 num_classes: int = 4,
                 in_channels: int = 3,
                 **kwargs):

    cfg = MODEL_CONFIGS.get(name)
    if cfg is None:
        raise ValueError(f"Unknown variant '{name}'. Choose from: {list(MODEL_CONFIGS)}")

    model = JujubeYOLO(
        num_classes=num_classes,
        in_channels=in_channels,
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

    x = torch.randn(2, 3, 224, 224)

    print(f"{'Variant':<14} {'Params':>12}  {'Size(MB)':>9}  {'Output'}")
    print("-" * 60)

    for name in MODEL_CONFIGS:
        m = build_model(name, num_classes=5)
        m.eval()
        with torch.no_grad():
            y = m(x)
        p = m.count_parameters()
        print(f"{name:<14} {p:>12,}  {p * 4 / 1024 / 1024:>9.2f}  {tuple(y.shape)}")
