"""
Jujube-YOLO v3 — DCSE + C3k2_RCM + MBCA (classification variant)
------------------------------------------------------------------------

Dựa trên: Wang, L. et al. (2025). "Jujube-YOLO: a precise jujube fruit recognition model in unstructured environments." Expert Systems With Applications, 291, 128530. https://doi.org/10.1016/j.eswa.2025.128530

Lịch sử thay đổi
-------------------
v1 -> v2: fix bug nối nhầm feature map P2/4 làm skip-connection cho nhánh đang ở P3/8 trong neck; chuyển từ detector sang classifier (GAP đa scale P3+P4+P5 -> Linear); thêm bộ tham số ablation (head_mode, rca_kernel,
rcm_mlp_ratio, rcm_n, mbca_*, c3k2_n_override, drop_path_rate, drop_rate) và các variant tối ưu tốc độ (fast/lite/ultra/flat).

v2 -> v3 (bản này): SỬA VẤN ĐỀ ACCURACY KÉM KHI INPUT GIẢM TỪ 224 XUỐNG 64.
====================================================================
Bối cảnh: file config huấn luyện thực tế (jujube_ours.yaml) đã hạ input_size từ 224 xuống 64 để tăng tốc độ. Benchmark thực đo cho thấy khi input=224, "jujube_yolo_flat" đạt accuracy CAO NHẤT (89.11%) trong mọi
model được so sánh; nhưng khi input=64 (giữ NGUYÊN kiến trúc v2, chỉ đổi input_size ở dataloader), "jujube_yolo_flat" tụt xuống 86.02% — THẤP HƠN cả juyolo_baseline (87.25%) và nhiều baseline nông khác (tinynet_e,
repghostnet_050, regnetx_002). Nguyên nhân không phải bug runtime (model vẫn chạy được, không crash) mà là SAI LỆCH THIẾT KẾ giữa độ sâu backbone và độ phân giải ảnh đầu vào:

1) Tổng stride backbone quá sâu so với input 64px.
   JujubeYOLOBackboneNeck (v2) luôn có 5 lớp stride-2 liên tiếp (stem1, stem2, down_p3, down_p4, down_p5) => tổng stride /32. Với input 224px điều này hợp lý (P5 = 7x7). Nhưng với input 64px, P5 chỉ còn 64/32 =
   2x2 — gần như toàn bộ thông tin không gian (vết nứt/mốc/sâu vốn là chi tiết nhỏ, cục bộ trên vỏ quả) đã bị "vắt cạn" trước khi model kịp học được gì có ý nghĩa ở tầng sâu.
   jujube_yolo_flat (FlatBackbone) còn tệ hơn: stem dùng conv 7x7 stride 4
   NGAY LỚP ĐẦU TIÊN (blur + downsample rất mạnh khi ảnh vốn đã nhỏ), rồi down2 stride 4 nữa => input 64px chỉ còn 2x2 ở output, tương tự.

2) RCA (trong RCM) dùng kernel trục lớn cố định (mặc định 11: conv (11,1) và (1,11)) trên MỌI feature map, bất kể feature map đó lớn hay nhỏ. Ở input 224px, kernel 11 trên feature map 28-56px là hợp lý (nhìn
   được ngữ cảnh rộng). Nhưng ở input 64px, feature map các stage chỉ còn 2-16px => kernel 11 (padding 5 mỗi bên) chủ yếu là ZERO-PADDING, phép tích chập gần như học nhiễu/hằng số thay vì thông tin không gian thật.
   Đây KHÔNG gây crash (padding vẫn cho ra shape đúng — khớp với việc model "vẫn chạy được" trong log benchmark), nhưng làm loãng tín hiệu. SPPF (maxpool k=5) trên P5 2x2 gặp vấn đề tương tự.

3) Neck FPN/PAN (upsample + concat + C3k2) vốn được chính v2 note trong docstring là "gần như vô nghĩa" ở feature map bé (2x2-8x8), nhưng variant "jujube_yolo" (không phải flat) vẫn BẬT neck mặc định kể cả
   khi dùng input 64px vì train.py chỉ đổi input_size ở dataloader chứ không đổi kiến trúc model theo.

Hướng sửa (đã áp dụng trong file này):
   a) Backbone nông có thể cấu hình được: thêm `stem_strides` (1 hoặc 2 lớp stride-2 ở stem) và `num_down_stages` (1-3 cặp down_*/stage_p* sau stem). Tổng stride = 2**stem_strides * 2**num_down_stages, MẶC ĐỊNH giữ nguyên 2 và 3 (tổng /32, giống v2 y hệt — KHÔNG đổi hành vi
      của các variant cũ dùng cho input >=224px). Các variant mới "*_64" dùng stem_strides=1, num_down_stages=2 => tổng stride chỉ /8, feature map cuối với input 64px là 8x8 (thay vì 2x2) — giữ lại đủ chi tiết không gian cho các vết nứt/mốc/sâu nhỏ.
   b) Khi num_down_stages < 3, neck FPN/PAN tự động bị TẮT (neck cần đủ 3 scale P3/P4/P5). forward() khi đó trả về TOÀN BỘ feature map thô của các stage đã xây (vd (P2,P3,P4)) thay vì chỉ 1 scale sâu nhất
      như v2 — nhờ vậy head multi_scale vẫn tận dụng được thông tin đa tầng (nông + sâu) mà không tốn chi phí neck vô nghĩa ở feature map bé.
   c) `input_size` (tuỳ chọn) được truyền xuống backbone để TỰ ĐỘNG giới
      hạn (clamp) rca_kernel và sppf_kernel không vượt quá kích thước
      feature map thực tế của từng stage — không còn tình trạng kernel
      lớn hơn hẳn feature map như mục (2) ở trên. Nếu không truyền
      input_size, hành vi giữ nguyên như v2 (không tự clamp).
   d) FlatBackbone: stem_k / stem_stride / down2_stride giờ cấu hình
      được (mặc định 7/4/4 — giữ nguyên v2). Variant "jujube_yolo_flat_64"
      dùng stem_k=3, stem_stride=2, down2_stride=2 => tổng stride /8
      thay vì /32, đồng thời kernel stem nhỏ hơn (3 thay vì 7) để không
      "làm mờ" chi tiết bề mặt quả ngay ở lớp đầu.
   e) Cảnh báo runtime (warnings.warn) một lần khi feature map sâu nhất
      dự kiến < 4x4, để tránh lặp lại sai lầm này với các input nhỏ khác
      trong tương lai.

   => 2 variant MỚI khuyến nghị dùng cho jujube_ours.yaml (input_size=64):
        - "jujube_yolo_64"       (thay cho "jujube_yolo")
        - "jujube_yolo_flat_64"  (đã sửa lại, thay cho bản "jujube_yolo_flat"
                                   cũ — entry cùng tên trong MODEL_CONFIGS
                                   giờ trỏ tới kiến trúc nông hơn)
   Variant "abl_input_64" được GIỮ NGUYÊN như v2 (kiến trúc sâu /32,
   không auto-clamp kiến trúc) để làm điểm đối chứng "input nhỏ nhưng
   không sửa kiến trúc" — chính là cấu hình đã cho kết quả kém trong
   benchmark thực tế.

Kiến trúc module (không đổi so với v2):
    1. DCSE  — stem
    2. C3k2_RCM — backbone (thay C3k2 chuẩn)
    3. MBCA  — conv fusion ở neck

Chạy:
    python jujube_yolo.py
"""

from __future__ import annotations

import math
import warnings
from typing import Optional

import torch
import torch.nn as nn


# ============================================================
# Registry metadata
# ============================================================

ARCH_NAME = "jujube_yolo"

MODEL_CONFIGS = {
    # ── kích thước gốc (thiết kế cho input ~224px, giống scaling n/s của
    #    YOLO). KHÔNG khuyến nghị dùng trực tiếp các variant này cho input
    #    nhỏ (<=96px) — xem nhóm "*_64" ở cuối dict, được thiết kế riêng
    #    cho input nhỏ (xem mục "v2 -> v3" trong docstring đầu file). ────
    "jujube_yolo":    {"width_mult": 1.00, "depth_mult": 1.00,
                        "use_dcse": True,  "use_rcm": True,  "use_mbca": True},
    "jujube_yolo_s":  {"width_mult": 0.75, "depth_mult": 0.67,
                        "use_dcse": True,  "use_rcm": True,  "use_mbca": True},
    "jujube_yolo_n":  {"width_mult": 0.50, "depth_mult": 0.34,
                        "use_dcse": True,  "use_rcm": True,  "use_mbca": True},

    # ── ablation, (baseline = YOLOv11n)
    "juyolo_baseline":            {"width_mult": 1.0, "depth_mult": 1.0,
                             "use_dcse": False, "use_rcm": False, "use_mbca": False},
    "juyolo_only_dcse":           {"width_mult": 1.0, "depth_mult": 1.0,
                             "use_dcse": True,  "use_rcm": False, "use_mbca": False},
    "juyolo_only_rcm":            {"width_mult": 1.0, "depth_mult": 1.0,
                             "use_dcse": False, "use_rcm": True,  "use_mbca": False},
    "juyolo_only_mbca":           {"width_mult": 1.0, "depth_mult": 1.0,
                             "use_dcse": False, "use_rcm": False, "use_mbca": True},
    "juyolo_dcse_rcm":            {"width_mult": 1.0, "depth_mult": 1.0,
                             "use_dcse": True,  "use_rcm": True,  "use_mbca": False},
    "juyolo_dcse_mbca":           {"width_mult": 1.0, "depth_mult": 1.0,
                             "use_dcse": True,  "use_rcm": False, "use_mbca": True},
    "juyolo_rcm_mbca":            {"width_mult": 1.0, "depth_mult": 1.0,
                             "use_dcse": False, "use_rcm": True,  "use_mbca": True},
    "juyolo_full":                {"width_mult": 1.0, "depth_mult": 1.0,
                             "use_dcse": True,  "use_rcm": True,  "use_mbca": True},

    # ── Ablation Study bổ sung ──────────────────────────────────────────
    # -- Group: Head Design --------------------------------------------
    "abl_head_single_p5":  {"width_mult": 1.0, "depth_mult": 1.0,
                             "use_dcse": True,  "use_rcm": True,  "use_mbca": True, "head_mode": "single_scale"},
    "abl_head_dual_p4p5":  {"width_mult": 1.0, "depth_mult": 1.0,
                             "use_dcse": True,  "use_rcm": True,  "use_mbca": True, "head_mode": "dual_scale"},
    "abl_head_no_neck":    {"width_mult": 1.0, "depth_mult": 1.0,
                             "use_dcse": True,  "use_rcm": True,  "use_mbca": True, "use_neck": False},

    # -- Group: Attention -------------------------------------------------
    "abl_rca_k7":           {"width_mult": 1.0, "depth_mult": 1.0,
                             "use_dcse": True,  "use_rcm": True,  "use_mbca": True, "rca_kernel": 7},
    "abl_rcm_no_mlp":       {"width_mult": 1.0, "depth_mult": 1.0,
                             "use_dcse": True,  "use_rcm": True,  "use_mbca": True, "rcm_mlp_ratio": 1},
    "abl_mbca_no_shuffle":  {"width_mult": 1.0, "depth_mult": 1.0,
                             "use_dcse": True,  "use_rcm": True,  "use_mbca": True, "mbca_use_channel_shuffle": False},
    "abl_mbca_no_global":   {"width_mult": 1.0, "depth_mult": 1.0,
                             "use_dcse": True,  "use_rcm": True,  "use_mbca": True, "mbca_use_global_branch": False},

    # -- Group: Depth -------------------------------------------------
    "abl_rcm_single":       {"width_mult": 1.0, "depth_mult": 1.0,
                             "use_dcse": True,  "use_rcm": True,  "use_mbca": True, "rcm_n": 1},
    "abl_c3k2_n1":          {"width_mult": 1.0, "depth_mult": 1.0,
                             "use_dcse": True,  "use_rcm": True,  "use_mbca": True, "c3k2_n_override": 1},

    # -- Group: Regularization -----------------------------------------
    "abl_droppath_01":      {"width_mult": 1.0, "depth_mult": 1.0, "use_dcse": True,  "use_rcm": True,  "use_mbca": True, "drop_path_rate": 0.1},
    "abl_dropout_02":       {"width_mult": 1.0, "depth_mult": 1.0, "use_dcse": True,  "use_rcm": True,  "use_mbca": True, "drop_rate": 0.2},

    # -- Group: Resolution ------------------------------------------------
    "abl_input_160":        {"width_mult": 1.0, "depth_mult": 1.0, "use_dcse": True,  "use_rcm": True,  "use_mbca": True, "input_size": 160},
    # GIỮ NGUYÊN kiến trúc sâu /32 của v2, chỉ đổi input_size xuống 64 —
    # đây CHÍNH LÀ cấu hình "input nhỏ nhưng không sửa kiến trúc" đã cho
    # kết quả kém trong benchmark thực tế (bảng đen trong yêu cầu sửa).
    # Dùng entry này làm ĐỐI CHỨNG cho ablation "resolution", KHÔNG dùng
    # để train thật — với input 64px thật sự, hãy dùng "jujube_yolo_64"
    # ở nhóm v3 bên dưới.
    "abl_input_64":         {"width_mult": 1.0, "depth_mult": 1.0, "use_dcse": True,  "use_rcm": True,  "use_mbca": True, "input_size": 64},

    # ── Điều chỉnh lần 1: speed-oriented (chap nhan giam accuracy de doi lay FPS) ─
    "jujube_yolo_flat":    {"width_mult": 0.50, "depth_mult": 1.0, "use_dcse": False, "use_rcm": False, "use_mbca": False, "use_neck": False, "use_flat_backbone": True},
    # (thiết kế cho input >=~128px; xem "jujube_yolo_flat_64" bên dưới cho
    # input 64px — đây là entry đã bị SỬA so với v2, không còn dùng
    # stem 7x7/stride4 + down2 stride4 nữa, xem nhóm v3 bên dưới)

    # ================================================================
    # NHÓM MỚI (v3): thiết kế riêng cho input NHỎ (jujube_ours.yaml hiện
    # dùng input_size: 64). Xem phân tích chi tiết ở docstring đầu file,
    # mục "v2 -> v3". Đây là các variant NÊN dùng thay cho "jujube_yolo"
    # / "jujube_yolo_flat" khi train với input 64px.
    # ================================================================
    "jujube_yolo_64": {
        "width_mult": 1.00, "depth_mult": 1.00,
        "use_dcse": True, "use_rcm": True, "use_mbca": True,
        "stem_strides": 1, "num_down_stages": 2,   # tổng stride /8 (thay vì /32)
        "use_neck": False,                          # neck vô nghĩa ở feature map bé, đã tự tắt do num_down_stages<3
        "input_size": 64,                           # dùng để auto-clamp rca_kernel/sppf_kernel theo từng stage
    },
    "jujube_yolo_64_s": {
        "width_mult": 0.75, "depth_mult": 0.67,
        "use_dcse": True, "use_rcm": True, "use_mbca": True,
        "stem_strides": 1, "num_down_stages": 2,
        "use_neck": False,
        "input_size": 64,
    },
    # SỬA lại "jujube_yolo_flat_64" (trước đây kế thừa nguyên FlatBackbone
    # của v2: stem 7x7/stride4 + down2 stride4 -> tổng stride /32 -> P5
    # chỉ 2x2 với input 64px). Bản v3 dùng stem 3x3/stride2 + down2
    # stride2 -> tổng stride /8 -> feature map cuối 8x8.
    "jujube_yolo_flat_64": {
        "width_mult": 0.50, "depth_mult": 1.0,
        "use_dcse": False, "use_rcm": False, "use_mbca": False,
        "use_neck": False, "use_flat_backbone": True,
        "flat_stem_k": 3, "flat_stem_stride": 2, "flat_down2_stride": 2,
        "input_size": 64,
    },
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


def _clamp_odd_kernel(k: int, feat_size: Optional[int]) -> int:
    """Giới hạn kernel k không vượt quá kích thước feature map (feat_size).

    Lý do (xem docstring đầu file, mục v2->v3): các module RCA/SPPF dùng
    kernel lớn cố định (11, 5, ...) được thiết kế cho input ~224px, nơi
    feature map đủ lớn để kernel đó "nhìn" được ngữ cảnh thật. Khi input
    nhỏ (64px), feature map có thể bé hơn hẳn kernel -> phần lớn kernel
    chỉ nhân với zero-padding, làm loãng tín hiệu không gian thay vì rút
    trích ngữ cảnh. Hàm này không đổi hành vi khi feat_size is None (giữ
    nguyên k như v2, không auto-clamp) hoặc khi k đã <= feat_size.
    """
    if feat_size is None or feat_size <= 0:
        return k
    k = min(k, feat_size)
    if k % 2 == 0:
        k -= 1
    return max(1, k)


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
    """Average over width only -> (B, C, H, 1)."""
    def forward(self, x):
        return x.mean(dim=3, keepdim=True)


class _GAPw(nn.Module):
    """Average over height only -> (B, C, 1, W)."""
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
    """Fout = rho( xiF(xiC(P_H(x) (+) P_V(x))) (x) x ) + x   -- Eq. (2).

    k_large: kernel lớn dùng trong RCA (mặc định 11 theo paper, có thể bị
    JujubeYOLOBackboneNeck tự clamp nhỏ hơn nếu biết input_size — xem
    _clamp_odd_kernel).
    """

    def __init__(self, c, mlp_ratio=2, k_large=11):
        super().__init__()
        self.rca = RCA(c, k_large=k_large)
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
    """N khối RCM residual liên tiếp (Fig. 6, phải; mặc định N=2)."""

    def __init__(self, c, mlp_ratio=2, rca_kernel=11, n_rcm=2, drop_path_rate=0.0):
        super().__init__()
        assert n_rcm in (1, 2), "RCMBottleneck: n_rcm chỉ hỗ trợ 1 hoặc 2 (theo Fig.6)"
        self.rcm1 = RCM(c, mlp_ratio=mlp_ratio, k_large=rca_kernel)
        self.rcm2 = RCM(c, mlp_ratio=mlp_ratio, k_large=rca_kernel) if n_rcm == 2 else None
        self.drop_path = DropPath(drop_path_rate)

    def forward(self, x):
        y = self.rcm1(x)
        if self.rcm2 is not None:
            y = self.rcm2(y)
        return x + self.drop_path(y)


class PlainBottleneck(nn.Module):
    """Bottleneck chuẩn (Conv3x3-Conv3x3 + residual) — dùng khi ablation use_rcm=False."""

    def __init__(self, c, drop_path_rate=0.0):
        super().__init__()
        self.cv1 = Conv(c, c, k=3)
        self.cv2 = Conv(c, c, k=3)
        self.drop_path = DropPath(drop_path_rate)

    def forward(self, x):
        return x + self.drop_path(self.cv2(self.cv1(x)))


class C3k2(nn.Module):
    """Khối CSP-style C3k2 của YOLOv11, inner-bottleneck chọn RCMBottleneck
    (use_rcm=True, -> C3k2_RCM) hoặc PlainBottleneck (ablation / neck)."""

    def __init__(self, c1, c2, n=1, e=0.5, use_rcm=True,
                 rca_kernel=11, rcm_mlp_ratio=2, rcm_n=2, drop_path_rate=0.0):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, 2 * c_, k=1)
        self.cv2 = Conv((2 + n) * c_, c2, k=1)
        if use_rcm:
            block = lambda: RCMBottleneck(c_, mlp_ratio=rcm_mlp_ratio, rca_kernel=rca_kernel,
                                           n_rcm=rcm_n, drop_path_rate=drop_path_rate)
        else:
            block = lambda: PlainBottleneck(c_, drop_path_rate=drop_path_rate)
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

    def __init__(self, c1, c2, k=3, reduction=8, groups_shuffle=4,
                 use_channel_shuffle=True, use_global_branch=True):
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

        self.use_global_branch = use_global_branch
        if use_global_branch:
            self.branch_global = nn.Sequential(
                _GAPFull(),
                nn.Conv2d(c1, c_b, kernel_size=1, bias=False),
            )
        else:
            self.branch_global = Conv(c1, c_b, k=1)

        self.use_channel_shuffle = use_channel_shuffle
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
        b4 = self.branch_global(x)
        if self.use_global_branch:
            b4 = b4.expand(-1, -1, x.shape[2], x.shape[3])

        feat = torch.cat([b1, b2, b3, b4], dim=1)
        if self.use_channel_shuffle:
            feat = self._channel_shuffle(feat, self.groups_shuffle)

        gate = self.sigmoid(self.fc2(self.relu(self.fc1(self.gap(feat)))))
        feat = feat * gate

        return self.act_out(feat + self.skip(x))


# ============================================================
# SPPF (khối chuẩn YOLO)
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
# FlatBackbone — toi gian SO LUONG LAYER (khong chi FLOPs)
# ------------------------------------------------------------
# v3: stem_k / stem_stride / down2_stride nay co the tuy chinh (truoc day
# co dinh k=7, s=4 cho stem va s=4 cho down2 -> tong stride /32, qua sau
# cho input nho). Mac dinh GIU NGUYEN gia tri v2 (khong doi hanh vi cua
# variant "jujube_yolo_flat", input >=224px). Variant "jujube_yolo_flat_64"
# trong MODEL_CONFIGS truyen stem_k=3, stem_stride=2, down2_stride=2 ->
# tong stride /8 thay vi /32 (feature map cuoi 8x8 thay vi 2x2 voi input
# 64px), dong thoi kernel stem nho hon (3 thay vi 7) giup giu chi tiet be
# mat qua (vet nut/moc/sau) thay vi bi "lam mo" ngay lop dau bo i conv 7x7
# stride 4.
# ============================================================

class FlatBackbone(nn.Module):

    def __init__(self, in_channels=3, width_mult=1.0,
                 stem_k=7, stem_stride=4, down2_stride=4):
        super().__init__()
        c = lambda x: _round_ch(x * width_mult)
        c1, c2, c3 = c(64), c(128), c(256)

        self.stem = Conv(in_channels, c1, k=stem_k, s=stem_stride)
        self.down1 = Conv(c1, c2, k=3, s=2)
        self.block1 = PlainBottleneck(c2)
        self.down2 = Conv(c2, c3, k=3, s=down2_stride)
        self.block2 = PlainBottleneck(c3)

        self.out_channels = (c3,)

    def forward(self, x):
        x = self.stem(x)
        x = self.down1(x)
        x = self.block1(x)
        x = self.down2(x)
        x = self.block2(x)
        return (x,)


# ============================================================
# Backbone + Neck
# ============================================================

class JujubeYOLOBackboneNeck(nn.Module):
    """
    Backbone (stem -> C3k2_RCM nhiều stage) [+ Neck FPN/PAN tuỳ chọn].

    v3 — backbone không còn cố định 5 lớp stride-2 (tổng /32) cho MỌI
    input (xem phân tích ở docstring đầu file). Hai tham số mới điều
    khiển độ sâu downsample:
        stem_strides    : 1 hoặc 2 lớp stride-2 ở stem (mặc định 2, giống
                           v2 — DCSE x2 -> P2/4).
        num_down_stages : số cặp (down_*, stage_p*) được xây SAU stem, tối
                           đa 3 (down_p3, down_p4, down_p5), mặc định 3
                           (giống v2 — đủ P3/P4/P5).
    Tổng stride = 2**stem_strides * 2**num_down_stages. Mặc định (2, 3)
    cho tổng /32 y hệt v2 — KHÔNG đổi hành vi của các variant cũ dùng cho
    input >=224px. Các variant "*_64" dùng (1, 2) -> tổng /8.

    Khi num_down_stages < 3, neck FPN/PAN KHÔNG được xây (neck cần đủ 3
    scale P3/P4/P5) — use_neck tự động bị ép về False (kèm cảnh báo nếu
    người dùng truyền use_neck=True). Khi use_neck=False, forward() trả
    về TOÀN BỘ feature map thô của các stage đã xây (vd (P2,P3,P4)), thay
    vì chỉ scale sâu nhất như v2 — head multi_scale của JujubeYOLO vẫn
    tận dụng được thông tin đa tầng mà không tốn chi phí neck (chính v2
    đã nhận định neck "gần như vô nghĩa" ở feature map bé).

    input_size (tuỳ chọn, pixel): nếu được truyền, dùng để TÍNH TRƯỚC kích
    thước feature map của từng stage, qua đó tự động giới hạn (clamp)
    rca_kernel và sppf_kernel không vượt quá feature map thực tế tại stage
    đó (xem _clamp_odd_kernel). Không truyền -> giữ hành vi v2 (không tự
    clamp).
    """

    def __init__(self, in_channels=3, width_mult=1.0, depth_mult=1.0,
                 use_dcse=True, use_rcm=True, use_mbca=True, use_neck=True,
                 rca_kernel: int = 11, rcm_mlp_ratio: int = 2, rcm_n: int = 2,
                 drop_path_rate: float = 0.0,
                 mbca_use_channel_shuffle: bool = True,
                 mbca_use_global_branch: bool = True,
                 c3k2_n_override: Optional[int] = None,
                 stem_strides: int = 2,
                 num_down_stages: int = 3,
                 input_size: Optional[int] = None,
                 sppf_kernel: int = 5):
        super().__init__()

        assert stem_strides in (1, 2), "stem_strides chỉ hỗ trợ 1 hoặc 2"
        assert num_down_stages in (1, 2, 3), "num_down_stages chỉ hỗ trợ 1, 2 hoặc 3"

        if num_down_stages < 3 and use_neck:
            warnings.warn(
                f"JujubeYOLOBackboneNeck: use_neck=True nhưng num_down_stages="
                f"{num_down_stages} < 3 -> neck FPN/PAN cần đủ P3/P4/P5 nên tự "
                f"động tắt neck (use_neck=False)."
            )
            use_neck = False
        self.use_neck = use_neck

        c = lambda x: _round_ch(x * width_mult)
        if c3k2_n_override is not None:
            n = lambda x: c3k2_n_override
        else:
            n = lambda x: _round_depth(x, depth_mult)

        rcm_common = dict(rcm_mlp_ratio=rcm_mlp_ratio, rcm_n=rcm_n,
                           drop_path_rate=drop_path_rate)

        ch1, ch2, ch3, ch4, ch5 = c(64), c(128), c(256), c(512), c(1024)

        stem_block = DCSE if use_dcse else (lambda c1, c2, s: Conv(c1, c2, k=3, s=s))

        # ---- stem: 1 hoặc 2 lớp stride-2 ----
        if stem_strides == 2:
            self.stem1 = stem_block(in_channels, ch1, s=2)
            self.stem2 = stem_block(ch1, ch2, s=2)
        else:
            # 1 lớp duy nhất -> giữ độ phân giải gấp đôi so với mặc định
            # ngay từ đầu mạng, quan trọng khi input đã nhỏ sẵn (64px)
            self.stem1 = stem_block(in_channels, ch2, s=2)
            self.stem2 = None
        stem_total_stride = 2 ** stem_strides

        # ---- kích thước feature map dự kiến từng stage (để auto-clamp kernel) ----
        def _feat_size(total_stride):
            if input_size is None:
                return None
            return max(1, input_size // total_stride)

        size_p2 = _feat_size(stem_total_stride)
        size_p3 = _feat_size(stem_total_stride * 2)
        size_p4 = _feat_size(stem_total_stride * 4)
        size_p5 = _feat_size(stem_total_stride * 8)

        def _rca_k(size):
            return _clamp_odd_kernel(rca_kernel, size)

        if input_size is not None:
            deepest_size = [size_p2, size_p3, size_p4, size_p5][num_down_stages]
            if deepest_size is not None and deepest_size < 4:
                warnings.warn(
                    f"JujubeYOLOBackboneNeck: với input_size={input_size} và cấu "
                    f"hình hiện tại (stem_strides={stem_strides}, "
                    f"num_down_stages={num_down_stages}), feature map sâu nhất chỉ "
                    f"còn {deepest_size}x{deepest_size} — khá bé, cân nhắc giảm "
                    f"num_down_stages hoặc dùng stem_strides=1 để giữ nhiều chi "
                    f"tiết không gian hơn."
                )

        # ---- backbone stages (P2 luôn có; P3/P4/P5 tuỳ num_down_stages) ----
        self.stage_p2 = C3k2(ch2, ch3, n=n(2), e=0.25, use_rcm=use_rcm,
                              rca_kernel=_rca_k(size_p2), **rcm_common)
        self._raw_ch = [ch3]

        self.down_p3 = self.stage_p3 = None
        self.down_p4 = self.stage_p4 = None
        self.down_p5 = self.stage_p5 = self.sppf = None

        if num_down_stages >= 1:
            self.down_p3 = Conv(ch3, ch3, k=3, s=2)
            self.stage_p3 = C3k2(ch3, ch4, n=n(2), e=0.25, use_rcm=use_rcm,
                                  rca_kernel=_rca_k(size_p3), **rcm_common)
            self._raw_ch.append(ch4)
        if num_down_stages >= 2:
            self.down_p4 = Conv(ch4, ch4, k=3, s=2)
            self.stage_p4 = C3k2(ch4, ch4, n=n(2), e=0.5, use_rcm=use_rcm,
                                  rca_kernel=_rca_k(size_p4), **rcm_common)
            self._raw_ch.append(ch4)
        if num_down_stages >= 3:
            self.down_p5 = Conv(ch4, ch5, k=3, s=2)
            self.stage_p5 = C3k2(ch5, ch5, n=n(2), e=0.5, use_rcm=use_rcm,
                                  rca_kernel=_rca_k(size_p5), **rcm_common)
            sk = _clamp_odd_kernel(sppf_kernel, size_p5) if size_p5 is not None else sppf_kernel
            self.sppf = SPPF(ch5, ch5, k=sk)
            self._raw_ch.append(ch5)

        if not self.use_neck:
            # classification-only, không dùng neck: trả về TẤT CẢ feature
            # map thô đã xây (đa scale, rẻ hơn neck FPN/PAN nhiều)
            self.out_channels = tuple(self._raw_ch)
            return

        # ---- neck (chỉ xây khi num_down_stages == 3, giữ nguyên như v2);
        # C3k2 ở neck là plain (use_rcm=False), giống Fig.8 ----
        self.up1 = nn.Upsample(scale_factor=2, mode="nearest")
        self.c3k2_n1 = C3k2(ch5 + ch4, ch4, n=n(2), e=0.5, use_rcm=False,
                             drop_path_rate=drop_path_rate)
        self.up2 = nn.Upsample(scale_factor=2, mode="nearest")
        self.c3k2_n2 = C3k2(ch4 + ch4, ch3, n=n(2), e=0.5, use_rcm=False,
                             drop_path_rate=drop_path_rate)

        if use_mbca:
            fuse_block = lambda c1, c2: MBCA(c1, c2,
                                              use_channel_shuffle=mbca_use_channel_shuffle,
                                              use_global_branch=mbca_use_global_branch)
        else:
            fuse_block = lambda c1, c2: Conv(c1, c2, k=3)
        self.fuse_p3 = fuse_block(ch3, ch3)

        self.down_n1 = Conv(ch3, ch3, k=3, s=2)
        self.c3k2_n3 = C3k2(ch3 + ch4, ch4, n=n(2), e=0.5, use_rcm=False,
                             drop_path_rate=drop_path_rate)
        self.fuse_p4 = fuse_block(ch4, ch4)

        self.down_n2 = Conv(ch4, ch4, k=3, s=2)
        self.c3k2_n4 = C3k2(ch4 + ch5, ch5, n=n(2), e=0.5, use_rcm=False,
                             drop_path_rate=drop_path_rate)

        self.out_channels = (ch3, ch4, ch5)

    def forward(self, x):
        x = self.stem1(x)
        if self.stem2 is not None:
            x = self.stem2(x)

        feat_p2 = self.stage_p2(x)
        raw_feats = [feat_p2]
        x = feat_p2

        feat_p3 = feat_p4 = feat_p5 = None
        if self.stage_p3 is not None:
            x = self.down_p3(x)
            feat_p3 = self.stage_p3(x)
            raw_feats.append(feat_p3)
            x = feat_p3
        if self.stage_p4 is not None:
            x = self.down_p4(x)
            feat_p4 = self.stage_p4(x)
            raw_feats.append(feat_p4)
            x = feat_p4
        if self.stage_p5 is not None:
            x = self.down_p5(x)
            feat_p5 = self.sppf(self.stage_p5(x))
            raw_feats.append(feat_p5)

        if not self.use_neck:
            return tuple(raw_feats)

        # top-down (FPN)
        u1 = self.up1(feat_p5)
        n1 = self.c3k2_n1(torch.cat([u1, feat_p4], dim=1))
        u2 = self.up2(n1)
        n2 = self.c3k2_n2(torch.cat([u2, feat_p3], dim=1))
        p3_out = self.fuse_p3(n2)

        # bottom-up (PAN)
        d1 = self.down_n1(p3_out)
        n3 = self.c3k2_n3(torch.cat([d1, n1], dim=1))
        p4_out = self.fuse_p4(n3)

        d2 = self.down_n2(p4_out)
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
        return outs


# ============================================================
# Model — bản CLASSIFICATION (GAP đa scale + Linear), tương thích train.py
# ============================================================

class JujubeYOLO(nn.Module):
    """
    Backbone/neck Jujube-YOLO dùng làm feature extractor cho bài toán phân
    loại ảnh: GAP riêng từng scale rồi concat -> Linear(num_classes).
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
        use_neck: bool = True,
        use_flat_backbone: bool = False,
        drop_rate: float = 0.0,
        head_mode: str = "multi_scale",
        rca_kernel: int = 11,
        rcm_mlp_ratio: int = 2,
        rcm_n: int = 2,
        drop_path_rate: float = 0.0,
        mbca_use_channel_shuffle: bool = True,
        mbca_use_global_branch: bool = True,
        c3k2_n_override: Optional[int] = None,
        # --- v3: backbone nông cho input nhỏ + kernel tự-clamp theo input_size ---
        stem_strides: int = 2,
        num_down_stages: int = 3,
        input_size: Optional[int] = None,
        sppf_kernel: int = 5,
        flat_stem_k: int = 7,
        flat_stem_stride: int = 4,
        flat_down2_stride: int = 4,
    ):
        super().__init__()

        # head_mode — nhóm ablation "Head Design":
        #   "multi_scale" (mặc định): GAP toàn bộ scale mà backbone trả về
        #                  (P3+P4+P5 nếu có neck, hoặc mọi raw stage nếu
        #                  không có neck)
        #   "dual_scale" : chỉ GAP 2 scale sâu nhất
        #   "single_scale": chỉ GAP scale sâu nhất
        assert head_mode in ("multi_scale", "dual_scale", "single_scale"), \
            f"head_mode không hợp lệ: {head_mode!r}"
        self.head_mode = head_mode

        if use_flat_backbone:
            # bo qua toan bo DCSE/RCM/MBCA/neck — toi gian so luong layer
            # de giam overhead o BS=1. flat_stem_k/stride/down2_stride
            # dieu chinh do sau backbone (xem FlatBackbone).
            self.backbone_neck = FlatBackbone(
                in_channels=in_channels, width_mult=width_mult,
                stem_k=flat_stem_k, stem_stride=flat_stem_stride,
                down2_stride=flat_down2_stride,
            )
        else:
            self.backbone_neck = JujubeYOLOBackboneNeck(
                in_channels=in_channels,
                width_mult=width_mult, depth_mult=depth_mult,
                use_dcse=use_dcse, use_rcm=use_rcm, use_mbca=use_mbca,
                use_neck=use_neck,
                rca_kernel=rca_kernel, rcm_mlp_ratio=rcm_mlp_ratio, rcm_n=rcm_n,
                drop_path_rate=drop_path_rate,
                mbca_use_channel_shuffle=mbca_use_channel_shuffle,
                mbca_use_global_branch=mbca_use_global_branch,
                c3k2_n_override=c3k2_n_override,
                stem_strides=stem_strides,
                num_down_stages=num_down_stages,
                input_size=input_size,
                sppf_kernel=sppf_kernel,
            )

        out_ch = self.backbone_neck.out_channels
        if head_mode == "single_scale":
            in_feat = out_ch[-1]
        elif head_mode == "dual_scale":
            assert len(out_ch) >= 2, (
                "head_mode='dual_scale' cần backbone_neck trả về >=2 scale; "
                f"hiện chỉ có {len(out_ch)} scale."
            )
            in_feat = sum(out_ch[-2:])
        else:
            in_feat = sum(out_ch)

        self.gap = _GAPFull()
        self.dropout = nn.Dropout(drop_rate) if drop_rate > 0 else nn.Identity()
        self.classifier = nn.Linear(in_feat, num_classes)

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
        feats = self.backbone_neck(x)
        if self.head_mode == "single_scale":
            feats = feats[-1:]
        elif self.head_mode == "dual_scale":
            feats = feats[-2:]
        feat = torch.cat([self.gap(f).flatten(1) for f in feats], dim=1)
        return self.classifier(self.dropout(feat))

    def forward_features(self, x):
        """Trả về các feature map thô, hữu ích cho detection/Grad-CAM."""
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


def jujube_yolo_64(num_classes=4):
    """Bản khuyến nghị cho input 64px (xem MODEL_CONFIGS['jujube_yolo_64'])."""
    return build_model("jujube_yolo_64", num_classes=num_classes)


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

    # input_size: ưu tiên giá trị truyền trực tiếp vào build_model (vd nếu
    # train.py truyền input_size=cfg['input_size'] từ yaml), nếu không có
    # thì lấy metadata khai báo sẵn trong MODEL_CONFIGS của variant. Giá
    # trị này chỉ dùng để auto-clamp rca_kernel/sppf_kernel theo từng
    # stage (xem JujubeYOLOBackboneNeck) — KHÔNG bắt buộc phải sửa
    # train.py để có hiệu quả sửa lỗi accuracy (các variant "*_64" đã tự
    # khai báo input_size=64 sẵn trong MODEL_CONFIGS).
    input_size = kwargs.pop("input_size", cfg.get("input_size", None))

    model = JujubeYOLO(
        num_classes=num_classes,
        in_channels=in_channels,
        width_mult=cfg["width_mult"],
        depth_mult=cfg["depth_mult"],
        use_dcse=cfg["use_dcse"],
        use_rcm=cfg["use_rcm"],
        use_mbca=cfg["use_mbca"],
        use_neck=cfg.get("use_neck", True),
        use_flat_backbone=cfg.get("use_flat_backbone", False),
        drop_rate=cfg.get("drop_rate", 0.0),
        head_mode=cfg.get("head_mode", "multi_scale"),
        rca_kernel=cfg.get("rca_kernel", 11),
        rcm_mlp_ratio=cfg.get("rcm_mlp_ratio", 2),
        rcm_n=cfg.get("rcm_n", 2),
        drop_path_rate=cfg.get("drop_path_rate", 0.0),
        mbca_use_channel_shuffle=cfg.get("mbca_use_channel_shuffle", True),
        mbca_use_global_branch=cfg.get("mbca_use_global_branch", True),
        c3k2_n_override=cfg.get("c3k2_n_override", None),
        # --- v3 ---
        stem_strides=cfg.get("stem_strides", 2),
        num_down_stages=cfg.get("num_down_stages", 3),
        input_size=input_size,
        sppf_kernel=cfg.get("sppf_kernel", 5),
        flat_stem_k=cfg.get("flat_stem_k", 7),
        flat_stem_stride=cfg.get("flat_stem_stride", 4),
        flat_down2_stride=cfg.get("flat_down2_stride", 4),
        **kwargs,
    )
    model.variant_name = name
    return model


# ============================================================
# Fuse Conv+BN cho inference
# ============================================================

def fuse_model(model: nn.Module) -> nn.Module:
    from torch.nn.utils import fuse_conv_bn_eval
    for m in model.modules():
        if isinstance(m, Conv) and isinstance(m.bn, nn.BatchNorm2d):
            m.conv = fuse_conv_bn_eval(m.conv, m.bn)
            m.bn = nn.Identity()
    return model


# ============================================================
# Quick self-test
# ============================================================

if __name__ == "__main__":

    print(f"{'Variant':<32} {'Input':>7}  {'Params':>12}  {'Size(MB)':>9}  {'#Scales':>7}  {'Output'}")
    print("-" * 95)

    for name, cfg in MODEL_CONFIGS.items():
        size = cfg.get("input_size", 224)
        x = torch.randn(2, 3, size, size)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            m = build_model(name, num_classes=5)
        m.eval()
        with torch.no_grad():
            y = m(x)
        p = m.count_parameters()
        n_scales = len(m.backbone_neck.out_channels)
        flag = " (!)" if w else ""
        print(f"{name:<32} {size:>4}px  {p:>12,}  {p * 4 / 1024 / 1024:>9.2f}  {n_scales:>7}  {tuple(y.shape)}{flag}")
