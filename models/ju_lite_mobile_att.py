"""
Ju-LiteMobileAtt: A Lightweight Attention Network for Efficient Jujube Defect Classification
Paper: Zhu X, Dang H, Jin X, Li X (2025). PLoS One 20(12): e0337898.
       https://doi.org/10.1371/journal.pone.0337898

Architecture Overview (bản gốc):
  - Backbone  : MobileNetV2 (inverted residual blocks, depth-wise separable conv)
  - Attention  : CoordAtt (cascaded SE + ECA modules, zero-init)
  - Feature module 1: EfficientRCAM (depth-wise conv + CoordAtt + residual)
  - Feature module 2: CascadedRCAM  (EfficientRCAM + residual fusion + 2x channel compression)
  - Strategy   : Selective attention — CoordAtt injected only at the LAST block of each stage

══════════════════════════════════════════════════════════════════════════
v5 (bản này) — DỰA TRÊN SỐ ĐO THẬT TRÊN T4, BS=1, INPUT 64x64
══════════════════════════════════════════════════════════════════════════
Kết quả đo của v4 (5 seed):
    model                      params   MACs*   Acc     F1      latency   FPS
    jujube_yolo_flat_64        463K     48M     88.90   84.13   1.76 ms   574
    ju_lite_mobile_att_64_fast  83K     7.8M    87.22   82.48   2.48 ms   404
    ju_lite_mobile_att_64       96K     9.1M    87.22   82.84   3.10 ms   323
    (*MACs tính từ cấu hình)

PHÁT HIỆN CHÍNH: FPS không đi theo params hay MACs mà đi theo SỐ OP (kernel
+ lời gọi module Python). Model 463K params nhanh hơn model 83K params, vì
ở BS=1 / ảnh 64x64 GPU gần như rảnh, thời gian bị chi phối bởi overhead
launch: ~60-80 µs cho mỗi op (flat_64: 25 op -> 1.76 ms; _64_fast: 32 op ->
2.48 ms; _64: 50 op -> 3.10 ms). Hệ quả:
    - Muốn nhanh hơn: GIẢM SỐ OP, không phải giảm params.
    - Muốn chính xác hơn: có thể TĂNG channel / dùng conv 3x3 đầy đủ gần như
      miễn phí về FPS (GPU đang rảnh). Depthwise conv có params ít nhưng tốn
      op và kém hiệu quả trên GPU; conv 3x3 thường đủ "đậm" hơn mỗi op.

Các thay đổi v5:
  1) Gộp Conv+BN TỰ ĐỘNG khi eval (ConvBNHSwish, ConvBN, InvertedResidual):
     chạy eval dùng trọng số đã gộp BN, giảm ~1/3 số op và nhiều lời gọi
     module. KHÔNG đổi tên tham số -> checkpoint v4 (_64, _64_fast) load
     được y nguyên và tự chạy nhanh hơn, không cần train lại. Có thể tắt
     bằng LAZY_FUSE = False để so sánh.
  2) Khối FusedMBConv (kiểu EfficientNetV2): conv3x3 mở rộng + conv1x1 nén,
     2 conv thay vì 3 (bỏ depthwise).
  3) 2 variant mới:
       ju_lite_mobile_att_64_turbo : ít op nhất (ưu tiên FPS)
       ju_lite_mobile_att_64_plus  : rộng hơn + pooling avg+max (ưu tiên F1)
  4) pool="avgmax": ghép global-avg và global-max. Vết nứt / mốc là chi tiết
     cục bộ nhỏ, max-pool giữ được tín hiệu này mà avg-pool làm loãng.
  5) get_loss(class_weights=...) + compute_class_weights(): lớp mold chỉ ~52
     ảnh test và F1 ~51% kéo thấp macro-F1; trọng số lớp giúp lớp hiếm.

Thứ cần tự kiểm chứng bằng train 5 seed: accuracy/F1 của _turbo và _plus,
và việc class_weights có cải thiện macro-F1 hay không.
══════════════════════════════════════════════════════════════════════════

(v4) Backbone nông cho input 64: stem s2 + 2 stage s2 => tổng stride /8,
feature map cuối 8x8 (thay vì 2x2 của thiết kế gốc /32); bỏ attention/tail;
h_swish dùng F.hardswish (1 kernel); pool dùng x.mean((2,3)).
(v3) variant "ju_lite_mobile_att_micro" cho input 224.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

ARCH_NAME       = "julitemobileatt"
ARCH_ALIASES    = ["ju_lite_mobile_att", "julitemobile"]
DEFAULT_VARIANT = "ju_lite_mobile_att_64_turbo"

# True: ở eval, tự gộp BN vào conv (nhanh hơn). False: chạy như v4 (để so sánh).
LAZY_FUSE = True

MODEL_CONFIGS = {
    # original variants (kept for backward compat)
    "ju_lite_mobile_att":       {"input_size": 224, "width_mult": 1.00, "params_M": 2.16,  "optimized": False},
    "ju_lite_mobile_att_s":     {"input_size": 224, "width_mult": 0.75, "params_M": None,  "optimized": False},
    "ju_lite_mobile_att_xs":    {"input_size": 224, "width_mult": 0.50, "params_M": None,  "optimized": False},
    "ju_lite_mobile_att_micro": {"input_size": 224, "width_mult": 0.50, "params_M": 0.55,  "optimized": True},

    # ── v4: input 64, InvertedResidual, không attention (GIỮ NGUYÊN để dùng lại checkpoint) ──
    "ju_lite_mobile_att_64":      {"input_size": 64, "width_mult": 1.00, "params_M": 0.096, "optimized": True,
                                   "stage_cfg": "fast_64", "stem_c": 16, "stem_stride": 2,
                                   "use_att": False, "use_tail": False, "head_c": 256},
    "ju_lite_mobile_att_64_fast": {"input_size": 64, "width_mult": 1.00, "params_M": 0.083, "optimized": True,
                                   "stage_cfg": "fast_64_xs", "stem_c": 16, "stem_stride": 2,
                                   "use_att": False, "use_tail": False, "head_c": 192},

    # ── v5: FusedMBConv (conv3x3 thay depthwise), ít op hơn ──
    # turbo: ~15 op khi eval (v4 _64_fast: 21 op) -> ưu tiên FPS
    "ju_lite_mobile_att_64_turbo": {"input_size": 64, "width_mult": 1.00, "params_M": 0.177, "optimized": True,
                                    "stage_cfg": "turbo_64", "stem_c": 32, "stem_stride": 2,
                                    "use_att": False, "use_tail": False, "head_c": 192,
                                    "block_type": "fused", "pool": "avg"},
    # plus: ~23 op khi eval, rộng hơn + avg+max pooling -> ưu tiên F1
    "ju_lite_mobile_att_64_plus":  {"input_size": 64, "width_mult": 1.00, "params_M": 0.405, "optimized": True,
                                    "stage_cfg": "plus_64", "stem_c": 32, "stem_stride": 2,
                                    "use_att": False, "use_tail": False, "head_c": 256,
                                    "block_type": "fused", "pool": "avgmax"},
}


# ─────────────────────────────────────────────────────────────
# 1. Hàm/khối cơ bản
# ─────────────────────────────────────────────────────────────

def h_swish(x: torch.Tensor) -> torch.Tensor:
    # F.hardswish = x * relu6(x + 3) / 6, đúng công thức cũ nhưng chỉ 1 kernel
    return F.hardswish(x)


class HSwish(nn.Module):
    def forward(self, x):
        return F.hardswish(x)


def _fold_bn(conv: nn.Conv2d, bn: nn.BatchNorm2d, dtype):
    """Gộp BatchNorm (chế độ eval) vào conv -> (weight, bias).

    BN eval:  y = (conv(x) - mean) * gamma / sqrt(var + eps) + beta
    => conv mới có  w' = w * s,  b' = beta - mean * s,  với s = gamma / sqrt(var + eps).
    Tính bằng fp32 cho chính xác rồi ép về dtype của đầu vào (fp16 nếu cần).
    """
    with torch.no_grad():
        scale = bn.weight.float() * torch.rsqrt(bn.running_var.float() + bn.eps)
        w = conv.weight.float() * scale.view(-1, 1, 1, 1)
        b = bn.bias.float() - bn.running_mean.float() * scale
        if conv.bias is not None:
            b = b + conv.bias.float() * scale
    return w.to(dtype).contiguous(), b.to(dtype).contiguous()


def _cache_stale(cache, x):
    return cache is None or cache[0].dtype != x.dtype or cache[0].device != x.device


class ConvBNHSwish(nn.Module):
    """Conv -> BN -> H-Swish.

    Train: chạy nn.Sequential bình thường (BN dùng thống kê batch).
    Eval : dùng trọng số ĐÃ GỘP BN (cache lần đầu), chỉ còn 2 op (conv, hardswish)
           thay vì 3 op + 4 lời gọi module. Cache bị xoá khi gọi .train()/.eval()
           hoặc load_state_dict, và tự tính lại nếu đổi device/dtype.
    Tên tham số (block.0 / block.1) giữ nguyên để load checkpoint cũ.
    """
    def __init__(self, in_c, out_c, kernel=1, stride=1, padding=0, groups=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_c),
            HSwish(),
        )
        self._stride, self._padding, self._groups = stride, padding, groups
        self._fused = None

    def train(self, mode=True):
        self._fused = None
        return super().train(mode)

    def forward(self, x):
        if self.training or not LAZY_FUSE:
            return self.block(x)
        f = self._fused
        if _cache_stale(f, x):
            f = self._fused = _fold_bn(self.block[0], self.block[1], x.dtype)
        return F.hardswish(F.conv2d(x, f[0], f[1], self._stride, self._padding, 1, self._groups))


class ConvBN(nn.Module):
    """Conv -> BN (không activation), cũng tự gộp BN khi eval. Dùng làm lớp nén 1x1."""
    def __init__(self, in_c, out_c, kernel=1, stride=1, padding=0, groups=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_c),
        )
        self._stride, self._padding, self._groups = stride, padding, groups
        self._fused = None

    def train(self, mode=True):
        self._fused = None
        return super().train(mode)

    def forward(self, x):
        if self.training or not LAZY_FUSE:
            return self.block(x)
        f = self._fused
        if _cache_stale(f, x):
            f = self._fused = _fold_bn(self.block[0], self.block[1], x.dtype)
        return F.conv2d(x, f[0], f[1], self._stride, self._padding, 1, self._groups)


# ─────────────────────────────────────────────────────────────
# 2. SE Module
# ─────────────────────────────────────────────────────────────

class SEModule(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.fc1 = nn.Linear(channels, mid,      bias=True)
        self.fc2 = nn.Linear(mid,      channels, bias=True)

    def forward(self, x):
        s = x.mean(dim=(2, 3))
        s = F.relu(self.fc1(s), inplace=True)
        s = torch.sigmoid(self.fc2(s))
        return x * s.unsqueeze(-1).unsqueeze(-1)


# ─────────────────────────────────────────────────────────────
# 3. ECA Module
# ─────────────────────────────────────────────────────────────

class ECAModule(nn.Module):
    def __init__(self, channels, kernel=3, groups=1):
        super().__init__()
        self.conv    = nn.Conv1d(1, 1, kernel_size=kernel,
                                 padding=kernel // 2, groups=groups, bias=False)

    def forward(self, x):
        y = x.mean(dim=(2, 3), keepdim=True)        # (B,C,1,1)
        y = y.squeeze(-1).transpose(-1, -2)          # (B,1,C)
        y = torch.sigmoid(self.conv(y))              # (B,1,C)
        y = y.transpose(-1, -2).unsqueeze(-1)        # (B,C,1,1)
        return x * y


# ─────────────────────────────────────────────────────────────
# 4. CoordAtt
# ─────────────────────────────────────────────────────────────

class CoordAtt(nn.Module):
    def __init__(self, channels, reduction=4, eca_kernel=3, groups=1):
        super().__init__()
        self.se         = SEModule(channels, reduction)
        self.eca        = ECAModule(channels, eca_kernel, groups)
        self.group_conv = nn.Conv2d(channels, channels, 1,
                                    groups=max(1, channels // 16), bias=False)
        nn.init.zeros_(self.group_conv.weight)

    def forward(self, x):
        s    = self.se(x)
        attn = torch.sigmoid(self.group_conv(s).mean(dim=(2, 3), keepdim=True))
        out  = self.eca(s * attn)
        return out + x


# ─────────────────────────────────────────────────────────────
# 5. EfficientRCAM
# ─────────────────────────────────────────────────────────────

class EfficientRCAM(nn.Module):
    def __init__(self, channels, expand_ratio=2, reduction=4):
        super().__init__()
        mid = channels * expand_ratio
        self.pw_expand = ConvBNHSwish(channels, mid, 1)
        self.dw_conv   = ConvBNHSwish(mid, mid, 3, 1, 1, groups=mid)
        self.coord_att = CoordAtt(mid, reduction)
        self.pw_proj   = nn.Sequential(
            nn.Conv2d(mid, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.act = HSwish()

    def forward(self, x):
        out = self.pw_expand(x)
        out = self.dw_conv(out)
        out = self.coord_att(out)
        out = self.pw_proj(out)
        return self.act(out + x)


# ─────────────────────────────────────────────────────────────
# 6. CascadedRCAM  (original variants only)
# ─────────────────────────────────────────────────────────────

class CascadedRCAM(nn.Module):
    def __init__(self, in_c, reduction=4):
        super().__init__()
        self.efficient_rcam = EfficientRCAM(in_c, expand_ratio=2, reduction=reduction)
        out_c = max(in_c // 2, 16)
        self.compress     = ConvBNHSwish(in_c, out_c, 1)
        self.out_channels = out_c

    def forward(self, x):
        return self.compress(x + self.efficient_rcam(x))


# ─────────────────────────────────────────────────────────────
# 7. LightHead  (micro variant)
# ─────────────────────────────────────────────────────────────

class LightHead(nn.Module):
    def __init__(self, in_c, reduction=4):
        super().__init__()
        self.rcam         = EfficientRCAM(in_c, expand_ratio=2, reduction=reduction)
        out_c             = max(in_c // 2, 16)
        self.compress     = ConvBNHSwish(in_c, out_c, 1)
        self.out_channels = out_c

    def forward(self, x):
        return self.compress(self.rcam(x))


# ─────────────────────────────────────────────────────────────
# 8a. MobileNetV2 InvertedResidual  (expand 1x1 -> depthwise 3x3 -> project 1x1)
#
# v5: lớp nén (Conv2d + BN cuối trong self.conv) tự gộp BN khi eval, nhưng
# vẫn giữ nguyên tên tham số conv.{i}.* để checkpoint v4 load được.
# ─────────────────────────────────────────────────────────────

class InvertedResidual(nn.Module):
    def __init__(self, in_c, out_c, stride, expand_ratio,
                 use_coord_att=False, att_reduction=4):
        super().__init__()
        self.use_res = (stride == 1 and in_c == out_c)
        mid = int(round(in_c * expand_ratio))

        layers = []
        if expand_ratio != 1:
            layers.append(ConvBNHSwish(in_c, mid, 1))
        layers += [
            ConvBNHSwish(mid, mid, 3, stride, 1, groups=mid),
            nn.Conv2d(mid, out_c, 1, bias=False),
            nn.BatchNorm2d(out_c),
        ]
        self.conv      = nn.Sequential(*layers)
        self.coord_att = CoordAtt(out_c, att_reduction) if use_coord_att else None
        self._n_pre    = len(layers) - 2       # số lớp đứng trước cặp (Conv2d, BN) cuối
        self._proj     = None

    def train(self, mode=True):
        self._proj = None
        return super().train(mode)

    def forward(self, x):
        n = self._n_pre
        h = x
        for i in range(n):
            h = self.conv[i](h)
        conv, bn = self.conv[n], self.conv[n + 1]
        if self.training or not LAZY_FUSE:
            out = bn(conv(h))
        else:
            p = self._proj
            if _cache_stale(p, h):
                p = self._proj = _fold_bn(conv, bn, h.dtype)
            out = F.conv2d(h, p[0], p[1])
        if self.coord_att is not None:
            out = self.coord_att(out)
        if self.use_res:
            return x + out
        return out


# ─────────────────────────────────────────────────────────────
# 8b. FusedMBConv (v5)  (conv3x3 mở rộng -> conv1x1 nén)
#
# So với InvertedResidual: bỏ depthwise và conv expand 1x1 -> 2 conv thay vì 3.
# expand_ratio == 1: chỉ 1 conv3x3 (+ residual nếu cùng shape).
# ─────────────────────────────────────────────────────────────

class FusedMBConv(nn.Module):
    def __init__(self, in_c, out_c, stride, expand_ratio):
        super().__init__()
        self.use_res = (stride == 1 and in_c == out_c)
        if expand_ratio == 1:
            self.expand  = ConvBNHSwish(in_c, out_c, 3, stride, 1)
            self.project = None
        else:
            mid = int(round(in_c * expand_ratio))
            self.expand  = ConvBNHSwish(in_c, mid, 3, stride, 1)
            self.project = ConvBN(mid, out_c, 1)

    def forward(self, x):
        out = self.expand(x)
        if self.project is not None:
            out = self.project(out)
        return x + out if self.use_res else out


# ─────────────────────────────────────────────────────────────
# 9. Stage configs   (n_blocks, out_channels, expand_ratio, [stride mỗi block])
# ─────────────────────────────────────────────────────────────

_STAGE_CFG_ORIGINAL = [
    (1, 16,  1, [1]),
    (2, 24,  6, [2, 1]),
    (3, 32,  6, [2, 1, 1]),
    (4, 64,  6, [2, 1, 1, 1]),
    (3, 96,  6, [1, 1, 1]),
    (3, 160, 6, [2, 1, 1]),
    (1, 320, 6, [1]),
]

_STAGE_CFG_OPTIMIZED = [
    (1, 16,  1, [1]),
    (2, 24,  6, [2, 1]),
    (3, 32,  6, [2, 1, 1]),
    (3, 64,  6, [2, 1, 1]),
    (3, 96,  6, [1, 1, 1]),
    (2, 160, 6, [2, 1]),
    (1, 320, 6, [1]),
]

# v4 — input 64: stem s2 (64->32) + 2 stage s2 (32->16->8) => /8, map cuối 8x8
_STAGE_CFG_FAST_64 = [
    (1, 24, 4, [2]),
    (1, 24, 4, [1]),
    (1, 48, 4, [2]),
    (1, 48, 4, [1]),
    (1, 96, 4, [1]),
]
_STAGE_CFG_FAST_64_XS = [
    (1, 32, 4, [2]),
    (1, 64, 4, [2]),
    (1, 96, 4, [1]),
]

# v5 — FusedMBConv. Mỗi tuple = 1 block.
_STAGE_CFG_TURBO_64 = [
    (1, 48, 2, [2]),     # 32x32 -> 16x16 : conv3x3 s2 32->64, nén 64->48
    (1, 96, 2, [2]),     # 16x16 -> 8x8   : conv3x3 s2 48->96, nén 96->96
    (1, 96, 1, [1]),     # 8x8            : conv3x3 96->96 + residual
]
_STAGE_CFG_PLUS_64 = [
    (1, 48, 2, [2]),     # -> 16x16
    (1, 48, 1, [1]),     # 16x16 residual (thêm sức chứa ở độ phân giải còn thấy chi tiết nhỏ)
    (1, 96, 2, [2]),     # -> 8x8
    (1, 96, 1, [1]),     # 8x8 residual
    (1, 128, 2, [1]),    # 8x8: conv3x3 96->192, nén 192->128
]

_STAGE_CFGS = {
    "original":   _STAGE_CFG_ORIGINAL,
    "optimized":  _STAGE_CFG_OPTIMIZED,
    "fast_64":    _STAGE_CFG_FAST_64,
    "fast_64_xs": _STAGE_CFG_FAST_64_XS,
    "turbo_64":   _STAGE_CFG_TURBO_64,
    "plus_64":    _STAGE_CFG_PLUS_64,
}


# ─────────────────────────────────────────────────────────────
# 10. JuLiteMobileAtt
# ─────────────────────────────────────────────────────────────

class JuLiteMobileAtt(nn.Module):
    """
    Các tham số từ v4 trở đi (stage_cfg, stem_c, stem_stride, use_att, use_tail,
    head_c, block_type, pool) đều có mặc định tái hiện đúng hành vi v3, nên các
    variant cũ không đổi cấu trúc / tên tham số / thứ tự khởi tạo.
    """
    def __init__(self, num_classes=6, width_mult=1.0,
                 att_reduction=4, optimized=False, dropout=0.2,
                 stage_cfg=None, stem_c=32, stem_stride=2,
                 use_att=True, use_tail=True, head_c=None,
                 block_type="ir", pool="avg"):
        super().__init__()
        assert block_type in ("ir", "fused"), f"block_type không hợp lệ: {block_type!r}"
        assert pool in ("avg", "avgmax"), f"pool không hợp lệ: {pool!r}"

        def _c(v): return max(1, int(v * width_mult))

        if stage_cfg is None:
            stage_cfg = _STAGE_CFG_OPTIMIZED if optimized else _STAGE_CFG_ORIGINAL
        elif isinstance(stage_cfg, str):
            stage_cfg = _STAGE_CFGS[stage_cfg]

        self.stem = ConvBNHSwish(3, _c(stem_c), 3, stride=stem_stride, padding=1)

        stages, in_c = [], _c(stem_c)
        for stage_idx, (n, out, t, strides) in enumerate(stage_cfg):
            out_c  = _c(out)
            blocks = []
            for i, s in enumerate(strides):
                if block_type == "ir":
                    blk_att = use_att and (i == n - 1) and stage_idx > 0
                    blocks.append(InvertedResidual(in_c, out_c, s, t,
                                                   use_coord_att=blk_att,
                                                   att_reduction=att_reduction))
                else:
                    blocks.append(FusedMBConv(in_c, out_c, s, t))
                in_c = out_c
            stages.append(nn.Sequential(*blocks))
        self.stages = nn.ModuleList(stages)

        if use_tail:
            if optimized:
                self.tail = LightHead(in_c, reduction=att_reduction)
                head_out  = _c(512)
            else:
                self.tail = CascadedRCAM(in_c, reduction=att_reduction)
                head_out  = _c(1280)
            tail_out = self.tail.out_channels
        else:
            self.tail = nn.Identity()
            head_out  = _c(512)
            tail_out  = in_c
        if head_c is not None:
            head_out = _c(head_c)

        self.head_conv   = ConvBNHSwish(tail_out, head_out, 1)
        self.pool_mode   = pool
        self.dropout     = nn.Dropout(p=dropout)
        self.classifier  = nn.Linear(head_out * (2 if pool == "avgmax" else 1), num_classes)
        self.num_classes = num_classes

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _reset_fusion_cache(self):
        for m in self.modules():
            if isinstance(m, (ConvBNHSwish, ConvBN)):
                m._fused = None
            elif isinstance(m, InvertedResidual):
                m._proj = None

    def load_state_dict(self, state_dict, strict=True, **kwargs):
        # nạp trọng số mới -> cache BN đã gộp (nếu có) không còn đúng
        out = super().load_state_dict(state_dict, strict=strict, **kwargs)
        self._reset_fusion_cache()
        return out

    def forward_features(self, x):
        """Feature map trước global pool, shape (B, C, H, W)."""
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        x = self.tail(x)
        return self.head_conv(x)

    def forward(self, x):
        x = self.forward_features(x)
        if self.pool_mode == "avgmax":
            x = torch.cat((x.mean(dim=(2, 3)), x.amax(dim=(2, 3))), dim=1)
        else:
            x = x.mean(dim=(2, 3))
        x = self.dropout(x)
        return self.classifier(x)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─────────────────────────────────────────────────────────────
# 11. build_model
# ─────────────────────────────────────────────────────────────

def build_model(name=DEFAULT_VARIANT,
                num_classes=6,
                in_channels=3,
                pretrained=False,
                pretrained_path=None,
                compile=False,
                fp16=False,
                **kwargs) -> JuLiteMobileAtt:
    """
    Args:
        name : 'ju_lite_mobile_att_64_turbo' (nhanh nhất), '..._64_plus' (F1),
               '..._64_fast' / '..._64' (v4, checkpoint cũ dùng lại được),
               '..._micro' (input 224).
        compile : torch.compile(mode='reduce-overhead'). Với model nhỏ này,
                  CUDA graph thủ công trong __main__ ổn định hơn.
        fp16 : model.half() (GPU only; nhớ cast input).
    """
    if name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown variant '{name}'. Available: {list(MODEL_CONFIGS.keys())}")
    if in_channels != 3:
        raise ValueError("JuLiteMobileAtt hiện chỉ hỗ trợ in_channels=3.")

    cfg   = MODEL_CONFIGS[name]
    model = JuLiteMobileAtt(
        num_classes=num_classes,
        width_mult=cfg["width_mult"],
        optimized=cfg.get("optimized", False),
        stage_cfg=cfg.get("stage_cfg", None),
        stem_c=cfg.get("stem_c", 32),
        stem_stride=cfg.get("stem_stride", 2),
        use_att=cfg.get("use_att", True),
        use_tail=cfg.get("use_tail", True),
        head_c=cfg.get("head_c", None),
        block_type=cfg.get("block_type", "ir"),
        pool=cfg.get("pool", "avg"),
        **kwargs,
    )
    model.variant_name = name

    if pretrained:
        if pretrained_path is None:
            raise ValueError("pretrained=True requires pretrained_path.")
        ckpt = torch.load(pretrained_path, map_location="cpu")
        sd   = ckpt.get("state_dict", ckpt.get("model", ckpt))
        model.load_state_dict(sd, strict=True)

    if fp16:
        model = model.half()

    if compile:
        model = torch.compile(model, mode="reduce-overhead")

    return model


def fuse_model(model: nn.Module) -> nn.Module:
    """Gộp Conv+BN thủ công (chỉ còn cần cho EfficientRCAM.pw_proj của các variant
    attention cũ). Các khối ConvBNHSwish / ConvBN / InvertedResidual đã tự gộp lúc eval
    nên bị bỏ qua ở đây. Gọi SAU load_state_dict."""
    from torch.nn.utils import fuse_conv_bn_eval
    model.eval()
    skip = set()
    for m in model.modules():
        if isinstance(m, (ConvBNHSwish, ConvBN)):
            skip.add(id(m.block))
        elif isinstance(m, InvertedResidual):
            skip.add(id(m.conv))
    for m in list(model.modules()):
        if isinstance(m, nn.Sequential) and id(m) not in skip:
            for i in range(len(m) - 1):
                if isinstance(m[i], nn.Conv2d) and isinstance(m[i + 1], nn.BatchNorm2d):
                    m[i] = fuse_conv_bn_eval(m[i], m[i + 1])
                    m[i + 1] = nn.Identity()
    return model


# ─────────────────────────────────────────────────────────────
# 12. Training utilities
# ─────────────────────────────────────────────────────────────

def get_optimizer(model, lr=5e-4, momentum=0.9, weight_decay=5e-2):
    """AdamW — paper Section 4.3."""
    return torch.optim.AdamW(model.parameters(),
                              lr=lr, betas=(momentum, 0.999),
                              weight_decay=weight_decay)


def get_scheduler(optimizer, total_epochs=200, warmup_epochs=1):
    """Cosine annealing with 1-epoch linear warm-up — paper Section 4.3."""
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1e-2, end_factor=1.0, total_iters=warmup_epochs)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_epochs - warmup_epochs, eta_min=1e-6)
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs])


def compute_class_weights(class_counts, power=0.5, device="cpu"):
    """Trọng số lớp theo nghịch đảo tần suất, làm mượt bằng `power`.

    w_c = (N / (C * n_c)) ** power, rồi chuẩn hoá để trung bình = 1.
    power=1.0: cân bằng hoàn toàn (dễ làm tụt accuracy lớp lớn);
    power=0.5: nhẹ tay, thường là điểm bắt đầu hợp lý; power=0: không dùng.
    class_counts: số ảnh train mỗi lớp, theo đúng thứ tự nhãn.
    """
    c = torch.as_tensor(class_counts, dtype=torch.float32)
    w = (c.sum() / (len(c) * c)) ** power
    return (w / w.mean()).to(device)


def get_loss(num_classes=6, label_smoothing=0.1, class_weights=None):
    """Cross-entropy + label smoothing. class_weights: tensor (num_classes,) tuỳ chọn
    (xem compute_class_weights) để tăng chú ý lên lớp hiếm như mold."""
    return nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)


def mixup_data(x, y, alpha=0.2, device="cpu"):
    """
    Mixup augmentation (Zhang et al., 2018).
    Với ảnh 64x64 và lỗi nhỏ, nên thử tắt (alpha=0) để so sánh.

        x_mix, y_a, y_b, lam = mixup_data(x, y, alpha=0.2, device=device)
        logits = model(x_mix)
        loss = lam * criterion(logits, y_a) + (1-lam) * criterion(logits, y_b)
    """
    lam = float(torch.distributions.Beta(alpha, alpha).sample()) if alpha > 0 else 1.0
    idx = torch.randperm(x.size(0), device=device)
    return lam * x + (1 - lam) * x[idx], y, y[idx], lam


# ─────────────────────────────────────────────────────────────
# 13. Data-augmentation pipeline  (paper Section 4.2)
#     input_size mặc định 224 (hành vi cũ); với variant 64 gọi input_size=64.
# ─────────────────────────────────────────────────────────────

def get_transforms(split="train", input_size=224):
    try:
        from torchvision import transforms
    except ImportError:
        raise ImportError("torchvision is required.")

    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    resize = int(round(input_size * 256 / 224))

    if split == "train":
        return transforms.Compose([
            transforms.RandomResizedCrop(
                input_size, scale=(0.70, 1.0), ratio=(0.8, 1.25),
                interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomAffine(degrees=15, shear=10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomErasing(p=0.5, scale=(0.02, 0.20)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    else:
        return transforms.Compose([
            transforms.Resize(resize, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])


# ─────────────────────────────────────────────────────────────
# 14. Self-test + benchmark FPS (BS=1, input 64x64)
#
#   python ju_lite_mobile_att.py
#
# Mỗi variant 64 đo các chế độ:
#   eager, KHÔNG gộp BN : như v4 (LAZY_FUSE=False)
#   eager + gộp BN      : mặc định v5
#   fp16                : + half precision (GPU)
#   fp16 + CUDA graph   : replay 1 graph, gần như hết overhead launch (GPU)
# Model khởi tạo ngẫu nhiên (đủ để đo tốc độ).
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import time

    device      = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_CLASSES = 5          # sửa cho khớp dataset
    RUNS, WARMUP = 300, 50

    # ---- 1) shape check ----
    print("Shape check:")
    for name, cfg in MODEL_CONFIGS.items():
        s = cfg["input_size"]
        m = build_model(name, num_classes=NUM_CLASSES).eval()
        with torch.no_grad():
            fmap = m.forward_features(torch.randn(2, 3, s, s))
            out  = m(torch.randn(2, 3, s, s))
        assert tuple(out.shape) == (2, NUM_CLASSES)
        print(f"  {name:<30} in={s:>3}px  params={m.count_parameters()/1e3:>8.1f}K  "
              f"fmap={tuple(fmap.shape[1:])}  logits={tuple(out.shape)}")

    # ---- 1b) gộp BN không được làm đổi kết quả ----
    print("\nKiểm tra gộp BN (sai lệch tối đa so với không gộp):")
    for name in ("ju_lite_mobile_att_64_fast", "ju_lite_mobile_att_64_turbo", "ju_lite_mobile_att_64_plus"):
        m = build_model(name, num_classes=NUM_CLASSES)
        # BN mới tạo có mean=0, var=1 (gộp lên cũng gần như không đổi) -> gán giá trị ngẫu nhiên để test có ý nghĩa
        for bn in (x for x in m.modules() if isinstance(x, nn.BatchNorm2d)):
            bn.running_mean.normal_(0, 0.1); bn.running_var.uniform_(0.5, 1.5)
            bn.weight.data.uniform_(0.5, 1.5); bn.bias.data.normal_(0, 0.1)
        m.eval()
        xx = torch.randn(4, 3, 64, 64)
        with torch.no_grad():
            LAZY_FUSE = False
            ref = m(xx)
            LAZY_FUSE = True
            m._reset_fusion_cache()
            fus = m(xx)
        print(f"  {name:<30} max|diff| = {(ref - fus).abs().max().item():.2e}")
    print()

    # ---- 2) benchmark ----
    def time_fn(fn):
        with torch.no_grad():
            for _ in range(WARMUP):
                fn()
            if device == "cuda":
                s_evt = torch.cuda.Event(enable_timing=True)
                e_evt = torch.cuda.Event(enable_timing=True)
                torch.cuda.synchronize()
                s_evt.record()
                for _ in range(RUNS):
                    fn()
                e_evt.record()
                torch.cuda.synchronize()
                return s_evt.elapsed_time(e_evt) / RUNS          # ms
            t0 = time.perf_counter()
            for _ in range(RUNS):
                fn()
            return (time.perf_counter() - t0) / RUNS * 1000

    def cuda_graph_fn(model, x):
        static_x = x.clone()
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side), torch.no_grad():
            for _ in range(3):                     # warm-up: tạo cache BN đã gộp trước khi capture
                model(static_x)
        torch.cuda.current_stream().wait_stream(side)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g), torch.no_grad():
            model(static_x)
        return g.replay

    print(f"PyTorch : {torch.__version__}")
    print(f"Device  : {device}" + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""))
    print(f"{'Variant | mode':<58} {'ms':>7} {'FPS':>8}")
    print("─" * 76)

    variants = ("ju_lite_mobile_att_64", "ju_lite_mobile_att_64_fast",
                "ju_lite_mobile_att_64_turbo", "ju_lite_mobile_att_64_plus")
    for name in variants:
        s = MODEL_CONFIGS[name]["input_size"]
        modes = [("eager, khong gop BN (v4)", False, False, False),
                 ("eager + gop BN (v5)",       True,  False, False)]
        if device == "cuda":
            modes += [("gop BN + fp16",              True, True, False),
                      ("gop BN + fp16 + CUDA graph", True, True, True)]
        for label, lazy, half, graph in modes:
            LAZY_FUSE = lazy
            m = build_model(name, num_classes=NUM_CLASSES).to(device).eval()
            x = torch.randn(1, 3, s, s, device=device)
            if half:
                m, x = m.half(), x.half()
            fn = cuda_graph_fn(m, x) if graph else (lambda m=m, x=x: m(x))
            ms  = time_fn(fn)
            fps = 1000.0 / ms
            tag = "  <- 500-1000" if 500 <= fps <= 1000 else ("  (>1000)" if fps > 1000 else "")
            short = name[len('ju_lite_mobile_att_'):]
            print(f"  {short + ' | ' + label:<56} {ms:>7.3f} {fps:>8.0f}{tag}")
        print()
    LAZY_FUSE = True
