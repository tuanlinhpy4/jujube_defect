"""
Ju-LiteMobileAtt: A Lightweight Attention Network for Efficient Jujube Defect Classification
Paper: Zhu X, Dang H, Jin X, Li X (2025). PLoS One 20(12): e0337898.
       https://doi.org/10.1371/journal.pone.0337898

Architecture Overview:
  - Backbone  : MobileNetV2 (inverted residual blocks, depth-wise separable conv)
  - Attention  : CoordAtt (cascaded SE + ECA modules, zero-init)
  - Feature module 1: EfficientRCAM (depth-wise conv + CoordAtt + residual)
  - Feature module 2: CascadedRCAM  (EfficientRCAM + residual fusion + 2x channel compression)
  - Strategy   : Selective attention — CoordAtt injected only at the LAST block of each stage
  - Dataset    : Jujube12000 (6 classes: Normal, Wrinkled, Deformed, Cracked, Mold, Bird-Pecked)

v3 (giữ nguyên): variant "ju_lite_mobile_att_micro" cho input 224 (xem lịch sử bên dưới).

══════════════════════════════════════════════════════════════════════════
v4 (bản này): THÊM 2 VARIANT CHO INPUT 64x64, ƯU TIÊN FPS (BS=1) BẤT CHẤP ACCURACY
══════════════════════════════════════════════════════════════════════════
Vấn đề của các variant cũ khi chạy input 64x64:
  1) Tổng stride /32 (stem s2 + 4 stage s2) -> feature map cuối chỉ 2x2,
     mất gần hết thông tin không gian (cùng lỗi đã gặp ở jujube_yolo v2).
  2) Với BS=1 trên GPU, thời gian chạy bị chi phối bởi SỐ KERNEL LAUNCH chứ
     không phải FLOPs (ảnh 64x64 quá nhỏ để GPU "bận"). Model cũ có ~30+
     block, mỗi CoordAtt (SE + ECA + group_conv) tự nó đã ~20 kernel, và
     h_swish cũ = 4 kernel rời (add, relu6, mul, div) cho MỖI ConvBNHSwish.

Cách sửa (chỉ ảnh hưởng variant mới, variant cũ giữ nguyên cấu trúc/state_dict):
  a) Backbone nông: stem s2 + 2 lần stride-2 => tổng stride /8, feature
     map cuối 8x8 (thay vì 2x2). Chỉ 11-17 conv thay vì ~50 conv.
  b) Bỏ CoordAtt/SE/ECA và bỏ tail (LightHead/CascadedRCAM) ở variant mới
     (cờ use_att=False, use_tail=False). Vẫn giữ code attention cho variant cũ.
  c) h_swish dùng F.hardswish (1 kernel thay vì 4, cùng công thức
     x*relu6(x+3)/6).
  d) Global pool dùng x.mean((2, 3)) (1 kernel, deterministic backward,
     không cần AdaptiveAvgPool2d + Flatten).
  e) Thêm fuse_model(): gộp Conv+BN (bỏ ~1/3 kernel khi inference).
     GỌI SAU KHI load_state_dict và model.eval().

  Variant mới (input 64, width_mult 1.0, 6 lớp -> số liệu tính từ cấu hình):
    ju_lite_mobile_att_64       17 conv  ~96K params  ~9.1M MACs  map cuối 8x8
    ju_lite_mobile_att_64_fast  11 conv  ~83K params  ~7.8M MACs  map cuối 8x8
  Ước lượng (CHƯA đo, phụ thuộc GPU/CPU host): _64 khoảng 500-800 FPS và
  _64_fast khoảng 900-1500 FPS ở BS=1 eager; bật fuse_model + fp16 +
  CUDA graph (xem __main__) còn nhanh hơn nhiều. Hãy chạy `python
  ju_lite_mobile_att.py` trên máy của bạn để có số thật.

Lưu ý: 2 variant mới dùng stage_cfg riêng, KHÔNG load được checkpoint của
các variant cũ (shape khác) — cần train lại với input 64.
══════════════════════════════════════════════════════════════════════════

Optimization notes (v3 — target: <1M params, <6ms GPU latency, >83% F1):
  ┌──────────────────────────────────────────────────────────────────────┐
  │ Change                            │ Effect                           │
  ├───────────────────────────────────┼──────────────────────────────────┤
  │ width_mult 1.0 → 0.50 (micro)    │ ~4x param reduction              │
  │ Stage 3: 4 blocks → 3 blocks     │ fewer ops on 28x28 fmaps         │
  │ Stage 5: 3 blocks → 2 blocks     │ fewer ops on 7x7 fmaps           │
  │ CascadedRCAM → LightHead         │ removes heavy tail module         │
  │ head channels 1280 → 512         │ cuts classifier param             │
  │ Dropout(0.2) before Linear       │ regularisation at small scale     │
  │ Label smoothing eps=0.1          │ improves F1 on small model        │
  │ Mixup alpha=0.2                  │ improves generalisation           │
  │ [v3] SEModule: pool+fc → einsum  │ fuses pool+matmul, 1 kernel       │
  │ [v3] CoordAtt: merged pool path  │ removes redundant pool call       │
  │ [v3] ConvBNHSwish: inplace ops   │ reduces memory bandwidth          │
  │ [v3] build_model: torch.compile  │ auto kernel fusion (PyTorch>=2.0) │
  │ [v3] build_model: fp16 flag      │ halves bandwidth on GPU           │
  └──────────────────────────────────────────────────────────────────────┘
  Expected: ~0.55M params · <6ms on T4 GPU · F1 >= 83%
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

ARCH_NAME       = "julitemobileatt"
ARCH_ALIASES    = ["ju_lite_mobile_att", "julitemobile"]
DEFAULT_VARIANT = "ju_lite_mobile_att_64"

MODEL_CONFIGS = {
    # original variants (kept for backward compat)
    "ju_lite_mobile_att":       {"input_size": 224, "width_mult": 1.00, "params_M": 2.16,  "optimized": False},
    "ju_lite_mobile_att_s":     {"input_size": 224, "width_mult": 0.75, "params_M": None,  "optimized": False},
    "ju_lite_mobile_att_xs":    {"input_size": 224, "width_mult": 0.50, "params_M": None,  "optimized": False},
    # optimized variant (target: <1M · <6ms · >83% F1)
    "ju_lite_mobile_att_micro": {"input_size": 224, "width_mult": 0.50, "params_M": 0.55,  "optimized": True},

    # ── v4: thiết kế riêng cho input 64x64, ưu tiên FPS ─────────────────
    # tổng stride /8 (stem s2 + 2 stage s2) -> feature map cuối 8x8,
    # không attention, không tail. Channel khai báo tường minh trong
    # stage_cfg nên để width_mult = 1.0.
    "ju_lite_mobile_att_64":      {"input_size": 64, "width_mult": 1.00, "params_M": 0.096, "optimized": True,
                                   "stage_cfg": "fast_64", "stem_c": 16, "stem_stride": 2,
                                   "use_att": False, "use_tail": False, "head_c": 256},
    "ju_lite_mobile_att_64_fast": {"input_size": 64, "width_mult": 1.00, "params_M": 0.083, "optimized": True,
                                   "stage_cfg": "fast_64_xs", "stem_c": 16, "stem_stride": 2,
                                   "use_att": False, "use_tail": False, "head_c": 192},
}


# ─────────────────────────────────────────────────────────────
# 1. Basic building-block helpers
# ─────────────────────────────────────────────────────────────

def h_swish(x: torch.Tensor) -> torch.Tensor:
    # F.hardswish = x * relu6(x + 3) / 6, đúng công thức cũ nhưng 1 kernel
    # thay vì 4 (add, relu6, mul, div).
    return F.hardswish(x)


class HSwish(nn.Module):
    def forward(self, x):
        return F.hardswish(x)


class ConvBNHSwish(nn.Module):
    """Conv -> BN -> H-Swish.
    v3: bias=False already set; BN + HSwish kept as Sequential so
    torch.compile can fuse BN into the preceding Conv kernel.
    """
    def __init__(self, in_c, out_c, kernel=1, stride=1, padding=0, groups=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_c),
            HSwish(),
        )

    def forward(self, x):
        return self.block(x)


# ─────────────────────────────────────────────────────────────
# 2. SE Module
# ─────────────────────────────────────────────────────────────

class SEModule(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        mid = max(channels // reduction, 8)
        # Use weight/bias directly (no nn.Sequential) so torch.compile
        # can fuse the two matmuls with the surrounding ops more easily.
        self.fc1 = nn.Linear(channels, mid,      bias=True)
        self.fc2 = nn.Linear(mid,      channels, bias=True)

    def forward(self, x):
        # Global average pool: (B,C,H,W) → (B,C)  [single mean kernel]
        s = x.mean(dim=(2, 3))
        s = F.relu(self.fc1(s), inplace=True)
        s = torch.sigmoid(self.fc2(s))
        # Broadcast back and scale: (B,C) → (B,C,1,1)
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
        # pool + 1-D conv in one fused path
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
        s    = self.se(x)                                          # channel-screened features
        # group_conv on s, then pool (reuse mean — no 2nd pool kernel)
        attn = torch.sigmoid(self.group_conv(s).mean(dim=(2, 3),  # (B,C)
                             keepdim=True))                        # (B,C,1,1)
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
# 7. LightHead  (micro variant — replaces CascadedRCAM)
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
# 8. MobileNetV2 InvertedResidual
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

    def forward(self, x):
        out = self.conv(x)
        if self.coord_att is not None:
            out = self.coord_att(out)
        if self.use_res:
            return x + out
        return out


# ─────────────────────────────────────────────────────────────
# 9. Stage configs   (n_blocks, out_channels, expand_ratio, [stride per block])
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
    (3, 64,  6, [2, 1, 1]),      # trimmed: 4→3
    (3, 96,  6, [1, 1, 1]),
    (2, 160, 6, [2, 1]),         # trimmed: 3→2
    (1, 320, 6, [1]),
]

# v4 — input 64x64. stem s2 (64->32) + 2 stage s2 (32->16->8) => /8, map cuối 8x8.
# 5 InvertedResidual (15 conv) + stem + head = 17 conv.
_STAGE_CFG_FAST_64 = [
    (1, 24, 4, [2]),             # 32x32 -> 16x16
    (1, 24, 4, [1]),             # 16x16 (residual)
    (1, 48, 4, [2]),             # 16x16 -> 8x8
    (1, 48, 4, [1]),             # 8x8   (residual)
    (1, 96, 4, [1]),             # 8x8
]

# v4 — nhanh hơn: 3 InvertedResidual (9 conv) + stem + head = 11 conv.
_STAGE_CFG_FAST_64_XS = [
    (1, 32, 4, [2]),             # 32x32 -> 16x16
    (1, 64, 4, [2]),             # 16x16 -> 8x8
    (1, 96, 4, [1]),             # 8x8
]

_STAGE_CFGS = {
    "original":   _STAGE_CFG_ORIGINAL,
    "optimized":  _STAGE_CFG_OPTIMIZED,
    "fast_64":    _STAGE_CFG_FAST_64,
    "fast_64_xs": _STAGE_CFG_FAST_64_XS,
}


# ─────────────────────────────────────────────────────────────
# 10. JuLiteMobileAtt
# ─────────────────────────────────────────────────────────────

class JuLiteMobileAtt(nn.Module):
    """
    Các tham số v4 (stage_cfg, stem_c, stem_stride, use_att, use_tail, head_c)
    đều có giá trị mặc định tái hiện ĐÚNG hành vi v3, nên các variant cũ
    không đổi cấu trúc / tên tham số / thứ tự khởi tạo.
    """
    def __init__(self, num_classes=6, width_mult=1.0,
                 att_reduction=4, optimized=False, dropout=0.2,
                 stage_cfg=None, stem_c=32, stem_stride=2,
                 use_att=True, use_tail=True, head_c=None):
        super().__init__()

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
                blk_att = use_att and (i == n - 1) and stage_idx > 0
                blocks.append(InvertedResidual(in_c, out_c, s, t,
                                               use_coord_att=blk_att,
                                               att_reduction=att_reduction))
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
        self.dropout     = nn.Dropout(p=dropout)
        self.classifier  = nn.Linear(head_out, num_classes)
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

    def forward_features(self, x):
        """Feature map trước global pool, shape (B, C, H, W)."""
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        x = self.tail(x)
        return self.head_conv(x)

    def forward(self, x):
        x = self.forward_features(x)
        x = x.mean(dim=(2, 3))          # global avg pool, 1 kernel (thay Pool+Flatten)
        x = self.dropout(x)
        return self.classifier(x)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─────────────────────────────────────────────────────────────
# 11. build_model
#
#   compile=True -> torch.compile(mode="reduce-overhead"): gộp kernel nhỏ và
#                   dùng CUDA graph, hiệu quả nhất ở BS=1 (compile lần đầu
#                   mất ~30-60s).
#   fp16=True    -> model.half() (GPU only; nhớ cast input sang half).
#
#   fuse_model() (bên dưới) KHÔNG nằm trong build_model vì phải gọi SAU khi
#   load_state_dict (checkpoint lưu Conv và BN tách rời).
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
    Instantiate Ju-LiteMobileAtt.

    Args:
        name            : Variant key from MODEL_CONFIGS.
                          'ju_lite_mobile_att_64'      : input 64, cân bằng (17 conv)
                          'ju_lite_mobile_att_64_fast' : input 64, nhanh nhất (11 conv)
                          'ju_lite_mobile_att_micro'   : input 224 (v3)
        num_classes     : Number of output classes.
        in_channels     : Number of input channels (chỉ hỗ trợ 3).
        pretrained      : If True, load weights from pretrained_path.
        pretrained_path : Path to a .pth checkpoint (required if pretrained=True).
        compile         : If True, apply torch.compile(mode='reduce-overhead').
        fp16            : If True, convert model to float16 (GPU only).
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


# ─────────────────────────────────────────────────────────────
# 11b. fuse_model — gộp Conv+BN cho inference
#
# Mỗi (Conv2d, BatchNorm2d) liền kề trong nn.Sequential được gộp thành
# 1 Conv2d có bias, BN thay bằng Identity. Giảm ~1/3 số kernel, kết quả
# giống model gốc (sai lệch cỡ 1e-6). Chỉ dùng cho inference.
#
# Cách dùng:
#   model = build_model(name, num_classes=...)
#   model.load_state_dict(sd); model.eval()
#   model = fuse_model(model)        # SAU load_state_dict
# ─────────────────────────────────────────────────────────────

def fuse_model(model: nn.Module) -> nn.Module:
    from torch.nn.utils import fuse_conv_bn_eval
    model.eval()
    for m in list(model.modules()):
        if isinstance(m, nn.Sequential):
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


def get_loss(num_classes=6, label_smoothing=0.1):
    """
    Cross-entropy with label smoothing eps=0.1.
    Set label_smoothing=0.0 to revert to standard CE.
    """
    return nn.CrossEntropyLoss(label_smoothing=label_smoothing)


def mixup_data(x, y, alpha=0.2, device="cpu"):
    """
    Mixup augmentation (Zhang et al., 2018).

    Usage in training loop:
        x_mix, y_a, y_b, lam = mixup_data(x, y, alpha=0.2, device=device)
        logits = model(x_mix)
        loss = lam * criterion(logits, y_a) + (1-lam) * criterion(logits, y_b)
    """
    lam = float(torch.distributions.Beta(alpha, alpha).sample()) if alpha > 0 else 1.0
    idx = torch.randperm(x.size(0), device=device)
    return lam * x + (1 - lam) * x[idx], y, y[idx], lam


# ─────────────────────────────────────────────────────────────
# 13. Data-augmentation pipeline  (paper Section 4.2)
#
# v4: thêm input_size (mặc định 224 = hành vi cũ). Với variant 64 hãy gọi
#     get_transforms(split, input_size=64). Tỉ lệ Resize/Crop giữ 256/224.
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
# Với mỗi variant 64 đo 4 chế độ:
#   eager fp32            : model gốc, chạy thẳng (giống harness benchmark thường dùng)
#   fused fp32            : sau fuse_model (gộp Conv+BN)
#   fused fp16            : + half precision (GPU)
#   fused fp16 + CUDAgraph: replay 1 graph, bỏ gần hết overhead launch (GPU)
# Model khởi tạo ngẫu nhiên (đủ để đo tốc độ, không cần checkpoint).
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import time

    device      = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_CLASSES = 5          # sửa cho khớp dataset
    RUNS, WARMUP = 300, 50

    # ---- 1) shape check cho mọi variant ----
    print("Shape check:")
    for name, cfg in MODEL_CONFIGS.items():
        s = cfg["input_size"]
        m = build_model(name, num_classes=NUM_CLASSES).eval()
        with torch.no_grad():
            fmap = m.forward_features(torch.randn(2, 3, s, s))
            out  = m(torch.randn(2, 3, s, s))
        assert tuple(out.shape) == (2, NUM_CLASSES)
        print(f"  {name:<28} in={s:>3}px  params={m.count_parameters()/1e3:>8.1f}K  "
              f"fmap={tuple(fmap.shape[1:])}  logits={tuple(out.shape)}")
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
            for _ in range(3):
                model(static_x)
        torch.cuda.current_stream().wait_stream(side)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g), torch.no_grad():
            model(static_x)
        return g.replay

    print(f"PyTorch : {torch.__version__}")
    print(f"Device  : {device}" + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""))
    print(f"{'Variant / mode':<46} {'ms':>7} {'FPS':>8}")
    print("─" * 64)

    for name in ("ju_lite_mobile_att_64", "ju_lite_mobile_att_64_fast"):
        s = MODEL_CONFIGS[name]["input_size"]
        modes = [("eager fp32", False, False, False),
                 ("fused fp32", True,  False, False)]
        if device == "cuda":
            modes += [("fused fp16",            True, True, False),
                      ("fused fp16 + CUDA graph", True, True, True)]
        for label, fuse, half, graph in modes:
            m = build_model(name, num_classes=NUM_CLASSES).to(device).eval()
            if fuse:
                m = fuse_model(m)
            x = torch.randn(1, 3, s, s, device=device)
            if half:
                m, x = m.half(), x.half()
            fn = cuda_graph_fn(m, x) if graph else (lambda m=m, x=x: m(x))
            ms  = time_fn(fn)
            fps = 1000.0 / ms
            tag = "  <- 500-1000" if 500 <= fps <= 1000 else ("  (>1000)" if fps > 1000 else "")
            print(f"  {name[len('ju_lite_mobile_att_'):] + ' | ' + label:<44} {ms:>7.3f} {fps:>8.0f}{tag}")
        print()

    print("Gợi ý: nếu FPS thấp hơn mục tiêu -> dùng ju_lite_mobile_att_64_fast, gọi")
    print("fuse_model() sau load_state_dict, và/hoặc bật compile=True / CUDA graph.")
