"""
Ju-LiteMobileAtt (64_turbo) - ban rut gon
------------------------------------------------------------------------

Dua tren: Zhu X, Dang H, Jin X, Li X (2025). "Ju-LiteMobileAtt: A Lightweight
Attention Network for Efficient Jujube Defect Classification." PLoS One 20(12):
e0337898. https://doi.org/10.1371/journal.pone.0337898

File nay la ban RUT GON cua ju_lite_mobile_att.py (v5): chi giu lai variant
"ju_lite_mobile_att_64_turbo" va nhung thanh phan variant nay thuc su dung
(ConvBNHSwish, ConvBN, FusedMBConv, head GAP + Linear). Cac variant khac
(ju_lite_mobile_att, _s, _xs, _micro, _64, _64_fast, _64_plus) va cac module
chi phuc vu chung (SE, ECA, CoordAtt, EfficientRCAM, CascadedRCAM, LightHead,
InvertedResidual, pooling avg+max) da bi xoa.

Dam bao KHONG doi ket qua so voi file goc (cung seed -> cung ket qua):
    - Ten module / ten tham so giu nguyen (stem, stages.i.j.expand/project,
      head_conv, classifier) -> state_dict cua checkpoint cu load duoc ngay.
    - Thu tu khoi tao module giu nguyen (stem -> stages -> head_conv ->
      classifier) va _init_weights giu nguyen -> cung seed cho cung bo trong
      so khoi tao.
    - Cong thuc forward (train va eval, ke ca gop BN luc eval) giu nguyen.

Kien truc (input 64x64, 5 lop):
    stem      Conv3x3/s2  3 -> 32            : 64 -> 32
    stage 0   FusedMBConv 32 -> 48 (t=2, s2) : 32 -> 16   (conv3x3 32->64, nen 64->48)
    stage 1   FusedMBConv 48 -> 96 (t=2, s2) : 16 -> 8    (conv3x3 48->96, nen 96->96)
    stage 2   FusedMBConv 96 -> 96 (t=1, s1) : 8x8, residual (1 conv3x3)
    head_conv Conv1x1     96 -> 192
    GAP (mean H,W) -> Dropout(0.2) -> Linear(192, num_classes)
    => tong stride /8, feature map cuoi 8x8, khoang 177K params.

Luc eval, moi (Conv + BatchNorm) duoc gop thanh 1 conv (xem _fold_bn) nen
chi con ~15 op; luc train chay Conv + BN binh thuong.

Chay self-test + benchmark:
    python ju_lite_mobile_att.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

ARCH_NAME       = "julitemobileatt"
ARCH_ALIASES    = ["ju_lite_mobile_att", "julitemobile"]
DEFAULT_VARIANT = "ju_lite_mobile_att_64_turbo"

# True: o eval, tu gop BN vao conv (nhanh hon). False: chay Conv + BN rieng (de so sanh).
LAZY_FUSE = True

MODEL_CONFIGS = {
    "ju_lite_mobile_att_64_turbo": {
        "input_size": 64, "width_mult": 1.00, "params_M": 0.177, "optimized": True,
        "stage_cfg": "turbo_64", "stem_c": 32, "stem_stride": 2,
        "use_att": False, "use_tail": False, "head_c": 192,
        "block_type": "fused", "pool": "avg",
    },
}

# (n_blocks, out_channels, expand_ratio, [stride moi block])
_STAGE_CFG_TURBO_64 = [
    (1, 48, 2, [2]),     # 32x32 -> 16x16 : conv3x3 s2 32->64, nen 64->48
    (1, 96, 2, [2]),     # 16x16 -> 8x8   : conv3x3 s2 48->96, nen 96->96
    (1, 96, 1, [1]),     # 8x8            : conv3x3 96->96 + residual
]


# ============================================================
# Khoi co ban
# ============================================================

class HSwish(nn.Module):
    """H-Swish = x * relu6(x + 3) / 6 (F.hardswish: 1 kernel)."""
    def forward(self, x):
        return F.hardswish(x)


def _fold_bn(conv: nn.Conv2d, bn: nn.BatchNorm2d, dtype):
    """Gop BatchNorm (che do eval) vao conv -> (weight, bias).

    BN eval:  y = (conv(x) - mean) * gamma / sqrt(var + eps) + beta
    => conv moi co  w' = w * s,  b' = beta - mean * s,  voi s = gamma / sqrt(var + eps).
    Tinh bang fp32 cho chinh xac roi ep ve dtype cua dau vao (fp16 neu can).
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

    Train: chay nn.Sequential binh thuong (BN dung thong ke batch).
    Eval : dung trong so DA GOP BN (cache lan dau), chi con 2 op (conv, hardswish).
           Cache bi xoa khi goi .train()/.eval() hoac load_state_dict, va tu
           tinh lai neu doi device/dtype.
    Ten tham so (block.0 / block.1) giu nguyen de load checkpoint cu.
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
    """Conv -> BN (khong activation), cung tu gop BN khi eval. Dung lam lop nen 1x1."""
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


class FusedMBConv(nn.Module):
    """conv3x3 mo rong -> conv1x1 nen (kieu EfficientNetV2).

    expand_ratio == 1: chi 1 conv3x3 (+ residual neu cung shape).
    """
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


# ============================================================
# Model - CLASSIFICATION (GAP + Linear)
# ============================================================

class JuLiteMobileAtt(nn.Module):
    """stem -> FusedMBConv stages -> head_conv 1x1 -> GAP -> Dropout -> Linear.

    forward() tra ve logits (B, num_classes).
    """
    def __init__(self, num_classes=6, width_mult=1.0, dropout=0.2,
                 stage_cfg=_STAGE_CFG_TURBO_64, stem_c=32, stem_stride=2, head_c=192):
        super().__init__()

        def _c(v): return max(1, int(v * width_mult))

        self.stem = ConvBNHSwish(3, _c(stem_c), 3, stride=stem_stride, padding=1)

        stages, in_c = [], _c(stem_c)
        for n, out, t, strides in stage_cfg:
            out_c  = _c(out)
            blocks = []
            for s in strides:
                blocks.append(FusedMBConv(in_c, out_c, s, t))
                in_c = out_c
            stages.append(nn.Sequential(*blocks))
        self.stages = nn.ModuleList(stages)

        head_out = _c(head_c)
        self.head_conv   = ConvBNHSwish(in_c, head_out, 1)
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

    def _reset_fusion_cache(self):
        for m in self.modules():
            if isinstance(m, (ConvBNHSwish, ConvBN)):
                m._fused = None

    def load_state_dict(self, state_dict, strict=True, **kwargs):
        # nap trong so moi -> cache BN da gop (neu co) khong con dung
        out = super().load_state_dict(state_dict, strict=strict, **kwargs)
        self._reset_fusion_cache()
        return out

    def forward_features(self, x):
        """Feature map truoc global pool, shape (B, C, H, W)."""
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        return self.head_conv(x)

    def forward(self, x):
        x = self.forward_features(x)
        x = x.mean(dim=(2, 3))          # global avg pool
        x = self.dropout(x)
        return self.classifier(x)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============================================================
# build_model  (bat buoc boi model registry)
# ============================================================

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
        name            : chi co 'ju_lite_mobile_att_64_turbo'.
        num_classes     : so lop dau ra.
        in_channels     : chi ho tro 3.
        pretrained      : True -> nap trong so tu pretrained_path.
        compile         : torch.compile(mode='reduce-overhead').
        fp16            : model.half() (GPU only; nho cast input).
    """
    if name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown variant '{name}'. Available: {list(MODEL_CONFIGS.keys())}")
    if in_channels != 3:
        raise ValueError("JuLiteMobileAtt hien chi ho tro in_channels=3.")

    # input_size van duoc chap nhan (train.py co the truyen) nhung khong dung toi.
    kwargs.pop("input_size", None)

    cfg   = MODEL_CONFIGS[name]
    model = JuLiteMobileAtt(
        num_classes=num_classes,
        width_mult=cfg["width_mult"],
        stage_cfg=_STAGE_CFG_TURBO_64,
        stem_c=cfg["stem_c"],
        stem_stride=cfg["stem_stride"],
        head_c=cfg["head_c"],
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
    """Giu lai de tuong thich voi script goi fuse_model(). Moi Conv+BN cua model nay
    da tu gop khi eval (ConvBNHSwish / ConvBN) nen ham chi chuyen sang eval()."""
    return model.eval()


# ============================================================
# Training utilities
# ============================================================

def get_optimizer(model, lr=5e-4, momentum=0.9, weight_decay=5e-2):
    """AdamW - paper Section 4.3."""
    return torch.optim.AdamW(model.parameters(),
                              lr=lr, betas=(momentum, 0.999),
                              weight_decay=weight_decay)


def get_scheduler(optimizer, total_epochs=200, warmup_epochs=1):
    """Cosine annealing with 1-epoch linear warm-up - paper Section 4.3."""
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1e-2, end_factor=1.0, total_iters=warmup_epochs)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_epochs - warmup_epochs, eta_min=1e-6)
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs])


def compute_class_weights(class_counts, power=0.5, device="cpu"):
    """Trong so lop theo nghich dao tan suat, lam muot bang `power`.

    w_c = (N / (C * n_c)) ** power, roi chuan hoa de trung binh = 1.
    power=1.0: can bang hoan toan; power=0.5: nhe tay; power=0: khong dung.
    class_counts: so anh train moi lop, theo dung thu tu nhan.
    """
    c = torch.as_tensor(class_counts, dtype=torch.float32)
    w = (c.sum() / (len(c) * c)) ** power
    return (w / w.mean()).to(device)


def get_loss(num_classes=6, label_smoothing=0.1, class_weights=None):
    """Cross-entropy + label smoothing. class_weights: tensor (num_classes,) tuy chon."""
    return nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)


def mixup_data(x, y, alpha=0.2, device="cpu"):
    """Mixup augmentation (Zhang et al., 2018).

        x_mix, y_a, y_b, lam = mixup_data(x, y, alpha=0.2, device=device)
        logits = model(x_mix)
        loss = lam * criterion(logits, y_a) + (1-lam) * criterion(logits, y_b)
    """
    lam = float(torch.distributions.Beta(alpha, alpha).sample()) if alpha > 0 else 1.0
    idx = torch.randperm(x.size(0), device=device)
    return lam * x + (1 - lam) * x[idx], y, y[idx], lam


def get_transforms(split="train", input_size=224):
    """Pipeline augmentation (paper Section 4.2). Voi model nay goi input_size=64."""
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


# ============================================================
# Self-test + benchmark FPS (BS=1, input 64x64)
#
#   python ju_lite_mobile_att.py
#
# Che do do:
#   eager, KHONG gop BN : LAZY_FUSE=False
#   eager + gop BN      : mac dinh
#   gop BN + fp16       : half precision (GPU)
#   gop BN + fp16 + CUDA graph : replay 1 graph (GPU)
# Model khoi tao ngau nhien (du de do toc do).
# ============================================================

if __name__ == "__main__":
    import time

    device      = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_CLASSES = 5          # sua cho khop dataset
    RUNS, WARMUP = 300, 50
    name = DEFAULT_VARIANT
    size = MODEL_CONFIGS[name]["input_size"]

    # ---- 1) shape check ----
    m = build_model(name, num_classes=NUM_CLASSES).eval()
    with torch.no_grad():
        fmap = m.forward_features(torch.randn(2, 3, size, size))
        out  = m(torch.randn(2, 3, size, size))
    print(f"variant     : {name}")
    print(f"params      : {m.count_parameters() / 1e3:.1f}K")
    print(f"feature map : {tuple(fmap.shape)}")
    print(f"logits      : {tuple(out.shape)}")
    assert tuple(fmap.shape) == (2, 192, 8, 8), "feature map cuoi phai la 8x8 x 192 kenh"
    assert tuple(out.shape) == (2, NUM_CLASSES)

    # ---- 1b) gop BN khong duoc lam doi ket qua ----
    for bn in (x for x in m.modules() if isinstance(x, nn.BatchNorm2d)):
        bn.running_mean.normal_(0, 0.1); bn.running_var.uniform_(0.5, 1.5)
        bn.weight.data.uniform_(0.5, 1.5); bn.bias.data.normal_(0, 0.1)
    xx = torch.randn(4, 3, size, size)
    with torch.no_grad():
        LAZY_FUSE = False
        ref = m(xx)
        LAZY_FUSE = True
        m._reset_fusion_cache()
        fus = m(xx)
    print(f"gop BN      : max|diff| = {(ref - fus).abs().max().item():.2e}")
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
            for _ in range(3):                     # warm-up: tao cache BN da gop truoc khi capture
                model(static_x)
        torch.cuda.current_stream().wait_stream(side)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g), torch.no_grad():
            model(static_x)
        return g.replay

    print(f"PyTorch : {torch.__version__}")
    print(f"Device  : {device}" + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""))
    print(f"{'Mode':<34} {'ms':>7} {'FPS':>8}")
    print("-" * 52)

    modes = [("eager, khong gop BN", False, False, False),
             ("eager + gop BN",      True,  False, False)]
    if device == "cuda":
        modes += [("gop BN + fp16",              True, True, False),
                  ("gop BN + fp16 + CUDA graph", True, True, True)]
    for label, lazy, half, graph in modes:
        LAZY_FUSE = lazy
        mm = build_model(name, num_classes=NUM_CLASSES).to(device).eval()
        x = torch.randn(1, 3, size, size, device=device)
        if half:
            mm, x = mm.half(), x.half()
        fn = cuda_graph_fn(mm, x) if graph else (lambda mm=mm, x=x: mm(x))
        ms  = time_fn(fn)
        fps = 1000.0 / ms
        tag = "  <- 500-1000" if 500 <= fps <= 1000 else ("  (>1000)" if fps > 1000 else "")
        print(f"  {label:<32} {ms:>7.3f} {fps:>8.0f}{tag}")
    LAZY_FUSE = True
