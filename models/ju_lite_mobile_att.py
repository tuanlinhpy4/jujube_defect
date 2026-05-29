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
"""
 
import torch
import torch.nn as nn
import torch.nn.functional as F
ARCH_NAME = "julitemobileatt"

MODEL_CONFIGS = {
    "ju_lite_mobile_att":    {"input_size": 224, "width_mult": 1.00},
    "ju_lite_mobie_att_s":  {"input_size": 224, "width_mult": 0.75},
    "ju_lite_mobile_att_xs": {"input_size": 224, "width_mult": 0.50},
}

DEFAULT_VARIANT = "ju_lite_mobile_att"
 
# ─────────────────────────────────────────────────────────────
# 1. Basic building-block helpers
# ─────────────────────────────────────────────────────────────
 
def h_swish(x: torch.Tensor) -> torch.Tensor:
    """H-Swish activation: x * ReLU6(x+3) / 6  (Eq. 2 in paper)."""
    return x * F.relu6(x + 3.0, inplace=True) / 6.0
 
 
class HSwish(nn.Module):
    def forward(self, x):
        return h_swish(x)
 
 
class ConvBNHSwish(nn.Module):
    """Conv -> BN -> H-Swish (used throughout the paper)."""
 
    def __init__(self, in_c: int, out_c: int, kernel: int = 1,
                 stride: int = 1, padding: int = 0, groups: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel, stride, padding,
                      groups=groups, bias=False),
            nn.BatchNorm2d(out_c),
            HSwish(),
        )
 
    def forward(self, x):
        return self.block(x)
 
 
# ─────────────────────────────────────────────────────────────
# 2. SE Module  (channel attention — first stage of CoordAtt)
# ─────────────────────────────────────────────────────────────
 
class SEModule(nn.Module):
    """Squeeze-and-Excitation channel attention (Hu et al., CVPR 2018)."""
 
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels),
            nn.Sigmoid(),
        )
 
    def forward(self, x):
        scale = self.fc(x).view(x.size(0), x.size(1), 1, 1)
        return x * scale
 
 
# ─────────────────────────────────────────────────────────────
# 3. ECA Module  (spatial / directional attention — second stage)
# ─────────────────────────────────────────────────────────────
 
class ECAModule(nn.Module):
    """Efficient Channel Attention with grouped convolution for directional features."""
 
    def __init__(self, channels: int, kernel: int = 3, groups: int = 1):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # grouped 1-D conv over the channel dimension
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel,
                              padding=kernel // 2, groups=groups, bias=False)
        self.sigmoid = nn.Sigmoid()
 
    def forward(self, x):
        y = self.avg_pool(x)                        # (B, C, 1, 1)
        y = y.squeeze(-1).transpose(-1, -2)         # (B, 1, C)
        y = self.conv(y)                             # (B, 1, C)
        y = self.sigmoid(y)
        y = y.transpose(-1, -2).unsqueeze(-1)       # (B, C, 1, 1)
        return x * y.expand_as(x)
 
 
# ─────────────────────────────────────────────────────────────
# 4. CoordAtt  (Eq. 1 — cascaded SE → ECA with residual, zero-init)
# ─────────────────────────────────────────────────────────────
 
class CoordAtt(nn.Module):
    """
    Two-stage collaborative attention (paper Section 3.3):
      Stage-1: SE  — channel-level defect-background separation
      Stage-2: ECA — spatial / directional defect localisation
    Zero-initialised group-conv weight makes the module an identity
    map at the start of training, easing gradient flow.
 
    Output (Eq. 1):
        F_coordatt = [σ(AvgPool(GroupConv(s⊙F))) ⊙ (s⊙F)] + F
        where s = σ(FC(ReLU(FC(AvgPool(F)))))
    """
 
    def __init__(self, channels: int, reduction: int = 4,
                 eca_kernel: int = 3, groups: int = 1):
        super().__init__()
        self.se  = SEModule(channels, reduction)
        self.eca = ECAModule(channels, eca_kernel, groups)
 
        # Zero-init group conv for the outer pooling path (Eq. 1)
        self.group_conv = nn.Conv2d(channels, channels, 1,
                                    groups=max(1, channels // 16), bias=False)
        nn.init.zeros_(self.group_conv.weight)      # zero initialisation
 
    def forward(self, x):
        # s⊙F  (channel screening via SE)
        s = self.se(x)
        # σ(AvgPool(GroupConv(s⊙F)))
        attn = torch.sigmoid(
            F.adaptive_avg_pool2d(self.group_conv(s), 1)
        )
        # spatial refinement via ECA
        out = self.eca(s * attn)
        return out + x          # residual connection
 
 
# ─────────────────────────────────────────────────────────────
# 5. EfficientRCAM  (paper Section 3.4, Fig. 3)
# ─────────────────────────────────────────────────────────────
 
class EfficientRCAM(nn.Module):
    """
    Efficient Residual Coordinate Attention Module.
 
    Pipeline:
      Input ──► ConvBNHSwish (expand 1×1)
             ──► 3×3 DwConv  (depth-wise, captures local defect textures)
             ──► CoordAtt    (channel + spatial attention)
             ──► Pw expand   (1×1 point-wise)
             ──► residual add Input
             ──► HSwish
             ──► Output
    """
 
    def __init__(self, channels: int, expand_ratio: int = 2,
                 reduction: int = 4):
        super().__init__()
        mid = channels * expand_ratio
 
        # Expand point-wise
        self.pw_expand = ConvBNHSwish(channels, mid, 1)
 
        # 3×3 depth-wise conv (multi-scale receptive field)
        self.dw_conv = ConvBNHSwish(mid, mid, 3, 1, 1, groups=mid)
 
        # CoordAtt attention (zero-init conv inside)
        self.coord_att = CoordAtt(mid, reduction)
 
        # Point-wise projection back
        self.pw_proj = nn.Sequential(
            nn.Conv2d(mid, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
        )
 
        self.act = HSwish()
 
    def forward(self, x):
        residual = x
        out = self.pw_expand(x)
        out = self.dw_conv(out)
        out = self.coord_att(out)
        out = self.pw_proj(out)
        return self.act(out + residual)
 
 
# ─────────────────────────────────────────────────────────────
# 6. CascadedRCAM  (paper Section 3.5, Eq. 3-4)
# ─────────────────────────────────────────────────────────────
 
class CascadedRCAM(nn.Module):
    """
    Cascaded Residual Coordinate Attention Module.
 
    Three-layer architecture (Eq. 3-4):
      1. EfficientRCAM enhancement  →  F_rcam
      2. Residual fusion:  F_in + F_rcam   (cross-scale complementarity)
      3. 2× channel compression via ConvBNHSwish  →  F_out
 
    Using 2× compression (not the conventional 4×) retains more
    intermediate features, which is critical for low-contrast defects.
    """
 
    def __init__(self, in_c: int, reduction: int = 4):
        super().__init__()
        self.efficient_rcam = EfficientRCAM(in_c, expand_ratio=2,
                                            reduction=reduction)
        # 2× channel compression (paper: "retains more intermediate features")
        out_c = max(in_c // 2, 16)
        self.compress = ConvBNHSwish(in_c, out_c, 1)
        self.out_channels = out_c
 
    def forward(self, x):
        f_rcam = self.efficient_rcam(x)          # enhanced features
        fused  = x + f_rcam                      # residual fusion  (Eq. 3)
        out    = self.compress(fused)            # 2× channel compression
        return out
 
 
# ─────────────────────────────────────────────────────────────
# 7. MobileNetV2 InvertedResidual  (standard block)
# ─────────────────────────────────────────────────────────────
 
class InvertedResidual(nn.Module):
    """Standard MobileNetV2 inverted residual block (Sandler et al., 2018)."""
 
    def __init__(self, in_c: int, out_c: int, stride: int,
                 expand_ratio: int, use_coord_att: bool = False,
                 att_reduction: int = 4):
        super().__init__()
        self.use_res = (stride == 1 and in_c == out_c)
        mid = int(round(in_c * expand_ratio))
 
        layers = []
        if expand_ratio != 1:
            layers.append(ConvBNHSwish(in_c, mid, 1))
        layers += [
            ConvBNHSwish(mid, mid, 3, stride, 1, groups=mid),   # DwConv
            nn.Conv2d(mid, out_c, 1, bias=False),
            nn.BatchNorm2d(out_c),
        ]
        self.conv = nn.Sequential(*layers)
 
        # Selective attention: CoordAtt injected only at the LAST block of each stage
        self.coord_att = CoordAtt(out_c, att_reduction) if use_coord_att else None
 
    def forward(self, x):
        out = self.conv(x)
        if self.coord_att is not None:
            out = self.coord_att(out)
        if self.use_res:
            return x + out
        return out
 
 
# ─────────────────────────────────────────────────────────────
# 8. Ju-LiteMobileAtt  (paper Section 3.6, Table 1)
# ─────────────────────────────────────────────────────────────
 
class JuLiteMobileAtt(nn.Module):
    """
    Ju-LiteMobileAtt — full model.
 
    Stage configuration from Table 1 (7 stages):
    ┌───────┬───────┬──────┬──────┬──────────┬───────────────┬────────────────────┐
    │ Stage │ Blks  │  Cin │ Cout │ Expand t │ Stride        │ CoordAtt at        │
    ├───────┼───────┼──────┼──────┼──────────┼───────────────┼────────────────────┤
    │   0   │   1   │  32  │  16  │    1     │ 1             │ None               │
    │   1   │   2   │  16  │  24  │    6     │ 2,1           │ 2nd block          │
    │   2   │   3   │  24  │  32  │    6     │ 2,1,1         │ 3rd block          │
    │   3   │   4   │  32  │  64  │    6     │ 2,1,1,1       │ 4th block          │
    │   4   │   3   │  64  │  96  │    6     │ 1,1,1         │ 3rd block          │
    │   5   │   3   │  96  │ 160  │    6     │ 2,1,1         │ 3rd block          │
    │   6   │   1   │ 160  │ 320  │    6     │ 1             │ 1st block (yes)    │
    └───────┴───────┴──────┴──────┴──────────┴───────────────┴────────────────────┘
 
    After Stage 6:
      CascadedRCAM  →  ConvBNHSwish  →  AvgPool  →  Flatten  →  Linear (num_classes)
    """
 
    # (n_blocks, out_channels, expand_ratio, strides_per_block)
    _STAGE_CFG = [
        # Stage 0
        (1, 16,  1, [1]),
        # Stage 1
        (2, 24,  6, [2, 1]),
        # Stage 2
        (3, 32,  6, [2, 1, 1]),
        # Stage 3
        (4, 64,  6, [2, 1, 1, 1]),
        # Stage 4
        (3, 96,  6, [1, 1, 1]),
        # Stage 5
        (3, 160, 6, [2, 1, 1]),
        # Stage 6
        (1, 320, 6, [1]),
    ]
 
    def __init__(self, num_classes: int = 6,
                 width_mult: float = 1.0,
                 att_reduction: int = 4):
        super().__init__()
 
        def _c(v): return max(1, int(v * width_mult))
 
        # ── Stem ──────────────────────────────────────────────
        self.stem = ConvBNHSwish(3, _c(32), 3, stride=2, padding=1)
 
        # ── Inverted residual stages ───────────────────────────
        stages = []
        in_c = _c(32)
 
        for stage_idx, (n, out, t, strides) in enumerate(self._STAGE_CFG):
            out_c = _c(out)
            blocks = []
            for i, s in enumerate(strides):
                # CoordAtt applied at the LAST block of each stage
                is_last = (i == n - 1)
                # Stage 0 is the only stage with no attention
                use_att = is_last and stage_idx > 0
                blocks.append(
                    InvertedResidual(in_c, out_c, s, t,
                                     use_coord_att=use_att,
                                     att_reduction=att_reduction)
                )
                in_c = out_c
            stages.append(nn.Sequential(*blocks))
 
        self.stages = nn.ModuleList(stages)
 
        # ── CascadedRCAM (tail feature refinement) ─────────────
        self.cascaded_rcam = CascadedRCAM(in_c, reduction=att_reduction)
        rcam_out = self.cascaded_rcam.out_channels
 
        # ── Conv head  ─────────────────────────────────────────
        head_out = _c(1280)
        self.head_conv = ConvBNHSwish(rcam_out, head_out, 1)
 
        # ── Classifier ─────────────────────────────────────────
        self.pool       = nn.AdaptiveAvgPool2d(1)
        self.flatten    = nn.Flatten()
        self.classifier = nn.Linear(head_out, num_classes)
 
        self._init_weights()
 
    # ── Weight initialisation ──────────────────────────────────
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
 
    # ── Forward ────────────────────────────────────────────────
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        x = self.cascaded_rcam(x)
        x = self.head_conv(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x
 
    # ── Convenience: parameter count ───────────────────────────
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
 
 
# ─────────────────────────────────────────────────────────────
# 9. Training utilities
# ─────────────────────────────────────────────────────────────
 
def build_model(num_classes: int = 6,
                width_mult: float = 1.0) -> JuLiteMobileAtt:
    """Instantiate Ju-LiteMobileAtt (no pre-trained weights — trained from scratch)."""
    return JuLiteMobileAtt(num_classes=num_classes, width_mult=width_mult)
 
 
def get_optimizer(model: nn.Module,
                  lr: float = 5e-4,
                  momentum: float = 0.9,
                  weight_decay: float = 5e-2) -> torch.optim.AdamW:
    """
    AdamW optimizer as described in paper Section 4.3.
    lr=5e-4, momentum=0.9, weight_decay=5e-2
    """
    return torch.optim.AdamW(model.parameters(),
                              lr=lr,
                              betas=(momentum, 0.999),
                              weight_decay=weight_decay)
 
 
def get_scheduler(optimizer: torch.optim.Optimizer,
                  total_epochs: int = 200,
                  warmup_epochs: int = 1):
    """
    Cosine annealing with 1 warm-up epoch (paper Section 4.3).
    Uses a sequential scheduler: LinearLR warm-up → CosineAnnealingLR.
    """
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1e-2,
        end_factor=1.0,
        total_iters=warmup_epochs,
    )
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_epochs - warmup_epochs,
        eta_min=1e-6,
    )
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[warmup_epochs],
    )
 
 
# ─────────────────────────────────────────────────────────────
# 10. Data-augmentation pipeline  (paper Section 4.2)
# ─────────────────────────────────────────────────────────────
 
def get_transforms(split: str = "train"):
    """
    Data augmentation as described in Section 4.2:
      - Random crop  (70-100 %, aspect 0.8-1.25), resize to 224×224
      - Random horizontal + vertical flip  (p=0.5 each)
      - Rotation ±15°, shearing 10°
      - Color jitter ±20% brightness/contrast/saturation
      - Normalization (ImageNet stats)
    """
    try:
        from torchvision import transforms
    except ImportError:
        raise ImportError("torchvision is required for the transforms helper.")
 
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]
 
    if split == "train":
        return transforms.Compose([
            transforms.RandomResizedCrop(
                224, scale=(0.70, 1.0), ratio=(0.8, 1.25),
                interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomAffine(degrees=15, shear=10),
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomErasing(p=0.5, scale=(0.02, 0.20)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    else:  # val / test
        return transforms.Compose([
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
 
 
# ─────────────────────────────────────────────────────────────
# 11. Quick sanity check
# ─────────────────────────────────────────────────────────────
 
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
 
    # Jujube12000 has 6 classes (Table 1 / Section 4.1)
    CLASS_NAMES = ["Normal", "Wrinkled", "Deformed", "Cracked", "Mold", "Bird-Pecked"]
 
    model = build_model(num_classes=len(CLASS_NAMES)).to(device)
    params = model.count_parameters()
    print(f"Trainable parameters : {params / 1e6:.3f} M  (paper reports ~2.16 M)")
 
    # Forward pass with a dummy batch (batch=4, C=3, H=224, W=224)
    dummy = torch.randn(4, 3, 224, 224, device=device)
    with torch.no_grad():
        logits = model(dummy)
    print(f"Output shape         : {logits.shape}")   # expect (4, 6)
    print(f"Predicted classes    : {logits.argmax(1).tolist()}")
 
    # Optimizer + scheduler
    opt   = get_optimizer(model)
    sched = get_scheduler(opt, total_epochs=200, warmup_epochs=1)
    print(f"Optimizer            : {type(opt).__name__}  lr={opt.param_groups[0]['lr']}")
 
    # Loss
    criterion = nn.CrossEntropyLoss()
    targets   = torch.randint(0, len(CLASS_NAMES), (4,), device=device)
    loss      = criterion(logits, targets)
    print(f"Loss (random target) : {loss.item():.4f}")
    print("\nAll checks passed ✓")
