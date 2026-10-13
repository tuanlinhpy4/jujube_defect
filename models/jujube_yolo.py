"""
Jujube-YOLO (flat_64) - classification variant rut gon
------------------------------------------------------------------------

Dua tren: Wang, L. et al. (2025). "Jujube-YOLO: a precise jujube fruit
recognition model in unstructured environments." Expert Systems With
Applications, 291, 128530. https://doi.org/10.1016/j.eswa.2025.128530

File nay la ban RUT GON cua jujube_yolo.py (v3): chi giu lai variant
"jujube_yolo_flat_64" va nhung thanh phan ma variant nay thuc su dung
(Conv, PlainBottleneck, FlatBackbone, head GAP + Linear). Cac variant khac
(jujube_yolo, _s, _n, ablation, jujube_yolo_64, ...) va cac module chi phuc
vu chung (DCSE, RCM/RCA, C3k2, MBCA, SPPF, neck FPN/PAN, DetectHead,
DropPath) da bi xoa.

Dam bao KHONG doi ket qua so voi file goc (cung seed -> cung ket qua):
    - Ten module / ten tham so giu nguyen (backbone_neck.stem, down1,
      block1, down2, block2, classifier) -> state_dict cua checkpoint cu
      load duoc truc tiep, khong can doi key.
    - Thu tu khoi tao module giu nguyen (stem -> down1 -> block1 -> down2 ->
      block2 -> classifier) va _init_weights giu nguyen -> cung seed cho
      cung bo trong so khoi tao.
    - Cong thuc forward giu nguyen.

Kien truc (width_mult=0.50, input 64x64):
    stem  Conv 3x3/s2   3 ->  32   : 64 -> 32
    down1 Conv 3x3/s2  32 ->  64   : 32 -> 16
    block1 PlainBottleneck(64)
    down2 Conv 3x3/s2  64 -> 128   : 16 -> 8
    block2 PlainBottleneck(128)
    GAP (mean H,W) -> Linear(128, num_classes)
    => tong stride /8, feature map cuoi 8x8.

Chay self-test:
    python jujube_yolo.py
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


# ============================================================
# Registry metadata
# ============================================================

ARCH_NAME = "jujube_yolo"

# Cac key use_dcse/use_rcm/use_mbca/use_neck/depth_mult duoc giu lai chi de
# tuong thich voi script ben ngoai co the doc MODEL_CONFIGS; build_model()
# ben duoi chi dung width_mult va flat_*.
MODEL_CONFIGS = {
    "jujube_yolo_flat_64": {
        "width_mult": 0.50, "depth_mult": 1.0,
        "use_dcse": False, "use_rcm": False, "use_mbca": False,
        "use_neck": False, "use_flat_backbone": True,
        "flat_stem_k": 3, "flat_stem_stride": 2, "flat_down2_stride": 2,
        "input_size": 64,
    },
}

DEFAULT_VARIANT = "jujube_yolo_flat_64"


# ============================================================
# Utility
# ============================================================

def _round_ch(c: float, divisor: int = 8):
    return int(math.ceil(c / divisor) * divisor)


def autopad(k, p=None, d=1):
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class Conv(nn.Module):
    """Conv2d -> BatchNorm2d -> SiLU (khoi conv chuan cua YOLO)."""

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class _GAPFull(nn.Module):
    """Global average pool -> (B, C, 1, 1).

    Dung x.mean(dim=[2, 3]) thay cho nn.AdaptiveAvgPool2d vi backward cua
    AdaptiveAvgPool2d khong deterministic tren CUDA, se crash khi train.py
    bat torch.use_deterministic_algorithms(True). Gia tri tinh ra giong het.
    """
    def forward(self, x):
        return x.mean(dim=[2, 3], keepdim=True)


# ============================================================
# Bottleneck + FlatBackbone
# ============================================================

class PlainBottleneck(nn.Module):
    """Bottleneck chuan (Conv3x3-Conv3x3 + residual)."""

    def __init__(self, c):
        super().__init__()
        self.cv1 = Conv(c, c, k=3)
        self.cv2 = Conv(c, c, k=3)

    def forward(self, x):
        return x + self.cv2(self.cv1(x))


class FlatBackbone(nn.Module):
    """Backbone phang, it layer: stem -> down1 -> block1 -> down2 -> block2.

    Voi flat_64: stem_k=3, stem_stride=2, down2_stride=2 -> tong stride /8
    (input 64px cho feature map cuoi 8x8).
    """

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
# Model - CLASSIFICATION (GAP + Linear), tuong thich train.py
# ============================================================

class JujubeYOLO(nn.Module):
    """FlatBackbone -> GAP -> Linear(num_classes).

    forward() tra ve logits (B, num_classes), dung truc tiep voi
    nn.CrossEntropyLoss. Ten thuoc tinh `backbone_neck` duoc giu nguyen
    nhu file goc de checkpoint cu load duoc.
    """

    def __init__(
        self,
        num_classes: int = 4,
        in_channels: int = 3,
        width_mult: float = 0.5,
        flat_stem_k: int = 3,
        flat_stem_stride: int = 2,
        flat_down2_stride: int = 2,
    ):
        super().__init__()

        self.backbone_neck = FlatBackbone(
            in_channels=in_channels, width_mult=width_mult,
            stem_k=flat_stem_k, stem_stride=flat_stem_stride,
            down2_stride=flat_down2_stride,
        )

        self.gap = _GAPFull()
        self.classifier = nn.Linear(self.backbone_neck.out_channels[-1], num_classes)

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
        feats = self.backbone_neck(x)               # tuple 1 phan tu (feature map cuoi)
        feat = self.gap(feats[-1]).flatten(1)
        return self.classifier(feat)

    def forward_features(self, x):
        """Tra ve feature map tho (huu ich cho Grad-CAM)."""
        return self.backbone_neck(x)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self):
        p = self.count_parameters()
        return f"JujubeYOLO(params={p:,}, size={p * 4 / 1024 / 1024:.2f} MB)"


# ============================================================
# build_model  (bat buoc boi model registry)
# ============================================================

def build_model(name: str = DEFAULT_VARIANT,
                num_classes: int = 4,
                in_channels: int = 3,
                **kwargs):

    cfg = MODEL_CONFIGS.get(name)
    if cfg is None:
        raise ValueError(f"Unknown variant '{name}'. Choose from: {list(MODEL_CONFIGS)}")

    # input_size van duoc chap nhan (train.py co the truyen vao) nhung
    # FlatBackbone khong dung toi -> bo qua, giong hanh vi file goc.
    kwargs.pop("input_size", None)
    if kwargs:
        raise TypeError(f"build_model() got unexpected keyword arguments: {sorted(kwargs)}")

    model = JujubeYOLO(
        num_classes=num_classes,
        in_channels=in_channels,
        width_mult=cfg["width_mult"],
        flat_stem_k=cfg["flat_stem_k"],
        flat_stem_stride=cfg["flat_stem_stride"],
        flat_down2_stride=cfg["flat_down2_stride"],
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
    name = DEFAULT_VARIANT
    size = MODEL_CONFIGS[name]["input_size"]

    m = build_model(name, num_classes=5).eval()
    x = torch.randn(2, 3, size, size)
    with torch.no_grad():
        y = m(x)
        f = m.forward_features(x)

    print(f"variant      : {name}")
    print(f"model        : {m}")
    print(f"input        : {tuple(x.shape)}")
    print(f"feature map  : {tuple(f[0].shape)}")
    print(f"logits       : {tuple(y.shape)}")

    assert tuple(f[0].shape) == (2, 128, 8, 8), "feature map cuoi phai la 8x8 x 128 kenh"
    assert tuple(y.shape) == (2, 5)
    print("OK")
