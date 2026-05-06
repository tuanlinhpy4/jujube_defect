"""EdgeJuNet — Lightweight CNN with Multi-Kernel IR + Coordinate Attention.

Architectures:
    EdgeJuNet: MK-IR (dilated DW) + CoordAttention + DropPath.
    Inspired by MvANet, MobileNetV2, JujubeNet, RJ-TinyViT, Ju-LiteMobileAtt.
"""

from __future__ import annotations

from typing import Any, Dict, List

import torch
import torch.nn as nn


ARCH_NAME = "EdgeJuNet"
ARCH_ALIASES = ["edgejunet"]


# ---------------------------------------------------------------------------
# DropPath / Stochastic Depth
# ---------------------------------------------------------------------------

def _drop_path(x: torch.Tensor, drop_prob: float, training: bool):
    if drop_prob == 0.0 or not training:
        return x
    keep = 1.0 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    mask = torch.bernoulli(torch.full(shape, keep, device=x.device, dtype=x.dtype))
    return x / keep * mask


class DropPath(nn.Module):
    """Per-sample stochastic depth (used in residual branches only)."""

    def __init__(self, p: float = 0.0):
        super().__init__()
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _drop_path(x, self.p, self.training)


# ---------------------------------------------------------------------------
# Coordinate Attention (Hou et al., CVPR 2021)
# ---------------------------------------------------------------------------

class CoordAttention(nn.Module):
    """Coordinate Attention — spatial-aware channel attention using only 1x1 convs.

    Cheaper than CBAM while preserving positional information.
    """

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.conv1 = nn.Conv2d(channels, mid, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid)
        self.act = nn.ReLU(inplace=True)
        self.conv_h = nn.Conv2d(mid, channels, 1, bias=False)
        self.conv_w = nn.Conv2d(mid, channels, 1, bias=False)

    def _weights(self, x: torch.Tensor) -> torch.Tensor:
        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        y = torch.cat([x_h, x_w], dim=2)
        y = self.act(self.bn1(self.conv1(y)))
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        return torch.sigmoid(self.conv_h(x_h)) * torch.sigmoid(self.conv_w(x_w))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self._weights(x)

    def get_attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        """Return attention weights for visualization."""
        return self._weights(x)


# ---------------------------------------------------------------------------
# Conv-BN-ReLU helper
# ---------------------------------------------------------------------------

def _cbr(in_ch: int, out_ch: int, kernel: int = 1, stride: int = 1,
         groups: int = 1, dilation: int = 1) -> nn.Sequential:
    pad = ((kernel - 1) // 2) * dilation
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel, stride=stride, padding=pad,
                  dilation=dilation, groups=groups, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


# ---------------------------------------------------------------------------
# Channel rounding utility
# ---------------------------------------------------------------------------

def _round_ch(c: float, divisor: int = 8, min_value: int = None) -> int:
    if min_value is None:
        min_value = divisor
    new_c = max(min_value, int(c + divisor / 2) // divisor * divisor)
    if new_c < 0.9 * c:
        new_c += divisor
    return new_c


# ---------------------------------------------------------------------------
# Inverted Residual (MobileNetV2-style)
# ---------------------------------------------------------------------------

class InvertedResidual(nn.Module):
    """Standard MobileNetV2 inverted residual block."""

    def __init__(self, in_ch: int, out_ch: int, stride: int,
                 expand_ratio: float, drop_path: float = 0.0):
        super().__init__()
        assert stride in (1, 2)
        hidden = int(in_ch * expand_ratio)
        self.use_res = (stride == 1) and (in_ch == out_ch)

        layers: List[nn.Module] = []
        if expand_ratio != 1:
            layers.append(_cbr(in_ch, hidden, 1))
        layers.extend([
            _cbr(hidden, hidden, 3, stride=stride, groups=hidden),
            nn.Conv2d(hidden, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
        ])
        self.block = nn.Sequential(*layers)
        self.drop = DropPath(drop_path) if (drop_path > 0 and self.use_res) else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.block(x)
        if self.use_res:
            out = self.drop(out) + x
        return out


# ---------------------------------------------------------------------------
# Multi-Kernel Inverted Residual — EdgeJuNet's signature block
# ---------------------------------------------------------------------------

class MKInvertedResidual(nn.Module):
    """Multi-Kernel IR: two parallel DWConv branches (dilation=1 and 2).

    Gives multi-scale receptive field (3x3 + effective 5x5) at the
    same FLOPs cost as a single 3x3 DWConv (channel-split trick).

    Inspired by MvANet's multi-eye design and RJ-TinyViT's MK Block.
    """

    def __init__(self, in_ch: int, out_ch: int, stride: int,
                 expand_ratio: float, drop_path: float = 0.0):
        super().__init__()
        assert stride in (1, 2)
        hidden = int(in_ch * expand_ratio)
        assert hidden % 2 == 0
        self.half = hidden // 2
        self.use_res = (stride == 1) and (in_ch == out_ch)

        self.expand = _cbr(in_ch, hidden, 1)
        self.dw1 = _cbr(self.half, self.half, 3, stride=stride,
                        groups=self.half, dilation=1)
        self.dw2 = _cbr(self.half, self.half, 3, stride=stride,
                        groups=self.half, dilation=2)
        self.project = nn.Sequential(
            nn.Conv2d(hidden, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.drop = DropPath(drop_path) if (drop_path > 0 and self.use_res) else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.expand(x)
        h1, h2 = torch.split(h, self.half, dim=1)
        h = torch.cat([self.dw1(h1), self.dw2(h2)], dim=1)
        out = self.project(h)
        if self.use_res:
            out = self.drop(out) + x
        return out


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class EdgeJuNet(nn.Module):
    """EdgeJuNet — jujube classifier optimized for Pi4 CPU deployment.

    Architecture:
        1. Stem       — Conv3x3 stride=2
        2. Block 1    — MK-IR + CoordAttention
        3. Block 2    — IR + CoordAttention
        4. Block 3    — IR (stride=2) + IR (stride=1 + DropPath) + CoordAttention
        5. Head       — 1x1 expand + GAP + FC
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 4,
        width_mult: float = 1.0,
        drop_path: float = 0.1,
    ):
        super().__init__()
        c = lambda x: _round_ch(x * width_mult)

        self.config: Dict[str, Any] = {
            "width_mult": width_mult,
            "drop_path": drop_path,
            "num_classes": num_classes,
        }

        c_stem = c(16); c_b1 = c(24); c_b2 = c(32)
        c_b3 = c(48);   c_head = c(96)

        self.stem = _cbr(in_channels, c_stem, 3, stride=2)

        self.block1 = nn.Sequential(
            MKInvertedResidual(c_stem, c_b1, stride=2, expand_ratio=4),
            CoordAttention(c_b1),
        )

        self.block2 = nn.Sequential(
            InvertedResidual(c_b1, c_b2, stride=2, expand_ratio=3),
            CoordAttention(c_b2),
        )

        self.block3 = nn.Sequential(
            InvertedResidual(c_b2, c_b3, stride=2, expand_ratio=3),
            InvertedResidual(c_b3, c_b3, stride=1, expand_ratio=3,
                             drop_path=drop_path),
            CoordAttention(c_b3),
        )

        self.head_conv = _cbr(c_b3, c_head, 1)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(c_head, num_classes)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.head_conv(x)
        x = self.global_pool(x).flatten(1)
        return self.classifier(x)

    def forward_with_features(self, x: torch.Tensor) -> Dict[str, Any]:
        """Forward pass returning intermediate features for visualization."""
        result: Dict[str, Any] = {}
        x = self.stem(x);         result["stem"] = x
        x = self.block1(x);       result["block1"] = x
        x = self.block2(x);       result["block2"] = x
        x = self.block3(x);       result["block3"] = x
        x = self.head_conv(x);    result["head"] = x
        x = self.global_pool(x).flatten(1)
        result["logits"] = self.classifier(x)
        return result

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        total = self.count_parameters()
        wm = self.config['width_mult']
        return (f"EdgeJuNet(w={wm}, "
                f"params={total:,} ({total * 4 / 1024:.1f} KB))")


# ---------------------------------------------------------------------------
# Model configs — params_M estimated from width_mult scaling
# ---------------------------------------------------------------------------

MODEL_CONFIGS = {
    "edgejunet":    dict(width_mult=1.0,  drop_path=0.1,
                         input_size=224, params_M=None),
    "edgejunet_xs": dict(width_mult=0.75, drop_path=0.1,
                         input_size=224, params_M=None),
    "edgejunet_s":  dict(width_mult=0.5,  drop_path=0.05,
                         input_size=224, params_M=None),
}

DEFAULT_VARIANT = "edgejunet"


def build_model(name: str = "edgejunet", num_classes: int = 4,
                in_channels: int = 3, **overrides) -> EdgeJuNet:
    """Build EdgeJuNet by variant name."""
    if name not in MODEL_CONFIGS:
        raise ValueError(
            f"Unknown model: {name}. Choose from {list(MODEL_CONFIGS.keys())}"
        )
    cfg = {k: v for k, v in MODEL_CONFIGS[name].items()
           if k not in ("input_size", "params_M")}
    cfg.update(overrides)
    return EdgeJuNet(in_channels=in_channels, num_classes=num_classes, **cfg)


def edgejunet(num_classes: int = 4, in_channels: int = 3) -> EdgeJuNet:
    return build_model("edgejunet", num_classes, in_channels)


def edgejunet_xs(num_classes: int = 4, in_channels: int = 3) -> EdgeJuNet:
    return build_model("edgejunet_xs", num_classes, in_channels)


def edgejunet_s(num_classes: int = 4, in_channels: int = 3) -> EdgeJuNet:
    return build_model("edgejunet_s", num_classes, in_channels)
