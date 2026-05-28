"""
EdgeJuFormer — Lightweight CNN-Token Hybrid Network
---------------------------------------------------

Core idea:
    Local CNN features + lightweight token interaction.

Designed for:
    - Edge AI
    - Fruit recognition
    - Embedded deployment
    - Small datasets

Main Contributions:
    1. Dynamic local feature extraction (multi-scale DWConv)
    2. Tiny token mixer for global context
    3. Coordinate attention for spatial awareness
    4. Lightweight hybrid architecture

Author: Your Research Prototype
"""

from __future__ import annotations

import math
from typing import Dict, Any

import torch
import torch.nn as nn


# ============================================================
# Registry metadata
# ============================================================

ARCH_NAME = "juformer"

MODEL_CONFIGS = {
    "juformer":    {"input_size": 224, "width_mult": 1.00},
    "juformer_s":  {"input_size": 224, "width_mult": 0.75},
    "juformer_xs": {"input_size": 224, "width_mult": 0.50},
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
# Coordinate Attention
# ============================================================

class CoordAttention(nn.Module):

    def __init__(self, channels, reduction=8):
        super().__init__()

        hidden = max(channels // reduction, 8)

        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        self.conv1 = nn.Conv2d(channels, hidden, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden)

        self.act = nn.ReLU(inplace=True)

        self.conv_h = nn.Conv2d(hidden, channels, 1, bias=False)
        self.conv_w = nn.Conv2d(hidden, channels, 1, bias=False)

    def forward(self, x):

        n, c, h, w = x.shape

        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)

        y = self.act(self.bn1(self.conv1(y)))

        x_h, x_w = torch.split(y, [h, w], dim=2)

        x_w = x_w.permute(0, 1, 3, 2)

        a_h = torch.sigmoid(self.conv_h(x_h))
        a_w = torch.sigmoid(self.conv_w(x_w))

        return x * a_h * a_w


# ============================================================
# Dynamic Multi-Scale Depthwise Conv
# ============================================================

class DynamicDWConv(nn.Module):
    """
    Adaptive receptive field selection.

    Learns dynamic weights between:
        - 3x3 DWConv
        - 5x5 DWConv
    """

    def __init__(self, channels):

        super().__init__()

        self.dw3 = _cbr(channels, channels, k=3, g=channels)
        self.dw5 = _cbr(channels, channels, k=5, g=channels)

        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, 2, 1)
        )

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):

        w = self.gate(x)

        w = self.softmax(w)

        w3 = w[:, 0:1]
        w5 = w[:, 1:2]

        y3 = self.dw3(x)
        y5 = self.dw5(x)

        return w3 * y3 + w5 * y5


# ============================================================
# Tiny Token Mixer
# ============================================================

class TinyTokenMixer(nn.Module):
    """
    Lightweight global interaction block.

    CNN feature map
        -> token projection
        -> MHSA
        -> reshape back
    """

    def __init__(self, channels, num_heads=4):

        super().__init__()

        self.norm = nn.LayerNorm(channels)

        self.attn = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=num_heads,
            batch_first=True
        )

        self.proj = nn.Linear(channels, channels)

    def forward(self, x):

        b, c, h, w = x.shape

        # BCHW -> BNC
        tokens = x.flatten(2).transpose(1, 2)

        tokens = self.norm(tokens)

        attn_out, _ = self.attn(tokens, tokens, tokens)

        tokens = tokens + attn_out

        tokens = self.proj(tokens)

        # BNC -> BCHW
        out = tokens.transpose(1, 2).reshape(b, c, h, w)

        return out


# ============================================================
# Hybrid Block
# ============================================================

class EdgeJuFormerBlock(nn.Module):

    def __init__(
        self,
        in_ch,
        out_ch,
        stride=1,
        expand_ratio=4,
        use_token=True,
        drop_path=0.0
    ):
        super().__init__()

        hidden = in_ch * expand_ratio

        self.use_res = (stride == 1 and in_ch == out_ch)

        # expansion
        self.expand = _cbr(in_ch, hidden, k=1)

        # dynamic receptive field
        self.dynamic_dw = DynamicDWConv(hidden)

        # downsample
        if stride == 2:
            self.downsample = _cbr(
                hidden,
                hidden,
                k=3,
                s=2,
                g=hidden
            )
        else:
            self.downsample = nn.Identity()

        # token mixer
        self.token = (
            TinyTokenMixer(hidden)
            if use_token else nn.Identity()
        )

        # projection
        self.project = nn.Sequential(
            nn.Conv2d(hidden, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch)
        )

        self.ca = CoordAttention(out_ch)

        self.drop = (
            DropPath(drop_path)
            if drop_path > 0 else nn.Identity()
        )

    def forward(self, x):

        identity = x

        x = self.expand(x)

        x = self.dynamic_dw(x)

        x = self.downsample(x)

        x = self.token(x)

        x = self.project(x)

        x = self.ca(x)

        if self.use_res:
            x = self.drop(x) + identity

        return x


# ============================================================
# Model
# ============================================================

class EdgeJuFormer(nn.Module):

    def __init__(
        self,
        num_classes=4,
        width_mult=1.0,
        drop_path=0.1
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

        # stages
        self.stage1 = EdgeJuFormerBlock(
            c1,
            c2,
            stride=2,
            use_token=False
        )

        self.stage2 = EdgeJuFormerBlock(
            c2,
            c3,
            stride=2,
            use_token=True
        )

        self.stage3 = nn.Sequential(

            EdgeJuFormerBlock(
                c3,
                c4,
                stride=2,
                use_token=True
            ),

            EdgeJuFormerBlock(
                c4,
                c4,
                stride=1,
                use_token=True,
                drop_path=drop_path
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
                    m.weight,
                    mode='fan_out',
                    nonlinearity='relu'
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

    def count_parameters(self):

        return sum(
            p.numel()
            for p in self.parameters()
            if p.requires_grad
        )

    # --------------------------------------------------------

    def forward_features(self, x):

        feats = {}

        x = self.stem(x)
        feats["stem"] = x

        x = self.stage1(x)
        feats["stage1"] = x

        x = self.stage2(x)
        feats["stage2"] = x

        x = self.stage3(x)
        feats["stage3"] = x

        x = self.head(x)
        feats["head"] = x

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

    return EdgeJuFormer(
        num_classes=num_classes,
        width_mult=1.0
    )


def edgejuformer_s(num_classes=4):

    return EdgeJuFormer(
        num_classes=num_classes,
        width_mult=0.75
    )


def edgejuformer_xs(num_classes=4):

    return EdgeJuFormer(
        num_classes=num_classes,
        width_mult=0.5
    )


# ============================================================
# build_model  (required by model registry)
# ============================================================

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

    model = EdgeJuFormer(
        num_classes=num_classes,
        width_mult=cfg["width_mult"],
    )

    if in_channels != 3:
        # stem[0] is the Conv2d inside _cbr(3, c1, k=3, s=2)
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
# Test
# ============================================================

if __name__ == "__main__":

    model = build_model(name="juformer", num_classes=4)

    x = torch.randn(1, 3, 224, 224)

    y = model(x)

    print(model)

    print("Output shape:", y.shape)

    print("Parameters:", model.count_parameters())
