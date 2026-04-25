"""
MvANet Building Blocks
======================
BNPReLUConv, SEBlock, AttentionLayer, VisualReceptor, VisualLayer,
SimpleVisualLayer, VisualReceptorsBlock.
"""

import torch
import torch.nn as nn
from typing import List


class BNPReLUConv(nn.Module):
    """Pre-activation block: BN -> PReLU -> Conv."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int,
                 stride: int = 1, padding: int = 0):
        super().__init__()
        self.bn = nn.BatchNorm2d(in_ch)
        self.prelu = nn.PReLU(in_ch)
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size,
                              stride=stride, padding=padding, bias=False)

    def forward(self, x):
        return self.conv(self.prelu(self.bn(x)))


class SEBlock(nn.Module):
    """Squeeze-and-Excitation channel attention.

    GlobalAvgPool -> FC -> PReLU -> FC -> Sigmoid -> channel scale.
    """

    def __init__(self, channels: int, reduction: int = 3):
        super().__init__()
        mid = max(channels // reduction, 1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(channels, mid)
        self.prelu = nn.PReLU(mid)
        self.fc2 = nn.Linear(mid, channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        w = self.pool(x).view(b, c)
        w = self.sigmoid(self.fc2(self.prelu(self.fc1(w))))
        return x * w.view(b, c, 1, 1)

    def get_attention_weights(self, x):
        """Return attention weights for visualization."""
        b, c, _, _ = x.size()
        w = self.pool(x).view(b, c)
        w = self.sigmoid(self.fc2(self.prelu(self.fc1(w))))
        return w  # (B, C)


class AttentionLayer(nn.Module):
    """Attention Layer: BN-PReLU-Conv1x1 + SE + AvgPool2x2.

    Combines channel recalibration with spatial downsampling.
    """

    def __init__(self, in_ch: int, scale_factor: float = 1.0,
                 reduction: int = 3):
        super().__init__()
        self.out_channels = max(int(scale_factor * in_ch), 1)
        self.conv = BNPReLUConv(in_ch, self.out_channels, kernel_size=1)
        self.se = SEBlock(self.out_channels, reduction=reduction)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv(x)
        x = self.se(x)
        return self.pool(x)

    def forward_with_attention(self, x):
        """Forward pass that also returns SE attention weights."""
        x = self.conv(x)
        attn = self.se.get_attention_weights(x)
        x = self.se(x)
        return self.pool(x), attn


class VisualReceptor(nn.Module):
    """Visual Receptor — bottleneck "eye" unit.

    BN-PReLU-Conv1x1 (compress) -> BN-PReLU-Conv3x3 (extract).
    """

    def __init__(self, in_ch: int, mid_ch: int, out_ch: int):
        super().__init__()
        self.conv1x1 = BNPReLUConv(in_ch, mid_ch, kernel_size=1)
        self.conv3x3 = BNPReLUConv(mid_ch, out_ch, kernel_size=3, padding=1)
        self.out_channels = out_ch

    def forward(self, x):
        return self.conv3x3(self.conv1x1(x))


class VisualLayer(nn.Module):
    """Visual Layer: p parallel VisualReceptors (Inception-style).

    Each receptor sees the same input; outputs are concatenated.
    """

    def __init__(self, in_ch: int, layer_k: int, p: int, t: float):
        super().__init__()
        alpha = 0.5 * p * (p + 1) / 10.0
        out_per_receptor = layer_k // p

        self.receptors = nn.ModuleList()
        for _ in range(p):
            mid = max(int(in_ch / (t + alpha)), 1)
            self.receptors.append(VisualReceptor(in_ch, mid, out_per_receptor))

        self.out_channels = out_per_receptor * p

    def forward(self, x):
        return torch.cat([r(x) for r in self.receptors], dim=1)

    def forward_per_receptor(self, x):
        """Return individual receptor outputs for visualization."""
        return [r(x) for r in self.receptors]


class SimpleVisualLayer(nn.Module):
    """Shallow Feature Extraction: p parallel Conv3x3 branches."""

    def __init__(self, in_ch: int, k0: int, p: int):
        super().__init__()
        ch = k0 // p
        self.branches = nn.ModuleList([
            nn.Conv2d(in_ch, ch, kernel_size=3, padding=1, bias=False)
            for _ in range(p)
        ])
        self.out_channels = ch * p

    def forward(self, x):
        return torch.cat([b(x) for b in self.branches], dim=1)

    def forward_per_branch(self, x):
        """Return individual branch outputs for visualization."""
        return [b(x) for b in self.branches]


class VisualReceptorsBlock(nn.Module):
    """Dense-connected Visual Layers + final fuse receptor.

    Dense connectivity: layer l receives concat(x_0, x_1, ..., x_{l-1}).
    After all layers, a single VisualReceptor fuses everything.
    """

    def __init__(self, in_ch: int, k_list: List[int], p: int, t: float):
        super().__init__()
        self.visual_layers = nn.ModuleList()

        cumul = in_ch
        for k in k_list:
            vl = VisualLayer(cumul, k, p, t)
            self.visual_layers.append(vl)
            cumul += vl.out_channels

        # Fuse receptor (p=1)
        alpha_fuse = 0.5 * 1 * 2 / 10.0
        fuse_mid = max(int(cumul / (t + alpha_fuse)), 1)
        self.fuse_receptor = VisualReceptor(cumul, fuse_mid, k_list[-1])
        self.out_channels = k_list[-1]

    def forward(self, x):
        features = [x]
        for layer in self.visual_layers:
            dense_in = torch.cat(features, dim=1)
            features.append(layer(dense_in))
        return self.fuse_receptor(torch.cat(features, dim=1))

    def forward_with_intermediates(self, x):
        """Forward pass returning intermediate feature maps per layer."""
        features = [x]
        layer_outputs = []

        for layer in self.visual_layers:
            dense_in = torch.cat(features, dim=1)
            out = layer(dense_in)
            features.append(out)
            # Also get per-receptor outputs
            per_receptor = layer.forward_per_receptor(dense_in)
            layer_outputs.append(per_receptor)

        fused = self.fuse_receptor(torch.cat(features, dim=1))
        return fused, layer_outputs
