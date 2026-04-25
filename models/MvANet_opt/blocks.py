"""
MvANet Building Blocks (Optimized v2)
======================================
Drop-in replacement — same class names, same signatures, same methods.

Optimizations:
  1. BNPReLUConv: PReLU → ReLU           (fewer params)
  2. SEBlock:     SE → ECA               (near-zero params vs 2 FC layers)
  3. VisualReceptor: Conv3x3 → GhostConv (up to 16x fewer params, same speed)
  4. VisualReceptorsBlock: + Stochastic Depth (better regularization → accuracy)
"""

import math
import torch
import torch.nn as nn
from typing import List


# ───────────────────────────────────────────────────────────────────────────
# Drop Path (Stochastic Depth)
# ───────────────────────────────────────────────────────────────────────────
def _drop_path(x: torch.Tensor, drop_prob: float, training: bool):
    if drop_prob == 0.0 or not training:
        return x
    keep = 1.0 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    mask = torch.bernoulli(torch.full(shape, keep, device=x.device, dtype=x.dtype))
    return x / keep * mask


class DropPath(nn.Module):
    def __init__(self, p: float = 0.0):
        super().__init__()
        self.p = p

    def forward(self, x):
        return _drop_path(x, self.p, self.training)


# ───────────────────────────────────────────────────────────────────────────
# Ghost Convolution (Huawei, CVPR 2020)
#
# Idea: generate primary features with cheap 1x1, then "ghost" the rest
# with depthwise 3x3.  Result: ~same accuracy, fraction of params/FLOPs.
#
# Params:  in_ch * (out/2) + (out/2) * 9   vs   in_ch * out * 9
# Example: 86→30:  1,425  vs  23,220  (16x fewer!)
# Speed:   faster or equal to standard conv on CPU (1x1 is memory-friendly)
# ───────────────────────────────────────────────────────────────────────────
class GhostConv(nn.Module):
    """Ghost Convolution: primary 1x1 + ghost 3x3-depthwise → concat.

    Generates `out_ch // 2` features via 1x1, then derives the other half
    with a cheap 3x3 depthwise conv.  Output = concat(primary, ghost).
    """

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.out_ch = out_ch
        # Following the original GhostNet approach:
        # primary 1x1 generates init_ch features, ghost 3x3 depthwise
        # generates the same number, then trim to exact out_ch.
        init_ch = (out_ch + 1) // 2  # ceil(out_ch / 2)
        self.primary = nn.Conv2d(in_ch, init_ch, 1, bias=False)
        self.ghost = nn.Conv2d(init_ch, init_ch, 3, padding=1,
                               groups=init_ch, bias=False)

    def forward(self, x):
        p = self.primary(x)
        g = self.ghost(p)
        return torch.cat([p, g], dim=1)[:, :self.out_ch, :, :]


# ───────────────────────────────────────────────────────────────────────────
# BNPReLUConv — BN → ReLU → Conv  (name kept for compatibility)
# ───────────────────────────────────────────────────────────────────────────
class BNPReLUConv(nn.Module):
    """Pre-activation block: BN -> ReLU -> Conv.

    For kernel_size=3: uses GhostConv (massive param/FLOP reduction).
    For kernel_size=1: uses standard Conv2d (already cheap).
    """

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int,
                 stride: int = 1, padding: int = 0):
        super().__init__()
        self.bn = nn.BatchNorm2d(in_ch)
        self.prelu = nn.ReLU(inplace=True)

        if kernel_size == 3:
            self.conv = GhostConv(in_ch, out_ch)
        else:
            self.conv = nn.Conv2d(in_ch, out_ch, kernel_size,
                                  stride=stride, padding=padding, bias=False)

    def forward(self, x):
        return self.conv(self.prelu(self.bn(x)))


# ───────────────────────────────────────────────────────────────────────────
# SEBlock — ECA internally (name & interface kept for compatibility)
# ───────────────────────────────────────────────────────────────────────────
class SEBlock(nn.Module):
    """Channel attention (ECA implementation).

    Original SE: GlobalAvgPool → FC(C→C/r) → PReLU → FC(C/r→C) → Sigmoid
    Optimized:   GlobalAvgPool → Conv1d(k)  → Sigmoid
    """

    def __init__(self, channels: int, reduction: int = 3):
        super().__init__()
        t = int(abs(math.log2(max(channels, 2)) / 2 + 0.5))
        k = t if t % 2 else t + 1
        k = max(k, 3)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        w = self.pool(x).view(b, 1, c)
        w = self.sigmoid(self.conv(w))
        return x * w.view(b, c, 1, 1)

    def get_attention_weights(self, x):
        """Return attention weights for visualization — shape (B, C)."""
        b, c, _, _ = x.size()
        w = self.pool(x).view(b, 1, c)
        w = self.sigmoid(self.conv(w))
        return w.view(b, c)


# ───────────────────────────────────────────────────────────────────────────
# AttentionLayer — identical interface
# ───────────────────────────────────────────────────────────────────────────
class AttentionLayer(nn.Module):
    """Attention Layer: BN-ReLU-Conv1x1 + ECA + AvgPool2x2."""

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


# ───────────────────────────────────────────────────────────────────────────
# VisualReceptor — bottleneck with GhostConv on 3x3
# ───────────────────────────────────────────────────────────────────────────
class VisualReceptor(nn.Module):
    """Visual Receptor — bottleneck "eye" unit.

    BN-ReLU-Conv1x1 (compress) -> BN-ReLU-GhostConv3x3 (extract).
    """

    def __init__(self, in_ch: int, mid_ch: int, out_ch: int):
        super().__init__()
        self.conv1x1 = BNPReLUConv(in_ch, mid_ch, kernel_size=1)
        self.conv3x3 = BNPReLUConv(mid_ch, out_ch, kernel_size=3, padding=1)
        self.out_channels = out_ch

    def forward(self, x):
        return self.conv3x3(self.conv1x1(x))


# ───────────────────────────────────────────────────────────────────────────
# VisualLayer — identical interface
# ───────────────────────────────────────────────────────────────────────────
class VisualLayer(nn.Module):
    """Visual Layer: p parallel VisualReceptors (Inception-style)."""

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


# ───────────────────────────────────────────────────────────────────────────
# SimpleVisualLayer — identical interface
# ───────────────────────────────────────────────────────────────────────────
class SimpleVisualLayer(nn.Module):
    """Shallow Feature Extraction: p parallel Conv3x3 branches.

    Note: keeps standard Conv3x3 here (in_ch=3, too small for Ghost).
    """

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


# ───────────────────────────────────────────────────────────────────────────
# VisualReceptorsBlock — identical interface + Stochastic Depth
# ───────────────────────────────────────────────────────────────────────────
class VisualReceptorsBlock(nn.Module):
    """Dense-connected Visual Layers + final fuse receptor.

    Dense connectivity: layer l receives concat(x_0, x_1, ..., x_{l-1}).
    After all layers, a single VisualReceptor fuses everything.

    Added: Stochastic Depth for regularization (improves generalization).
    Linear schedule: layer 0 = 0%, last layer = drop_path_rate.
    """

    # Class-level default; can be overridden per-instance
    drop_path_rate = 0.1

    def __init__(self, in_ch: int, k_list: List[int], p: int, t: float):
        super().__init__()
        self.visual_layers = nn.ModuleList()
        self.drop_paths = nn.ModuleList()

        n_layers = len(k_list)
        cumul = in_ch
        for i, k in enumerate(k_list):
            vl = VisualLayer(cumul, k, p, t)
            self.visual_layers.append(vl)
            dp_rate = self.drop_path_rate * (i / max(n_layers - 1, 1))
            self.drop_paths.append(DropPath(dp_rate))
            cumul += vl.out_channels

        # Fuse receptor (p=1)
        alpha_fuse = 0.5 * 1 * 2 / 10.0
        fuse_mid = max(int(cumul / (t + alpha_fuse)), 1)
        self.fuse_receptor = VisualReceptor(cumul, fuse_mid, k_list[-1])
        self.out_channels = k_list[-1]

    def forward(self, x):
        features = [x]
        for layer, dp in zip(self.visual_layers, self.drop_paths):
            dense_in = torch.cat(features, dim=1)
            features.append(dp(layer(dense_in)))
        return self.fuse_receptor(torch.cat(features, dim=1))

    def forward_with_intermediates(self, x):
        """Forward pass returning intermediate feature maps per layer."""
        features = [x]
        layer_outputs = []

        for layer, dp in zip(self.visual_layers, self.drop_paths):
            dense_in = torch.cat(features, dim=1)
            out = dp(layer(dense_in))
            features.append(out)
            per_receptor = layer.forward_per_receptor(dense_in)
            layer_outputs.append(per_receptor)

        fused = self.fuse_receptor(torch.cat(features, dim=1))
        return fused, layer_outputs
