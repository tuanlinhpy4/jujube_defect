"""Single-file optimized MvANet variant.

This architecture keeps the same public interface as the original MvANet while
using ReLU, ECA-style attention, GhostConv blocks, and stochastic depth.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn


ARCH_NAME = "MvANet_opt"
ARCH_ALIASES = ["mvanet_opt", "MvANetOpt"]


def _drop_path(x: torch.Tensor, drop_prob: float, training: bool):
    if drop_prob == 0.0 or not training:
        return x
    keep = 1.0 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    mask = torch.bernoulli(torch.full(shape, keep, device=x.device, dtype=x.dtype))
    return x / keep * mask


class DropPath(nn.Module):
    """Per-sample stochastic depth."""

    def __init__(self, p: float = 0.0):
        super().__init__()
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _drop_path(x, self.p, self.training)


class GhostConv(nn.Module):
    """Ghost convolution: primary 1x1 plus cheap depthwise 3x3 features."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.out_ch = out_ch
        init_ch = (out_ch + 1) // 2
        self.primary = nn.Conv2d(in_ch, init_ch, 1, bias=False)
        self.ghost = nn.Conv2d(
            init_ch, init_ch, 3, padding=1, groups=init_ch, bias=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        primary = self.primary(x)
        ghost = self.ghost(primary)
        return torch.cat([primary, ghost], dim=1)[:, :self.out_ch, :, :]


class BNPReLUConv(nn.Module):
    """Pre-activation block: BatchNorm -> ReLU -> Conv."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int,
                 stride: int = 1, padding: int = 0):
        super().__init__()
        self.bn = nn.BatchNorm2d(in_ch)
        self.prelu = nn.ReLU(inplace=True)
        if kernel_size == 3:
            self.conv = GhostConv(in_ch, out_ch)
        else:
            self.conv = nn.Conv2d(
                in_ch, out_ch, kernel_size, stride=stride, padding=padding,
                bias=False,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.prelu(self.bn(x)))


class SEBlock(nn.Module):
    """ECA-style channel attention, keeping the SEBlock interface."""

    def __init__(self, channels: int, reduction: int = 3):
        super().__init__()
        t = int(abs(math.log2(max(channels, 2)) / 2 + 0.5))
        k = t if t % 2 else t + 1
        k = max(k, 3)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def _weights(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        w = self.pool(x).view(b, 1, c)
        return self.sigmoid(self.conv(w)).view(b, c)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        return x * self._weights(x).view(b, c, 1, 1)

    def get_attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        """Return attention weights for visualization."""

        return self._weights(x)


class AttentionLayer(nn.Module):
    """Attention layer: BN-ReLU-Conv1x1 + ECA + AvgPool2x2."""

    def __init__(self, in_ch: int, scale_factor: float = 1.0,
                 reduction: int = 3):
        super().__init__()
        self.out_channels = max(int(scale_factor * in_ch), 1)
        self.conv = BNPReLUConv(in_ch, self.out_channels, kernel_size=1)
        self.se = SEBlock(self.out_channels, reduction=reduction)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.se(x)
        return self.pool(x)

    def forward_with_attention(self, x: torch.Tensor):
        """Forward pass that also returns attention weights."""

        x = self.conv(x)
        attn = self.se.get_attention_weights(x)
        x = self.se(x)
        return self.pool(x), attn


class VisualReceptor(nn.Module):
    """Bottleneck unit with GhostConv on the 3x3 extraction layer."""

    def __init__(self, in_ch: int, mid_ch: int, out_ch: int):
        super().__init__()
        self.conv1x1 = BNPReLUConv(in_ch, mid_ch, kernel_size=1)
        self.conv3x3 = BNPReLUConv(mid_ch, out_ch, kernel_size=3, padding=1)
        self.out_channels = out_ch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv3x3(self.conv1x1(x))


class VisualLayer(nn.Module):
    """Visual layer: p parallel VisualReceptors concatenated by channel."""

    def __init__(self, in_ch: int, layer_k: int, p: int, t: float):
        super().__init__()
        alpha = 0.5 * p * (p + 1) / 10.0
        out_per_receptor = layer_k // p
        mid = max(int(in_ch / (t + alpha)), 1)
        self.receptors = nn.ModuleList(
            [VisualReceptor(in_ch, mid, out_per_receptor) for _ in range(p)]
        )
        self.out_channels = out_per_receptor * p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([receptor(x) for receptor in self.receptors], dim=1)

    def forward_per_receptor(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Return individual receptor outputs for visualization."""

        return [receptor(x) for receptor in self.receptors]


class SimpleVisualLayer(nn.Module):
    """Shallow feature extraction with p parallel Conv3x3 branches."""

    def __init__(self, in_ch: int, k0: int, p: int):
        super().__init__()
        ch = k0 // p
        self.branches = nn.ModuleList(
            [nn.Conv2d(in_ch, ch, kernel_size=3, padding=1, bias=False)
             for _ in range(p)]
        )
        self.out_channels = ch * p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([branch(x) for branch in self.branches], dim=1)

    def forward_per_branch(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Return individual branch outputs for visualization."""

        return [branch(x) for branch in self.branches]


class VisualReceptorsBlock(nn.Module):
    """Dense-connected visual layers plus stochastic depth and fuse receptor."""

    drop_path_rate = 0.1

    def __init__(self, in_ch: int, k_list: List[int], p: int, t: float):
        super().__init__()
        self.visual_layers = nn.ModuleList()
        self.drop_paths = nn.ModuleList()

        n_layers = len(k_list)
        cumul = in_ch
        for idx, k in enumerate(k_list):
            layer = VisualLayer(cumul, k, p, t)
            self.visual_layers.append(layer)
            dp_rate = self.drop_path_rate * (idx / max(n_layers - 1, 1))
            self.drop_paths.append(DropPath(dp_rate))
            cumul += layer.out_channels

        alpha_fuse = 0.5 * 1 * 2 / 10.0
        fuse_mid = max(int(cumul / (t + alpha_fuse)), 1)
        self.fuse_receptor = VisualReceptor(cumul, fuse_mid, k_list[-1])
        self.out_channels = k_list[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = [x]
        for layer, drop_path in zip(self.visual_layers, self.drop_paths):
            dense_in = torch.cat(features, dim=1)
            features.append(drop_path(layer(dense_in)))
        return self.fuse_receptor(torch.cat(features, dim=1))

    def forward_with_intermediates(self, x: torch.Tensor):
        """Return fused output and per-receptor intermediate feature maps."""

        features = [x]
        layer_outputs = []
        for layer, drop_path in zip(self.visual_layers, self.drop_paths):
            dense_in = torch.cat(features, dim=1)
            out = drop_path(layer(dense_in))
            features.append(out)
            layer_outputs.append(layer.forward_per_receptor(dense_in))

        fused = self.fuse_receptor(torch.cat(features, dim=1))
        return fused, layer_outputs


class MvANet(nn.Module):
    """Optimized Multi-vision Attention Network."""

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 4,
        k0: int = 84,
        k_list: Optional[List[int]] = None,
        t: float = 1.5,
        a1: float = 0.7,
        a2: float = 0.6,
        p: int = 3,
    ):
        super().__init__()
        if k_list is None:
            k_list = [96, 138, 192]

        self.config = {
            "k0": k0,
            "k_list": k_list,
            "t": t,
            "a1": a1,
            "a2": a2,
            "p": p,
            "num_classes": num_classes,
        }

        self.shallow = SimpleVisualLayer(in_channels, k0, p)
        ch = self.shallow.out_channels

        self.attention_a = AttentionLayer(ch, scale_factor=a1)
        ch = self.attention_a.out_channels

        self.visual_block = VisualReceptorsBlock(ch, k_list, p, t)
        ch = self.visual_block.out_channels

        self.attention_b = AttentionLayer(ch, scale_factor=a2)
        ch = self.attention_b.out_channels

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(ch, num_classes)

        self._init_weights()

    def _init_weights(self) -> None:
        """He initialization for ReLU-based blocks."""

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu"
                )
            elif isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu"
                )
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu"
                )
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.shallow(x)
        x = self.attention_a(x)
        x = self.visual_block(x)
        x = self.attention_b(x)
        x = self.global_pool(x).flatten(1)
        return self.classifier(x)

    def forward_with_features(self, x: torch.Tensor) -> Dict[str, Any]:
        """Forward pass returning intermediate features for visualization."""

        result: Dict[str, Any] = {}
        shallow_out = self.shallow(x)
        result["shallow"] = shallow_out
        result["shallow_per_branch"] = self.shallow.forward_per_branch(x)

        attn_a_out, attn_a_weights = self.attention_a.forward_with_attention(
            shallow_out
        )
        result["attn_a"] = attn_a_out
        result["attn_a_weights"] = attn_a_weights

        vb_out, vb_layer_outputs = self.visual_block.forward_with_intermediates(
            attn_a_out
        )
        result["visual_block"] = vb_out
        result["visual_layers"] = vb_layer_outputs

        attn_b_out, attn_b_weights = self.attention_b.forward_with_attention(
            vb_out
        )
        result["attn_b"] = attn_b_out
        result["attn_b_weights"] = attn_b_weights

        pooled = self.global_pool(attn_b_out).flatten(1)
        result["logits"] = self.classifier(pooled)
        return result

    def count_parameters(self) -> int:
        return sum(param.numel() for param in self.parameters()
                   if param.requires_grad)

    def __repr__(self) -> str:
        total = self.count_parameters()
        return (
            f"MvANet(p={self.config['p']}, "
            f"params={total:,} ({total * 4 / 1024:.1f} KB))"
        )


MODEL_CONFIGS = {
    "mvanet_opt_1_tiny": dict(k0=84, k_list=[18, 24, 30],
                              t=1.7, a1=1.0, a2=0.8, p=1),
    "mvanet_opt_1":      dict(k0=84, k_list=[96, 138, 192],
                              t=1.5, a1=0.7, a2=0.6, p=1),
    "mvanet_opt_2":      dict(k0=84, k_list=[96, 138, 192],
                              t=1.5, a1=0.7, a2=0.6, p=2),
    "mvanet_opt_3":      dict(k0=84, k_list=[96, 138, 192],
                              t=1.5, a1=0.7, a2=0.6, p=3),
}

DEFAULT_VARIANT = "mvanet_opt_3"


def build_model(name: str = "mvanet_opt_3", num_classes: int = 4,
                in_channels: int = 3, **overrides) -> MvANet:
    """Build optimized MvANet by variant name."""

    if name not in MODEL_CONFIGS:
        raise ValueError(
            f"Unknown model: {name}. Choose from {list(MODEL_CONFIGS.keys())}"
        )
    cfg = {**MODEL_CONFIGS[name], **overrides}
    return MvANet(in_channels=in_channels, num_classes=num_classes, **cfg)


def mvanet_opt_1_tiny(num_classes=4, in_channels=3):
    return build_model("mvanet_opt_1_tiny", num_classes, in_channels)


def mvanet_opt_1(num_classes=4, in_channels=3):
    return build_model("mvanet_opt_1", num_classes, in_channels)


def mvanet_opt_2(num_classes=4, in_channels=3):
    return build_model("mvanet_opt_2", num_classes, in_channels)


def mvanet_opt_3(num_classes=4, in_channels=3):
    return build_model("mvanet_opt_3", num_classes, in_channels)
