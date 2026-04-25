"""
MvANet — Multi-vision Attention Network
========================================
Full model definition with factory functions for all 4 variants.
"""

import torch
import torch.nn as nn
from typing import List, Optional, Dict, Any

from .blocks import (
    SimpleVisualLayer, AttentionLayer, VisualReceptorsBlock
)


class MvANet(nn.Module):
    """Multi-vision Attention Network.

    Architecture (Fig. 4):
        1. SimpleVisualLayer    — p parallel Conv3x3
        2. AttentionLayer A     — SE + downsample
        3. VisualReceptorsBlock — dense visual layers
        4. AttentionLayer B     — SE + downsample
        5. GlobalAvgPool + FC   — classification
    """

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
            "k0": k0, "k_list": k_list, "t": t,
            "a1": a1, "a2": a2, "p": p,
            "num_classes": num_classes
        }

        # 1. Shallow Feature Extraction
        self.shallow = SimpleVisualLayer(in_channels, k0, p)
        ch = self.shallow.out_channels

        # 2. Attention Layer A
        self.attention_a = AttentionLayer(ch, scale_factor=a1)
        ch = self.attention_a.out_channels

        # 3. Visual Receptors Block
        self.visual_block = VisualReceptorsBlock(ch, k_list, p, t)
        ch = self.visual_block.out_channels

        # 4. Attention Layer B
        self.attention_b = AttentionLayer(ch, scale_factor=a2)
        ch = self.attention_b.out_channels

        # 5. Classification
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(ch, num_classes)

        self._init_weights()

    def _init_weights(self):
        """He initialization (paper ref [34])."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.shallow(x)
        x = self.attention_a(x)
        x = self.visual_block(x)
        x = self.attention_b(x)
        x = self.global_pool(x).flatten(1)
        return self.classifier(x)

    def forward_with_features(self, x: torch.Tensor) -> Dict[str, Any]:
        """Forward pass returning intermediate features for visualization."""
        result = {}

        # Shallow
        shallow_out = self.shallow(x)
        result["shallow"] = shallow_out
        result["shallow_per_branch"] = self.shallow.forward_per_branch(x)

        # Attention A
        attn_a_out, attn_a_weights = self.attention_a.forward_with_attention(
            shallow_out
        )
        result["attn_a"] = attn_a_out
        result["attn_a_weights"] = attn_a_weights

        # Visual Receptors Block
        vb_out, vb_layer_outputs = self.visual_block.forward_with_intermediates(
            attn_a_out
        )
        result["visual_block"] = vb_out
        result["visual_layers"] = vb_layer_outputs  # list of lists

        # Attention B
        attn_b_out, attn_b_weights = self.attention_b.forward_with_attention(
            vb_out
        )
        result["attn_b"] = attn_b_out
        result["attn_b_weights"] = attn_b_weights

        # Classification
        pooled = self.global_pool(attn_b_out).flatten(1)
        logits = self.classifier(pooled)
        result["logits"] = logits

        return result

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self):
        total = self.count_parameters()
        return (f"MvANet(p={self.config['p']}, "
                f"params={total:,} ({total * 4 / 1024:.1f} KB))")


# ===========================================================================
# Factory functions (Table 2 in paper)
# ===========================================================================

MODEL_CONFIGS = {
    "mvanet_1_tiny": dict(k0=84, k_list=[18, 24, 30], t=1.7, a1=1.0, a2=0.8, p=1),
    "mvanet_1":      dict(k0=84, k_list=[96, 138, 192], t=1.5, a1=0.7, a2=0.6, p=1),
    "mvanet_2":      dict(k0=84, k_list=[96, 138, 192], t=1.5, a1=0.7, a2=0.6, p=2),
    "mvanet_3":      dict(k0=84, k_list=[96, 138, 192], t=1.5, a1=0.7, a2=0.6, p=3),
}


def build_model(name: str = "mvanet_3", num_classes: int = 4,
                in_channels: int = 3, **overrides) -> MvANet:
    """Build MvANet by config name.

    Args:
        name: one of mvanet_1_tiny, mvanet_1, mvanet_2, mvanet_3
        num_classes: number of output classes
        in_channels: input channels
        **overrides: override any config parameter

    Returns:
        MvANet instance
    """
    if name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model: {name}. "
                         f"Choose from {list(MODEL_CONFIGS.keys())}")

    cfg = {**MODEL_CONFIGS[name], **overrides}
    return MvANet(in_channels=in_channels, num_classes=num_classes, **cfg)


def mvanet_1_tiny(num_classes=4, in_channels=3):
    return build_model("mvanet_1_tiny", num_classes, in_channels)

def mvanet_1(num_classes=4, in_channels=3):
    return build_model("mvanet_1", num_classes, in_channels)

def mvanet_2(num_classes=4, in_channels=3):
    return build_model("mvanet_2", num_classes, in_channels)

def mvanet_3(num_classes=4, in_channels=3):
    return build_model("mvanet_3", num_classes, in_channels)
