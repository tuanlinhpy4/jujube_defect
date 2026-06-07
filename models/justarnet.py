"""JuStarNet — texture-aware jujube classifier built for the hard *mold* class.

Motivation (jujube_ours benchmark, see paper table)
---------------------------------------------------
A `timm` model-zoo sweep (StarNet, MobileNetV4, ConvNeXt-Atto/Femto, EdgeNeXt,
FastViT, LeViT, GhostNetV2, RepGhostNet, RepViT, EfficientNet-lite0, PoolFormer,
MixNet, ...) — 25 CNN / hybrid / transformer families spanning 0.3M–11M params —
**all plateau at 80–82% F1** on jujube_ours. When that many different backbones
hit the same wall, the bottleneck is not the backbone; it is something they all
share: **Global Average Pooling (GAP)** + the **mold** class.

  * mold is a *minority* class (≈346 imgs vs ≈1469 for dry → 4.2:1 imbalance).
  * mold is a *localized, fine fungal texture* — a fuzzy patch covering part of
    the fruit. GAP averages the feature map, so a small bright mold patch is
    averaged toward zero and washed out. Every GAP model loses it the same way.

JuStarNet attacks the shared bottleneck instead of reshuffling backbone blocks
(the repo lineage already did that 9 times). Two ideas, both novel to this repo
and directly citeable against the benchmark table:

1. Star-operation backbone (StarNet, "Rewrite the Stars", CVPR 2024 — row
   `Starnet s1` in the table). The star op  ``act(W1 x) * (W2 x)``  realises
   implicit *high-dimensional* nonlinear features at low parameter cost — strong
   for fine-grained separation (mold vs cracked vs dry) on a small budget.

2. Second-Order Multi-Statistic Pooling head (the centrepiece). Replaces GAP
   with ``concat[ mean, std, GeM ]`` per channel:
     - mean : 1st-order, the usual GAP signal (global appearance).
     - std  : 2nd-order *spatial variance* per channel. A localized mold patch
              creates HIGH variance in texture channels; a uniformly dry fruit
              has LOW variance. This is exactly the cue GAP throws away, and the
              repo grep confirms no existing model uses any 2nd-order statistic.
     - GeM  : generalized mean (learnable p) — softly emphasises the most
              salient regions (the mold patch) instead of the whole fruit.

   A weight-normalized **cosine classifier** closes the head: it removes the
   norm-bias that majority classes accumulate on imbalanced data and composes
   with the inverse-frequency class weights the pipeline already builds. No
   change to train.py / loss / data is required — JuStarNet is a drop-in
   ``--model justarnet``.

Self-contained on purpose (same convention as junet_v3): every block lives in
this file so it can be dropped into a Colab pipeline without the rest of models/.
"""

from __future__ import annotations

from typing import Any, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F


ARCH_NAME = "JuStarNet"
ARCH_ALIASES = ["justarnet", "JuStarNet", "justar"]


# ===========================================================================
# Stochastic depth
# ===========================================================================

def _drop_path(x: torch.Tensor, drop_prob: float, training: bool) -> torch.Tensor:
    if drop_prob <= 0.0 or not training:
        return x
    keep = 1.0 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    mask = torch.bernoulli(torch.full(shape, keep, device=x.device, dtype=x.dtype))
    return x / keep * mask


class DropPath(nn.Module):
    """Per-sample stochastic depth — applied only in residual branches."""

    def __init__(self, p: float = 0.0):
        super().__init__()
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _drop_path(x, self.p, self.training)


# ===========================================================================
# Conv-BN(-Act) helper
# ===========================================================================

class ConvBN(nn.Sequential):
    """Conv2d(bias=False) → BatchNorm2d, with optional ReLU6."""

    def __init__(self, in_ch: int, out_ch: int, kernel: int = 1, stride: int = 1,
                 groups: int = 1, dilation: int = 1, act: bool = False):
        pad = ((kernel - 1) // 2) * dilation
        layers: List[nn.Module] = [
            nn.Conv2d(in_ch, out_ch, kernel, stride=stride, padding=pad,
                      dilation=dilation, groups=groups, bias=False),
            nn.BatchNorm2d(out_ch),
        ]
        if act:
            layers.append(nn.ReLU6(inplace=True))
        super().__init__(*layers)


# ===========================================================================
# Star block (Ma et al., "Rewrite the Stars", CVPR 2024)
# ===========================================================================

class StarBlock(nn.Module):
    """StarNet demo block — the star operation gives implicit high-dim features.

        x   = dwconv(x)                 # 7x7 depthwise, large receptive field
        x   = act(f1 x) * (f2 x)        # <-- STAR operation (element-wise mul)
        x   = dwconv2(g(x))             # project back + depthwise mix
        out = input + drop_path(x)

    The element-wise product of two linear projections is equivalent to a
    polynomial feature map of much higher dimension than the channel count —
    cheap nonlinear capacity that helps fine-grained defect separation. The 7x7
    depthwise kernels give a wide receptive field so mold texture is seen in
    context rather than pixel-by-pixel.
    """

    def __init__(self, dim: int, mlp_ratio: int = 3, drop_path: float = 0.0,
                 kernel: int = 7):
        super().__init__()
        hidden = mlp_ratio * dim
        self.dwconv = ConvBN(dim, dim, kernel, groups=dim, act=False)
        self.f1 = ConvBN(dim, hidden, 1, act=False)
        self.f2 = ConvBN(dim, hidden, 1, act=False)
        self.act = nn.ReLU6(inplace=True)
        self.g = ConvBN(hidden, dim, 1, act=False)
        self.dwconv2 = ConvBN(dim, dim, kernel, groups=dim, act=False)
        self.drop_path = DropPath(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        inp = x
        x = self.dwconv(x)
        x1, x2 = self.f1(x), self.f2(x)
        x = self.act(x1) * x2
        x = self.dwconv2(self.g(x))
        return inp + self.drop_path(x)


# ===========================================================================
# Second-Order Multi-Statistic Pooling  (the centrepiece)
# ===========================================================================

class SecondOrderPool(nn.Module):
    """Pool a feature map into ``concat[ mean, std, GeM ]`` per channel.

    Given F of shape (B, C, H, W) this returns (B, 3C):

        mean_c = (1/HW) Σ F                         # 1st order (GAP)
        std_c  = sqrt( E[F^2] - E[F]^2 + eps )      # 2nd order spatial variance
        gem_c  = ( E[ clamp(F, eps)^p ] )^(1/p)     # generalized mean, p learnable

    The std term is what GAP-only models lack: a localized mold patch raises the
    per-channel spatial variance, separating mold from the smooth, uniform
    appearance of dry / invalid fruit.
    """

    def __init__(self, p_init: float = 3.0, eps: float = 1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.full((1,), float(p_init)))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        flat = x.flatten(2)                          # (B, C, HW)

        mean = flat.mean(dim=2)                      # (B, C)
        var = flat.var(dim=2, unbiased=False)        # (B, C)
        std = torch.sqrt(var + self.eps)

        p = self.p.clamp(min=1.0)
        gem = flat.clamp(min=self.eps).pow(p).mean(dim=2).pow(1.0 / p)

        return torch.cat([mean, std, gem], dim=1)    # (B, 3C)

    @property
    def out_mult(self) -> int:
        return 3


# ===========================================================================
# Cosine classifier — removes norm-bias from majority classes
# ===========================================================================

class CosineClassifier(nn.Module):
    """logits = scale · cos(feature, weight). Composes with class-weighted CE."""

    def __init__(self, in_features: int, num_classes: int, scale: float = 16.0):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_classes, in_features))
        nn.init.kaiming_normal_(self.weight, nonlinearity="linear")
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.normalize(x, dim=1)
        w = F.normalize(self.weight, dim=1)
        return self.scale * F.linear(x, w)


# ===========================================================================
# Head: SecondOrderPool → LayerNorm → Linear → GELU → Dropout → Cosine
# ===========================================================================

class TextureHead(nn.Module):
    def __init__(self, in_ch: int, num_classes: int, fc_dim: int = 256,
                 dropout: float = 0.1, cosine_scale: float = 16.0,
                 gem_p_init: float = 3.0):
        super().__init__()
        self.pool = SecondOrderPool(p_init=gem_p_init)
        pooled_dim = in_ch * self.pool.out_mult
        # LayerNorm is batch-size independent (safe at bs=1 inference) and keeps
        # the three statistics on a common scale before fusion.
        self.norm = nn.LayerNorm(pooled_dim)
        self.proj = nn.Linear(pooled_dim, fc_dim, bias=False)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.classifier = CosineClassifier(fc_dim, num_classes, scale=cosine_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.pool(x)
        f = self.norm(f)
        f = self.drop(self.act(self.proj(f)))
        return self.classifier(f)

    def forward_with_features(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        pooled = self.pool(x)
        f = self.drop(self.act(self.proj(self.norm(pooled))))
        return {"pooled": pooled, "embedding": f, "logits": self.classifier(f)}


# ===========================================================================
# JuStarNet
# ===========================================================================

class JuStarNet(nn.Module):
    """Star-operation backbone + Second-Order pooling head.

    Layout (input 224x224):
        stem    3  → C0, s2                                   112x112
        stage1  C0 → C1, downsample s2, depth d0              56x56
        stage2  C1 → C2, downsample s2, depth d1              28x28
        stage3  C2 → C3, downsample s2, depth d2              14x14
        stage4  C3 → C3, NO downsample,  depth d3             14x14   <- keep
                resolution so the second-order pool sees mold texture in detail
        head    SecondOrderPool(C3) → LN → Linear → GELU → Dropout → Cosine
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 5,
        channels: List[int] = (32, 64, 96, 128),
        depths: List[int] = (2, 2, 4, 2),
        mlp_ratio: int = 3,
        drop_path: float = 0.1,
        fc_dim: int = 256,
        dropout: float = 0.1,
        cosine_scale: float = 16.0,
    ):
        super().__init__()
        c0, c1, c2, c3 = channels
        d0, d1, d2, d3 = depths

        self.config: Dict[str, Any] = {
            "channels": list(channels),
            "depths": list(depths),
            "mlp_ratio": mlp_ratio,
            "drop_path": drop_path,
            "num_classes": num_classes,
        }

        # Progressive (linearly increasing) stochastic-depth schedule.
        total_blocks = d0 + d1 + d2 + d3
        dpr = [drop_path * i / max(total_blocks - 1, 1) for i in range(total_blocks)]
        idx = 0

        def make_stage(in_ch: int, out_ch: int, depth: int, downsample: bool):
            nonlocal idx
            layers: List[nn.Module] = []
            if downsample:
                layers.append(ConvBN(in_ch, out_ch, 3, stride=2, act=True))
            elif in_ch != out_ch:
                layers.append(ConvBN(in_ch, out_ch, 1, act=True))
            for _ in range(depth):
                layers.append(StarBlock(out_ch, mlp_ratio=mlp_ratio,
                                        drop_path=dpr[idx]))
                idx += 1
            return nn.Sequential(*layers)

        self.stem = ConvBN(in_channels, c0, 3, stride=2, act=True)
        self.stage1 = make_stage(c0, c1, d0, downsample=True)
        self.stage2 = make_stage(c1, c2, d1, downsample=True)
        self.stage3 = make_stage(c2, c3, d2, downsample=True)
        self.stage4 = make_stage(c3, c3, d3, downsample=False)

        self.head = TextureHead(c3, num_classes, fc_dim=fc_dim,
                                dropout=dropout, cosine_scale=cosine_scale)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _forward_backbone(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self._forward_backbone(x))

    def forward_with_features(self, x: torch.Tensor) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        x = self.stem(x);    result["stem"] = x
        x = self.stage1(x);  result["stage1"] = x
        x = self.stage2(x);  result["stage2"] = x
        x = self.stage3(x);  result["stage3"] = x
        x = self.stage4(x);  result["stage4"] = x
        result.update(self.head.forward_with_features(x))
        return result

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        total = self.count_parameters()
        return f"JuStarNet(params={total:,} ({total / 1e6:.3f} M))"


# ===========================================================================
# Registry
# ===========================================================================

MODEL_CONFIGS: Dict[str, Dict] = {
    # primary target — ~1.3M params, competitive with RepGhostNet/ConvNeXt-Femto
    "justarnet": dict(
        channels=(32, 64, 96, 128),
        depths=(2, 2, 4, 2),
        mlp_ratio=3,
        drop_path=0.1,
        fc_dim=256,
        dropout=0.1,
        cosine_scale=16.0,
        input_size=224,
        params_M=None,
    ),
    # compact variant for tighter latency / param budgets
    "justarnet_s": dict(
        channels=(24, 48, 80, 112),
        depths=(1, 2, 3, 2),
        mlp_ratio=3,
        drop_path=0.05,
        fc_dim=192,
        dropout=0.1,
        cosine_scale=16.0,
        input_size=224,
        params_M=None,
    ),
    # smallest reference variant
    "justarnet_xs": dict(
        channels=(24, 40, 64, 96),
        depths=(1, 1, 2, 2),
        mlp_ratio=3,
        drop_path=0.05,
        fc_dim=160,
        dropout=0.1,
        cosine_scale=16.0,
        input_size=224,
        params_M=None,
    ),
}

DEFAULT_VARIANT = "justarnet"


def build_model(name: str = "justarnet", num_classes: int = 5,
                in_channels: int = 3, **overrides) -> JuStarNet:
    if name not in MODEL_CONFIGS:
        raise ValueError(
            f"Unknown model: {name}. Choose from {list(MODEL_CONFIGS.keys())}"
        )
    cfg = {k: v for k, v in MODEL_CONFIGS[name].items()
           if k not in ("input_size", "params_M")}
    cfg.update(overrides)
    return JuStarNet(in_channels=in_channels, num_classes=num_classes, **cfg)


def justarnet(num_classes: int = 5, in_channels: int = 3) -> JuStarNet:
    return build_model("justarnet", num_classes, in_channels)


def justarnet_s(num_classes: int = 5, in_channels: int = 3) -> JuStarNet:
    return build_model("justarnet_s", num_classes, in_channels)


def justarnet_xs(num_classes: int = 5, in_channels: int = 3) -> JuStarNet:
    return build_model("justarnet_xs", num_classes, in_channels)
