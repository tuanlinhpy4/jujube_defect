"""JuNet-Edge: the ship model for STM32H750VBT6 at 64x64, distilled from the
v1/v2 ablations (``models/junet_mcu.py`` / ``junet_mcu_v2.py``) and the on-target
ST Edge AI int8 analysis (``Analyzing.docx``).

Why this architecture (every choice is backed by a measured ablation result on
``jujube_64``, 3-5 seeds, macro-F1):

1. **KeepHi -- strides (2,2,1) so the final feature map stays 8x8.** This was the
   single strongest int8-clean lever (``keephi`` 93.84 in v1 -> 94.60 in v2). It
   matters even more here: the baselines ingest the *same* 64x64 jujube images but
   the pipeline upscales them to 224 before inference, so they run on larger
   feature maps. JuNet-Edge runs at *native 64* on the MCU; keeping an 8x8 (not
   4x4) final map is how it claws that spatial resolution back.
2. **RepVGG-style reparam depthwise (RepDW) in the two high-res stages.** ``rep``
   tied ``keephi`` (93.38 / 94.58). A RepDW trains as a 3x3 + 1x1 (+ identity)
   multi-branch depthwise but ``fuse_for_deploy()`` collapses it to a *single*
   3x3 depthwise conv -> identical int8 cost to a plain depthwise, free
   train-time capacity. Call ``model.fuse_for_deploy()`` before ONNX export.
3. **Ghost inverted residual + a single late SE in the deep stage.** Ghost ~ full
   1x1 expand for accuracy (``conv_expand`` 92.58 ~ ``full``) but cheaper; stage SE
   is int8-clean (``op_u8_u8``) and helps the deepest stage. SE is used *once*
   (late) -- ``no_attn`` ~ ``full`` so attention everywhere is not worth it.
4. **ReLU everywhere.** ``hswish``/``silu`` won FP32 (94.01 / 93.48) but are
   int8-costly (LUT / float). The ship recipe stays ReLU so the int8 graph is
   honest.
5. **No multi-kernel / dilated / mixconv.** ST Edge AI / CMSIS-NN has no int8
   dilated-depthwise kernel; the dilated branch fell back to float in the op-dump
   (~5.5% of ops). Every depthwise here is a plain 3x3 -> fully int8.

Sizing for STM32H750VBT6 (1 MB SRAM, 128 KB internal flash):
  * peak int8 activation arena at 64x64 is ~120 KB << 1 MB SRAM -> RAM is not the
    binding constraint; capacity was pushed accordingly.
  * int8 weight flash ~= #params bytes. ``junet_edge_s`` fits the 128 KB internal
    flash; ``junet_edge`` / ``junet_edge_l`` expect the weight segment in external
    QSPI/OCTOSPI (supported by the H750VBT6).

Self-contained on purpose (like ``models/junet.py``): no imports from sibling
ablation files, so the registry auto-scan picks it up on any checkout/clone.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, replace
from typing import Any, Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


ARCH_NAME = "JuNetEdge"
ARCH_ALIASES = ["junet_edge", "junetedge"]


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------
def _round_ch(channels: float, divisor: int = 8) -> int:
    """Round channels to a multiple of 8 (better for int8 SIMD kernels)."""
    rounded = max(divisor, int(channels + divisor / 2) // divisor * divisor)
    if rounded < 0.9 * channels:
        rounded += divisor
    return rounded


def act_layer(name: str) -> nn.Module:
    # Ship recipe is ReLU-only; relu6 kept as an int8-safe alternative.
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name == "relu6":
        return nn.ReLU6(inplace=True)
    raise ValueError(f"Unknown / non-int8-clean activation for JuNet-Edge: {name}")


class DropPath(nn.Module):
    """Per-sample stochastic depth for residual branches."""

    def __init__(self, p: float = 0.0):
        super().__init__()
        self.p = float(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.p <= 0.0 or not self.training:
            return x
        keep = 1.0 - self.p
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep)
        return x.div(keep) * mask


class ConvBNAct(nn.Sequential):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel: int = 1,
        stride: int = 1,
        groups: int = 1,
        act: str | None = "relu",
    ):
        padding = (kernel - 1) // 2
        layers: List[nn.Module] = [
            nn.Conv2d(in_ch, out_ch, kernel, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_ch),
        ]
        if act is not None:
            layers.append(act_layer(act))
        super().__init__(*layers)


class GhostConv(nn.Module):
    """Ghost convolution: primary 1x1 features plus cheap depthwise features."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, ratio: int = 2,
                 dw_size: int = 3, act: str = "relu"):
        super().__init__()
        self.out_ch = out_ch
        primary_ch = math.ceil(out_ch / ratio)
        cheap_ch = primary_ch * (ratio - 1)
        self.primary = ConvBNAct(in_ch, primary_ch, 1, stride=stride, act=act)
        self.cheap = ConvBNAct(primary_ch, cheap_ch, dw_size, groups=primary_ch, act=act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.primary(x)
        z = self.cheap(y)
        return torch.cat([y, z], dim=1)[:, : self.out_ch]


class SqueezeExcitation(nn.Module):
    """Stage-level channel attention (int8-clean: pool + 1x1 + ReLU + sigmoid gate)."""

    def __init__(self, channels: int, reduction: int = 4, act: str = "relu"):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, mid, bias=False),
            act_layer(act),
            nn.Linear(mid, channels, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = torch.sigmoid(self.fc(x))
        return x * w.view(x.size(0), x.size(1), 1, 1)


class RepDW(nn.Module):
    """RepVGG-style depthwise: 3x3 + 1x1 (+ identity) branches at train time,
    reparameterized to a single 3x3 depthwise conv for deployment."""

    def __init__(self, channels: int, stride: int, act: str = "relu"):
        super().__init__()
        self.channels = channels
        self.stride = stride
        self.dw3 = nn.Sequential(
            nn.Conv2d(channels, channels, 3, stride, 1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.dw1 = nn.Sequential(
            nn.Conv2d(channels, channels, 1, stride, 0, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.bn_id = nn.BatchNorm2d(channels) if stride == 1 else None
        self.act = act_layer(act)
        self.fused = False
        self.reparam: nn.Conv2d | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.fused:
            return self.act(self.reparam(x))
        out = self.dw3(x) + self.dw1(x)
        if self.bn_id is not None:
            out = out + self.bn_id(x)
        return self.act(out)

    @staticmethod
    def _fuse_conv_bn(weight: torch.Tensor, bn: nn.BatchNorm2d) -> Tuple[torch.Tensor, torch.Tensor]:
        std = torch.sqrt(bn.running_var + bn.eps)
        t = (bn.weight / std).reshape(-1, 1, 1, 1)
        kernel = weight * t
        bias = bn.bias - bn.running_mean * bn.weight / std
        return kernel, bias

    @torch.no_grad()
    def reparameterize(self) -> None:
        """Fold the multi-branch depthwise into one 3x3 depthwise conv (+ bias)."""
        if self.fused:
            return
        k3, b3 = self._fuse_conv_bn(self.dw3[0].weight, self.dw3[1])
        k1, b1 = self._fuse_conv_bn(self.dw1[0].weight, self.dw1[1])
        k1 = F.pad(k1, [1, 1, 1, 1])  # 1x1 -> 3x3 (centered)
        kernel = k3 + k1
        bias = b3 + b1
        if self.bn_id is not None:
            # identity branch == 1x1 depthwise with unit weight, then BN.
            id_w = torch.ones(self.channels, 1, 1, 1, device=k3.device, dtype=k3.dtype)
            kid, bid = self._fuse_conv_bn(id_w, self.bn_id)
            kernel = kernel + F.pad(kid, [1, 1, 1, 1])
            bias = bias + bid

        conv = nn.Conv2d(self.channels, self.channels, 3, self.stride, 1,
                         groups=self.channels, bias=True)
        conv.weight.data.copy_(kernel)
        conv.bias.data.copy_(bias)
        self.reparam = conv
        for name in ("dw3", "dw1", "bn_id"):
            if getattr(self, name, None) is not None:
                delattr(self, name)
            elif name in self._modules:
                del self._modules[name]
        self.fused = True


# ---------------------------------------------------------------------------
# Blocks
# ---------------------------------------------------------------------------
class GhostIRBlock(nn.Module):
    """Inverted residual: Ghost expand -> single 3x3 depthwise -> 1x1 project."""

    def __init__(self, in_ch: int, out_ch: int, stride: int, expand: float,
                 act: str = "relu", drop_path: float = 0.0):
        super().__init__()
        hidden = _round_ch(in_ch * expand)
        self.use_res = stride == 1 and in_ch == out_ch
        self.expand = GhostConv(in_ch, hidden, act=act)
        self.dw = ConvBNAct(hidden, hidden, 3, stride=stride, groups=hidden, act=act)
        self.project = ConvBNAct(hidden, out_ch, 1, act=None)
        self.drop = DropPath(drop_path) if self.use_res and drop_path > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.project(self.dw(self.expand(x)))
        return x + self.drop(y) if self.use_res else y


class RepIRBlock(nn.Module):
    """Inverted residual: Ghost expand -> RepDW (reparam) -> 1x1 project."""

    def __init__(self, in_ch: int, out_ch: int, stride: int, expand: float,
                 act: str = "relu", drop_path: float = 0.0):
        super().__init__()
        hidden = _round_ch(in_ch * expand)
        self.use_res = stride == 1 and in_ch == out_ch
        self.expand = GhostConv(in_ch, hidden, act=act)
        self.dw = RepDW(hidden, stride, act=act)
        self.project = ConvBNAct(hidden, out_ch, 1, act=None)
        self.drop = DropPath(drop_path) if self.use_res and drop_path > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.project(self.dw(self.expand(x)))
        return x + self.drop(y) if self.use_res else y


_BLOCKS = {"ghost_ir": GhostIRBlock, "rep": RepIRBlock}


# ---------------------------------------------------------------------------
# Stage specification
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class StageSpec:
    out_ch: int
    depth: int
    stride: int
    block: str = "ghost_ir"     # ghost_ir | rep
    expand: float = 3.0
    attn: str = "none"          # none | se  (stage-level, applied after blocks)


def make_stages(
    blocks: Tuple[str, str, str] = ("rep", "rep", "ghost_ir"),
    channels: Tuple[int, int, int] = (48, 72, 104),
    expand: Tuple[float, float, float] = (4.0, 3.0, 3.0),
    attn: Tuple[str, str, str] = ("none", "none", "se"),
    depths: Tuple[int, int, int] = (1, 1, 2),
    strides: Tuple[int, int, int] = (2, 2, 1),   # keephi: final map stays 8x8
) -> List[StageSpec]:
    return [
        StageSpec(channels[i], depths[i], strides[i], blocks[i], expand[i], attn[i])
        for i in range(3)
    ]


def _attention(kind: str, channels: int, se_reduction: int, act: str) -> nn.Module:
    if kind == "se":
        return SqueezeExcitation(channels, reduction=se_reduction, act=act)
    if kind == "none":
        return nn.Identity()
    raise ValueError(f"Unknown attention: {kind}")


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class JuNetEdge(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 4,
        stem_ch: int = 32,
        stages: Sequence[StageSpec] | None = None,
        head_ch: int = 160,
        act: str = "relu",
        se_reduction: int = 4,
        dropout: float = 0.0,
        drop_path: float = 0.1,
        width_mult: float = 1.0,
    ):
        super().__init__()
        stages = list(stages if stages is not None else make_stages())
        self.config: Dict[str, Any] = {
            "stem_ch": stem_ch,
            "stages": [vars(s) for s in stages],
            "head_ch": head_ch,
            "act": act,
            "dropout": dropout,
            "drop_path": drop_path,
            "width_mult": width_mult,
            "num_classes": num_classes,
        }

        if width_mult != 1.0:
            stem_ch = _round_ch(stem_ch * width_mult)
            head_ch = _round_ch(head_ch * width_mult)
            stages = [replace(s, out_ch=_round_ch(s.out_ch * width_mult)) for s in stages]

        # Linear stochastic-depth schedule over residual blocks.
        total_res, prev = 0, stem_ch
        for s in stages:
            for d in range(s.depth):
                stride = s.stride if d == 0 else 1
                if stride == 1 and prev == s.out_ch:
                    total_res += 1
                prev = s.out_ch
        res_idx = 0

        self.stem = ConvBNAct(in_channels, stem_ch, 3, stride=2, act=act)

        stage_modules: List[nn.Module] = []
        prev = stem_ch
        for s in stages:
            layers: List[nn.Module] = []
            for d in range(s.depth):
                stride = s.stride if d == 0 else 1
                use_res = stride == 1 and prev == s.out_ch
                dp = 0.0
                if use_res:
                    dp = drop_path * res_idx / max(total_res - 1, 1) if total_res > 1 else drop_path
                    res_idx += 1
                layers.append(_BLOCKS[s.block](prev, s.out_ch, stride, s.expand, act=act, drop_path=dp))
                prev = s.out_ch
            layers.append(_attention(s.attn, s.out_ch, se_reduction, act))
            stage_modules.append(nn.Sequential(*layers))
        self.stages = nn.Sequential(*stage_modules)

        self.head_conv = ConvBNAct(prev, head_ch, 1, act=act)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.classifier = nn.Linear(head_ch, num_classes)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stages(x)
        x = self.head_conv(x)
        x = self.global_pool(x).flatten(1)
        return self.classifier(self.dropout(x))

    @torch.no_grad()
    def fuse_for_deploy(self) -> "JuNetEdge":
        """Reparameterize every RepDW to a single 3x3 depthwise conv. Call in
        eval() mode before int8 ONNX export. Idempotent."""
        was_training = self.training
        self.eval()
        for m in self.modules():
            if isinstance(m, RepDW):
                m.reparameterize()
        if was_training:
            self.train()
        return self

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        total = self.count_parameters()
        return f"JuNetEdge(params={total:,} ({total / 1e6:.4f}M))"


# ---------------------------------------------------------------------------
# Variant configs
# ---------------------------------------------------------------------------
def _cfg(stem_ch: int = 32, head_ch: int = 160, act: str = "relu", dropout: float = 0.0,
         drop_path: float = 0.1, width_mult: float = 1.0,
         stages: List[StageSpec] | None = None, **meta) -> Dict[str, Any]:
    cfg = {
        "stem_ch": stem_ch,
        "head_ch": head_ch,
        "act": act,
        "dropout": dropout,
        "drop_path": drop_path,
        "width_mult": width_mult,
        "stages": stages if stages is not None else make_stages(),
        "input_size": 64,
        "params_M": None,
    }
    cfg.update(meta)
    return cfg


MODEL_CONFIGS: Dict[str, Dict[str, Any]] = {
    # -- Default ship model: rep front + Ghost deep + keephi (8x8) + late SE.
    "junet_edge": _cfg(),

    # -- Small: fits the 128 KB internal flash (no external QSPI needed).
    "junet_edge_s": _cfg(stem_ch=24, head_ch=128, stages=make_stages(
        channels=(32, 48, 72))),

    # -- Large: max-accuracy push (wider + deeper stage3). Weights -> QSPI.
    "junet_edge_l": _cfg(stem_ch=32, head_ch=192, dropout=0.1, drop_path=0.12,
        stages=make_stages(channels=(56, 88, 128), depths=(1, 2, 3))),
}


DEFAULT_VARIANT = "junet_edge"


_NOTES = {
    "junet_edge": "Ship model: RepDW front + Ghost deep + keephi 8x8 + late SE, ReLU, int8-clean.",
    "junet_edge_s": "Smaller variant sized to fit 128 KB internal flash.",
    "junet_edge_l": "Wider/deeper max-accuracy variant (weights expect external QSPI flash).",
}


def build_model(name: str = DEFAULT_VARIANT, num_classes: int = 4, in_channels: int = 3,
                **overrides) -> JuNetEdge:
    if name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown JuNet-Edge variant: {name}. Choose from {sorted(MODEL_CONFIGS)}")
    cfg = copy.deepcopy(MODEL_CONFIGS[name])
    cfg.update(overrides)
    cfg = {k: v for k, v in cfg.items() if k not in ("input_size", "params_M")}
    return JuNetEdge(in_channels=in_channels, num_classes=num_classes, **cfg)


def list_variants() -> List[str]:
    return sorted(MODEL_CONFIGS)


def junet_edge(num_classes: int = 4, in_channels: int = 3) -> JuNetEdge:
    return build_model(DEFAULT_VARIANT, num_classes=num_classes, in_channels=in_channels)


if __name__ == "__main__":
    # On-device sizing + int8-cleanliness + fuse round-trip check for H750VBT6.
    RAM = 1024 * 1024
    INT_FLASH = 128 * 1024
    print(f"{'variant':16s} {'params':>9s} {'w~KB':>7s} {'peakAct~KB':>11s}  flags")
    for variant in MODEL_CONFIGS:
        model = build_model(variant, num_classes=4).eval()
        params = model.count_parameters()

        peak = {"v": 0}

        def _hook(_m, inp, out, _p=peak):
            if inp and torch.is_tensor(inp[0]) and torch.is_tensor(out):
                _p["v"] = max(_p["v"], inp[0].numel() + out.numel())

        handles = [m.register_forward_hook(_hook) for m in model.modules()]
        x = torch.zeros(1, 3, 64, 64)
        with torch.no_grad():
            y_ref = model(x)
        for h in handles:
            h.remove()

        # fuse round-trip: output must be unchanged after reparameterization.
        model.fuse_for_deploy()
        with torch.no_grad():
            y_fused = model(x)
        max_err = (y_ref - y_fused).abs().max().item()

        # int8-clean audit: no dilated convs, only ReLU/ReLU6 nonlinearities.
        dil = sum(1 for m in model.modules()
                  if isinstance(m, nn.Conv2d) and tuple(m.dilation) != (1, 1))
        bad_act = sum(1 for m in model.modules()
                      if isinstance(m, nn.Module)
                      and m.__class__.__name__ in {"SiLU", "Hardswish", "GELU", "Mish"})

        flags = []
        if params > INT_FLASH:
            flags.append("ext-flash")
        if peak["v"] > RAM:
            flags.append("OVER-RAM!")
        if dil:
            flags.append(f"DILATED:{dil}")
        if bad_act:
            flags.append(f"NON-INT8-ACT:{bad_act}")
        if max_err > 1e-3:
            flags.append(f"FUSE-MISMATCH:{max_err:.2e}")
        print(f"{variant:16s} {params:>9,} {params/1024:>6.1f}K "
              f"{peak['v']/1024:>10.1f}K  fuse_err={max_err:.1e}  {' '.join(flags) or 'OK'}")
