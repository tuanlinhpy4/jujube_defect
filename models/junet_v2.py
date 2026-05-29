"""JuNet v2 — JuNet mở rộng theo Phương án B.

Khác biệt chính so với JuNet gốc (36K):
  - 4 stages thay vì 3 (thêm stage4 stride=2 → resolution 7×7)
  - Channels mở rộng: 16→24→40→64→80 (thay vì 16→24→32→48)
  - Depth tăng: [2, 2, 3, 2] (thay vì [1, 1, 2])
  - Đa dạng dilation tại stage 3 (alternating [1, 3]) và stage 4 (dilation=2 ở
    sub-block cuối) — mở rộng RF cho texture mold/insect mà gần như miễn phí
    về params.
  - Vẫn giữ DNA của JuNet: SE attention (không thay bằng ECA/CS_STAM —
    đó là biên giới của JuPicoNet).

Triết lý: tăng capacity một cách có kỷ luật, không port linh tinh từ JuPicoNet.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn

from .edgejunet_ghost import (
    DropPath,
    GhostConv,
    _cbr,
)


ARCH_NAME = "JuNetV2"
ARCH_ALIASES = ["junetv2", "junet_v2"]


# ---------------------------------------------------------------------------
# Squeeze-Excitation (giữ giống JuNet gốc)
# ---------------------------------------------------------------------------

class SqueezeExcitation(nn.Module):
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = torch.sigmoid(self.fc(x))
        return x * w.view(x.size(0), x.size(1), 1, 1)


# ---------------------------------------------------------------------------
# Ghost Inverted Residual với dilation tùy chỉnh
# ---------------------------------------------------------------------------

class GhostIR(nn.Module):
    """Ghost IR đơn nhánh với dilation cấu hình được."""

    def __init__(self, in_ch: int, out_ch: int, stride: int,
                 expand_ratio: float, dilation: int = 1,
                 drop_path: float = 0.0):
        super().__init__()
        assert stride in (1, 2)
        hidden = int(in_ch * expand_ratio)
        self.use_res = (stride == 1) and (in_ch == out_ch)

        self.expand = GhostConv(in_ch, hidden, 1, relu=True)
        self.dw = _cbr(hidden, hidden, 3, stride=stride,
                        groups=hidden, dilation=dilation)
        self.project = nn.Sequential(
            nn.Conv2d(hidden, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.drop = DropPath(drop_path) if (drop_path > 0 and self.use_res) else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.project(self.dw(self.expand(x)))
        if self.use_res:
            out = self.drop(out) + x
        return out


# ---------------------------------------------------------------------------
# Multi-Kernel Ghost IR với 2 dilation tùy chỉnh
# ---------------------------------------------------------------------------

class MKGhostIR(nn.Module):
    """MK-IR Ghost: chia hidden thành 2 nửa, mỗi nửa 1 dilation khác nhau."""

    def __init__(self, in_ch: int, out_ch: int, stride: int,
                 expand_ratio: float,
                 dilations: Tuple[int, int] = (1, 2),
                 drop_path: float = 0.0):
        super().__init__()
        assert stride in (1, 2)
        hidden = int(in_ch * expand_ratio)
        # Bảo đảm hidden chẵn để chia đôi
        if hidden % 2 != 0:
            hidden += 1
        self.half = hidden // 2
        self.use_res = (stride == 1) and (in_ch == out_ch)

        self.expand = GhostConv(in_ch, hidden, 1, relu=True)
        d1, d2 = dilations
        self.dw1 = _cbr(self.half, self.half, 3, stride=stride,
                         groups=self.half, dilation=d1)
        self.dw2 = _cbr(self.half, self.half, 3, stride=stride,
                         groups=self.half, dilation=d2)
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
# Stage builder — list các sub-block + 1 SE cuối stage
# ---------------------------------------------------------------------------

def _build_stage(blocks_cfg: List[Dict[str, Any]],
                  out_ch: int,
                  se_reduction: int = 4) -> nn.Sequential:
    """Mỗi cfg: dict(type, in_ch, out_ch, stride, expand, dilations|dilation, drop_path)."""
    layers: List[nn.Module] = []
    for cfg in blocks_cfg:
        btype = cfg["type"]
        if btype == "ir":
            layers.append(GhostIR(
                cfg["in_ch"], cfg["out_ch"], stride=cfg["stride"],
                expand_ratio=cfg["expand"],
                dilation=cfg.get("dilation", 1),
                drop_path=cfg.get("drop_path", 0.0),
            ))
        elif btype == "mk":
            layers.append(MKGhostIR(
                cfg["in_ch"], cfg["out_ch"], stride=cfg["stride"],
                expand_ratio=cfg["expand"],
                dilations=cfg.get("dilations", (1, 2)),
                drop_path=cfg.get("drop_path", 0.0),
            ))
        else:
            raise ValueError(f"Unknown block type: {btype}")
    layers.append(SqueezeExcitation(out_ch, reduction=se_reduction))
    return nn.Sequential(*layers)


# ---------------------------------------------------------------------------
# JuNet V2 — 4 stages, multi-dilation, depth [2,2,3,2]
# ---------------------------------------------------------------------------

class JuNetV2(nn.Module):
    """JuNet v2 — phương án B: 4 stages mở rộng, multi-dilation.

    Layout (input 224×224×3):
      stem  (Conv 3×3 s=2):       3 → 16        (112×112)
      stage1 [MK-IR + IR]:       16 → 24        (56×56)
      stage2 [IR + MK-IR]:       24 → 40        (28×28)
      stage3 [IR + MK-IR + IR]:  40 → 64        (14×14)   ← multi-dilation [1,3]
      stage4 [IR + IR(d=2)]:     64 → 80        (7×7)     ← deep semantic, d=2 cuối
      head   (1×1 + GAP + FC):   80 → 128 → C
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 4,
        drop_path: float = 0.1,
        se_reduction: int = 4,
    ):
        super().__init__()

        self.config: Dict[str, Any] = {
            "drop_path": drop_path,
            "se_reduction": se_reduction,
            "num_classes": num_classes,
        }

        c_stem = 16
        c_s1 = 24
        c_s2 = 40
        c_s3 = 64
        c_s4 = 80
        c_head = 128

        self.stem = _cbr(in_channels, c_stem, 3, stride=2)

        # Stage 1: MK-IR (dilation [1,2]) + IR — 16→24
        self.stage1 = _build_stage([
            dict(type="mk", in_ch=c_stem, out_ch=c_s1, stride=2,
                  expand=4, dilations=(1, 2)),
            dict(type="ir", in_ch=c_s1, out_ch=c_s1, stride=1,
                  expand=3, dilation=1, drop_path=drop_path * 0.25),
        ], c_s1, se_reduction=se_reduction)

        # Stage 2: IR + MK-IR (dilation [1,2]) — 24→40
        self.stage2 = _build_stage([
            dict(type="ir", in_ch=c_s1, out_ch=c_s2, stride=2, expand=3),
            dict(type="mk", in_ch=c_s2, out_ch=c_s2, stride=1,
                  expand=3, dilations=(1, 2),
                  drop_path=drop_path * 0.5),
        ], c_s2, se_reduction=se_reduction)

        # Stage 3: IR + MK-IR (dilation [1,3]) + IR — 40→64
        # Sub-block giữa dùng dilation 3 → effective RF lớn cho texture rộng
        self.stage3 = _build_stage([
            dict(type="ir", in_ch=c_s2, out_ch=c_s3, stride=2, expand=3),
            dict(type="mk", in_ch=c_s3, out_ch=c_s3, stride=1,
                  expand=3, dilations=(1, 3),
                  drop_path=drop_path * 0.75),
            dict(type="ir", in_ch=c_s3, out_ch=c_s3, stride=1,
                  expand=3, dilation=1, drop_path=drop_path),
        ], c_s3, se_reduction=se_reduction)

        # Stage 4: IR + IR(dilation=2) — 64→80
        # Sub-block 2 dùng dilation=2 → mở rộng RF khi resolution đã 7×7
        self.stage4 = _build_stage([
            dict(type="ir", in_ch=c_s3, out_ch=c_s4, stride=2, expand=3),
            dict(type="ir", in_ch=c_s4, out_ch=c_s4, stride=1,
                  expand=3, dilation=2, drop_path=drop_path),
        ], c_s4, se_reduction=se_reduction)

        # Head — giống pattern JuNet (single-scale, không multi-scale fusion)
        self.head_conv = _cbr(c_s4, c_head, 1)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(c_head, num_classes)

        self._init_weights()

    def _init_weights(self) -> None:
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
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                         nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.head_conv(x)
        x = self.global_pool(x).flatten(1)
        return self.classifier(x)

    def forward_with_features(self, x: torch.Tensor) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        x = self.stem(x);   result["stem"] = x
        x = self.stage1(x); result["stage1"] = x
        x = self.stage2(x); result["stage2"] = x
        x = self.stage3(x); result["stage3"] = x
        x = self.stage4(x); result["stage4"] = x
        x = self.head_conv(x); result["head"] = x
        x = self.global_pool(x).flatten(1)
        result["logits"] = self.classifier(x)
        return result

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        total = self.count_parameters()
        return f"JuNetV2(params={total:,} ({total * 4 / 1024:.1f} KB))"


# ---------------------------------------------------------------------------
# Registry — chỉ 1 size mặc định, không variant
# ---------------------------------------------------------------------------

MODEL_CONFIGS = {
    "junetv2": dict(
        drop_path=0.1,
        se_reduction=4,
        input_size=224,
        params_M=0.154,
    ),
}

DEFAULT_VARIANT = "junetv2"


def build_model(name: str = "junetv2", num_classes: int = 4,
                in_channels: int = 3, **overrides) -> JuNetV2:
    if name not in MODEL_CONFIGS:
        raise ValueError(
            f"Unknown model: {name}. Choose from {list(MODEL_CONFIGS.keys())}"
        )
    cfg = {k: v for k, v in MODEL_CONFIGS[name].items()
           if k not in ("input_size", "params_M")}
    cfg.update(overrides)
    return JuNetV2(in_channels=in_channels, num_classes=num_classes, **cfg)


def junetv2(num_classes: int = 4, in_channels: int = 3) -> JuNetV2:
    return build_model("junetv2", num_classes, in_channels)
