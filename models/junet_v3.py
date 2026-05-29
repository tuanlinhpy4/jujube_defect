"""JuNetV3 — JuNetV2.1 sau khi chốt debate Codex/Opus.

Mục tiêu:
  - Giữ backbone GhostIR/MKGhostIR nhẹ của JuNetV2.
  - Sửa lỗi mất texture của JuNetV2 v1: stage4 giữ 14x14, không dilation lớn.
  - Bỏ SE hoàn toàn; chỉ dùng attention muộn, nhẹ.
  - Stage3: ECA -> CS-STAM để học texture/spatial pattern cho mold/insect.
  - Stage4: ECA.
  - Head: multi-scale GAP stage3 + stage4/head, đưa thẳng vào CosineClassifier.

Không sửa pipeline train, loss, sampler, augmentation, hay input size.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn

from .edgejunet_ghost import (
    DropPath,
    GhostConv,
    _cbr,
)
from .jupiconet import (
    CS_STAM,
    CosineClassifier,
    ECA,
)


ARCH_NAME = "JuNetV3"
ARCH_ALIASES = ["junetv3", "junet_v3"]


# ---------------------------------------------------------------------------
# Ghost Inverted Residual với dilation tùy chỉnh
# ---------------------------------------------------------------------------

class GhostIR(nn.Module):
    """Ghost inverted residual đơn nhánh."""

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
# Multi-Kernel Ghost IR với 2 dilation
# ---------------------------------------------------------------------------

class MKGhostIR(nn.Module):
    """Ghost IR chia hidden thành 2 nhánh depthwise dilation khác nhau."""

    def __init__(self, in_ch: int, out_ch: int, stride: int,
                 expand_ratio: float,
                 dilations: Tuple[int, int] = (1, 2),
                 drop_path: float = 0.0):
        super().__init__()
        assert stride in (1, 2)
        hidden = int(in_ch * expand_ratio)
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


def _build_stage(blocks_cfg: List[Dict[str, Any]]) -> nn.Sequential:
    """Build stage từ list cfg: type, in_ch, out_ch, stride, expand, dilation(s)."""
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
    return nn.Sequential(*layers)


# ---------------------------------------------------------------------------
# Multi-scale head: GAP(stage3) + GAP(head(stage4)) -> CosineClassifier
# ---------------------------------------------------------------------------

class MultiScaleCosineHead(nn.Module):
    def __init__(self, ch_s3: int, ch_s4: int, head_ch: int,
                 num_classes: int, scale: float = 16.0):
        super().__init__()
        self.head_conv = _cbr(ch_s4, head_ch, 1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = CosineClassifier(ch_s3 + head_ch, num_classes,
                                           scale=scale)

    def forward(self, x_s3: torch.Tensor, x_s4: torch.Tensor) -> torch.Tensor:
        h4 = self.head_conv(x_s4)
        f3 = self.pool(x_s3).flatten(1)
        f4 = self.pool(h4).flatten(1)
        return self.classifier(torch.cat([f3, f4], dim=1))

    def forward_with_head(self, x_s3: torch.Tensor,
                          x_s4: torch.Tensor) -> Dict[str, torch.Tensor]:
        h4 = self.head_conv(x_s4)
        f3 = self.pool(x_s3).flatten(1)
        f4 = self.pool(h4).flatten(1)
        fused = torch.cat([f3, f4], dim=1)
        return {
            "head": h4,
            "fused": fused,
            "logits": self.classifier(fused),
        }


# ---------------------------------------------------------------------------
# JuNetV3
# ---------------------------------------------------------------------------

class JuNetV3(nn.Module):
    """JuNetV3: JuNetV2 backbone + late ECA/CS-STAM + multi-scale cosine head.

    Layout input 224x224:
      stem     3 -> 16, stride 2                         112x112
      stage1  16 -> 24, MKGhostIR(1,2) + GhostIR          56x56
      stage2  24 -> 40, GhostIR + MKGhostIR(1,2)          28x28
      stage3  40 -> 64, GhostIR + MKGhostIR(1,3) + GhostIR
                      -> ECA -> CS-STAM                  14x14
      stage4  64 -> 80, GhostIR + GhostIR, stride 1
                      -> ECA                             14x14
      head    GAP(stage3) + GAP(1x1(stage4 80->128))
                      -> CosineClassifier
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 4,
        drop_path: float = 0.1,
        cosine_scale: float = 16.0,
    ):
        super().__init__()

        self.config: Dict[str, Any] = {
            "drop_path": drop_path,
            "cosine_scale": cosine_scale,
            "num_classes": num_classes,
        }

        c_stem = 16
        c_s1 = 24
        c_s2 = 40
        c_s3 = 64
        c_s4 = 80
        c_head = 128

        self.stem = _cbr(in_channels, c_stem, 3, stride=2)

        # Stage1: no attention.
        self.stage1 = _build_stage([
            dict(type="mk", in_ch=c_stem, out_ch=c_s1, stride=2,
                 expand=4, dilations=(1, 2)),
            dict(type="ir", in_ch=c_s1, out_ch=c_s1, stride=1,
                 expand=3, dilation=1, drop_path=drop_path * 0.25),
        ])

        # Stage2: no attention.
        self.stage2 = _build_stage([
            dict(type="ir", in_ch=c_s1, out_ch=c_s2, stride=2, expand=3),
            dict(type="mk", in_ch=c_s2, out_ch=c_s2, stride=1,
                 expand=3, dilations=(1, 2),
                 drop_path=drop_path * 0.5),
        ])

        # Stage3: ECA -> CS-STAM, matching the verified JuPicoNet order.
        self.stage3 = _build_stage([
            dict(type="ir", in_ch=c_s2, out_ch=c_s3, stride=2, expand=3),
            dict(type="mk", in_ch=c_s3, out_ch=c_s3, stride=1,
                 expand=3, dilations=(1, 3),
                 drop_path=drop_path * 0.75),
            dict(type="ir", in_ch=c_s3, out_ch=c_s3, stride=1,
                 expand=3, dilation=1, drop_path=drop_path),
        ])
        self.stage3_eca = ECA(c_s3)
        self.stage3_csstam = CS_STAM(c_s3)

        # Stage4: keep 14x14, no dilation>1, ECA only.
        self.stage4 = _build_stage([
            dict(type="ir", in_ch=c_s3, out_ch=c_s4, stride=1, expand=3),
            dict(type="ir", in_ch=c_s4, out_ch=c_s4, stride=1,
                 expand=3, dilation=1, drop_path=drop_path),
        ])
        self.stage4_eca = ECA(c_s4)

        self.head = MultiScaleCosineHead(c_s3, c_s4, c_head, num_classes,
                                         scale=cosine_scale)

        self._init_weights()

    def _init_weights(self) -> None:
        # KaimingNormal fan_out cho Conv/Linear backbone, nhưng EXCLUDE các module
        # attention đã được init kỹ trong constructor riêng (ECA, CS_STAM):
        #   - ECA: Conv1d kernel 3-5, fan_out quá nhỏ sẽ làm std lớn
        #   - CS_STAM: fusion Conv2d 1x1 fan_out=1 → std=1.41 quá lớn,
        #     dễ làm sigmoid mask saturate ngay từ epoch đầu
        # Giữ default Pytorch init (kaiming_uniform a=sqrt(5)) cho 2 module này.
        attention_modules = (ECA, CS_STAM)
        attention_param_ids = set()
        for m in self.modules():
            if isinstance(m, attention_modules):
                for p in m.parameters(recurse=True):
                    attention_param_ids.add(id(p))

        for m in self.modules():
            if isinstance(m, attention_modules):
                continue
            if isinstance(m, (nn.Conv1d, nn.Conv2d)):
                if id(m.weight) in attention_param_ids:
                    continue
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

    def _forward_backbone(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x_s3 = self.stage3(x)
        x_s3 = self.stage3_eca(x_s3)
        x_s3 = self.stage3_csstam(x_s3)
        x_s4 = self.stage4(x_s3)
        x_s4 = self.stage4_eca(x_s4)
        return x_s3, x_s4

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_s3, x_s4 = self._forward_backbone(x)
        return self.head(x_s3, x_s4)

    def forward_with_features(self, x: torch.Tensor) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        x = self.stem(x); result["stem"] = x
        x = self.stage1(x); result["stage1"] = x
        x = self.stage2(x); result["stage2"] = x
        x = self.stage3(x); result["stage3_raw"] = x
        x = self.stage3_eca(x); result["stage3_eca"] = x
        x = self.stage3_csstam(x); result["stage3"] = x
        x = self.stage4(x); result["stage4_raw"] = x
        x = self.stage4_eca(x); result["stage4"] = x
        result.update(self.head.forward_with_head(result["stage3"], x))
        return result

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        total = self.count_parameters()
        return f"JuNetV3(params={total:,} ({total * 4 / 1024:.1f} KB))"


# ---------------------------------------------------------------------------
# Registry — một size mặc định
# ---------------------------------------------------------------------------

MODEL_CONFIGS = {
    "junetv3": dict(
        drop_path=0.1,
        cosine_scale=16.0,
        input_size=224,
        params_M=0.149,
    ),
}

DEFAULT_VARIANT = "junetv3"


def build_model(name: str = "junetv3", num_classes: int = 4,
                in_channels: int = 3, **overrides) -> JuNetV3:
    if name not in MODEL_CONFIGS:
        raise ValueError(
            f"Unknown model: {name}. Choose from {list(MODEL_CONFIGS.keys())}"
        )
    cfg = {k: v for k, v in MODEL_CONFIGS[name].items()
           if k not in ("input_size", "params_M")}
    cfg.update(overrides)
    return JuNetV3(in_channels=in_channels, num_classes=num_classes, **cfg)


def junetv3(num_classes: int = 4, in_channels: int = 3) -> JuNetV3:
    return build_model("junetv3", num_classes, in_channels)
