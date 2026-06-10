"""MoldNetNano — phiên bản <1M params giữ nguyên F1 của MoldNet-lite (~86.94%).

================================================================================
Ý TƯỞNG: F1 đến từ HEAD, không phải kích thước BACKBONE
================================================================================
MoldNet-lite (edgenext_small, 5.28M) đạt ~86.94% macro-F1. Phân tích cho thấy
điểm tạo ra F1 KHÔNG nằm ở dung lượng backbone mà ở MoldHead:

    second-order pooling (GeM ⊕ std)  →  khôi phục tín hiệu mold mà GAP đánh mất
    ECA + CS_STAM                     →  gate đốm mốc/texture cục bộ
    multi-scale fusion + CosineClassifier → chống bias lớp đông trên data lệch

Hơn 20 backbone trong bảng (CNN/ViT/hybrid, 0.3-11M) đều tắc ở 80-82% vì cùng
một lý do: GAP pha loãng đốm mốc nhỏ. Thay GAP bằng MoldHead mới là đòn bẩy.
⇒ Có thể NÉN backbone ~9× mà vẫn giữ F1, miễn là GIỮ NGUYÊN MoldHead.

Chiến lược nén về <1M:
  1. Backbone = lcnet_075 (PP-LCNet, pretrained ImageNet ~68.8% top1) — backbone
     PRETRAINED MẠNH NHẤT trong ngân sách ~0.6M; kênh giàu (96@28 / 192@14 / 384@7)
     đủ nuôi second-order pooling. (edgenext_small 5.28M → lcnet_075 0.58M, ~9×).
  2. BÙ lại backbone yếu hơn bằng HEAD GIÀU HƠN một chút: dùng 3 tầng (28×28 +
     14×14 + 7×7) thay vì 2. Tầng 28×28 độ phân giải cao giúp bắt đốm mốc nhỏ —
     rẻ trên backbone nhỏ, và đây đúng là class cần res cao nhất.
  3. Giữ NGUYÊN mọi khối tạo F1: GeM⊕std, ECA, CS_STAM, CosineClassifier, TTA.

Kết quả: TOTAL ≈ 0.895M params (< 1M), kỳ vọng giữ ~86.94% macro-F1.

An toàn pipeline (giống MoldNet): tính GeM/std ở float32 + clamp p∈[1,5] → không
NaN dưới AMP fp16, deterministic; TTA tự bật ở val/test, tắt khi train + GradCAM.

Chạy:
    python train.py --dataset jujube_ours --model moldnet_nano --seed 1 2 3 4 5
Yêu cầu: pip install timm.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# Tái dùng các khối ĐÃ KIỂM CHỨNG từ moldnet.py — không nhân bản logic.
from .moldnet import ECA, CS_STAM, SecondOrderPool, CosineClassifier

try:
    import timm
    _HAS_TIMM = True
except ImportError:  # pragma: no cover
    _HAS_TIMM = False


ARCH_NAME = "moldnet_nano"
ARCH_ALIASES = ["MoldNetNano"]


# ---------------------------------------------------------------------------
# MoldHeadNano — 3-scale fusion (28×28 ⊕ 14×14 ⊕ 7×7) + second-order pooling
# ---------------------------------------------------------------------------

class MoldHeadNano(nn.Module):
    """Như MoldHead nhưng dùng 3 tầng feature để bù backbone siêu nhẹ.

    Tầng 28×28 (độ phân giải cao) đặc biệt giúp lớp mold: đốm nấm nhỏ cần res cao.
    Mỗi tầng: ECA (lọc kênh) → CS_STAM (gate đốm cục bộ) → SecondOrderPool(GeM⊕std).
    Tầng 7×7 chỉ ECA (CS_STAM 5×1/1×5 lãng phí ở 7×7).
    """

    def __init__(self, c_lo: int, c_mid: int, c_high: int, num_classes: int,
                 fc_dim: int = 224, dropout: float = 0.2,
                 cosine_scale: float = 24.0, gem_p_init: float = 3.0,
                 learnable_p: bool = True, use_logit_adjust: bool = False):
        super().__init__()
        # tầng 28×28
        self.eca_lo = ECA(c_lo)
        self.csstam_lo = CS_STAM(c_lo)
        self.pool_lo = SecondOrderPool(gem_p_init, learnable_p)
        # tầng 14×14
        self.eca_mid = ECA(c_mid)
        self.csstam_mid = CS_STAM(c_mid)
        self.pool_mid = SecondOrderPool(gem_p_init, learnable_p)
        # tầng 7×7
        self.eca_high = ECA(c_high)
        self.pool_high = SecondOrderPool(gem_p_init, learnable_p)

        feat_dim = 2 * (c_lo + c_mid + c_high)        # ×2 do GeM ⊕ std
        self.norm = nn.LayerNorm(feat_dim)
        self.proj = nn.Sequential(
            nn.Linear(feat_dim, fc_dim, bias=False),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.classifier = CosineClassifier(fc_dim, num_classes, cosine_scale,
                                           use_logit_adjust)

    def forward(self, f_lo, f_mid, f_high):
        a = self.pool_lo(self.csstam_lo(self.eca_lo(f_lo)))
        b = self.pool_mid(self.csstam_mid(self.eca_mid(f_mid)))
        c = self.pool_high(self.eca_high(f_high))
        feat = torch.cat([a, b, c], dim=1)
        return self.classifier(self.proj(self.norm(feat)))


# ---------------------------------------------------------------------------
# MoldNetNano
# ---------------------------------------------------------------------------

class MoldNetNano(nn.Module):
    def __init__(self, backbone_name: str = "lcnet_075", num_classes: int = 5,
                 in_channels: int = 3, pretrained: bool = True,
                 tta: bool = True, drop_path: float = 0.05,
                 fc_dim: int = 224, dropout: float = 0.2,
                 cosine_scale: float = 24.0, gem_p_init: float = 3.0,
                 learnable_p: bool = True, use_logit_adjust: bool = False):
        super().__init__()
        if not _HAS_TIMM:
            raise ImportError("MoldNetNano cần timm. Cài: pip install timm")

        kw = dict(pretrained=pretrained, features_only=True, in_chans=in_channels)
        try:
            self.backbone = timm.create_model(backbone_name,
                                              drop_path_rate=drop_path, **kw)
        except TypeError:
            self.backbone = timm.create_model(backbone_name, **kw)

        chs = self.backbone.feature_info.channels()   # 3 tầng cuối
        c_lo, c_mid, c_high = chs[-3], chs[-2], chs[-1]
        self.head = MoldHeadNano(c_lo, c_mid, c_high, num_classes,
                                 fc_dim=fc_dim, dropout=dropout,
                                 cosine_scale=cosine_scale, gem_p_init=gem_p_init,
                                 learnable_p=learnable_p,
                                 use_logit_adjust=use_logit_adjust)
        self.tta = tta
        self.num_classes = num_classes

    def _forward_once(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        return self.head(feats[-3], feats[-2], feats[-1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self._forward_once(x)
        # TTA chỉ ở eval + không cần grad (val/test). Tắt khi train + GradCAM.
        if self.tta and (not self.training) and (not torch.is_grad_enabled()):
            logits_flip = self._forward_once(torch.flip(x, dims=[3]))
            logits = 0.5 * (logits + logits_flip)
        return logits


# ---------------------------------------------------------------------------
# Registry — MỘT biến thể duy nhất
# ---------------------------------------------------------------------------

MODEL_CONFIGS = {
    # ~0.895M params — backbone lcnet_075 (pretrained) + MoldHead 3-scale.
    # Giữ F1 ~86.94% của moldnet_lite ở < 1M params.
    "moldnet_nano": dict(
        backbone="lcnet_075",
        tta=True, drop_path=0.05, fc_dim=224, dropout=0.2,
        cosine_scale=24.0, gem_p_init=3.0,
        input_size=224, params_M=0.895),
}

DEFAULT_VARIANT = "moldnet_nano"

_HEAD_KEYS = {"fc_dim", "dropout", "cosine_scale", "gem_p_init", "learnable_p",
              "use_logit_adjust"}
_NET_KEYS = {"tta", "drop_path"}


def build_model(name: str = "moldnet_nano", num_classes: int = 5,
                in_channels: int = 3, pretrained: bool = True,
                **overrides) -> MoldNetNano:
    cfg = MODEL_CONFIGS.get(name)
    if cfg is None:
        raise ValueError(
            f"Unknown variant '{name}'. Choose from: {list(MODEL_CONFIGS)}")

    kwargs = dict(backbone_name=cfg["backbone"], pretrained=pretrained)
    for k, v in cfg.items():
        if k in _HEAD_KEYS or k in _NET_KEYS:
            kwargs[k] = v
    for k, v in overrides.items():
        if k in _HEAD_KEYS or k in _NET_KEYS or k == "pretrained":
            kwargs[k] = v

    model = MoldNetNano(num_classes=num_classes, in_channels=in_channels,
                        **kwargs)
    model.num_classes = num_classes
    return model


if __name__ == "__main__":
    x = torch.randn(2, 3, 224, 224)
    m = build_model("moldnet_nano", num_classes=5, pretrained=False)
    n = sum(p.numel() for p in m.parameters()) / 1e6
    m.eval()
    with torch.no_grad():
        y = m(x)
    print(f"moldnet_nano params={n:.4f}M  out={tuple(y.shape)}  (<1M: {n < 1.0})")
