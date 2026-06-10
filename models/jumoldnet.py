"""JuMoldNet — thân juformer (from-scratch) + MoldHead, <1M params, nhắm vượt MoldNet-lite.

================================================================================
Ý TƯỞNG: cộng đúng phần mold đang thiếu vào một thân vốn đã khỏe
================================================================================
juformer (EdgeJuFormer v4, ~254K, train FROM-SCRATCH) tự đạt ~84.9% macro-F1 trên
jujube_ours — gần chạm MoldNet-lite (85.04%, edgenext_small 5.28M PRETRAINED). Khác
moldnet_nano (lcnet_075 pretrained nhưng yếu, chỉ 83.89%), thân juformer đã chứng minh
fit ĐÚNG 2.6k ảnh này. Nhưng juformer kết bằng ``GAP → Linear`` — đúng cụm pha loãng
tín hiệu mold (đốm texture nhỏ chìm vào nền đen + quả lành).

JuMoldNet GIỮ NGUYÊN thân juformer, chỉ THAY cụm pooling/classifier cuối bằng MoldHead
3-scale ĐÃ KIỂM CHỨNG (tái dùng MoldHeadNano):

    second-order pooling (GeM ⊕ std)      → khôi phục tín hiệu mold GAP đánh mất
    ECA + CS_STAM                          → gate đốm mốc/texture cục bộ
    multi-scale (28×28 ⊕ 14×14 ⊕ 14×14)    → tận dụng đốm nhỏ ở độ phân giải cao
    CosineClassifier                       → chống bias lớp đông (dry 1469 vs mold 346)
    TTA (lật ngang, eval-only)

Tap feature (đo từ juformer.forward_features, input 224 — juformer chỉ downsample /16
nên sâu nhất là 14×14, KHÔNG có 7×7; đây là lợi thế cho mold vì giữ res cao):

    stage2  28×28×40  → LO   (res cao, bắt đốm mốc nhỏ)   ECA + CS_STAM
    stage3  14×14×64  → MID  (texture-context)            ECA + CS_STAM
    head    14×14×96  → HIGH (semantic, post-ReLU)        ECA

Ngân sách: thân ~253K + MoldHead ~94K ≈ 0.35M params (< 1M, nhẹ hơn moldnet_nano 0.895M
~2.5×). Train from-scratch như juformer — KHÔNG cần timm/pretrained.

An toàn pipeline (giống MoldNet): GeM/std tính ở float32 + clamp p∈[1,5] → không NaN
dưới AMP fp16, deterministic; TTA tự bật ở val/test (đều @torch.no_grad), tự tắt khi
train + GradCAM; DropPath/reparameterize của thân giữ nguyên.

Chạy:
    python train.py --dataset jujube_ours --model jumoldnet --seed 1 2 3 4 5
"""

from __future__ import annotations

import torch
import torch.nn as nn

# Tái dùng thân juformer (đã đạt 84.9% từ-scratch) — không nhân bản kiến trúc.
from .juformer import EdgeJuFormer
# Tái dùng head 3-scale ĐÃ KIỂM CHỨNG (GeM⊕std + ECA + CS_STAM + Cosine).
from .moldnet_nano import MoldHeadNano


ARCH_NAME = "jumoldnet"
ARCH_ALIASES = ["JuMoldNet"]


# ---------------------------------------------------------------------------
# JuMoldNet — thân EdgeJuFormer (bỏ GAP+Linear) + MoldHead 3-scale + TTA
# ---------------------------------------------------------------------------

class JuMoldNet(nn.Module):
    def __init__(self, num_classes: int = 5, in_channels: int = 3,
                 width_mult: float = 1.0, drop_path: float = 0.1,
                 tta: bool = True, fc_dim: int = 224, dropout: float = 0.2,
                 cosine_scale: float = 24.0, gem_p_init: float = 3.0,
                 learnable_p: bool = True, use_logit_adjust: bool = False):
        super().__init__()

        # ── Thân juformer ───────────────────────────────────────────────────
        self.body = EdgeJuFormer(num_classes=num_classes, width_mult=width_mult,
                                 drop_path=drop_path, ablation_mode=None)
        # stem nhận in_channels ≠ 3 (vd. grayscale) — như juformer.build_model
        if in_channels != 3:
            old = self.body.stem[0]
            self.body.stem[0] = nn.Conv2d(in_channels, old.out_channels,
                                          kernel_size=old.kernel_size,
                                          stride=old.stride, padding=old.padding,
                                          bias=False)
        # Gỡ GAP + classifier cũ (forward_features không gọi tới) để không đếm dư.
        self.body.pool = nn.Identity()
        self.body.classifier = nn.Identity()

        # ── Channels của 3 tap (đọc trực tiếp từ thân → robust theo width_mult) ─
        c_lo = self.body.stage2.project[0].out_channels    # stage2 out  (28×28)
        c_mid = self.body.head[0].in_channels              # stage3 out  (14×14)
        c_high = self.body.head[0].out_channels            # head   out  (14×14)

        # ── MoldHead 3-scale (lo/mid: ECA+CS_STAM ; high: ECA) ───────────────
        self.head = MoldHeadNano(c_lo, c_mid, c_high, num_classes,
                                 fc_dim=fc_dim, dropout=dropout,
                                 cosine_scale=cosine_scale, gem_p_init=gem_p_init,
                                 learnable_p=learnable_p,
                                 use_logit_adjust=use_logit_adjust)
        self.tta = tta
        self.num_classes = num_classes

    def _forward_once(self, x: torch.Tensor) -> torch.Tensor:
        f = self.body.forward_features(x)
        return self.head(f["stage2"], f["stage3"], f["head"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self._forward_once(x)
        # TTA chỉ ở eval + không cần grad (val/test). Tắt khi train + GradCAM.
        if self.tta and (not self.training) and (not torch.is_grad_enabled()):
            logits_flip = self._forward_once(torch.flip(x, dims=[3]))
            logits = 0.5 * (logits + logits_flip)
        return logits

    def reparameterize(self):
        """Fuse RepDWConv của thân → deploy (giữ nguyên API juformer)."""
        self.body.reparameterize()


# ---------------------------------------------------------------------------
# Registry — MỘT biến thể duy nhất
# ---------------------------------------------------------------------------

MODEL_CONFIGS = {
    # ~0.35M params — thân juformer from-scratch + MoldHead 3-scale.
    # Nhắm vượt moldnet_nano (83.89%) và chạm/vượt moldnet_lite (85.04%) ở < 1M.
    "jumoldnet": dict(
        width_mult=1.0, drop_path=0.1, tta=True,
        fc_dim=224, dropout=0.2, cosine_scale=24.0, gem_p_init=3.0,
        input_size=224, params_M=0.35),
}

DEFAULT_VARIANT = "jumoldnet"

_HEAD_KEYS = {"fc_dim", "dropout", "cosine_scale", "gem_p_init", "learnable_p",
              "use_logit_adjust"}
_NET_KEYS = {"tta", "drop_path", "width_mult"}


def build_model(name: str = "jumoldnet", num_classes: int = 5,
                in_channels: int = 3, pretrained: bool = False,
                **overrides) -> JuMoldNet:
    # pretrained bị bỏ qua: JuMoldNet train FROM-SCRATCH (như juformer).
    cfg = MODEL_CONFIGS.get(name)
    if cfg is None:
        raise ValueError(
            f"Unknown variant '{name}'. Choose from: {list(MODEL_CONFIGS)}")

    kwargs = {k: v for k, v in cfg.items() if k in _HEAD_KEYS or k in _NET_KEYS}
    for k, v in overrides.items():
        if k in _HEAD_KEYS or k in _NET_KEYS:
            kwargs[k] = v

    model = JuMoldNet(num_classes=num_classes, in_channels=in_channels, **kwargs)
    model.num_classes = num_classes
    return model


if __name__ == "__main__":
    x = torch.randn(2, 3, 224, 224)
    m = build_model("jumoldnet", num_classes=5)
    n = sum(p.numel() for p in m.parameters()) / 1e6
    m.eval()
    with torch.no_grad():
        y = m(x)
    print(f"jumoldnet params={n:.4f}M  out={tuple(y.shape)}  (<1M: {n < 1.0})")
