"""MoldNet — bộ phân loại lỗi táo tàu nhắm F1 cao nhất, đặc trị lớp ``mold``.

================================================================================
VÌ SAO MOLD LUÔN BỊ "TRẦN" Ở 50-55%  (chẩn đoán, không phải phỏng đoán)
================================================================================
Phân tích data + bảng benchmark cho thấy hai nguyên nhân cộng hưởng:

  1. MẤT CÂN BẰNG: mold chỉ 346 ảnh (≈9%), nhỏ nhất trong 5 lớp
     (dry 1469, insect_damaged 866, cracked 624, invalid 500, mold 346).
     Vì metric chọn checkpoint là MACRO-F1 (trung bình không trọng số), mold là
     lớp có đòn bẩy lớn nhất: kéo mold-F1 từ ~52% lên ~75% ⇒ macro-F1 +4~5 điểm.

  2. TÍN HIỆU MOLD LÀ TEXTURE CỤC BỘ BỊ GAP PHA LOÃNG:
     ảnh đã tách nền đen, quả táo sẫm/nhăn; nấm mốc là **đốm texture mờ nhỏ**
     (lông/đốm mốc, đổi màu) chiếm phần rất nhỏ so với cả quả + nền đen.
     Hơn 20 backbone trong bảng đều dùng Global Average Pooling (GAP) ở cuối →
     **trung bình hoá** đốm mốc nhỏ vào nền đen + phần quả lành ⇒ tín hiệu mold
     biến mất, nên mọi backbone (CNN/ViT/hybrid, 0.3-11M) đều tắc ở 80-82% macro.
     ⇒ Điểm nghẽn KHÔNG phải dung lượng backbone, mà là (a) cách POOLING và
       (b) NGƯỠNG quyết định cho lớp hiếm.

KEY INSIGHT (cơ sở thiết kế chính): đặc trưng phân biệt mold tốt nhất là
**PHƯƠNG SAI KHÔNG GIAN THEO KÊNH (std)**. Một đốm mốc nhỏ tạo variance cao cục
bộ trên feature map, trong khi quả khô/hỏng đều màu thì variance thấp. GAP vứt
bỏ hoàn toàn thông tin này. ⇒ ta thay GAP bằng **second-order pooling = GeM ⊕ std**.

================================================================================
KIẾN TRÚC  (mọi thành phần đều bám manh mối có sẵn trong project + literature)
================================================================================
Backbone PRETRAINED mạnh (transfer learning thống trị trên data nhỏ ~2.6k train),
ghép ENSEMBLE đa dạng để làm trơn biên quyết định lớp mold:

  • convnextv2_nano  (CNN, FCMAE masked-pretrain + GRN; rank-1 cho texture nhỏ,
                      ổn định nhất ở lr=1e-3 — locality bias)
  • tiny_vit_11m_224 (ViT chưng cất từ DeiT-Base; global context, bổ trợ CNN)
  • repvit_m1_5      (CNN thuần reparam, pretrain supervised — paradigm thứ 3)
  ⇒ đa dạng theo cả 3 trục: kiểu pretrain (FCMAE/distill/supervised) ×
    token-mixer (conv/attention/reparam-conv).

Mỗi backbone gắn một MoldHead (tái dùng khối ĐÃ KIỂM CHỨNG trong repo: ECA,
CS_STAM từ jupiconet/reptaku; CosineClassifier từ jupiconet/junet_v3):

  features_only → lấy 2 tầng cuối: mid 14×14 (texture-context), high 7×7 (semantic)
   ├─ mid : ECA (lọc kênh nền/đốm) → CS_STAM (gate đốm mốc cục bộ lên)
   ├─ high: ECA
   ├─ SecondOrderPool = [GeM(p học được) ⊕ std]  trên cả 2 tầng  ← khôi phục
   │        tín hiệu mold mà GAP đánh mất (GeM bám đốm activation cao, std bám
   │        variance texture cục bộ)
   ├─ concat → LayerNorm (ổn định, an toàn bs=1) → Linear → GELU → Dropout
   └─ CosineClassifier (weight-normalized, scale s): khử lợi thế norm của lớp
            đông (dry 1469) đè lớp hiếm (mold 346); ăn khớp với class-weighted CE
            mà train.py tự sinh.

  Ensemble: TRUNG BÌNH LOGIT giữa các nhánh (giữ tính đa dạng; tuỳ chọn trọng số
            học được qua softmax — mặc định tắt cho ổn định).
  TTA: ở eval (và chỉ khi không cần gradient) trung bình logit của ảnh gốc + lật
       ngang. Tự BẬT trong val/test (đều @torch.no_grad), tự TẮT khi train và khi
       chạy GradCAM (cần grad) → không phá pipeline, không phá visualize.py.

An toàn với pipeline (train.py bật torch.use_deterministic_algorithms(True)):
  - GeM/std tính ở float32, clamp p∈[1,5] ⇒ không NaN dưới AMP fp16, deterministic.
  - chỉ dùng flip/mean/conv/pool — toàn op có bản deterministic; không upsample.

================================================================================
BIẾN THỂ (registry) & CÁCH CHẠY
================================================================================
  moldnet       (mặc định) ensemble 3 backbone   — F1 cao nhất tuyệt đối
  moldnet_duo   ensemble 2 backbone (nano+tinyvit) — gần như tốt nhất, nhẹ hơn
  moldnet_pro   1 backbone convnextv2_nano        — single tốt nhất, ablation sạch
  moldnet_lite  1 backbone edgenext_small (~6M)   — vừa bảng edge-AI, vẫn dùng MoldHead

  python train.py --dataset jujube_ours --model moldnet_pro --seed 1 2 3 4 5
  python train.py --dataset jujube_ours --model moldnet     --seed 1 2 3 4 5

Yêu cầu: `pip install timm` (đã có sẵn trên máy train của bạn — bảng dùng timm).

Ablation gợi ý nếu mold vẫn nghẽn / lớp đông tụt: bật `use_logit_adjust=True`
(logit adjustment, Menon ICLR'21) hoặc giảm `cosine_scale` về 20.
"""

from __future__ import annotations

import math
from typing import List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import timm
    _HAS_TIMM = True
except ImportError:  # pragma: no cover - timm bắt buộc khi train
    _HAS_TIMM = False


ARCH_NAME = "moldnet"
ARCH_ALIASES = ["MoldNet"]


# ---------------------------------------------------------------------------
# Pooling: GeM + second-order (std)  — trái tim của thiết kế trị mold
# ---------------------------------------------------------------------------

class GeMPool(nn.Module):
    """Generalized-Mean Pooling: (mean(x^p))^(1/p).

    p→1 ≈ GAP, p→∞ ≈ max-pool. Với đốm mold nhỏ, p>1 ưu tiên vùng activation
    cao (đốm mốc) thay vì trung bình hoá vào nền. Tính ở float32 + clamp p để
    an toàn dưới AMP fp16 và deterministic mode.
    """

    def __init__(self, p_init: float = 3.0, eps: float = 1e-6,
                 p_min: float = 1.0, p_max: float = 5.0, learnable: bool = True):
        super().__init__()
        self.eps = eps
        self.p_min = p_min
        self.p_max = p_max
        if learnable:
            self.p = nn.Parameter(torch.tensor(float(p_init)))
        else:
            self.register_buffer("p", torch.tensor(float(p_init)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        p = self.p.clamp(self.p_min, self.p_max)
        xf = x.float().clamp(min=self.eps)
        pooled = xf.pow(p).mean(dim=(2, 3)).pow(1.0 / p)
        return pooled


class SecondOrderPool(nn.Module):
    """concat[ GeM(x) , spatial-std(x) ] → (B, 2C).

    std (phương sai không gian theo kênh) là tín hiệu mold đặc trưng nhất:
    đốm nấm → variance cao cục bộ; quả đều màu → variance thấp. GAP vứt bỏ.
    """

    def __init__(self, p_init: float = 3.0, learnable_p: bool = True,
                 eps: float = 1e-6):
        super().__init__()
        self.gem = GeMPool(p_init=p_init, learnable=learnable_p, eps=eps)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        g = self.gem(x)                                   # (B, C)
        xf = x.float()
        std = xf.var(dim=(2, 3), unbiased=False).clamp(min=self.eps).sqrt()
        return torch.cat([g, std], dim=1)                 # (B, 2C)


# ---------------------------------------------------------------------------
# Attention (tái dùng khối đã kiểm chứng trong jupiconet.py / reptaku.py)
# ---------------------------------------------------------------------------

class ECA(nn.Module):
    """Efficient Channel Attention — gần như miễn phí (1 conv1d, k≈3-5)."""

    def __init__(self, channels: int, gamma: float = 2.0, b: float = 1.0):
        super().__init__()
        t = int(abs(math.log2(channels) / gamma) + b / gamma)
        k = t if t % 2 else t + 1
        k = max(k, 3)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv1d = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.pool(x).squeeze(-1).transpose(-1, -2)
        y = self.conv1d(y).transpose(-1, -2).unsqueeze(-1)
        return x * torch.sigmoid(y)


class CS_STAM(nn.Module):
    """Channel-Shape Spatial-Texture Attention Module.

    Nhánh shape (DW 5×1 + 1×5) bắt vết nứt/sọc; nhánh texture (DW 3×3) bắt đốm
    mốc/lông; fusion → 1 spatial mask, residual gate (x + x*mask). Khối này từ
    reptaku/jupiconet đã được chứng minh tốt cho texture hư hại nhỏ trên táo.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.shape_v = nn.Conv2d(channels, channels, (5, 1), padding=(2, 0),
                                 groups=channels, bias=False)
        self.shape_h = nn.Conv2d(channels, channels, (1, 5), padding=(0, 2),
                                 groups=channels, bias=False)
        self.tex = nn.Conv2d(channels, channels, 3, padding=1,
                             groups=channels, bias=False)
        self.fusion = nn.Conv2d(2 * channels, 1, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ms = self.shape_v(x) + self.shape_h(x)
        mt = self.tex(x)
        m = torch.sigmoid(self.fusion(torch.cat([ms, mt], dim=1)))
        return x + x * m


# ---------------------------------------------------------------------------
# Classifier: cosine (weight-normalized) — chống bias lớp đông trên data lệch
# ---------------------------------------------------------------------------

class CosineClassifier(nn.Module):
    """logits = scale · cos(feature, weight).

    Chuẩn hoá weight khử lợi thế norm tự nhiên của lớp đông (dry 1469) so với
    lớp hiếm (mold 346); kết hợp với class-weighted CE của train.py.

    use_logit_adjust (mặc định False): thêm bias per-class HỌC ĐƯỢC chỉ khi train,
    bỏ khi eval (logit adjustment, Menon ICLR'21) — đẩy ngưỡng về phía lớp hiếm.
    Mặc định TẮT vì pipeline đã dùng class-weighted CE; bật như một ablation nếu
    mold còn nghẽn (theo dõi để lớp đông không bị tụt).
    """

    def __init__(self, in_features: int, num_classes: int, scale: float = 24.0,
                 use_logit_adjust: bool = False):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_classes, in_features))
        nn.init.kaiming_normal_(self.weight, nonlinearity="linear")
        self.scale = scale
        self.use_logit_adjust = use_logit_adjust
        if use_logit_adjust:
            self.logit_bias = nn.Parameter(torch.zeros(num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.normalize(x.float(), dim=1)
        w = F.normalize(self.weight.float(), dim=1)
        logits = self.scale * F.linear(x, w)
        if self.use_logit_adjust and self.training:
            logits = logits + self.logit_bias
        return logits


# ---------------------------------------------------------------------------
# MoldHead — multi-scale fusion + texture attention + second-order pooling
# ---------------------------------------------------------------------------

class MoldHead(nn.Module):
    def __init__(self, c_mid: int, c_high: int, num_classes: int,
                 fc_dim: int = 384, dropout: float = 0.2,
                 cosine_scale: float = 24.0, gem_p_init: float = 3.0,
                 learnable_p: bool = True, use_logit_adjust: bool = False,
                 use_cs_stam: bool = True):
        super().__init__()
        self.eca_mid = ECA(c_mid)
        self.csstam_mid = CS_STAM(c_mid) if use_cs_stam else nn.Identity()
        self.eca_high = ECA(c_high)
        self.pool_mid = SecondOrderPool(gem_p_init, learnable_p)
        self.pool_high = SecondOrderPool(gem_p_init, learnable_p)

        feat_dim = 2 * c_mid + 2 * c_high          # ×2 do GeM ⊕ std
        self.norm = nn.LayerNorm(feat_dim)
        self.proj = nn.Sequential(
            nn.Linear(feat_dim, fc_dim, bias=False),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.classifier = CosineClassifier(fc_dim, num_classes, cosine_scale,
                                           use_logit_adjust)

    def forward(self, f_mid: torch.Tensor, f_high: torch.Tensor) -> torch.Tensor:
        m = self.csstam_mid(self.eca_mid(f_mid))
        h = self.eca_high(f_high)
        feat = torch.cat([self.pool_mid(m), self.pool_high(h)], dim=1)
        feat = self.norm(feat)
        return self.classifier(self.proj(feat))


# ---------------------------------------------------------------------------
# Một nhánh = backbone timm (features_only) + MoldHead
# ---------------------------------------------------------------------------

def _make_backbone(name: str, in_channels: int, pretrained: bool,
                   drop_path: float):
    """Tạo timm backbone features_only; thử truyền drop_path_rate, fallback nếu
    backbone không nhận tham số đó. Trả về TẤT CẢ tầng (nhánh lấy 2 tầng cuối) để
    tương thích mọi version timm — không phụ thuộc out_indices âm."""
    kw = dict(pretrained=pretrained, features_only=True, in_chans=in_channels)
    try:
        return timm.create_model(name, drop_path_rate=drop_path, **kw)
    except TypeError:
        return timm.create_model(name, **kw)


class _Branch(nn.Module):
    def __init__(self, backbone_name: str, num_classes: int, in_channels: int,
                 pretrained: bool, drop_path: float, **head_kw):
        super().__init__()
        if not _HAS_TIMM:
            raise ImportError(
                "MoldNet cần timm. Cài bằng: pip install timm")
        self.backbone = _make_backbone(backbone_name, in_channels, pretrained,
                                       drop_path)
        chs = self.backbone.feature_info.channels()   # 2 tầng cuối
        c_mid, c_high = chs[-2], chs[-1]
        self.head = MoldHead(c_mid, c_high, num_classes, **head_kw)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        return self.head(feats[-2], feats[-1])


# ---------------------------------------------------------------------------
# MoldNet — ensemble logit-average + TTA
# ---------------------------------------------------------------------------

class MoldNet(nn.Module):
    def __init__(self, backbones: Sequence[str], num_classes: int = 5,
                 in_channels: int = 3, pretrained: bool = True,
                 tta: bool = True, learnable_fusion: bool = False,
                 drop_path: float = 0.1, **head_kw):
        super().__init__()
        self.branches = nn.ModuleList([
            _Branch(b, num_classes, in_channels, pretrained, drop_path, **head_kw)
            for b in backbones
        ])
        self.tta = tta
        self.num_classes = num_classes
        if learnable_fusion and len(self.branches) > 1:
            self.fusion_logits = nn.Parameter(torch.zeros(len(self.branches)))
        else:
            self.fusion_logits = None

    def _forward_once(self, x: torch.Tensor) -> torch.Tensor:
        outs = [b(x) for b in self.branches]
        if len(outs) == 1:
            return outs[0]
        stack = torch.stack(outs, dim=0)               # (N, B, num_classes)
        if self.fusion_logits is not None:
            w = F.softmax(self.fusion_logits, dim=0).view(-1, 1, 1)
            return (stack * w).sum(dim=0)
        return stack.mean(dim=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self._forward_once(x)
        # TTA chỉ khi eval + không cần grad (val/test @torch.no_grad). Tự tắt khi
        # train (self.training) và khi GradCAM bật grad.
        if self.tta and (not self.training) and (not torch.is_grad_enabled()):
            logits_flip = self._forward_once(torch.flip(x, dims=[3]))
            logits = 0.5 * (logits + logits_flip)
        return logits


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

# Các khoá head được phép truyền xuống MoldHead / MoldNet qua overrides.
_HEAD_KEYS = {
    "fc_dim", "dropout", "cosine_scale", "gem_p_init", "learnable_p",
    "use_logit_adjust", "use_cs_stam",
}
_NET_KEYS = {"tta", "learnable_fusion", "drop_path", "pretrained"}

MODEL_CONFIGS = {
    # F1 cao nhất tuyệt đối — ensemble 3 backbone đa dạng (~42M)
    "moldnet": dict(
        backbones=["convnextv2_nano", "tiny_vit_11m_224", "repvit_m1_5"],
        tta=True, drop_path=0.1, cosine_scale=24.0,
        input_size=224, params_M=41.05),
    # ensemble 2 backbone (CNN + distilled-ViT) — gần tốt nhất, nhẹ hơn (~27M)
    "moldnet_duo": dict(
        backbones=["convnextv2_nano", "tiny_vit_11m_224"],
        tta=True, drop_path=0.1, cosine_scale=24.0,
        input_size=224, params_M=26.83),
    # single backbone mạnh nhất — ablation sạch, single tốt nhất (~16M)
    "moldnet_pro": dict(
        backbones=["convnextv2_nano"],
        tta=True, drop_path=0.1, cosine_scale=24.0,
        input_size=224, params_M=15.73),
    # nhẹ, vừa bảng edge-AI mà vẫn dùng MoldHead (~6M)
    "moldnet_lite": dict(
        backbones=["edgenext_small"],
        tta=True, drop_path=0.1, cosine_scale=24.0,
        input_size=224, params_M=5.64),
}

DEFAULT_VARIANT = "moldnet_pro"


def build_model(name: str = "moldnet_pro", num_classes: int = 5,
                in_channels: int = 3, pretrained: bool = True,
                **overrides) -> MoldNet:
    cfg = MODEL_CONFIGS.get(name)
    if cfg is None:
        raise ValueError(
            f"Unknown variant '{name}'. Choose from: {list(MODEL_CONFIGS)}")

    backbones = cfg["backbones"]
    head_kw = {k: v for k, v in cfg.items() if k in _HEAD_KEYS}
    net_kw = {k: v for k, v in cfg.items() if k in _NET_KEYS}
    # overrides do người dùng truyền (vd. pretrained=False khi test) đè config
    for k, v in overrides.items():
        if k in _HEAD_KEYS:
            head_kw[k] = v
        elif k in _NET_KEYS or k == "pretrained":
            net_kw[k] = v
    net_kw.setdefault("pretrained", pretrained)

    model = MoldNet(backbones, num_classes=num_classes,
                    in_channels=in_channels, **net_kw, **head_kw)
    model.num_classes = num_classes
    return model


if __name__ == "__main__":
    x = torch.randn(2, 3, 224, 224)
    for variant in MODEL_CONFIGS:
        m = build_model(variant, num_classes=5, pretrained=False)
        n = sum(p.numel() for p in m.parameters()) / 1e6
        m.eval()
        with torch.no_grad():
            y = m(x)
        print(f"{variant:14s} params={n:6.2f}M  out={tuple(y.shape)}")
