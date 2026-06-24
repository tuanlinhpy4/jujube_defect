"""JedgeNet — model triển khai chính thức cho STM32H750VBT6 (jujube 4 lớp, 64x64, INT8).

JedgeNet là bản "chốt" rút ra từ nghiên cứu ablation ``junet_edge_lab`` (biến thể
``jel_reppw``). Nó = kiến trúc ``junet_edge_l`` (model nhanh nhất + nhỏ nhất đo được
trên H750) **cộng tái-tham-số-hoá pointwise (RepPW)** ở nhánh 1x1-project của hai
stage độ-phân-giải-cao.

Vì sao RepPW là "F1 miễn phí về latency":
  * Lúc TRAIN, mỗi 1x1-project là tổng hai nhánh 1x1 song song -> nhiều capacity hơn.
  * Khi gọi ``fuse_for_deploy()``, hai nhánh gập tuyến tính về **một** conv 1x1 duy
    nhất. Đồ thị INT8 sau fuse *giống hệt cấu trúc* ``junet_edge_l``: cùng op, cùng
    kernel, cùng shape, cùng số byte weight — chỉ khác giá trị weight. Trên MCU này
    latency bị chi phối bởi số byte weight đọc từ QSPI + op int8-clean, KHÔNG bởi MACs;
    nên JedgeNet đo được **iso-latency với junet_edge_l** mà macro-F1 cao hơn.

Đo trên thiết bị (STM32H750 @400MHz, 519 ảnh, DWT, INT8 CMSIS-NN, 5 seed):
  * latency 286.0 ms (junet_edge_l 287.6 ms — ngang bằng), 437.7 KB tflite, ~0.32 M.
  * macro-F1 0.9523 ± 0.005 (junet_edge_l 0.9468) — Pareto-dominant.

Nguyên tắc thiết kế kế thừa junet_edge (mọi lựa chọn đều int8-clean):
  1. KeepHi strides (2,2,1) — feature map cuối giữ 8x8 (đòn bẩy int8 mạnh nhất).
  2. RepVGG-style reparam (depthwise RepConv 2 stage đầu + RepPW 1x1-project) — capacity
     lúc train, gập biến mất lúc deploy.
  3. Ghost inverted residual + một SE muộn ở stage sâu.
  4. ReLU toàn bộ (không h-swish/SiLU) -> đồ thị int8 trung thực.
  5. Không multi-kernel/dilated -> mọi depthwise là 3x3, full int8 (không rơi reference kernel).

Cây module + tên thuộc tính giữ trùng khít với ``junet_edge_lab`` biến thể ``jel_reppw``,
nên checkpoint ``.pth`` đã train của jel_reppw load thẳng vào JedgeNet (state_dict khớp key).

Tự chứa hoàn toàn (như ``models/junet_edge.py``): không import từ file anh em, để registry
auto-scan nhận diện trên mọi checkout/clone. Gọi ``model.fuse_for_deploy()`` ở eval()
trước khi export ONNX INT8 (``pth_to_int8_onnx.py`` đã tự gọi nếu có).
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, replace
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


ARCH_NAME = "JedgeNet"
ARCH_ALIASES = ["jedgenet", "jedge"]


# ---------------------------------------------------------------------------
# Primitives (tự chứa — không import từ file anh em)
# ---------------------------------------------------------------------------
def _round_ch(channels: float, divisor: int = 8) -> int:
    """Làm tròn số channel về bội của 8 (tốt cho kernel int8 SIMD)."""
    rounded = max(divisor, int(channels + divisor / 2) // divisor * divisor)
    if rounded < 0.9 * channels:
        rounded += divisor
    return rounded


def act_layer(name: str) -> nn.Module:
    # Recipe ship là ReLU-only; relu6 giữ làm phương án int8-safe.
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name == "relu6":
        return nn.ReLU6(inplace=True)
    raise ValueError(f"Activation không int8-clean cho JedgeNet: {name}")


class DropPath(nn.Module):
    """Stochastic depth theo từng sample cho nhánh residual."""

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
    """Ghost convolution: feature chính 1x1 + feature 'rẻ' depthwise."""

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
    """Channel attention mức stage (int8-clean: pool + 1x1 + ReLU + sigmoid gate)."""

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


# ---------------------------------------------------------------------------
# RepConv — conv tái-tham-số-hoá (depthwise 3x3 hoặc pointwise 1x1)
# ---------------------------------------------------------------------------
class RepConv(nn.Module):
    """RepVGG-style: huấn luyện đa-nhánh, deploy gập về MỘT conv duy nhất.

    Hai chế độ:
      * mode='dw' (depthwise 3x3, groups=channels):
          nhánh {3x3, 1x1, identity nếu stride==1} -> 1 conv 3x3.
      * mode='pw' (pointwise 1x1, groups=1, in->out, KHÔNG activation):
          nhánh {1x1 A, 1x1 B} -> 1 conv 1x1. KHÔNG có identity (project đổi channel;
          identity của groups=1 là ma trận eye(C) chứ không phải unit-weight depthwise).

    Sau ``reparameterize()`` đồ thị deploy *giống hệt cấu trúc* một conv thường: cùng
    op / kernel / shape — chỉ khác giá trị weight (BN từng nhánh được nướng vào). Đó là
    cơ sở của tuyên bố "iso-latency vs junet_edge_l".
    """

    def __init__(self, in_ch: int, out_ch: int, mode: str = "dw", stride: int = 1,
                 act: str = "relu", use_act: bool = True):
        super().__init__()
        assert mode in ("dw", "pw")
        if mode == "dw":
            assert in_ch == out_ch, "depthwise yêu cầu in_ch == out_ch"
            kernel, groups = 3, in_ch
        else:  # pw
            kernel, groups = 1, 1
        self.mode = mode
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.kernel = kernel
        self.stride = stride
        self.groups = groups
        self.use_act = use_act
        self.act = act_layer(act)
        self.fused = False
        self.reparam: nn.Conv2d | None = None

        def cbn(k: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, k, stride, (k - 1) // 2, groups=groups, bias=False),
                nn.BatchNorm2d(out_ch),
            )

        # Nhánh chính (3x3 với dw, 1x1 'A' với pw) + nhánh nhỏ (1x1 với dw, 1x1 'B' với pw).
        self.main = cbn(kernel)
        self.small = cbn(1)
        # Nhánh identity (chỉ depthwise, stride==1, in==out).
        self.bn_id = nn.BatchNorm2d(out_ch) if (mode == "dw" and stride == 1 and in_ch == out_ch) else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.fused:
            y = self.reparam(x)
            return self.act(y) if self.use_act else y
        out = self.main(x) + self.small(x)
        if self.bn_id is not None:
            out = out + self.bn_id(x)
        return self.act(out) if self.use_act else out

    @staticmethod
    def _fuse_conv_bn(weight: torch.Tensor, bn: nn.BatchNorm2d) -> Tuple[torch.Tensor, torch.Tensor]:
        std = torch.sqrt(bn.running_var + bn.eps)
        t = (bn.weight / std).reshape(-1, 1, 1, 1)
        kernel = weight * t
        bias = bn.bias - bn.running_mean * bn.weight / std
        return kernel, bias

    @torch.no_grad()
    def reparameterize(self) -> None:
        """Gập mọi nhánh về MỘT conv (3x3 với dw, 1x1 với pw) + bias. Idempotent."""
        if self.fused:
            return
        K = self.kernel

        def pad_to_K(k: torch.Tensor) -> torch.Tensor:
            p = (K - k.shape[-1]) // 2
            return k if p == 0 else F.pad(k, [p, p, p, p])

        # main (3x3 / 1x1 A)
        weight_acc, bias_acc = self._fuse_conv_bn(self.main[0].weight, self.main[1])
        weight_acc = pad_to_K(weight_acc)
        # small (1x1 / 1x1 B)
        ks, bs = self._fuse_conv_bn(self.small[0].weight, self.small[1])
        weight_acc = weight_acc + pad_to_K(ks)
        bias_acc = bias_acc + bs
        # identity (depthwise unit-weight)
        if self.bn_id is not None:
            id_w = torch.ones(self.out_ch, 1, 1, 1, device=weight_acc.device, dtype=weight_acc.dtype)
            kid, bid = self._fuse_conv_bn(id_w, self.bn_id)
            weight_acc = weight_acc + pad_to_K(kid)
            bias_acc = bias_acc + bid

        conv = nn.Conv2d(self.in_ch, self.out_ch, K, self.stride, (K - 1) // 2,
                         groups=self.groups, bias=True)
        conv.weight.data.copy_(weight_acc)
        conv.bias.data.copy_(bias_acc)
        self.reparam = conv
        for name in ("main", "small", "bn_id"):
            if getattr(self, name, None) is not None:
                delattr(self, name)
            elif name in self._modules:
                del self._modules[name]
        self.fused = True


# ---------------------------------------------------------------------------
# Blocks
# ---------------------------------------------------------------------------
class GhostIRBlock(nn.Module):
    """Inverted residual: Ghost expand -> depthwise 3x3 thường -> 1x1 project."""

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
    """Inverted residual: Ghost expand -> RepConv depthwise 3x3 -> project
    (1x1 thường hoặc RepPW)."""

    def __init__(self, in_ch: int, out_ch: int, stride: int, expand: float,
                 act: str = "relu", drop_path: float = 0.0, reppw: bool = False):
        super().__init__()
        hidden = _round_ch(in_ch * expand)
        self.use_res = stride == 1 and in_ch == out_ch
        self.expand = GhostConv(in_ch, hidden, act=act)
        self.dw = RepConv(hidden, hidden, mode="dw", stride=stride, act=act, use_act=True)
        if reppw:
            self.project = RepConv(hidden, out_ch, mode="pw", stride=1, act=act, use_act=False)
        else:
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
    attn: str = "none"          # none | se  (mức stage, sau các block)
    reppw: bool = False         # chỉ 'rep': bọc 1x1 project bằng RepPW


def make_stages(
    blocks: Tuple[str, str, str] = ("rep", "rep", "ghost_ir"),
    channels: Tuple[int, int, int] = (56, 88, 128),
    expand: Tuple[float, float, float] = (4.0, 3.0, 3.0),
    attn: Tuple[str, str, str] = ("none", "none", "se"),
    depths: Tuple[int, int, int] = (1, 2, 3),
    strides: Tuple[int, int, int] = (2, 2, 1),   # keephi: final map giữ 8x8
    reppw: Tuple[bool, bool, bool] = (True, True, False),
) -> List[StageSpec]:
    return [
        StageSpec(channels[i], depths[i], strides[i], blocks[i], expand[i], attn[i], reppw[i])
        for i in range(3)
    ]


def _attention(kind: str, channels: int, se_reduction: int, act: str) -> nn.Module:
    if kind == "se":
        return SqueezeExcitation(channels, reduction=se_reduction, act=act)
    if kind == "none":
        return nn.Identity()
    raise ValueError(f"Attention không hỗ trợ: {kind}")


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class JedgeNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 4,
        stem_ch: int = 32,
        stages: Sequence[StageSpec] | None = None,
        head_ch: int = 192,
        act: str = "relu",
        se_reduction: int = 4,
        dropout: float = 0.1,
        drop_path: float = 0.12,
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

        # Lịch stochastic-depth tuyến tính trên các block residual.
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
                kwargs: Dict[str, Any] = dict(act=act, drop_path=dp)
                if s.block == "rep":
                    kwargs["reppw"] = s.reppw
                layers.append(_BLOCKS[s.block](prev, s.out_ch, stride, s.expand, **kwargs))
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
    def fuse_for_deploy(self) -> "JedgeNet":
        """Tái-tham-số-hoá mọi RepConv về 1 conv. Gọi ở eval() trước export INT8 ONNX.
        Idempotent."""
        was_training = self.training
        self.eval()
        for m in self.modules():
            if isinstance(m, RepConv):
                m.reparameterize()
        if was_training:
            self.train()
        return self

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        total = self.count_parameters()
        return f"JedgeNet(params={total:,} ({total / 1e6:.4f}M))"


# ---------------------------------------------------------------------------
# Variant config (một biến thể chính thức duy nhất ≙ jel_reppw)
# ---------------------------------------------------------------------------
def _cfg(stem_ch: int = 32, head_ch: int = 192, act: str = "relu", dropout: float = 0.1,
         drop_path: float = 0.12, width_mult: float = 1.0,
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
        "params_M": 0.32,
    }
    cfg.update(meta)
    return cfg


# Cấu hình = junet_edge_l (stem 32, head 192, channels 56/88/128, depths 1/2/3,
# blocks rep/rep/ghost, expand 4/3/3, SE muộn, keephi strides 2/2/1, drop_path 0.12,
# dropout 0.1) + RepPW ở 1x1-project của hai stage rep đầu.
MODEL_CONFIGS: Dict[str, Dict[str, Any]] = {
    "jedgenet": _cfg(stages=make_stages(reppw=(True, True, False))),
}


DEFAULT_VARIANT = "jedgenet"


_NOTES = {
    "jedgenet": ("Model triển khai chính thức: junet_edge_l + RepPW (gập về đồ thị "
                 "INT8 trùng junet_edge_l). Iso-latency 286 ms, macro-F1 0.9523 (5-seed, H750)."),
}


def build_model(name: str = DEFAULT_VARIANT, num_classes: int = 4, in_channels: int = 3,
                **overrides) -> JedgeNet:
    if name not in MODEL_CONFIGS:
        raise ValueError(f"Biến thể JedgeNet không rõ: {name}. Chọn: {sorted(MODEL_CONFIGS)}")
    cfg = copy.deepcopy(MODEL_CONFIGS[name])
    cfg.update(overrides)
    cfg = {k: v for k, v in cfg.items() if k not in ("input_size", "params_M")}
    return JedgeNet(in_channels=in_channels, num_classes=num_classes, **cfg)


def list_variants() -> List[str]:
    return sorted(MODEL_CONFIGS)


def jedgenet(num_classes: int = 4, in_channels: int = 3) -> JedgeNet:
    return build_model(DEFAULT_VARIANT, num_classes=num_classes, in_channels=in_channels)


# ---------------------------------------------------------------------------
# Self-test: sizing + int8-cleanliness + fuse round-trip
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    RAM = 1024 * 1024
    INT_FLASH = 128 * 1024
    x = torch.zeros(1, 3, 64, 64)

    print(f"{'variant':10s} {'train':>9s} {'deploy':>9s} {'peakAct~KB':>11s}  flags")
    for variant in MODEL_CONFIGS:
        model = build_model(variant, num_classes=4).eval()
        train_params = model.count_parameters()

        peak = {"v": 0}

        def _hook(_m, inp, out, _p=peak):
            if inp and torch.is_tensor(inp[0]) and torch.is_tensor(out):
                _p["v"] = max(_p["v"], inp[0].numel() + out.numel())

        handles = [m.register_forward_hook(_hook) for m in model.modules()]
        with torch.no_grad():
            y_ref = model(x)
        for h in handles:
            h.remove()

        # fuse round-trip ở EVAL (DropPath là no-op cả 2 phía); idempotent check.
        model.fuse_for_deploy()
        model.fuse_for_deploy()
        with torch.no_grad():
            y_fused = model(x)
        max_err = (y_ref - y_fused).abs().max().item()
        deploy_params = model.count_parameters()

        # int8-clean audit.
        dil = sum(1 for m in model.modules()
                  if isinstance(m, nn.Conv2d) and tuple(m.dilation) != (1, 1))
        bad_act = sum(1 for m in model.modules()
                      if m.__class__.__name__ in {"SiLU", "Hardswish", "GELU", "Mish"})
        unfused = sum(1 for m in model.modules()
                      if isinstance(m, RepConv) and not m.fused)
        misaligned = sum(1 for m in model.modules()
                         if isinstance(m, nn.Conv2d) and (m.out_channels % 8 != 0))

        flags = []
        if deploy_params > INT_FLASH:
            flags.append("ext-flash")
        if peak["v"] > RAM:
            flags.append("OVER-RAM!")
        if dil:
            flags.append(f"DILATED:{dil}")
        if bad_act:
            flags.append(f"NON-INT8-ACT:{bad_act}")
        if unfused:
            flags.append(f"UNFUSED:{unfused}")
        if misaligned:
            flags.append(f"CH%8:{misaligned}")
        if max_err > 1e-3:
            flags.append(f"FUSE-MISMATCH:{max_err:.2e}")

        print(f"{variant:10s} {train_params:>9,} {deploy_params:>9,} "
              f"{peak['v']/1024:>10.1f}K  fuse_err={max_err:.1e}  {' '.join(flags) or 'OK'}")
