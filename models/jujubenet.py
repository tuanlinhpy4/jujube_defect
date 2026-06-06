import math
import torch
import torch.nn as nn


ARCH_NAME = "jujubenet"

MODEL_CONFIGS = {
    "jujubenet": {"input_size": 224, "params_M": 8.50},
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def drop_path(x: torch.Tensor, drop_prob: float = 0.0, training: bool = False) -> torch.Tensor:
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor = torch.floor(random_tensor + keep_prob)
    return x / keep_prob * random_tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


# ---------------------------------------------------------------------------
# CBAM – Convolutional Block Attention Module (Woo et al., 2018)
# ---------------------------------------------------------------------------

class ChannelAttention(nn.Module):
    def __init__(self, in_channels: int, reduction_ratio: int = 16):
        super().__init__()
        mid = max(in_channels // reduction_ratio, 1)
        self.shared_mlp = nn.Sequential(
            nn.Linear(in_channels, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, in_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.shared_mlp(x.mean(dim=[2, 3]))
        max_out = self.shared_mlp(x.amax(dim=[2, 3]))
        scale = torch.sigmoid(avg_out + max_out).unsqueeze(-1).unsqueeze(-1)
        return x * scale


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = x.mean(dim=1, keepdim=True)
        max_out = x.amax(dim=1, keepdim=True)
        scale = torch.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))
        return x * scale


class CBAM(nn.Module):
    def __init__(self, in_channels: int, reduction_ratio: int = 16, spatial_kernel: int = 7):
        super().__init__()
        self.channel_att = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_att = SpatialAttention(spatial_kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_att(x)
        x = self.spatial_att(x)
        return x


# ---------------------------------------------------------------------------
# MDC Block – Multi-branching Depthwise separable Convolution block
# ---------------------------------------------------------------------------

class MDCBlock(nn.Module):
    def __init__(self, dim: int, drop_path_rate: float = 0.0, kernel_size: int = 7):
        super().__init__()
        half = dim // 2
        pad = kernel_size // 2

        self.stem_dw = nn.Conv2d(dim, dim, kernel_size, padding=pad, groups=dim, bias=False)
        self.stem_norm = nn.LayerNorm(dim, eps=1e-6)

        self.branch_a = nn.Sequential(
            nn.GELU(),
            nn.Conv2d(half, half, 3, padding=1, groups=half, bias=False),
            nn.BatchNorm2d(half),
            nn.GELU(),
            nn.Conv2d(half, half, 3, padding=1, groups=half, bias=False),
        )

        self.branch_b = nn.Sequential(
            nn.Conv2d(half, half, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(half, half, 1, bias=False),
        )

        self.merge_norm = nn.BatchNorm2d(dim)
        self.merge_act = nn.GELU()
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x

        out = self.stem_dw(x)
        out = out.permute(0, 2, 3, 1)
        out = self.stem_norm(out)
        out = out.permute(0, 3, 1, 2)

        a, b = out.chunk(2, dim=1)
        a = self.branch_a(a)
        b = self.branch_b(b)

        out = torch.cat([a, b], dim=1)
        out = self.merge_norm(out)
        out = self.merge_act(out)
        out = self.drop_path(out)
        return out + shortcut


# ---------------------------------------------------------------------------
# Stem & DownSample (from ConvNeXt-Tiny)
# ---------------------------------------------------------------------------

class Stem(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 96):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=4)
        self.norm = nn.LayerNorm(out_channels, eps=1e-6)

    def forward(self, x):
        x = self.conv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)
        return x


class DownSample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.norm = nn.LayerNorm(in_channels, eps=1e-6)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)
        x = self.conv(x)
        return x


# ---------------------------------------------------------------------------
# JujubeNet
# ---------------------------------------------------------------------------

class JujubeNet(nn.Module):
    def __init__(
        self,
        num_classes: int = 6,
        in_channels: int = 3,
        depths: list = None,
        dims: list = None,
        drop_path_rate: float = 0.1,
    ):
        super().__init__()

        if depths is None:
            depths = [3, 3, 9, 3]
        if dims is None:
            dims = [96, 192, 384, 768]

        total_blocks = sum(depths)
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, total_blocks)]

        # Stage 1
        self.stem = Stem(in_channels, dims[0])
        self.cbam0 = CBAM(dims[0])
        block_idx = 0
        self.stage1 = nn.Sequential(*[
            MDCBlock(dims[0], drop_path_rate=dp_rates[block_idx + i])
            for i in range(depths[0])
        ])
        block_idx += depths[0]

        # Stage 2
        self.down1 = DownSample(dims[0], dims[1])
        self.cbam1 = CBAM(dims[1])
        self.stage2 = nn.Sequential(*[
            MDCBlock(dims[1], drop_path_rate=dp_rates[block_idx + i])
            for i in range(depths[1])
        ])
        block_idx += depths[1]

        # Stage 3
        self.down2 = DownSample(dims[1], dims[2])
        self.cbam2 = CBAM(dims[2])
        self.stage3 = nn.Sequential(*[
            MDCBlock(dims[2], drop_path_rate=dp_rates[block_idx + i])
            for i in range(depths[2])
        ])
        block_idx += depths[2]

        # Stage 4
        self.down3 = DownSample(dims[2], dims[3])
        self.cbam3 = CBAM(dims[3])
        self.stage4 = nn.Sequential(*[
            MDCBlock(dims[3], drop_path_rate=dp_rates[block_idx + i])
            for i in range(depths[3])
        ])

        # Classifier head
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.LayerNorm(dims[3], eps=1e-6),
            nn.Linear(dims[3], num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.cbam0(x)
        x = self.stage1(x)

        x = self.down1(x)
        x = self.cbam1(x)
        x = self.stage2(x)

        x = self.down2(x)
        x = self.cbam2(x)
        x = self.stage3(x)

        x = self.down3(x)
        x = self.cbam3(x)
        x = self.stage4(x)

        x = self.classifier(x)
        return x


# ---------------------------------------------------------------------------
# build_model
# ---------------------------------------------------------------------------

def build_model(name="jujubenet", num_classes=6, in_channels=3, **kwargs):
    model = JujubeNet(num_classes=num_classes, in_channels=in_channels, **kwargs)
    model.num_classes = num_classes
    return model
