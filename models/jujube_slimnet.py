import math
import torch
import torch.nn as nn


ARCH_NAME = "jujube_slimnet"

MODEL_CONFIGS = {
    "jujube_slimnet": {"input_size": 224, "params_M": 0.80},
}


# ---------------------------------------------------------------------------
# PConv – Partial Convolution
# Applies a regular conv only on a fraction (1/n_div) of the channels and
# leaves the rest untouched, then concatenates back. This keeps spatial
# feature extraction cheap in terms of both FLOPs and parameters.
# ---------------------------------------------------------------------------

class PConv(nn.Module):
    def __init__(self, dim, n_div=4, kernel_size=3):
        super().__init__()
        self.dim_conv = dim // n_div
        self.dim_untouched = dim - self.dim_conv
        self.conv = nn.Conv2d(
            self.dim_conv, self.dim_conv, kernel_size,
            padding=kernel_size // 2, bias=False,
        )

    def forward(self, x):
        x1, x2 = torch.split(x, [self.dim_conv, self.dim_untouched], dim=1)
        x1 = self.conv(x1)
        return torch.cat([x1, x2], dim=1)


# ---------------------------------------------------------------------------
# ECA – Efficient Channel Attention (ECA-Net)
# Channel attention via a single 1D conv over the channel-pooled descriptor,
# with no dimensionality reduction — only a handful of parameters.
# ---------------------------------------------------------------------------

class ECALayer(nn.Module):
    def __init__(self, channels, gamma=2, b=1):
        super().__init__()
        t = int(abs((math.log2(channels) + b) / gamma))
        k = t if t % 2 else t + 1
        k = max(k, 3)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)                       # (B, C, 1, 1)
        y = y.squeeze(-1).transpose(-1, -2)         # (B, 1, C)
        y = self.conv(y)                            # (B, 1, C)
        y = y.transpose(-1, -2).unsqueeze(-1)       # (B, C, 1, 1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)


# ---------------------------------------------------------------------------
# PConv-ECA Block
# PConv (cheap spatial mixing) -> pointwise expand -> BN -> Act ->
# pointwise project -> BN -> ECA (channel attention) -> residual add.
# ---------------------------------------------------------------------------

class PConvECABlock(nn.Module):
    def __init__(self, dim, expand_ratio=2, n_div=4):
        super().__init__()
        hidden_dim = dim * expand_ratio

        self.pconv = PConv(dim, n_div=n_div)
        self.pw1 = nn.Conv2d(dim, hidden_dim, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden_dim)
        self.act = nn.ReLU(inplace=True)
        self.pw2 = nn.Conv2d(hidden_dim, dim, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(dim)
        self.eca = ECALayer(dim)

    def forward(self, x):
        shortcut = x
        out = self.pconv(x)
        out = self.pw1(out)
        out = self.bn1(out)
        out = self.act(out)
        out = self.pw2(out)
        out = self.bn2(out)
        out = self.eca(out)
        return out + shortcut


# ---------------------------------------------------------------------------
# Stem – lightweight Conv3x3 stride=2 to shrink the input quickly (latency)
# ---------------------------------------------------------------------------

class Stem(nn.Module):
    def __init__(self, in_channels=3, out_channels=16):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


# ---------------------------------------------------------------------------
# DownSample – Conv3x3 stride=2 between stages
# ---------------------------------------------------------------------------

class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


# ---------------------------------------------------------------------------
# Jujube-SlimNet
#   Stem (Conv3x3, s=2)
#   -> 4x [ DownSample (s=2) -> PConv-ECA Block ]
#   -> Head: GAP + GMP concat -> FC
# ~0.80 M parameters, no structural reparameterization.
# ---------------------------------------------------------------------------

class JujubeSlimNet(nn.Module):
    def __init__(
        self,
        num_classes=6,
        in_channels=3,
        channels=None,
        expand_ratio=2,
        n_div=4,
    ):
        super().__init__()

        if channels is None:
            channels = [16, 32, 64, 128, 256]

        self.stem = Stem(in_channels, channels[0])

        stages = []
        c_in = channels[0]
        for c_out in channels[1:]:
            stages.append(DownSample(c_in, c_out))
            stages.append(PConvECABlock(c_out, expand_ratio=expand_ratio, n_div=n_div))
            c_in = c_out
        self.body = nn.Sequential(*stages)

        self.gap = nn.AdaptiveAvgPool2d(1)
        # NOTE: nn.AdaptiveMaxPool2d's CUDA backward has no deterministic
        # implementation, which raises a RuntimeError under
        # torch.use_deterministic_algorithms(True). We compute the same
        # Global Max Pooling manually via tensor.amax, which does have a
        # deterministic backward, instead of changing the pooling itself.
        self.fc = nn.Linear(c_in * 2, num_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.stem(x)
        x = self.body(x)
        avg = self.gap(x).flatten(1)
        mx = x.amax(dim=[2, 3])
        x = torch.cat([avg, mx], dim=1)
        return self.fc(x)


# ---------------------------------------------------------------------------
# build_model
# ---------------------------------------------------------------------------

def build_model(name="jujube_slimnet", num_classes=6, in_channels=3, **kwargs):
    model = JujubeSlimNet(num_classes=num_classes, in_channels=in_channels, **kwargs)
    model.num_classes = num_classes
    return model
