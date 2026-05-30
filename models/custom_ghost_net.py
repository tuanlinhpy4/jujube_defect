import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ===========================================================================
# Model Registry Information (Đã đổi tên)
# ===========================================================================
ARCH_NAME = "custom_ghost_net"
MODEL_CONFIGS = {"custom_ghost_net": {"input_size": 224, "params_M": None}}

# ===========================================================================
# Components (Đã đổi tên class & Sửa lỗi tham số padding)
# ===========================================================================
class CustomGhostConv(nn.Module):
    """
    Đã thêm tham số padding=1 vào __init__ để fix lỗi TypeError của bạn.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, ratio=2, dw_size=3, relu=True):
        super().__init__()
        self.out_channels = out_channels
        init_channels = math.ceil(out_channels / ratio)
        new_channels = init_channels * (ratio - 1)

        self.primary_conv = nn.Sequential(
            # Truyền padding xuống Conv2d
            nn.Conv2d(in_channels, init_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Identity()
        )

        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size // 2, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True) if relu else nn.Identity()
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        return out[:, :self.out_channels, :, :]


class CustomSEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class CustomGhostBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.ghost1 = CustomGhostConv(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, relu=True)
        self.ghost2 = CustomGhostConv(out_channels, out_channels, kernel_size=3, stride=1, padding=1, relu=False)
        self.se = CustomSEBlock(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.ghost1(x)
        out = self.ghost2(out)
        out = self.se(out)
        out += self.shortcut(x)
        return self.relu(out)

# ===========================================================================
# Network Architecture (Đã đổi tên)
# ===========================================================================
class CustomGhostNet(nn.Module):
    def __init__(self, in_channels: int = 3, num_classes: int = 4):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        self.stage1 = CustomGhostBlock(32, 64, stride=2)
        self.stage2 = CustomGhostBlock(64, 128, stride=2)
        self.stage3 = CustomGhostBlock(128, 256, stride=2)
        
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(256, num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.pool(x).flatten(1)
        return self.classifier(x)

# ===========================================================================
# Registration Build Function (Đã đổi tên)
# ===========================================================================
def build_model(name="custom_ghost_net", num_classes=4, in_channels=3, **kwargs):
    model = CustomGhostNet(in_channels=in_channels, num_classes=num_classes)
    model.num_classes = num_classes
    return model
