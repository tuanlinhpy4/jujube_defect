import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ===========================================================================
# Model Registry Information
# ===========================================================================
ARCH_NAME = "se_ghost_resnet_bct"
MODEL_CONFIGS = {"se_ghost_resnet_bct": {"input_size": 224, "params_M": None}}

# ===========================================================================
# GhostNet & SE Components
# ===========================================================================
class GhostModule(nn.Module):
    """
    Ghost Module thay thế cho Conv2d truyền thống.
    ratio = 2 nghĩa là 1/2 feature map từ Conv thường, 1/2 từ Depthwise Conv.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, ratio=2, dw_size=3, relu=True):
        super().__init__()
        self.out_channels = out_channels
        init_channels = math.ceil(out_channels / ratio)
        new_channels = init_channels * (ratio - 1)

        # Primary Convolution
        self.primary_conv = nn.Sequential(
            nn.Conv2d(in_channels, init_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Identity()
        )

        # Cheap Operation (Depthwise Convolution)
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

class SEBlock(nn.Module):
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

class GhostResNetBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        # Thay thế Conv2d + BN + ReLU bằng GhostModule (relu=True)
        self.ghost1 = GhostModule(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, relu=True)
        
        # Thay thế Conv2d + BN bằng GhostModule (relu=False vì ta cộng shortcut trước khi qua ReLU)
        self.ghost2 = GhostModule(out_channels, out_channels, kernel_size=3, stride=1, padding=1, relu=False)
        
        self.se = SEBlock(out_channels)
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
# Network Architecture
# ===========================================================================
class SEGhostResNet(nn.Module):
    def __init__(self, in_channels: int = 3, num_classes: int = 4):
        super().__init__()
        # Stem để giảm độ phân giải ban đầu (Giữ nguyên hoặc có thể thay bằng Ghost, nhưng Stem thường dùng Conv chuẩn)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        # Stages với GhostResNetBlock
        self.stage1 = GhostResNetBlock(32, 64, stride=2)
        self.stage2 = GhostResNetBlock(64, 128, stride=2)
        self.stage3 = GhostResNetBlock(128, 256, stride=2)
        
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
# Registration Build Function
# ===========================================================================
def build_model(name="se_ghost_resnet_bct", num_classes=4, in_channels=3, **kwargs):
    model = SEGhostResNet(in_channels=in_channels, num_classes=num_classes)
    model.num_classes = num_classes
    return model
