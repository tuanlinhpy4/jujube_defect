import torch
import torch.nn as nn
import math

# ===========================================================================
# Model Registry Information
# ===========================================================================
ARCH_NAME = "se_ghost_resnet"
MODEL_CONFIGS = {"se_ghost_resnet": {"input_size": 224, "params_M": None}}

# ===========================================================================
# Components
# ===========================================================================
class GhostModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, ratio=2, dw_size=3):
        super().__init__()
        init_channels = math.ceil(out_channels / ratio)
        new_channels = init_channels * (ratio - 1)
        self.primary_conv = nn.Conv2d(in_channels, init_channels, kernel_size, padding=kernel_size//2, bias=False)
        self.cheap_operation = nn.Conv2d(init_channels, new_channels, dw_size, padding=dw_size//2, groups=init_channels, bias=False)
        self.out_channels = out_channels

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        return out[:, :self.out_channels, :, :]

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class GhostSEBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.ghost1 = GhostModule(in_channels, out_channels)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.ghost2 = GhostModule(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.se = SEBlock(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.ghost1(x)))
        out = self.bn2(self.ghost2(out))
        out = self.se(out)
        out += self.shortcut(x)
        return self.relu(out)

# ===========================================================================
# Main Architecture
# ===========================================================================
class SEGhostResNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=4):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        self.stage1 = GhostSEBlock(32, 64, stride=2)
        self.stage2 = GhostSEBlock(64, 128, stride=2)
        self.stage3 = GhostSEBlock(128, 256, stride=2)
        
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(256, num_classes)
        
    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.pool(x).flatten(1)
        return self.classifier(x)

# ===========================================================================
# Registration Build Function
# ===========================================================================
def build_model(name="se_ghost_resnet", num_classes=4, in_channels=3, **kwargs):
    return SEGhostResNet(in_channels=in_channels, num_classes=num_classes)
