import torch
import torch.nn as nn
import math

# ===========================================================================
# Model Registry Information
# ===========================================================================
ARCH_NAME = "se_ghost_resnet"
MODEL_CONFIGS = {
    "se_ghost_resnet": {
        "input_size": 224, 
        "description": "GhostNet-integrated SE-ResNet for efficient feature extraction"
    }
}

# ===========================================================================
# Ghost Module: Tạo các feature maps "ảo" bằng phép biến đổi rẻ tiền
# ===========================================================================
class GhostModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, ratio=2, dw_size=3, stride=1):
        super().__init__()
        self.out_channels = out_channels
        init_channels = math.ceil(out_channels / ratio)
        new_channels = init_channels * (ratio - 1)

        self.primary_conv = nn.Sequential(
            nn.Conv2d(in_channels, init_channels, kernel_size, stride, kernel_size // 2, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True)
        )

        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size // 2, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        return out[:, :self.out_channels, :, :]

# ===========================================================================
# SE Block & Ghost ResNet Block
# ===========================================================================
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
        self.ghost1 = GhostModule(in_channels, out_channels, stride=stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.ghost2 = GhostModule(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.se = SEBlock(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.bn1(self.ghost1(x)))
        out = self.bn2(self.ghost2(out))
        out = self.se(out)
        out += self.shortcut(x)
        return self.relu(out)

# ===========================================================================
# Kiến trúc mạng chính
# ===========================================================================
class GhostSEResNet(nn.Module):
    def __init__(self, in_channels: int = 3, num_classes: int = 4):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
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

def build_model(name="se_ghost_resnet", num_classes=4, in_channels=3, **kwargs):
    if name not in MODEL_CONFIGS:
        raise ValueError(f"Model {name} not found in registry.")
    return GhostSEResNet(in_channels=in_channels, num_classes=num_classes)

# ===========================================================================
# Test
# ===========================================================================
if __name__ == "__main__":
    model = build_model(name=ARCH_NAME)
    print(f"Model {ARCH_NAME} created successfully.")
    dummy_input = torch.randn(1, 3, 224, 224)
    print(f"Output shape: {model(dummy_input).shape}")
