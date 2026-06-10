import math
import torch
import torch.nn as nn
import torch.nn.functional as F
# ECA + ghostnet các lớp cuối + resnet
# ===========================================================================
# Model Registry Information (Đã cập nhật tên)
# ===========================================================================
ARCH_NAME = "hybrid_ghost_resnet"
MODEL_CONFIGS = {"hybrid_ghost_resnet": {"input_size": 224, "params_M": None}}

# ===========================================================================
# 1. Khối ECA (Efficient Channel Attention) - Siêu nhẹ, siêu nhanh
# ===========================================================================
class CustomECABlock(nn.Module):
    """
    Efficient Channel Attention (ECA) - Không dùng lớp tuyến tính (Linear),
    thay vào đó dùng Conv1D giúp số parameter gần như bằng 0 và latency tối ưu hơn SE.
    """
    def __init__(self, channels: int, k_size: int = 3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.avg_pool(x)
        y = y.squeeze(-1).transpose(-1, -2)
        y = self.conv(y)
        y = y.transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)

# ===========================================================================
# 2. Standard ResNet Block với ECA (Dùng cho Stage đầu - Tối ưu Latency)
# ===========================================================================
class StandardResNetBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.eca = CustomECABlock(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.eca(out)
        
        out += self.shortcut(x)
        return self.relu2(out)

# ===========================================================================
# 3. Ghost Components với ECA (Dùng cho Stage sau - Tiết kiệm Param)
# ===========================================================================
class CustomGhostConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, ratio=2, dw_size=3, relu=True):
        super().__init__()
        self.out_channels = out_channels
        init_channels = math.ceil(out_channels / ratio)
        new_channels = init_channels * (ratio - 1)

        self.primary_conv = nn.Sequential(
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


class CustomGhostBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.ghost1 = CustomGhostConv(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, relu=True)
        self.ghost2 = CustomGhostConv(out_channels, out_channels, kernel_size=3, stride=1, padding=1, relu=False)
        
        self.eca = CustomECABlock(out_channels)
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
        out = self.eca(out)
        out += self.shortcut(x)
        return self.relu(out)

# ===========================================================================
# 4. Hybrid Network Architecture
# ===========================================================================
class CustomHybridNet(nn.Module):
    def __init__(self, in_channels: int = 3, num_classes: int = 4):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        # Stage đầu: Dùng ResNet chuẩn kết hợp ECA để tối ưu Latency
        self.stage1 = StandardResNetBlock(32, 64, stride=2)
        self.stage2 = StandardResNetBlock(64, 128, stride=2)
        
        # Stage cuối: Dùng GhostNet kết hợp ECA để giảm bớt số Parameters
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
# Registration Build Function
# ===========================================================================
def build_model(name="hybrid_ghost_resnet", num_classes=4, in_channels=3, **kwargs):
    model = CustomHybridNet(in_channels=in_channels, num_classes=num_classes)
    model.num_classes = num_classes
    return model

if __name__ == "__main__":
    # Test thử kiến trúc và xuất output shape
    model = build_model()
    test_tensor = torch.randn(1, 3, 224, 224)
    output = model(test_tensor)
    print(f"Khởi tạo thành công mô hình: {ARCH_NAME}")
    print("Mô hình cấu hình:", MODEL_CONFIGS[ARCH_NAME])
    print("Output shape mong đợi:", output.shape)
