import torch
import torch.nn as nn
import torch.nn.functional as F
import math
# eca -> 25% ghostnet 75% shufflenet 
# ===========================================================================
# Model Registry Information
# ===========================================================================
ARCH_NAME = "eca_shufflenet_v2"
MODEL_CONFIGS = {"eca_shufflenet_v2": {"input_size": 224, "params_M": None}}

# ===========================================================================
# ECA-ShuffleNet-V2 Components (75% ShuffleNet / 25% GhostNet)
# ===========================================================================
class ECAModule(nn.Module):
    """Khối ECA chuẩn tính toán trọng số động theo chiều kênh"""
    def __init__(self, channels: int, gamma: int = 2, b: int = 1):
        super().__init__()
        t = int(abs((math.log(channels, 2) + b) / gamma))
        k = t if t % 2 != 0 else t + 1
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, 1, c)
        y = self.conv(y).transpose(1, 2).view(b, c, 1, 1)
        return self.sigmoid(y)

class ShuffleNetBlock(nn.Module):
    """Khối tích chập nhóm kết hợp xáo trộn kênh (Channel Shuffle)"""
    def __init__(self, channels: int, groups: int = 4):
        super().__init__()
        self.groups = groups
        self.group_conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        
    def channel_shuffle(self, x: torch.Tensor, groups: int) -> torch.Tensor:
        batch, channels, height, width = x.size()
        channels_per_group = channels // groups
        x = x.view(batch, groups, channels_per_group, height, width)
        x = torch.transpose(x, 1, 2).contiguous()
        x = x.view(batch, channels, height, width)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.group_conv(x)
        out = self.bn(out)
        out = self.relu(out)
        return self.channel_shuffle(out, self.groups)

class GhostModule(nn.Module):
    """
    Khối Ghost Module từ GhostNet:
    Tách phép tích chập thông thường thành: 1 nửa tích chập chuẩn + 1 nửa tích chập chiều sâu giá rẻ.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, ratio: int = 2, dw_size: int = 3):
        super().__init__()
        self.out_channels = out_channels
        init_channels = math.ceil(out_channels / ratio)
        new_channels = init_channels * (ratio - 1)

        # Tích chập gốc (Primary convolution)
        self.primary_conv = nn.Sequential(
            nn.Conv2d(in_channels, init_channels, kernel_size, padding=kernel_size//2, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True)
        )

        # Tích chập rẻ tạo đặc trưng ma (Cheap operation / Depthwise convolution)
        self.cheap_conv = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, padding=dw_size//2, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.primary_conv(x)
        x2 = self.cheap_conv(x1)
        out = torch.cat([x1, x2], dim=1)
        return out[:, :self.out_channels, :, :]

class ECAShuffleNetBlock(nn.Module):
    """
    Khối lai kết hợp cải tiến:
    - 75% số kênh đưa vào ShuffleNet để tính toán diện rộng với chi phí thấp.
    - 25% số kênh đưa vào Ghost Module để tạo ra các đặc trưng giàu thông tin một cách siêu nhẹ.
    - Kết nối Skip Connection từ tầng đầu vào.
    """
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        # 1. Định cấu hình phân chia kênh theo tỷ lệ 75 / 25
        self.c_shuffle = int(out_channels * 0.75)
        self.c_ghost = out_channels - self.c_shuffle
        
        assert self.c_shuffle % 4 == 0, f"Số kênh ShuffleNet ({self.c_shuffle}) phải chia hết cho 4!"

        # 2. Xử lý giảm độ phân giải không gian và thay đổi số kênh (nếu stride > 1 hoặc đổi kênh)
        self.pre_conv = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.pre_conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        
        # 3. Các thành phần chính của khối lai
        self.eca = ECAModule(out_channels)
        self.shuffle_branch = ShuffleNetBlock(self.c_shuffle, groups=4)
        self.ghost_branch = GhostModule(self.c_ghost, self.c_ghost, kernel_size=3)
        
        # 4. Tầng kết hợp thông tin và Skip Connection
        self.fuse_conv = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Đường tắt shortcut từ input gốc
        identity = self.shortcut(x)
        
        # Đưa qua tầng pre_conv (nếu có biến đổi chiều dữ liệu)
        out = self.pre_conv(x)
        
        # Bước 1: Tính toán trọng số kênh bằng khối ECA
        weights = self.eca(out)
        out_weighted = out * weights
        
        # Bước 2: Phân tách Tensor theo cấu hình 75% và 25% số kênh
        x_shuffle = out_weighted[:, :self.c_shuffle, :, :]
        x_ghost = out_weighted[:, self.c_shuffle:, :, :]
        
        # Bước 3: Trích xuất song song qua các nhánh riêng biệt (ShuffleNet & Ghost)
        out_shuffle = self.shuffle_branch(x_shuffle)
        out_ghost = self.ghost_branch(x_ghost)
        
        # Bước 4: Khôi phục cấu trúc kênh bằng phương pháp Concat
        out_combined = torch.cat([out_shuffle, out_ghost], dim=1)
        
        # Bước 5: Tổng hợp thông tin qua Conv 1x1 và thực hiện Skip Connection
        out = self.fuse_conv(out_combined)
        out += identity
        return self.relu(out)

# ===========================================================================
# Network Architecture
# ===========================================================================
class ECAShuffleNetV2(nn.Module):
    def __init__(self, in_channels: int = 3, num_classes: int = 4):
        super().__init__()
        # Tầng chuẩn hóa ảnh đầu vào ban đầu (Stem Layer)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        # Các khối Stages chính (Các giá trị kênh 64, 128, 256 đều tối ưu với tỷ lệ 75%)
        self.stage1 = ECAShuffleNetBlock(32, 64, stride=2)
        self.stage2 = ECAShuffleNetBlock(64, 128, stride=2)
        self.stage3 = ECAShuffleNetBlock(128, 256, stride=2)
        
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
def build_model(name="eca_shufflenet_v2", num_classes=4, in_channels=3, **kwargs):
    model = ECAShuffleNetV2(in_channels=in_channels, num_classes=num_classes)
    model.num_classes = num_classes
    return model


   
