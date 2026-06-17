import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ===========================================================================
# Model Registry Information
# ===========================================================================
ARCH_NAME = "micronet"
MODEL_CONFIGS = {"micronet": {"input_size": 224, "params_M": None}}

# ===========================================================================
# MicroNet Components (Micro-Factorized Conv & Dynamic Shift-Max)
# ===========================================================================
class DynamicShiftMax(nn.Module):
    """
    Hàm kích hoạt Dynamic Shift-Max của MicroNet:
    Tự động tính toán tham số kích hoạt theo từng ảnh, dịch chuyển kênh 
    và dùng phép toán Max để giữ tính phi tuyến mà không làm mất thông tin âm.
    """
    def __init__(self, channels: int, groups: int = 4):
        super().__init__()
        self.channels = channels
        self.groups = groups
        
        # Bộ tạo trọng số động siêu nhẹ (Router) dựa trên Global Average Pooling
        self.router = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 2, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 2, channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.size()
        
        # 1. Tính toán trọng số động theo ngữ cảnh của bức ảnh
        dynamic_weights = self.router(x)
        x_weighted = x * dynamic_weights
        
        # 2. Phép dịch chuyển kênh (Shift) - Thao tác chỉ số ô nhớ (Zero FLOPs)
        # Dịch chuyển các kênh theo cấu trúc nhóm để chúng giao thoa nhau
        channels_per_group = c // self.groups
        x_shifted = x_weighted.view(b, self.groups, channels_per_group, h, w)
        x_shifted = torch.roll(x_shifted, shifts=1, dims=2) # Dịch chuyển vòng tròn trong nhóm
        x_shifted = x_shifted.view(b, c, h, w)
        
        # 3. Lấy giá trị lớn nhất (Max) giữa đặc trưng gốc và đặc trưng dịch chuyển
        return torch.max(x, x_shifted)

class MicroFactorizedSpatialConv(nn.Module):
    """
    Tích chập không gian phân mảnh (Spatial-wise Factorization):
    Thay vì dùng kernel 3x3 (9 phép tính), ta phân rã thành 3x1 nối tiếp 1x3 (3+3=6 phép tính).
    Tích hợp luôn phép tích chập chiều sâu (Depthwise) để siêu nhẹ.
    """
    def __init__(self, channels: int, stride: int = 1):
        super().__init__()
        # Tích chập dọc 3x1 (Depthwise)
        self.conv_v = nn.Conv2d(
            channels, channels, kernel_size=(3, 1), stride=(stride, 1), 
            padding=(1, 0), groups=channels, bias=False
        )
        # Tích chập ngang 1x3 (Depthwise)
        self.conv_h = nn.Conv2d(
            channels, channels, kernel_size=(1, 3), stride=(1, stride), 
            padding=(0, 1), groups=channels, bias=False
        )
        self.bn = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bn(self.conv_h(self.conv_v(x)))

class MicroNetBlock(nn.Module):
    """
    Khối MicroNet chuẩn chỉnh:
    - Sử dụng Tích chập không gian phân mảnh (3x1 và 1x3).
    - Sử dụng Tích chập điểm phân mảnh (1x1 Group Conv) để giảm chi phí liên kênh.
    - Kích hoạt bằng hàm Dynamic Shift-Max.
    - Kết nối đường tắt Skip Connection.
    """
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        
        # 1. Tích chập không gian phân mảnh
        self.spatial_conv = MicroFactorizedSpatialConv(in_channels, stride=stride)
        
        # 2. Tích chập điểm phân mảnh (Pointwise Group Conv) để hòa trộn kênh giá rẻ
        self.channel_merge = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, groups=4, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        
        # 3. Hàm kích hoạt tinh hoa Dynamic Shift-Max
        self.act = DynamicShiftMax(out_channels, groups=4)
        
        # 4. Xử lý đường tắt Skip Connection khi lệch kích thước hoặc lệch kênh
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        
        out = self.spatial_conv(x)
        out = self.channel_merge(out)
        out = self.act(out)
        
        out += identity
        return out

# ===========================================================================
# Network Architecture
# ===========================================================================
class MicroNet(nn.Module):
    def __init__(self, in_channels: int = 3, num_classes: int = 4):
        super().__init__()
        # Stem Layer siêu phẳng
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            DynamicShiftMax(32, groups=4)
        )
        
        # Các Stages được xây dựng hoàn toàn bằng khối MicroNet đặc trưng
        # Các số lượng kênh nên chia hết cho 4 để phù hợp với số nhóm (groups=4)
        self.stage1 = MicroNetBlock(32, 64, stride=2)
        self.stage2 = MicroNetBlock(64, 128, stride=2)
        self.stage3 = MicroNetBlock(128, 256, stride=2)
        
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
def build_model(name="micronet", num_classes=4, in_channels=3, **kwargs):
    model = MicroNet(in_channels=in_channels, num_classes=num_classes)
    model.num_classes = num_classes
    return model

# --- Khối chạy thử nghiệm cục bộ ---
if __name__ == "__main__":
    model = build_model(name=ARCH_NAME)
    # Khởi tạo tensor giả lập chuẩn kích thước dự án [Batch=2, Channels=3, H=224, W=224]
    dummy_input = torch.randn(2, 3, 224, 224)
    output = model(dummy_input)
    print(f"Mô hình khởi tạo thành công: {ARCH_NAME}")
    print(f"Kích thước tensor đầu ra: {output.shape}")
