import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ===========================================================================
# Model Registry Information (Đã đổi tên thành quant_se_ghostnet)
# ===========================================================================
ARCH_NAME = "quant_se_ghostnet"
MODEL_CONFIGS = {"quant_se_ghostnet": {"input_size": 224, "params_M": None}}

# ===========================================================================
# Components (Hỗ trợ Quantization)
# ===========================================================================
class QuantSEGhostConv(nn.Module):
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

        # Toán tử cat lượng tử hóa
        self.f_cat = nn.quantized.FloatFunctional()

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = self.f_cat.cat([x1, x2], dim=1)
        return out[:, :self.out_channels, :, :]


class QuantSEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
        # Toán tử nhân lượng tử hóa
        self.f_mul = nn.quantized.FloatFunctional()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return self.f_mul.mul(x, y)


class QuantSEGhostBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.ghost1 = QuantSEGhostConv(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, relu=True)
        self.ghost2 = QuantSEGhostConv(out_channels, out_channels, kernel_size=3, stride=1, padding=1, relu=False)
        self.se = QuantSEBlock(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        # Toán tử cộng shortcut lượng tử hóa
        self.f_add = nn.quantized.FloatFunctional()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.ghost1(x)
        out = self.ghost2(out)
        out = self.se(out)
        out = self.f_add.add(out, self.shortcut(x))
        return self.relu(out)

# ===========================================================================
# Network Architecture (Đã đổi tên thành QuantSEGhostNet)
# ===========================================================================
class QuantSEGhostNet(nn.Module):
    def __init__(self, in_channels: int = 3, num_classes: int = 4):
        super().__init__()
        
        # Các cổng chuyển đổi kiểu dữ liệu (Quant/Dequant stubs)
        self.quant = torch.ao.quantization.QuantStub()
        self.dequant = torch.ao.quantization.DeQuantStub()
        
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        self.stage1 = QuantSEGhostBlock(32, 64, stride=2)
        self.stage2 = QuantSEGhostBlock(64, 128, stride=2)
        self.stage3 = QuantSEGhostBlock(128, 256, stride=2)
        
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(256, num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.quant(x)  # FP32 -> INT8
        
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.pool(x).flatten(1)
        x = self.classifier(x)
        
        x = self.dequant(x)  # INT8 -> FP32
        return x

# ===========================================================================
# Registration Build Function & Quantize Helper
# ===========================================================================
def build_model(name="quant_se_ghostnet", num_classes=4, in_channels=3, **kwargs):
    model = QuantSEGhostNet(in_channels=in_channels, num_classes=num_classes)
    model.num_classes = num_classes
    return model

def calibrate_and_convert_to_int8(model, calibration_dataloader):
    """
    Hàm thực hiện lượng tử hóa tĩnh (Static Quantization) cho QuantSEGhostNet
    """
    model.eval()
    
    # Cấu hình backend lượng tử hóa (fbgemm cho CPU x86, qnnpack cho ARM/Mobile)
    model.qconfig = torch.ao.quantization.get_default_qconfig('fbgemm')
    
    # Chuẩn bị mô hình (Chèn các bộ quan sát - observers)
    model_prepared = torch.ao.quantization.prepare(model, inplace=False)
    
    # Chạy Calibration với một lượng nhỏ dữ liệu mẫu
    with torch.no_grad():
        for images, _ in calibration_dataloader:
            model_prepared(images)
            
    # Chuyển đổi mô hình hoàn toàn sang kiểu tính toán số nguyên INT
