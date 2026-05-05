import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ===========================================================================
# Model Registry Information
# ===========================================================================
ARCH_NAME = "reptakunet"
MODEL_CONFIGS = {"reptakunet": {"input_size": 224, "params_M": None}} # params_M để None nếu bạn chưa đo tham số

# ===========================================================================
# Network Architecture
# ===========================================================================
class ECA(nn.Module):
    def __init__(self, in_channels: int, gamma: int = 2, b: int = 1, k_size: int = None):
        super().__init__()
        if k_size is None:
            t = int(abs(math.log2(in_channels) / gamma) + b / gamma)
            k = t if t % 2 else t + 1
        else:
            k = k_size
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1d   = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.avg_pool(x).squeeze(-1).transpose(-1, -2)
        y = self.conv1d(y).transpose(-1, -2).unsqueeze(-1)
        return x * torch.sigmoid(y).expand_as(x)

class CS_STAM(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.shape_vertical = nn.Conv2d(in_channels, in_channels, kernel_size=(5, 1), 
                                        padding=(2, 0), groups=in_channels, bias=False)
        self.shape_horizontal = nn.Conv2d(in_channels, in_channels, kernel_size=(1, 5), 
                                          padding=(0, 2), groups=in_channels, bias=False)
        self.texture_branch = nn.Conv2d(in_channels, in_channels, kernel_size=3, 
                                        padding=1, groups=in_channels, bias=False)
        self.fusion = nn.Conv2d(2 * in_channels, 1, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ms = self.shape_vertical(x) + self.shape_horizontal(x)
        mt = self.texture_branch(x)
        m = torch.sigmoid(self.fusion(torch.cat([ms, mt], dim=1)))
        return x + x * m

class ECA_CS_STAM(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.eca = ECA(in_channels)
        self.cs_stam = CS_STAM(in_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cs_stam(self.eca(x))

class RepTakuBlock(nn.Module):
    def __init__(self, channels: int, expand_ratio: float = 1.5, deploy: bool = False):
        super().__init__()
        self.deploy = deploy
        self.channels = channels
        hidden_dim = int(channels * expand_ratio)
        
        if deploy:
            self.dw_reparam = nn.Conv2d(channels, channels, kernel_size=3, padding=1, 
                                        groups=channels, bias=True)
        else:
            self.dw_3x3 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False)
            self.dw_1x1 = nn.Conv2d(channels, channels, kernel_size=1, padding=0, groups=channels, bias=False)
            self.bn_3x3 = nn.BatchNorm2d(channels)
            self.bn_1x1 = nn.BatchNorm2d(channels)
            self.bn_id  = nn.BatchNorm2d(channels)
            
        self.pw_expand = nn.Conv2d(channels, hidden_dim, kernel_size=1, bias=False)
        self.bn_exp    = nn.BatchNorm2d(hidden_dim)
        self.act       = nn.ReLU6(inplace=True)
        self.pw_project= nn.Conv2d(hidden_dim, channels, kernel_size=1, bias=False)
        self.bn_proj   = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.deploy:
            out = self.dw_reparam(x)
        else:
            out = self.bn_3x3(self.dw_3x3(x)) + self.bn_1x1(self.dw_1x1(x)) + self.bn_id(x)
            
        out = self.bn_proj(self.pw_project(self.act(self.bn_exp(self.pw_expand(out)))))
        return x + out

    def switch_to_deploy(self):
        if self.deploy:
            return
        kernel, bias = self._get_equivalent_kernel_bias()
        self.dw_reparam = nn.Conv2d(self.channels, self.channels, kernel_size=3, 
                                    padding=1, groups=self.channels, bias=True)
        self.dw_reparam.weight.data = kernel
        self.dw_reparam.bias.data = bias
        
        for attr in ['dw_3x3', 'dw_1x1', 'bn_3x3', 'bn_1x1', 'bn_id']:
            self.__delattr__(attr)
        self.deploy = True

    def _get_equivalent_kernel_bias(self):
        def _fuse_bn(conv_w, bn):
            gamma = bn.weight
            std = (bn.running_var + bn.eps).sqrt()
            t = (gamma / std).reshape(-1, 1, 1, 1)
            return conv_w * t, bn.bias - bn.running_mean * gamma / std
        
        k3, b3 = _fuse_bn(self.dw_3x3.weight, self.bn_3x3)
        k1, b1 = _fuse_bn(self.dw_1x1.weight, self.bn_1x1)
        k1 = F.pad(k1, [1, 1, 1, 1])
        device = self.bn_id.weight.device
        k_id = torch.zeros(self.channels, 1, 3, 3, device=device)
        for i in range(self.channels):
            k_id[i, 0, 1, 1] = 1.0
        k_id, b_id = _fuse_bn(k_id, self.bn_id)
        
        return k3 + k1 + k_id, b3 + b1 + b_id

class GlobalResponseNorm(nn.Module):
    def __init__(self, num_channels: int):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        self.beta  = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gx = torch.norm(x, p=2, dim=(2, 3), keepdim=True)
        nx = gx / (gx.mean(dim=1, keepdim=True) + 1e-6)
        return self.gamma * (x * nx) + self.beta + x

class Downsampler(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.pw = nn.Conv2d(in_ch * 2, out_ch, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU6(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.grn = GlobalResponseNorm(out_ch)

    def forward(self, x_in: torch.Tensor, x_attn: torch.Tensor) -> torch.Tensor:
        x = torch.cat([x_in, x_attn], dim=1)
        x = self.pool(self.act(self.bn(self.pw(x))))
        return self.grn(x)

class Stem(nn.Module):
    # Đã thêm tham số in_channels để nhận dynamic input
    def __init__(self, in_channels: int = 3, out_ch: int = 16):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_ch, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.act1  = nn.ReLU6(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act1(self.bn1(self.conv1(x)))

class RepTakuStage(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, depth: int):
        super().__init__()
        self.blocks = nn.Sequential(*[RepTakuBlock(in_ch) for _ in range(depth)])
        self.attention = ECA_CS_STAM(in_ch)
        self.downsampler = Downsampler(in_ch, out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_in = x
        x_blocks = self.blocks(x)
        x_attn = self.attention(x_blocks)
        return self.downsampler(x_in, x_attn)

class RepTakuNet(nn.Module):
    # Đã thêm in_channels và đẩy nó qua Stem
    def __init__(self, in_channels: int = 3, num_classes: int = 4):
        super().__init__()
        stem_ch = 16
        stage_ch = [32, 64, 128, 256]
        depths = [3, 3, 5, 2] 
        
        self.stem = Stem(in_channels=in_channels, out_ch=stem_ch)
        self.stage1 = RepTakuStage(stem_ch, stage_ch[0], depths[0])
        self.stage2 = RepTakuStage(stage_ch[0], stage_ch[1], depths[1])
        self.stage3 = RepTakuStage(stage_ch[1], stage_ch[2], depths[2])
        self.stage4 = RepTakuStage(stage_ch[2], stage_ch[3], depths[3])
        
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(stage_ch[3], num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.pool(x).flatten(1)
        return self.classifier(x)

    def switch_to_deploy(self):
        for m in self.modules():
            if hasattr(m, 'switch_to_deploy'):
                m.switch_to_deploy()

# ===========================================================================
# Registration Build Function
# ===========================================================================
def build_model(name="reptakunet", num_classes=4, in_channels=3, **kwargs):
    model = RepTakuNet(in_channels=in_channels, num_classes=num_classes)
    model.num_classes = num_classes
    return model