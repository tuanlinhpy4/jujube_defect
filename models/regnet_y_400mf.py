import torch.nn as nn
from torchvision.models import regnet_y_400mf, RegNet_Y_400MF_Weights

ARCH_NAME      = "regnet_y_400mf"
ARCH_ALIASES   = ["regnet"]
DEFAULT_VARIANT = "regnet_y_400mf"
MODEL_CONFIGS  = {
    "regnet_y_400mf": {"input_size": 224, "params_M": 4.34},
}

def build_model(name="regnet_y_400mf", num_classes=4, in_channels=3, **kwargs):
    model = regnet_y_400mf(weights=RegNet_Y_400MF_Weights.IMAGENET1K_V1)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    if in_channels != 3:
        model.stem[0] = nn.Conv2d(
            in_channels, 32, kernel_size=3, stride=2, padding=1, bias=False
        )
    model.num_classes = num_classes
    return model
