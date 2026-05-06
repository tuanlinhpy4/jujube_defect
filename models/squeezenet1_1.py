import torch.nn as nn
from torchvision.models import squeezenet1_1, SqueezeNet1_1_Weights

ARCH_NAME = "squeezenet1_1"
MODEL_CONFIGS = {"squeezenet1_1": {"input_size": 224, "params_M": 1.24}}

def build_model(name="squeezenet1_1", num_classes=4, in_channels=3, **kwargs):
    model = squeezenet1_1(weights=SqueezeNet1_1_Weights.IMAGENET1K_V1)
    model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=1)
    model.num_classes = num_classes
    if in_channels != 3:
        model.features[0] = nn.Conv2d(in_channels, 64, kernel_size=3, stride=2)
    return model
