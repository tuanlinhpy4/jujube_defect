import torch.nn as nn
from torchvision.models import squeezenet1_0, SqueezeNet1_0_Weights

ARCH_NAME = "squeezenet1_0"
MODEL_CONFIGS = {"squeezenet1_0": {"input_size": 224, "params_M": 1.25}}


def build_model(name="squeezenet1_0", num_classes=4, in_channels=3, **kwargs):
    model = squeezenet1_0(weights=SqueezeNet1_0_Weights.IMAGENET1K_V1)
    model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=1)
    model.num_classes = num_classes
    if in_channels != 3:
        model.features[0] = nn.Conv2d(in_channels, 96, kernel_size=7, stride=2)
    return model
