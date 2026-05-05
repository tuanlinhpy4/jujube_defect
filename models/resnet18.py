import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

ARCH_NAME = "resnet18"
MODEL_CONFIGS = {"resnet18": {"input_size": 224, "params_M": 11.69}}


def build_model(name="resnet18", num_classes=4, in_channels=3, **kwargs):
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    if in_channels != 3:
        model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.num_classes = num_classes
    return model
