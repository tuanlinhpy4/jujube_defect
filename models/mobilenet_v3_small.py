import torch.nn as nn
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

ARCH_NAME = "mobilenet_v3_small"
MODEL_CONFIGS = {"mobilenet_v3_small": {"input_size": 224, "params_M": 2.54}}


def build_model(name="mobilenet_v3_small", num_classes=4, in_channels=3, **kwargs):
    model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)
    if in_channels != 3:
        model.features[0][0] = nn.Conv2d(in_channels, 16, 3, stride=2, padding=1, bias=False)
    model.num_classes = num_classes
    return model
