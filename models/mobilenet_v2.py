import torch.nn as nn
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

ARCH_NAME = "mobilenet_v2"
MODEL_CONFIGS = {"mobilenet_v2": {"input_size": 224, "params_M": 3.50}}


def build_model(name="mobilenet_v2", num_classes=4, in_channels=3, **kwargs):
    model = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    if in_channels != 3:
        model.features[0][0] = nn.Conv2d(in_channels, 32, 3, stride=2, padding=1, bias=False)
    model.num_classes = num_classes
    return model
