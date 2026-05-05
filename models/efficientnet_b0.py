import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

ARCH_NAME = "efficientnet_b0"
MODEL_CONFIGS = {"efficientnet_b0": {"input_size": 224, "params_M": 5.29}}


def build_model(name="efficientnet_b0", num_classes=4, in_channels=3, **kwargs):
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    if in_channels != 3:
        model.features[0][0] = nn.Conv2d(in_channels, 32, 3, stride=2, padding=1, bias=False)
    model.num_classes = num_classes
    return model
