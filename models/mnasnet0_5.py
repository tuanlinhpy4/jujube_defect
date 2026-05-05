import torch.nn as nn
from torchvision.models import mnasnet0_5, MNASNet0_5_Weights

ARCH_NAME = "mnasnet0_5"
MODEL_CONFIGS = {"mnasnet0_5": {"input_size": 224, "params_M": 2.22}}


def build_model(name="mnasnet0_5", num_classes=4, in_channels=3, **kwargs):
    model = mnasnet0_5(weights=MNASNet0_5_Weights.IMAGENET1K_V1)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    if in_channels != 3:
        model.layers[0] = nn.Conv2d(in_channels, 16, 3, stride=2, padding=1, bias=False)
    model.num_classes = num_classes
    return model
