import torch.nn as nn
from torchvision.models import shufflenet_v2_x0_5, ShuffleNet_V2_X0_5_Weights

ARCH_NAME = "shufflenet_v2_x0_5"
MODEL_CONFIGS = {"shufflenet_v2_x0_5": {"input_size": 224, "params_M": 1.37}}


def build_model(name="shufflenet_v2_x0_5", num_classes=4, in_channels=3, **kwargs):
    model = shufflenet_v2_x0_5(weights=ShuffleNet_V2_X0_5_Weights.IMAGENET1K_V1)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    if in_channels != 3:
        model.conv1[0] = nn.Conv2d(in_channels, 24, 3, stride=2, padding=1, bias=False)
    model.num_classes = num_classes
    return model
