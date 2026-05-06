import timm
import torch.nn as nn

ARCH_NAME = "ghostnetv2_100"
MODEL_CONFIGS = {"ghostnetv2_100": {"input_size": 224, "params_M": 6.16}}

def build_model(name="ghostnetv2_100", num_classes=4, in_channels=3, **kwargs):
    model = timm.create_model(
        "ghostnetv2_100",
        pretrained=True,
        num_classes=num_classes,
        in_chans=in_channels,
    )
    model.num_classes = num_classes
    return model
