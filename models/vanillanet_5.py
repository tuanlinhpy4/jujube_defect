import timm
import torch.nn as nn

ARCH_NAME = "vanillanet_5"
MODEL_CONFIGS = {"vanillanet_5": {"input_size": 224, "params_M": 15.52}}

def build_model(name="vanillanet_5", num_classes=4, in_channels=3, **kwargs):
    model = timm.create_model(
        "vanillanet_5",
        pretrained=True,
        num_classes=num_classes,
        in_chans=in_channels,
    )
    model.num_classes = num_classes
    return model
