import timm
import torch.nn as nn

ARCH_NAME = "tinynet_e"
MODEL_CONFIGS = {"tinynet_e": {"input_size": 224, "params_M": 2.04}}

def build_model(name="tinynet_e", num_classes=4, in_channels=3, **kwargs):
    model = timm.create_model(
        "tinynet_e",
        pretrained=True,
        num_classes=num_classes,
        in_chans=in_channels,
    )
    model.num_classes = num_classes
    return model
