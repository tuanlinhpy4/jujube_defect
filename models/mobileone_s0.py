import timm
import torch.nn as nn

ARCH_NAME = "mobileone_s0"
MODEL_CONFIGS = {"mobileone_s0": {"input_size": 224, "params_M": 5.29}}

def build_model(name="mobileone_s0", num_classes=4, in_channels=3, **kwargs):
    model = timm.create_model(
        "mobileone_s0",
        pretrained=True,
        num_classes=num_classes,
        in_chans=in_channels,
    )
    model.num_classes = num_classes
    return model
