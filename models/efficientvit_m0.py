import timm
import torch.nn as nn

ARCH_NAME      = "efficientvit_m0"
ARCH_ALIASES   = ["efficientvit"]
DEFAULT_VARIANT = "efficientvit_m0"
MODEL_CONFIGS  = {
    "efficientvit_m0": {"input_size": 224, "params_M": 2.35},
}

def build_model(name="efficientvit_m0", num_classes=4, in_channels=3, **kwargs):
    model = timm.create_model(
        "efficientvit_m0",
        pretrained=True,
        num_classes=num_classes,
        in_chans=in_channels,
    )
    model.num_classes = num_classes
    return model
