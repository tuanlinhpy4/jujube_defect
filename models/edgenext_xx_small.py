import timm
import torch.nn as nn

ARCH_NAME      = "edgenext_xx_small"
ARCH_ALIASES   = ["edgenext"]
DEFAULT_VARIANT = "edgenext_xx_small"
MODEL_CONFIGS  = {
    "edgenext_xx_small": {"input_size": 256, "params_M": 1.33},
}

def build_model(name="edgenext_xx_small", num_classes=4, in_channels=3, **kwargs):
    model = timm.create_model(
        "edgenext_xx_small",
        pretrained=True,
        num_classes=num_classes,
        in_chans=in_channels,
    )
    model.num_classes = num_classes
    return model
