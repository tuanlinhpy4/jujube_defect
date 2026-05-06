import timm
import torch.nn as nn

ARCH_NAME      = "convnext_atto"
ARCH_ALIASES   = ["convnext"]
DEFAULT_VARIANT = "convnext_atto"
MODEL_CONFIGS  = {
    "convnext_atto": {"input_size": 224, "params_M": 3.70},
}

def build_model(name="convnext_atto", num_classes=4, in_channels=3, **kwargs):
    model = timm.create_model(
        "convnext_atto",
        pretrained=True,
        num_classes=num_classes,
        in_chans=in_channels,
    )
    model.num_classes = num_classes
    return model
