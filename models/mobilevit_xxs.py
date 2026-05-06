import timm
import torch.nn as nn

ARCH_NAME      = "mobilevit_xxs"
ARCH_ALIASES   = ["mobilevit"]
DEFAULT_VARIANT = "mobilevit_xxs"
MODEL_CONFIGS  = {
    "mobilevit_xxs": {"input_size": 256, "params_M": 1.27},
}

def build_model(name="mobilevit_xxs", num_classes=4, in_channels=3, **kwargs):
    model = timm.create_model(
        "mobilevit_xxs",
        pretrained=True,
        num_classes=num_classes,
        in_chans=in_channels,
    )
    model.num_classes = num_classes
    return model
