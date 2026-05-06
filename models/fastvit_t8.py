import timm
import torch.nn as nn

ARCH_NAME      = "fastvit_t8"
ARCH_ALIASES   = ["fastvit"]
DEFAULT_VARIANT = "fastvit_t8"
MODEL_CONFIGS  = {
    "fastvit_t8": {"input_size": 256, "params_M": 3.97},
}

def build_model(name="fastvit_t8", num_classes=4, in_channels=3, **kwargs):
    model = timm.create_model(
        "fastvit_t8",
        pretrained=True,
        num_classes=num_classes,
        in_chans=in_channels,
    )
    model.num_classes = num_classes
    return model
