import timm
import torch.nn as nn

ARCH_NAME      = "ghostnet_050"
ARCH_ALIASES   = ["ghostnet"]
DEFAULT_VARIANT = "ghostnet_050"
MODEL_CONFIGS  = {
    "ghostnet_050": {"input_size": 224, "params_M": 2.59},
}

def build_model(name="ghostnet_050", num_classes=4, in_channels=3, **kwargs):
    model = timm.create_model(
        "ghostnet_050",
        pretrained=True,
        num_classes=num_classes,
        in_chans=in_channels,
    )
    model.num_classes = num_classes
    return model
