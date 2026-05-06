import timm
import torch.nn as nn

ARCH_NAME      = "pp_lcnet_050"
ARCH_ALIASES   = ["pp_lcnet"]
DEFAULT_VARIANT = "pp_lcnet_050"
MODEL_CONFIGS  = {
    "pp_lcnet_050": {"input_size": 224, "params_M": 1.88},
}

def build_model(name="pp_lcnet_050", num_classes=4, in_channels=3, **kwargs):
    model = timm.create_model(
        "pp_lcnet_050",
        pretrained=True,
        num_classes=num_classes,
        in_chans=in_channels,
    )
    model.num_classes = num_classes
    return model
