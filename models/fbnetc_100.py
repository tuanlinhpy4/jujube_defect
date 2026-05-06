import timm
import torch.nn as nn

ARCH_NAME = "fbnetc_100"
MODEL_CONFIGS = {"fbnetc_100": {"input_size": 224, "params_M": 5.57}}

def build_model(name="fbnetc_100", num_classes=4, in_channels=3, **kwargs):
    model = timm.create_model(
        "fbnetc_100",
        pretrained=True,
        num_classes=num_classes,
        in_chans=in_channels,
    )
    model.num_classes = num_classes
    return model
