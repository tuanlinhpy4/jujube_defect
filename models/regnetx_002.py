import timm
import torch.nn as nn

ARCH_NAME = "regnetx_002"
MODEL_CONFIGS = {"regnetx_002": {"input_size": 224, "params_M": 2.68}}

def build_model(name="regnetx_002", num_classes=4, in_channels=3, **kwargs):
    model = timm.create_model(
        "regnetx_002",
        pretrained=True,
        num_classes=num_classes,
        in_chans=in_channels,
    )
    model.num_classes = num_classes
    return model
