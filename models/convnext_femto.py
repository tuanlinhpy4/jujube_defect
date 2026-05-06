import timm
import torch.nn as nn

ARCH_NAME = "convnext_femto"
MODEL_CONFIGS = {"convnext_femto": {"input_size": 224, "params_M": 5.22}}

def build_model(name="convnext_femto", num_classes=4, in_channels=3, **kwargs):
    model = timm.create_model(
        "convnext_femto",
        pretrained=True,
        num_classes=num_classes,
        in_chans=in_channels,
    )
    model.num_classes = num_classes
    return model
