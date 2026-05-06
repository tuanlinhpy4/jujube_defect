import timm
import torch.nn as nn

ARCH_NAME = "deit_tiny_patch16_224"
MODEL_CONFIGS = {"deit_tiny_patch16_224": {"input_size": 224, "params_M": 5.72}}

def build_model(name="deit_tiny_patch16_224", num_classes=4, in_channels=3, **kwargs):
    model = timm.create_model(
        "deit_tiny_patch16_224",
        pretrained=True,
        num_classes=num_classes,
        in_chans=in_channels,
    )
    model.num_classes = num_classes
    return model
