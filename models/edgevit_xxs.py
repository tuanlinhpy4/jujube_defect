import timm
import torch.nn as nn

ARCH_NAME = "edgevit_xxs"
MODEL_CONFIGS = {"edgevit_xxs": {"input_size": 224, "params_M": 4.13}}

def build_model(name="edgevit_xxs", num_classes=4, in_channels=3, **kwargs):
    model = timm.create_model(
        "edgevit_xxs",
        pretrained=True,
        num_classes=num_classes,
        in_chans=in_channels,
    )
    model.num_classes = num_classes
    return model
