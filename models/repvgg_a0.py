import timm
import torch.nn as nn

ARCH_NAME = "repvgg_a0"
MODEL_CONFIGS = {"repvgg_a0": {"input_size": 224, "params_M": 8.31}}

def build_model(name="repvgg_a0", num_classes=4, in_channels=3, **kwargs):
    model = timm.create_model(
        "repvgg_a0",
        pretrained=True,
        num_classes=num_classes,
        in_chans=in_channels,
    )
    model.num_classes = num_classes
    return model
