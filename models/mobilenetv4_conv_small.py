import timm
import torch.nn as nn

ARCH_NAME = "mobilenetv4_conv_small"
MODEL_CONFIGS = {"mobilenetv4_conv_small": {"input_size": 224, "params_M": 3.77}}

def build_model(name="mobilenetv4_conv_small", num_classes=4, in_channels=3, **kwargs):
    model = timm.create_model(
        "mobilenetv4_conv_small",
        pretrained=True,
        num_classes=num_classes,
        in_chans=in_channels,
    )
    model.num_classes = num_classes
    return model
