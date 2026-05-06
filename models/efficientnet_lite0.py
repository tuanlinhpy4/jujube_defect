import timm
import torch.nn as nn

ARCH_NAME = "efficientnet_lite0"
MODEL_CONFIGS = {"efficientnet_lite0": {"input_size": 224, "params_M": 4.65}}

def build_model(name="efficientnet_lite0", num_classes=4, in_channels=3, **kwargs):
    model = timm.create_model(
        "efficientnet_lite0",
        pretrained=True,
        num_classes=num_classes,
        in_chans=in_channels,
    )
    model.num_classes = num_classes
    return model
