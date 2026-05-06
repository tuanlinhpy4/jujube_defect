import timm
import torch.nn as nn

ARCH_NAME = "starnet_s1"
MODEL_CONFIGS = {"starnet_s1": {"input_size": 224, "params_M": 3.64}}

def build_model(name="starnet_s1", num_classes=4, in_channels=3, **kwargs):
    model = timm.create_model(
        "starnet_s1",
        pretrained=True,
        num_classes=num_classes,
        in_chans=in_channels,
    )
    model.num_classes = num_classes
    return model
