import timm
import torch.nn as nn

ARCH_NAME = "mixnet_s"
MODEL_CONFIGS = {"mixnet_s": {"input_size": 224, "params_M": 4.13}}

def build_model(name="mixnet_s", num_classes=4, in_channels=3, **kwargs):
    model = timm.create_model(
        "mixnet_s",
        pretrained=True,
        num_classes=num_classes,
        in_chans=in_channels,
    )
    model.num_classes = num_classes
    return model
