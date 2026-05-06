import timm
import torch.nn as nn

ARCH_NAME = "poolformer_s12"
MODEL_CONFIGS = {"poolformer_s12": {"input_size": 224, "params_M": 11.92}}

def build_model(name="poolformer_s12", num_classes=4, in_channels=3, **kwargs):
    model = timm.create_model(
        "poolformer_s12",
        pretrained=True,
        num_classes=num_classes,
        in_chans=in_channels,
    )
    model.num_classes = num_classes
    return model
