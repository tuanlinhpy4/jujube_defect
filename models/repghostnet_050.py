import timm
import torch.nn as nn

ARCH_NAME = "repghostnet_050"
MODEL_CONFIGS = {"repghostnet_050": {"input_size": 224, "params_M": 2.68}}

def build_model(name="repghostnet_050", num_classes=4, in_channels=3, **kwargs):
    model = timm.create_model(
        "repghostnet_050",
        pretrained=True,
        num_classes=num_classes,
        in_chans=in_channels,
    )
    model.num_classes = num_classes
    return model
