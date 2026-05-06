import timm
import torch.nn as nn

ARCH_NAME = "efficientformer_l1"
MODEL_CONFIGS = {"efficientformer_l1": {"input_size": 224, "params_M": 12.29}}

def build_model(name="efficientformer_l1", num_classes=4, in_channels=3, **kwargs):
    model = timm.create_model(
        "efficientformer_l1",
        pretrained=True,
        num_classes=num_classes,
        in_chans=in_channels,
    )
    model.num_classes = num_classes
    return model
