import torch.nn as nn


def build_loss(num_classes=None, **kwargs):
    return nn.CrossEntropyLoss()
