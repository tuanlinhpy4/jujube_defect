import torch.nn as nn


def build_loss(num_classes=None, label_smoothing=0.1, **kwargs):
    return nn.CrossEntropyLoss(label_smoothing=label_smoothing)
