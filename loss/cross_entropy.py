import torch
import torch.nn as nn
from typing import List, Optional, Union


def build_loss(
    num_classes: Optional[int] = None,
    class_weights: Optional[Union[List[float], torch.Tensor]] = None,
    **kwargs,
) -> nn.CrossEntropyLoss:
    """Build CrossEntropyLoss, optionally with per-class weights.

    Args:
        num_classes: number of classes (unused, kept for API consistency).
        class_weights: per-class weights to down/up-weight each class.
            - list[float]: converted to tensor automatically.
            - torch.Tensor: used directly (must match num_classes length).
            - None: all classes weighted equally (standard CE).
    """
    weight: Optional[torch.Tensor] = None

    if class_weights is not None:
        if isinstance(class_weights, torch.Tensor):
            weight = class_weights.float()
        else:
            weight = torch.tensor(class_weights, dtype=torch.float32)

    return nn.CrossEntropyLoss(weight=weight)
