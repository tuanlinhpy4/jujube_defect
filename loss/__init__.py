from importlib import import_module


AVAILABLE = ["cross_entropy", "focal", "label_smoothing"]


def build_loss(name, num_classes=None, loss_cfg=None):
    if name not in AVAILABLE:
        raise ValueError(f"Loss '{name}' not found. Available: {AVAILABLE}")
    cfg = loss_cfg or {}
    mod = import_module(f"loss.{name}")
    return mod.build_loss(num_classes=num_classes, **cfg)
