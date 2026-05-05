from .cross_entropy import build_loss as _build_ce

AVAILABLE = ["cross_entropy"]


def build_loss(name: str = "cross_entropy", num_classes=None, loss_cfg=None):
    if name != "cross_entropy":
        raise ValueError(
            f"Loss '{name}' not supported. Only 'cross_entropy' is available."
        )
    cfg = loss_cfg or {}
    return _build_ce(num_classes=num_classes, **cfg)
