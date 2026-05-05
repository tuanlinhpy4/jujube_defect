from importlib import import_module
from pathlib import Path

_MODELS = {}

def _scan():
    models_dir = Path(__file__).parent
    for f in sorted(models_dir.glob("*.py")):
        if f.name.startswith("_"):
            continue
        try:
            mod = import_module(f"models.{f.stem}")
            if hasattr(mod, "ARCH_NAME") and hasattr(mod, "build_model"):
                _MODELS[mod.ARCH_NAME] = mod
        except Exception as e:
            print(f"[models] skip {f.name}: {e}")

_scan()


def build_model(name, num_classes, in_channels=3, **kwargs):
    if name not in _MODELS:
        raise ValueError(f"Model '{name}' not found. Available: {list(_MODELS.keys())}")
    return _MODELS[name].build_model(name=name, num_classes=num_classes, in_channels=in_channels, **kwargs)


def list_models():
    return list(_MODELS.keys())


def model_params_M(name):
    if name not in _MODELS:
        return None
    cfgs = _MODELS[name].MODEL_CONFIGS
    cfg = cfgs.get(name, next(iter(cfgs.values())))
    return cfg.get("params_M")
