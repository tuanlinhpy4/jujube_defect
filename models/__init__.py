"""Model registry.

Scans all .py files in this directory and registers every variant from their
MODEL_CONFIGS dict into a flat lookup table.

Callable names
--------------
Direct variant:
    --model mvanet_3          original MvANet, p=3
    --model mvanet_1_tiny     original MvANet tiny
    --model mvanet_opt_3      optimised MvANet, p=3

Arch alias (→ DEFAULT_VARIANT of that file):
    --model mvanet            → mvanet_3
    --model mvanet_opt        → mvanet_opt_3
    --model MvANet            → mvanet_3  (case-insensitive alias)
"""

from __future__ import annotations

from importlib import import_module
from pathlib import Path
from typing import List, Optional

# variant_name  → module  (e.g. "mvanet_3"     → mvanet module)
_VARIANTS: dict = {}
# arch_name     → module  (e.g. "MvANet"        → mvanet module)
_MODULES: dict  = {}
# alias/arch    → default variant  (e.g. "mvanet" → "mvanet_3")
_ALIASES: dict  = {}


def _scan() -> None:
    models_dir = Path(__file__).parent
    for f in sorted(models_dir.glob("*.py")):
        if f.name.startswith("_"):
            continue
        try:
            mod = import_module(f"models.{f.stem}")
        except Exception as exc:
            print(f"[models] skip {f.name}: {exc}")
            continue

        if not (hasattr(mod, "ARCH_NAME")
                and hasattr(mod, "build_model")
                and hasattr(mod, "MODEL_CONFIGS")):
            continue

        arch = mod.ARCH_NAME
        _MODULES[arch] = mod

        for variant in mod.MODEL_CONFIGS:
            _VARIANTS[variant] = mod

        default = getattr(mod, "DEFAULT_VARIANT", list(mod.MODEL_CONFIGS)[-1])
        for key in (arch, arch.lower()):
            _ALIASES.setdefault(key, default)
        for alias in getattr(mod, "ARCH_ALIASES", []):
            _ALIASES.setdefault(alias, default)


_scan()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_model(name: str, num_classes: int,
                in_channels: int = 3, **kwargs):
    """Build a model by variant name or arch alias.

    Raises ValueError with the full list of valid names if *name* is unknown.
    """
    if name in _VARIANTS:
        return _VARIANTS[name].build_model(
            name=name, num_classes=num_classes,
            in_channels=in_channels, **kwargs,
        )
    if name in _ALIASES:
        variant = _ALIASES[name]
        return _VARIANTS[variant].build_model(
            name=variant, num_classes=num_classes,
            in_channels=in_channels, **kwargs,
        )
    raise ValueError(
        f"Model '{name}' not found.\n"
        f"  Variants : {sorted(_VARIANTS)}\n"
        f"  Aliases  : {sorted(_ALIASES)}"
    )


def list_models() -> List[str]:
    """All callable names — variants + aliases."""
    return sorted(set(_VARIANTS) | set(_ALIASES))


def list_archs() -> List[str]:
    """One entry per registered model file (ARCH_NAME)."""
    return list(_MODULES)


def list_variants(arch_name: str) -> List[str]:
    """All variant names that belong to *arch_name*."""
    mod = _MODULES.get(arch_name)
    if mod is None:
        return []
    return [v for v, m in _VARIANTS.items() if m is mod]


def model_params_M(name: str) -> Optional[float]:
    if name in _VARIANTS:
        mod, key = _VARIANTS[name], name
    elif name in _ALIASES:
        key = _ALIASES[name]
        mod = _VARIANTS.get(key)
    else:
        return None
    return mod.MODEL_CONFIGS.get(key, {}).get("params_M")
