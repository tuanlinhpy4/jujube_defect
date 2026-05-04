"""Dynamic model registry.

Every ``models/*.py`` file or legacy subpackage that exports ``build_model`` and
``MODEL_CONFIGS`` is registered automatically. New architectures should be one
file only.

To add a model:
    1. Create models/my_model.py.
    2. Export ARCH_NAME, MODEL_CONFIGS, and build_model.
    3. Use --arch ARCH_NAME --model variant_name.

Usage:
    from models import registry
    model = registry.build("MvANet", "mvanet_2", num_classes=4)

    from models import build_model
    model = build_model(arch="MvANet", name="mvanet_2", num_classes=4)
"""

import importlib
from pathlib import Path
from typing import Dict, Any, List


class _ModelRegistry:
    """Singleton registry for model architecture modules."""

    def __init__(self):
        self._archs: Dict[str, Any] = {}  # arch_name -> module
        self._aliases: Dict[str, str] = {}  # lower-case alias -> canonical name
        self._scanned = False

    def _scan(self):
        """Scan model files first, then legacy subpackages."""
        if self._scanned:
            return
        models_dir = Path(__file__).parent

        for entry in sorted(models_dir.glob("*.py")):
            if entry.name == "__init__.py" or entry.name.startswith("_"):
                continue
            try:
                mod = importlib.import_module(f"models.{entry.stem}")
                self._register_module(entry.stem, mod)
            except Exception as e:
                print(f"[registry] Skip models/{entry.name}: {e}")

        for entry in sorted(models_dir.iterdir()):
            if not entry.is_dir():
                continue
            if entry.name.startswith(("_", ".")):
                continue
            init_file = entry / "__init__.py"
            if not init_file.exists():
                continue
            try:
                mod = importlib.import_module(f"models.{entry.name}")
                self._register_module(entry.name, mod)
            except Exception as e:
                print(f"[registry] Skip models/{entry.name}: {e}")
        self._scanned = True

    def _register_module(self, default_name: str, mod: Any) -> None:
        """Register a module that follows the model-file contract."""

        if not (hasattr(mod, "build_model") and hasattr(mod, "MODEL_CONFIGS")):
            return

        arch_name = str(getattr(mod, "ARCH_NAME", default_name))
        if arch_name in self._archs:
            return

        self._archs[arch_name] = mod
        self._aliases[arch_name.lower()] = arch_name
        self._aliases[default_name.lower()] = arch_name
        for alias in getattr(mod, "ARCH_ALIASES", []):
            self._aliases[str(alias).lower()] = arch_name

    def _resolve_arch(self, arch: str) -> str:
        self._scan()
        if arch in self._archs:
            return arch
        canonical = self._aliases.get(str(arch).lower())
        if canonical:
            return canonical
        raise ValueError(
            f"Architecture '{arch}' was not found. "
            f"Available: {self.list_archs()}"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def list_archs(self) -> List[str]:
        """Return registered architecture names."""
        self._scan()
        return list(self._archs.keys())

    def canonical_name(self, arch: str) -> str:
        """Return the canonical architecture name for an alias or exact name."""

        return self._resolve_arch(arch)

    def list_variants(self, arch: str) -> List[str]:
        """Return variant names for one architecture."""
        canonical = self._resolve_arch(arch)
        return list(self._archs[canonical].MODEL_CONFIGS.keys())

    def list_all(self) -> Dict[str, List[str]]:
        """Return {arch: [variants]} for all registered architectures."""
        self._scan()
        return {
            arch: list(mod.MODEL_CONFIGS.keys())
            for arch, mod in self._archs.items()
        }

    def build(self, arch: str, name: str, num_classes: int = 4,
              in_channels: int = 3, **overrides):
        """Build a model from an architecture and variant name.

        Args:
            arch: architecture name, for example "MvANet" or "MvANet_opt"
            name: variant name, for example "mvanet_1_tiny"
            num_classes: number of output classes
            in_channels: number of input channels
            **overrides: values overriding MODEL_CONFIGS

        Returns:
            nn.Module instance
        """
        canonical = self._resolve_arch(arch)
        mod = self._archs[canonical]
        if name not in mod.MODEL_CONFIGS:
            raise ValueError(
                f"Variant '{name}' was not found in '{canonical}'. "
                f"Available: {list(mod.MODEL_CONFIGS.keys())}"
            )
        return mod.build_model(name, num_classes=num_classes,
                               in_channels=in_channels, **overrides)


# Singleton
registry = _ModelRegistry()


# ------------------------------------------------------------------
# Convenience function (backward-compatible)
# ------------------------------------------------------------------
def build_model(arch: str = "MvANet", name: str = "mvanet_3",
                num_classes: int = 4, in_channels: int = 3,
                **overrides):
    """Shortcut: models.build_model(arch, name, ...)"""
    return registry.build(arch, name, num_classes=num_classes,
                          in_channels=in_channels, **overrides)
