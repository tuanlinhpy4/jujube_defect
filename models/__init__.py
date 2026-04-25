"""
Dynamic Model Registry
======================
Tự động quét tất cả subfolder trong models/ và đăng ký chúng.

Cách thêm model mới:
    1. Tạo folder  models/MyModel/
    2. Đặt vào đó  blocks.py, mvanet.py (hoặc tên bất kỳ)
    3. Tạo __init__.py export 2 symbol: build_model, MODEL_CONFIGS
    4. Done — registry sẽ tự phát hiện.

Sử dụng:
    from models import registry
    model = registry.build("MvANet", "mvanet_2", num_classes=4)
    # hoặc
    from models import build_model
    model = build_model(arch="MvANet", name="mvanet_2", num_classes=4)
"""

import importlib
import os
from pathlib import Path
from typing import Dict, Any, List, Optional


class _ModelRegistry:
    """Singleton registry quản lý tất cả model architectures."""

    def __init__(self):
        self._archs: Dict[str, Any] = {}  # arch_name -> module
        self._scanned = False

    # ------------------------------------------------------------------
    # Quét tự động
    # ------------------------------------------------------------------
    def _scan(self):
        """Quét models/ để tìm các sub-package có build_model & MODEL_CONFIGS."""
        if self._scanned:
            return
        models_dir = Path(__file__).parent
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
                if hasattr(mod, "build_model") and hasattr(mod, "MODEL_CONFIGS"):
                    self._archs[entry.name] = mod
            except Exception as e:
                print(f"[registry] Skip models/{entry.name}: {e}")
        self._scanned = True

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def list_archs(self) -> List[str]:
        """Trả về danh sách architecture đã đăng ký."""
        self._scan()
        return list(self._archs.keys())

    def list_variants(self, arch: str) -> List[str]:
        """Trả về danh sách variant của 1 architecture."""
        self._scan()
        if arch not in self._archs:
            raise ValueError(
                f"Architecture '{arch}' không tìm thấy. "
                f"Có sẵn: {self.list_archs()}"
            )
        return list(self._archs[arch].MODEL_CONFIGS.keys())

    def list_all(self) -> Dict[str, List[str]]:
        """Trả về dict {arch: [variants]}."""
        self._scan()
        return {
            arch: list(mod.MODEL_CONFIGS.keys())
            for arch, mod in self._archs.items()
        }

    def build(self, arch: str, name: str, num_classes: int = 4,
              in_channels: int = 3, **overrides):
        """Build model từ tên architecture + variant.

        Args:
            arch: tên folder (e.g. "MvANet", "MvANet_opt")
            name: tên variant (e.g. "mvanet_1_tiny", "mvanet_2")
            num_classes: số lớp phân loại
            in_channels: số kênh đầu vào
            **overrides: ghi đè bất kỳ tham số nào trong MODEL_CONFIGS

        Returns:
            nn.Module instance
        """
        self._scan()
        if arch not in self._archs:
            raise ValueError(
                f"Architecture '{arch}' không tìm thấy. "
                f"Có sẵn: {self.list_archs()}"
            )
        mod = self._archs[arch]
        if name not in mod.MODEL_CONFIGS:
            raise ValueError(
                f"Variant '{name}' không có trong '{arch}'. "
                f"Có sẵn: {list(mod.MODEL_CONFIGS.keys())}"
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
