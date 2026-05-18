from __future__ import annotations

import os
import platform
import random
import subprocess
import sys
from typing import Any, Dict, Optional

import numpy as np
import torch


def seed_everything(seed: int, deterministic: bool = True) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = not deterministic

    try:
        torch.use_deterministic_algorithms(deterministic, warn_only=False)
    except TypeError:
        torch.use_deterministic_algorithms(deterministic)


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_generator(seed: int) -> torch.Generator:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def get_rng_state(train_generator: Optional[torch.Generator] = None) -> Dict[str, Any]:
    state: Dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    if train_generator is not None:
        state["train_generator"] = train_generator.get_state()
    return state


def set_rng_state(state: Optional[Dict[str, Any]],
                  train_generator: Optional[torch.Generator] = None) -> None:
    if not state:
        return
    if "python" in state:
        random.setstate(state["python"])
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    if "torch" in state:
        torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])
    if train_generator is not None and "train_generator" in state:
        train_generator.set_state(state["train_generator"])


def _git_value(args: list[str]) -> Optional[str]:
    """Return a git metadata value, or None when git is unavailable."""

    try:
        result = subprocess.run(
            ["git", *args],
            cwd=os.getcwd(),
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except Exception:
        return None

    value = result.stdout.strip()
    return value or None


def collect_environment() -> Dict[str, Any]:
    """Collect runtime metadata saved beside each experiment result."""

    env: Dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version()
        if hasattr(torch.backends, "cudnn") else None,
        "git_commit": _git_value(["rev-parse", "HEAD"]),
        "git_dirty": bool(_git_value(["status", "--short"])),
    }

    try:
        import torchvision
        env["torchvision"] = torchvision.__version__
    except Exception:
        env["torchvision"] = None

    try:
        import PIL
        env["pillow"] = PIL.__version__
    except Exception:
        env["pillow"] = None

    try:
        import yaml
        env["pyyaml"] = yaml.__version__
    except Exception:
        env["pyyaml"] = None

    try:
        import matplotlib
        env["matplotlib"] = matplotlib.__version__
    except Exception:
        env["matplotlib"] = None

    env["numpy"] = np.__version__
    env["devices"] = []
    if torch.cuda.is_available():
        for idx in range(torch.cuda.device_count()):
            env["devices"].append(torch.cuda.get_device_name(idx))

    return env