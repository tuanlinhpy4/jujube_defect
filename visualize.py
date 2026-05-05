"""
Visualization Utilities
=======================
- GradCAM heatmap (standalone script + importable)
- Confusion matrix heatmap
- Training loss/accuracy curves
- Per-class metric bar charts
- Feature map visualization
- SE attention weight visualization

Standalone GradCAM usage:
    # Auto-detect paths from dataset/model/seed (recommended)
    python visualize.py --dataset jujube --model mobilenet_v3_small --seed 1

    # Explicit paths
    python visualize.py \
        --weights runs/jujube/mobilenet_v3_small/cross_entropy/seed_1/best.pth \
        --data-root data/jujube

    # Multiple samples per class
    python visualize.py --dataset jujube --model resnet18 --seed 1 --n-samples 3
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("Warning: matplotlib not installed. Visualization disabled.")


def _ensure_dir(path: str):
    if path:
        os.makedirs(path, exist_ok=True)


# ===========================================================================
# Auto-detect helpers
# ===========================================================================

def _auto_weights(dataset: str, model: str, loss: str, seed: int,
                  runs_root: str = "runs") -> str:
    """Build the canonical best.pth path from run metadata."""
    return os.path.join(runs_root, dataset, model, loss, f"seed_{seed}", "best.pth")


def _auto_data_root(dataset: str, data_root_base: str = "data") -> str:
    """Build the canonical data root path from dataset name."""
    return os.path.join(data_root_base, dataset)


def _resolve_paths(args) -> Tuple[str, str]:
    """Return (weights_path, data_root), resolving auto-detect vs explicit."""
    weights = args.weights
    data_root = args.data_root

    if weights is None:
        if args.dataset is None or args.model is None:
            raise SystemExit(
                "ERROR: provide either --weights or both --dataset and --model.\n"
                "Run with -h for usage examples."
            )
        weights = _auto_weights(
            args.dataset, args.model, args.loss, args.seed, args.runs_root
        )
        print(f"  Auto-detected weights  : {weights}")

    if data_root is None:
        if args.dataset is None:
            raise SystemExit(
                "ERROR: provide either --data-root or --dataset.\n"
                "Run with -h for usage examples."
            )
        data_root = _auto_data_root(args.dataset, args.data_root_base)
        print(f"  Auto-detected data-root: {data_root}")

    if not os.path.isfile(weights):
        raise SystemExit(f"ERROR: weights file not found: {weights}")
    if not os.path.isdir(data_root):
        raise SystemExit(f"ERROR: data-root directory not found: {data_root}")

    return weights, data_root


# ===========================================================================
# GradCAM
# ===========================================================================

class GradCAM:
    """Gradient-weighted Class Activation Mapping.

    Works with any CNN that has at least one Conv2d layer.
    Hooks onto the *last* Conv2d by default, which gives the best spatial
    resolution trade-off for standard classification CNNs (ResNet, MobileNet,
    EfficientNet, ShuffleNet, SqueezeNet, MNASNet).

    Usage:
        cam = GradCAM(model)
        heatmap, pred_class = cam(image_tensor)   # image_tensor: (1,C,H,W)
        cam.remove_hooks()
    """

    def __init__(self, model, target_layer=None):
        import torch.nn as nn
        self.model = model
        self._activations: Optional[np.ndarray] = None
        self._gradients: Optional[np.ndarray] = None
        self._hooks: list = []

        if target_layer is None:
            last_conv = None
            for m in model.modules():
                if isinstance(m, nn.Conv2d):
                    last_conv = m
            if last_conv is None:
                raise ValueError("No Conv2d layer found in model.")
            target_layer = last_conv

        self._register_hooks(target_layer)

    def _register_hooks(self, layer):
        def fwd(module, inp, out):
            self._activations = out.detach().cpu().numpy()

        def bwd(module, grad_in, grad_out):
            self._gradients = grad_out[0].detach().cpu().numpy()

        self._hooks.append(layer.register_forward_hook(fwd))
        self._hooks.append(layer.register_full_backward_hook(bwd))

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def __call__(self, input_tensor, target_class: Optional[int] = None
                 ) -> Tuple[np.ndarray, int]:
        """
        Args:
            input_tensor: (1, C, H, W) float tensor on model's device.
            target_class: class index to explain. None → argmax (predicted class).

        Returns:
            heatmap: (H, W) numpy array in [0, 1], same spatial size as input.
            target_class: the class index that was explained.
        """
        import torch
        import torch.nn.functional as F

        self.model.eval()
        self.model.zero_grad()

        output = self.model(input_tensor)

        if target_class is None:
            target_class = int(output.argmax(dim=1).item())

        score = output[0, target_class]
        score.backward()

        weights = self._gradients.mean(axis=(2, 3), keepdims=True)
        cam = (weights * self._activations).sum(axis=1, keepdims=True)
        cam = np.maximum(cam, 0)

        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        h, w = input_tensor.shape[2], input_tensor.shape[3]
        cam_tensor = torch.from_numpy(cam).float()
        cam_resized = F.interpolate(
            cam_tensor, size=(h, w), mode="bilinear", align_corners=False
        )
        return cam_resized[0, 0].numpy(), target_class


def _load_image_tensor(img_path: str, input_size: int, normalize_mean, normalize_std):
    """Load a PIL image and return (display_array, model_tensor)."""
    from PIL import Image
    import torch
    from torchvision import transforms

    img = Image.open(img_path).convert("RGB")
    display = np.array(img.resize((input_size, input_size)))

    tfm = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=normalize_mean, std=normalize_std),
    ])
    tensor = tfm(img).unsqueeze(0)
    return display, tensor


def _overlay_heatmap(image_rgb: np.ndarray, heatmap: np.ndarray,
                     alpha: float = 0.45) -> np.ndarray:
    """Overlay a [0,1] heatmap on an RGB image (uint8)."""
    import matplotlib.cm as cm
    colored = (cm.jet(heatmap)[:, :, :3] * 255).astype(np.uint8)
    blended = (image_rgb * (1 - alpha) + colored * alpha).clip(0, 255).astype(np.uint8)
    return blended


def plot_gradcam_grid(
    model,
    class_sample_paths: Dict[str, str],
    class_names: List[str],
    input_size: int = 224,
    normalize_mean: tuple = (0.485, 0.456, 0.406),
    normalize_std: tuple = (0.229, 0.224, 0.225),
    save_path: Optional[str] = None,
    device=None,
):
    """Generate GradCAM for one sample per class and save as a grid figure.

    Each column = one class: original image | GradCAM overlay.
    """
    if not HAS_MPL:
        return

    import torch

    if device is None:
        device = next(model.parameters()).device

    cam_engine = GradCAM(model)
    n = len(class_names)
    fig, axes = plt.subplots(2, n, figsize=(3 * n, 6))
    if n == 1:
        axes = axes.reshape(2, 1)

    fig.suptitle("GradCAM — Last Conv Layer", fontsize=13, y=1.01)

    for j, cls in enumerate(class_names):
        img_path = class_sample_paths.get(cls)
        if img_path is None or not os.path.isfile(img_path):
            for row in range(2):
                axes[row, j].axis("off")
                axes[row, j].set_title(f"{cls}\n(no sample)", fontsize=8)
            continue

        display, tensor = _load_image_tensor(
            img_path, input_size, normalize_mean, normalize_std)
        tensor = tensor.to(device)

        with torch.enable_grad():
            heatmap, pred_idx = cam_engine(tensor)

        pred_name = class_names[pred_idx] if pred_idx < len(class_names) else str(pred_idx)
        overlay = _overlay_heatmap(display, heatmap)

        axes[0, j].imshow(display)
        axes[0, j].set_title(f"{cls}", fontsize=9, fontweight="bold")
        axes[0, j].axis("off")

        axes[1, j].imshow(overlay)
        color = "green" if pred_name == cls else "red"
        axes[1, j].set_title(f"pred: {pred_name}", fontsize=8, color=color)
        axes[1, j].axis("off")

    cam_engine.remove_hooks()
    plt.tight_layout()

    if save_path:
        _ensure_dir(os.path.dirname(save_path))
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved GradCAM grid: {save_path}")
    plt.close()


def _pick_samples(data_root: str, class_names: List[str],
                  n_per_class: int = 1) -> Dict[str, List[str]]:
    """Pick up to n_per_class image paths per class from data_root."""
    EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    result: Dict[str, List[str]] = {}
    for cls in class_names:
        cls_dir = Path(data_root) / cls
        if not cls_dir.is_dir():
            result[cls] = []
            continue
        imgs = sorted([p for p in cls_dir.iterdir()
                       if p.suffix.lower() in EXTS])[:n_per_class]
        result[cls] = [str(p) for p in imgs]
    return result


# ===========================================================================
# Confusion Matrix
# ===========================================================================

def plot_confusion_matrix(cm: np.ndarray, class_names: List[str],
                          save_path: Optional[str] = None,
                          title: str = "Confusion Matrix",
                          normalize: bool = False):
    if not HAS_MPL:
        return
    if normalize:
        cm_plot = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-6) * 100
        fmt = ".1f"
        title += " (Normalized %)"
    else:
        cm_plot = cm
        fmt = "d"

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm_plot, interpolation='nearest', cmap='Blues')
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=range(len(class_names)), yticks=range(len(class_names)),
           xticklabels=class_names, yticklabels=class_names,
           title=title, ylabel='True Label', xlabel='Predicted Label')
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    thresh = cm_plot.max() / 2.0
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            val = cm_plot[i, j]
            text = f"{val:{fmt}}" if isinstance(val, float) else f"{val}"
            ax.text(j, i, text, ha="center", va="center",
                    color="white" if val > thresh else "black", fontsize=10)
    plt.tight_layout()
    if save_path:
        _ensure_dir(os.path.dirname(save_path))
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved confusion matrix: {save_path}")
    plt.close()


# ===========================================================================
# Training Curves
# ===========================================================================

def plot_training_curves(history: Dict[str, List[float]],
                         save_path: Optional[str] = None):
    if not HAS_MPL:
        return

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    ax = axes[0]
    if "train_loss" in history:
        ax.plot(history["train_loss"], label="Train", alpha=0.8)
    if "val_loss" in history:
        ax.plot(history["val_loss"], label="Val", alpha=0.8)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss"); ax.set_title("Loss")
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1]
    if "train_acc" in history:
        ax.plot(history["train_acc"], label="Train Acc", alpha=0.8)
    if "val_acc" in history:
        ax.plot(history["val_acc"], label="Val Acc", alpha=0.8)
    if "val_error" in history:
        ax.plot(history["val_error"], label="Val Error", alpha=0.8, linestyle='--')
    ax.set_xlabel("Epoch"); ax.set_ylabel("%"); ax.set_title("Accuracy / Error")
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[2]
    if "lr" in history:
        ax.plot(history["lr"], color='green', alpha=0.8)
        ax.set_xlabel("Epoch"); ax.set_ylabel("LR")
        ax.set_title("Learning Rate Schedule")
        ax.set_yscale('log'); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        _ensure_dir(os.path.dirname(save_path))
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved training curves: {save_path}")
    plt.close()


# ===========================================================================
# Per-Class Metrics Bar Chart
# ===========================================================================

def plot_per_class_metrics(per_class: Dict[str, Dict[str, float]],
                           save_path: Optional[str] = None,
                           title: str = "Per-Class Metrics"):
    if not HAS_MPL:
        return
    classes = list(per_class.keys())
    precision = [per_class[c]["precision"] for c in classes]
    recall = [per_class[c]["recall"] for c in classes]
    f1 = [per_class[c]["f1"] for c in classes]
    x = np.arange(len(classes))
    width = 0.25
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width, recall, width, label='Recall', color='#2196F3')
    ax.bar(x, precision, width, label='Precision', color='#FF9800')
    ax.bar(x + width, f1, width, label='F1', color='#4CAF50')
    ax.set_ylabel('Score (%)'); ax.set_title(title)
    ax.set_xticks(x); ax.set_xticklabels(classes)
    ax.legend(); ax.set_ylim(0, 105); ax.grid(axis='y', alpha=0.3)
    for bars in ax.containers:
        ax.bar_label(bars, fmt='%.1f', fontsize=7, padding=2)
    plt.tight_layout()
    if save_path:
        _ensure_dir(os.path.dirname(save_path))
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved per-class metrics: {save_path}")
    plt.close()


# ===========================================================================
# Feature Maps
# ===========================================================================

def plot_feature_maps(features: Dict, class_name: str = "",
                      save_path: Optional[str] = None, max_channels: int = 16):
    if not HAS_MPL:
        return
    stages = []
    if "shallow" in features:
        stages.append(("Shallow Output", features["shallow"]))
    if "visual_layers" in features:
        for i, layer_receptors in enumerate(features["visual_layers"]):
            for j, receptor_out in enumerate(layer_receptors):
                stages.append((f"VL{i+1} VR{j+1}", receptor_out))
    if not stages:
        return
    n_stages = len(stages)
    fig, axes = plt.subplots(1, n_stages, figsize=(3 * n_stages, 3))
    if n_stages == 1:
        axes = [axes]
    fig.suptitle(f"Feature Maps - {class_name}", fontsize=14, y=1.02)
    for ax, (name, feat) in zip(axes, stages):
        feat_map = feat[0].detach().cpu() if feat.dim() == 4 else feat.detach().cpu()
        mean_map = feat_map[:max_channels].mean(dim=0).numpy()
        ax.imshow(mean_map, cmap='hot', interpolation='nearest')
        ax.set_title(name, fontsize=8); ax.axis('off')
    plt.tight_layout()
    if save_path:
        _ensure_dir(os.path.dirname(save_path))
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved feature map: {save_path}")
    plt.close()


def plot_feature_maps_detailed(features: Dict, class_name: str = "",
                                save_path: Optional[str] = None):
    if not HAS_MPL:
        return
    visual_layers = features.get("visual_layers", [])
    if not visual_layers:
        return
    n_layers = len(visual_layers)
    max_receptors = max(len(vl) for vl in visual_layers)
    fig, axes = plt.subplots(n_layers, max_receptors,
                             figsize=(4 * max_receptors, 3 * n_layers))
    if n_layers == 1:
        axes = axes.reshape(1, -1)
    if max_receptors == 1:
        axes = axes.reshape(-1, 1)
    fig.suptitle(f"Visual Receptor Outputs - {class_name}", fontsize=14)
    for i, layer_receptors in enumerate(visual_layers):
        for j in range(max_receptors):
            ax = axes[i, j]
            if j < len(layer_receptors):
                feat = layer_receptors[j][0].detach().cpu()
                ax.imshow(feat.mean(dim=0).numpy(), cmap='hot', interpolation='nearest')
                ax.set_title(f"VL{i+1} VR{j+1}\n({feat.shape[0]}ch)", fontsize=8)
            ax.axis('off')
    plt.tight_layout()
    if save_path:
        _ensure_dir(os.path.dirname(save_path))
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved detailed feature map: {save_path}")
    plt.close()


# ===========================================================================
# SE / ECA Attention Weights
# ===========================================================================

def plot_attention_weights(attn_a: np.ndarray, attn_b: np.ndarray,
                           save_path: Optional[str] = None):
    if not HAS_MPL:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 3))
    axes[0].bar(range(len(attn_a)), attn_a, color='#2196F3', alpha=0.7)
    axes[0].set_title("Attention Layer A Weights")
    axes[0].set_xlabel("Channel"); axes[0].set_ylabel("Weight")
    axes[1].bar(range(len(attn_b)), attn_b, color='#FF9800', alpha=0.7)
    axes[1].set_title("Attention Layer B Weights")
    axes[1].set_xlabel("Channel"); axes[1].set_ylabel("Weight")
    plt.tight_layout()
    if save_path:
        _ensure_dir(os.path.dirname(save_path))
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved attention weights: {save_path}")
    plt.close()


# ===========================================================================
# Standalone CLI
# ===========================================================================

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate GradCAM heatmaps from a saved checkpoint.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # Recommended — auto-detect paths:\n"
            "  python visualize.py --dataset jujube --model mobilenet_v3_small --seed 1\n\n"
            "  # Explicit paths:\n"
            "  python visualize.py \\\n"
            "      --weights runs/jujube/mobilenet_v3_small/cross_entropy/seed_1/best.pth \\\n"
            "      --data-root data/jujube\n\n"
            "  # Multiple samples per class:\n"
            "  python visualize.py --dataset jujube --model resnet18 --seed 1 --n-samples 3\n"
        ),
    )

    # ── Auto-detect group ────────────────────────────────────────────────
    auto = p.add_argument_group(
        "auto-detect",
        "Resolve --weights and --data-root automatically from the standard "
        "runs/<dataset>/<model>/<loss>/seed_<n>/ and data/<dataset>/ layout.",
    )
    auto.add_argument("--dataset", type=str, default=None,
                      help="Dataset name, e.g. jujube or jujube_ours")
    auto.add_argument("--model", type=str, default=None,
                      help="Model name, e.g. mobilenet_v3_small, resnet18")
    auto.add_argument("--seed", type=int, default=1,
                      help="Training seed index (default: 1)")
    auto.add_argument("--loss", type=str, default="cross_entropy",
                      help="Loss name used during training (default: cross_entropy)")
    auto.add_argument("--runs-root", type=str, default="runs",
                      help="Root of the runs directory (default: runs/)")
    auto.add_argument("--data-root-base", type=str, default="data",
                      help="Parent of dataset directories (default: data/)")

    # ── Explicit paths group ─────────────────────────────────────────────
    explicit = p.add_argument_group(
        "explicit paths",
        "Override auto-detected paths when needed.",
    )
    explicit.add_argument("--weights", type=str, default=None,
                          help="Direct path to best.pth checkpoint")
    explicit.add_argument("--data-root", type=str, default=None,
                          help="Direct path to dataset root (class sub-folders inside)")

    # ── Common options ───────────────────────────────────────────────────
    p.add_argument("--class-names", nargs="+", default=None,
                   help="Override class names (default: inferred alphabetically from folders)")
    p.add_argument("--output-dir", default=None,
                   help="Where to save figures (default: <checkpoint_dir>/gradcam/)")
    p.add_argument("--input-size", type=int, default=224)
    p.add_argument("--n-samples", type=int, default=1,
                   help="Images per class to visualize (default: 1)")
    p.add_argument("--device", default="auto")
    return p.parse_args()


def main():
    import sys
    import torch
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from models import build_model

    args = _parse_args()

    # ── device ──────────────────────────────────────────────────────────
    device = (torch.device("cuda" if torch.cuda.is_available() else "cpu")
              if args.device == "auto" else torch.device(args.device))

    # ── resolve weights & data-root ─────────────────────────────────────
    weights_path, data_root = _resolve_paths(args)

    # ── load checkpoint ─────────────────────────────────────────────────
    ckpt = torch.load(weights_path, map_location=device)
    cfg = ckpt.get("config", {})
    model_name = cfg.get("model")
    num_classes = cfg.get("num_classes")

    if model_name is None or num_classes is None:
        raise SystemExit(
            "ERROR: checkpoint missing 'config.model' or 'config.num_classes'.\n"
            "Ensure the checkpoint was saved by this training script."
        )

    # ── class names ──────────────────────────────────────────────────────
    if args.class_names:
        class_names = args.class_names
    else:
        class_names = sorted([
            d.name for d in Path(data_root).iterdir() if d.is_dir()
        ])
        if len(class_names) != num_classes:
            raise SystemExit(
                f"ERROR: found {len(class_names)} class folders in '{data_root}' "
                f"but checkpoint says num_classes={num_classes}.\n"
                f"Use --class-names to specify them explicitly."
            )

    print(f"\n  Model      : {model_name}")
    print(f"  Classes    : {class_names}")
    print(f"  Device     : {device}")
    print(f"  Weights    : {weights_path}")
    print(f"  Data root  : {data_root}")

    # ── build & load model ───────────────────────────────────────────────
    model = build_model(model_name, num_classes).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # ── output directory ─────────────────────────────────────────────────
    out_dir = args.output_dir or str(Path(weights_path).parent / "gradcam")
    os.makedirs(out_dir, exist_ok=True)

    # ── pick samples ─────────────────────────────────────────────────────
    all_samples = _pick_samples(data_root, class_names, n_per_class=args.n_samples)

    # ── GradCAM grid: one column per class, first sample only ────────────
    single_samples = {cls: paths[0] for cls, paths in all_samples.items() if paths}
    plot_gradcam_grid(
        model=model,
        class_sample_paths=single_samples,
        class_names=class_names,
        input_size=args.input_size,
        save_path=os.path.join(out_dir, "gradcam_grid.png"),
        device=device,
    )

    # ── per-class multi-sample grids (only when n_samples > 1) ───────────
    # Each saved as a single-column figure (one class, one sample per file).
    # FIX: pass class_names=[cls] so the grid has exactly 1 column instead
    # of N columns where N-1 are empty "no sample" placeholders.
    if args.n_samples > 1:
        for cls, paths in all_samples.items():
            if not paths:
                continue
            for k, img_path in enumerate(paths):
                plot_gradcam_grid(
                    model=model,
                    class_sample_paths={cls: img_path},
                    class_names=[cls],          # single-column grid
                    input_size=args.input_size,
                    save_path=os.path.join(out_dir, f"gradcam_{cls}_{k + 1}.png"),
                    device=device,
                )

    print(f"\n  GradCAM figures saved to: {out_dir}")


if __name__ == "__main__":
    main()
