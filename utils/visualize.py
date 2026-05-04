"""
Visualization Utilities
=======================
- Feature map visualization (matching paper Fig. 8)
- Confusion matrix heatmap
- Training loss/accuracy curves
- Per-class metric bar charts
- SE attention weight visualization
"""

import os
import numpy as np
from typing import Dict, List, Optional

try:
    import matplotlib
    matplotlib.use('Agg')  # non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("Warning: matplotlib not installed. Visualization disabled.")


def _ensure_dir(path: str):
    if path:
        os.makedirs(path, exist_ok=True)


# ===========================================================================
# Feature Map Visualization (Paper Fig. 8)
# ===========================================================================

def plot_feature_maps(features: Dict, class_name: str = "",
                      save_path: Optional[str] = None, max_channels: int = 16):
    """Visualize feature maps from each stage (like paper Fig. 8).

    Args:
        features: dict from model.forward_with_features()
        class_name: label for the input image
        save_path: where to save the figure
        max_channels: max channels to show per feature map
    """
    if not HAS_MPL:
        return

    stages = []

    # Shallow output
    if "shallow" in features:
        stages.append(("Shallow Output", features["shallow"]))

    # Visual layer outputs
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
        # Take first sample, average across channels
        if feat.dim() == 4:
            feat_map = feat[0].detach().cpu()
        else:
            feat_map = feat.detach().cpu()

        # Show mean activation across channels
        mean_map = feat_map[:max_channels].mean(dim=0).numpy()
        im = ax.imshow(mean_map, cmap='hot', interpolation='nearest')
        ax.set_title(name, fontsize=8)
        ax.axis('off')

    plt.tight_layout()
    if save_path:
        _ensure_dir(os.path.dirname(save_path))
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved feature map: {save_path}")
    plt.close()


def plot_feature_maps_detailed(features: Dict, class_name: str = "",
                                save_path: Optional[str] = None):
    """Detailed feature map visualization showing individual VR channels.

    Closer to paper Fig. 8 layout: rows = visual layers, cols = visual receptors.
    """
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
                mean_map = feat.mean(dim=0).numpy()
                ax.imshow(mean_map, cmap='hot', interpolation='nearest')
                ax.set_title(f"VL{i+1} VR{j+1}\n({feat.shape[0]}ch)",
                             fontsize=8)
            ax.axis('off')

    plt.tight_layout()
    if save_path:
        _ensure_dir(os.path.dirname(save_path))
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved detailed feature map: {save_path}")
    plt.close()


# ===========================================================================
# Confusion Matrix
# ===========================================================================

def plot_confusion_matrix(cm: np.ndarray, class_names: List[str],
                          save_path: Optional[str] = None,
                          title: str = "Confusion Matrix",
                          normalize: bool = False):
    """Plot confusion matrix as heatmap."""
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

    ax.set(xticks=range(len(class_names)),
           yticks=range(len(class_names)),
           xticklabels=class_names,
           yticklabels=class_names,
           title=title,
           ylabel='True Label',
           xlabel='Predicted Label')

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    # Text annotations
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
    """Plot loss and accuracy curves over training.

    Args:
        history: dict with keys like 'train_loss', 'val_loss',
                 'train_acc', 'val_acc', 'lr'
    """
    if not HAS_MPL:
        return

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    # Loss
    ax = axes[0]
    if "train_loss" in history:
        ax.plot(history["train_loss"], label="Train", alpha=0.8)
    if "val_loss" in history:
        ax.plot(history["val_loss"], label="Val", alpha=0.8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Accuracy / Error Rate
    ax = axes[1]
    if "train_acc" in history:
        ax.plot(history["train_acc"], label="Train Acc", alpha=0.8)
    if "val_acc" in history:
        ax.plot(history["val_acc"], label="Val Acc", alpha=0.8)
    if "val_error" in history:
        ax.plot(history["val_error"], label="Val Error", alpha=0.8,
                linestyle='--')
    ax.set_xlabel("Epoch")
    ax.set_ylabel("%")
    ax.set_title("Accuracy / Error")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Learning Rate
    ax = axes[2]
    if "lr" in history:
        ax.plot(history["lr"], color='green', alpha=0.8)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("LR")
        ax.set_title("Learning Rate Schedule")
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)

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
    """Bar chart of precision, recall, F1 per class (paper Table 4 style)."""
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

    ax.set_ylabel('Score (%)')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.legend()
    ax.set_ylim(0, 105)
    ax.grid(axis='y', alpha=0.3)

    # Value labels
    for bars in ax.containers:
        ax.bar_label(bars, fmt='%.1f', fontsize=7, padding=2)

    plt.tight_layout()
    if save_path:
        _ensure_dir(os.path.dirname(save_path))
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved per-class metrics: {save_path}")
    plt.close()


# ===========================================================================
# SE Attention Weights
# ===========================================================================

def plot_attention_weights(attn_a: np.ndarray, attn_b: np.ndarray,
                           save_path: Optional[str] = None):
    """Visualize SE attention weights from both attention layers."""
    if not HAS_MPL:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 3))

    axes[0].bar(range(len(attn_a)), attn_a, color='#2196F3', alpha=0.7)
    axes[0].set_title("Attention Layer A Weights")
    axes[0].set_xlabel("Channel")
    axes[0].set_ylabel("Weight")

    axes[1].bar(range(len(attn_b)), attn_b, color='#FF9800', alpha=0.7)
    axes[1].set_title("Attention Layer B Weights")
    axes[1].set_xlabel("Channel")
    axes[1].set_ylabel("Weight")

    plt.tight_layout()
    if save_path:
        _ensure_dir(os.path.dirname(save_path))
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved attention weights: {save_path}")
    plt.close()
