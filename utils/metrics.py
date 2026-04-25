"""
Evaluation Metrics
==================
Computes accuracy, precision, recall, F1, and confusion matrix.
Matches the paper's Table 4 format (per-class precision & recall).
"""

import torch
import numpy as np
from typing import Dict, List, Optional


class MetricTracker:
    """Tracks predictions and computes metrics after an epoch."""

    def __init__(self, num_classes: int, class_names: Optional[List[str]] = None):
        self.num_classes = num_classes
        self.class_names = class_names or [str(i) for i in range(num_classes)]
        self.reset()

    def reset(self):
        self.all_preds = []
        self.all_labels = []
        self.running_loss = 0.0
        self.num_samples = 0

    def update(self, preds: torch.Tensor, labels: torch.Tensor,
               loss: float = 0.0, batch_size: int = 1):
        """Update with batch predictions.

        Args:
            preds:  predicted class indices (B,)
            labels: ground truth class indices (B,)
            loss:   batch loss value
            batch_size: number of samples in batch
        """
        self.all_preds.extend(preds.cpu().numpy())
        self.all_labels.extend(labels.cpu().numpy())
        self.running_loss += loss * batch_size
        self.num_samples += batch_size

    @property
    def avg_loss(self) -> float:
        return self.running_loss / max(self.num_samples, 1)

    @property
    def accuracy(self) -> float:
        preds = np.array(self.all_preds)
        labels = np.array(self.all_labels)
        return (preds == labels).mean() * 100.0

    @property
    def error_rate(self) -> float:
        return 100.0 - self.accuracy

    def confusion_matrix(self) -> np.ndarray:
        """Return NxN confusion matrix."""
        cm = np.zeros((self.num_classes, self.num_classes), dtype=int)
        for pred, label in zip(self.all_preds, self.all_labels):
            cm[label][pred] += 1
        return cm

    def per_class_metrics(self) -> Dict[str, Dict[str, float]]:
        """Compute per-class precision, recall, F1 (matching paper Table 4)."""
        cm = self.confusion_matrix()
        metrics = {}

        for i, name in enumerate(self.class_names):
            tp = cm[i, i]
            fp = cm[:, i].sum() - tp
            fn = cm[i, :].sum() - tp

            precision = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0.0
            f1 = (2 * precision * recall / (precision + recall)
                  if (precision + recall) > 0 else 0.0)

            metrics[name] = {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": int(cm[i, :].sum()),
            }

        return metrics

    def summary(self) -> Dict[str, float]:
        """Return summary dict for logging."""
        per_class = self.per_class_metrics()
        avg_precision = np.mean([m["precision"] for m in per_class.values()])
        avg_recall = np.mean([m["recall"] for m in per_class.values()])
        avg_f1 = np.mean([m["f1"] for m in per_class.values()])

        return {
            "loss": self.avg_loss,
            "accuracy": self.accuracy,
            "error_rate": self.error_rate,
            "avg_precision": avg_precision,
            "avg_recall": avg_recall,
            "avg_f1": avg_f1,
        }

    def print_report(self):
        """Print formatted classification report (paper Table 4 style)."""
        per_class = self.per_class_metrics()
        cm = self.confusion_matrix()

        print(f"\n{'='*60}")
        print(f"  Overall Accuracy: {self.accuracy:.2f}% "
              f"(Error: {self.error_rate:.2f}%)")
        print(f"  Average Loss: {self.avg_loss:.4f}")
        print(f"{'='*60}")
        print(f"  {'Category':<12} {'Recall':>8} {'Precision':>10} "
              f"{'F1':>8} {'Support':>8}")
        print(f"  {'-'*48}")

        for name, m in per_class.items():
            print(f"  {name:<12} {m['recall']:>7.2f}% {m['precision']:>9.2f}% "
                  f"{m['f1']:>7.2f}% {m['support']:>8d}")

        print(f"{'='*60}")

        # Confusion matrix
        print(f"\n  Confusion Matrix:")
        header = "  " + " " * 12 + "".join(f"{n:>8s}" for n in self.class_names)
        print(header)
        for i, name in enumerate(self.class_names):
            row = "  " + f"{name:<12}" + "".join(f"{cm[i, j]:>8d}" for j in range(self.num_classes))
            print(row)
        print()
