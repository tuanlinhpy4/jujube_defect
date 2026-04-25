"""
Training Logger
===============
Logs metrics to console and JSON file. Tracks training history.
"""

import json
import os
import time
from typing import Dict, List, Optional
from collections import defaultdict


class TrainingLogger:
    """Logs training metrics per epoch."""

    def __init__(self, save_dir: str, run_name: str = "run"):
        self.save_dir = save_dir
        self.run_name = run_name
        os.makedirs(save_dir, exist_ok=True)

        self.history: Dict[str, List[float]] = defaultdict(list)
        self.best_error = float('inf')
        self.best_epoch = 0
        self.start_time = time.time()

    def log_epoch(self, epoch: int, train_metrics: Dict, val_metrics: Dict,
                  lr: float):
        """Log metrics for one epoch."""
        self.history["train_loss"].append(train_metrics.get("loss", 0))
        self.history["train_acc"].append(train_metrics.get("accuracy", 0))
        self.history["val_loss"].append(val_metrics.get("loss", 0))
        self.history["val_acc"].append(val_metrics.get("accuracy", 0))
        self.history["val_error"].append(val_metrics.get("error_rate", 0))
        self.history["lr"].append(lr)

        # Track best
        val_error = val_metrics.get("error_rate", float('inf'))
        is_best = val_error < self.best_error
        if is_best:
            self.best_error = val_error
            self.best_epoch = epoch

        elapsed = time.time() - self.start_time

        print(f"  Epoch [{epoch:3d}] "
              f"Train Loss: {train_metrics.get('loss', 0):.4f} | "
              f"Train Acc: {train_metrics.get('accuracy', 0):.2f}% | "
              f"Val Acc: {val_metrics.get('accuracy', 0):.2f}% | "
              f"Val Error: {val_error:.2f}% | "
              f"LR: {lr:.6f} | "
              f"Time: {elapsed:.0f}s"
              f"{' *BEST*' if is_best else ''}")

        return is_best

    def save_history(self):
        """Save training history to JSON."""
        path = os.path.join(self.save_dir, f"{self.run_name}_history.json")
        with open(path, 'w') as f:
            json.dump(dict(self.history), f, indent=2)
        print(f"  History saved to: {path}")
        return path

    def get_history(self) -> Dict[str, List[float]]:
        return dict(self.history)

    def print_summary(self):
        """Print final training summary."""
        elapsed = time.time() - self.start_time
        print(f"\n{'='*60}")
        print(f"  Training Complete — {self.run_name}")
        print(f"  Best Val Error: {self.best_error:.2f}% (epoch {self.best_epoch})")
        print(f"  Best Val Accuracy: {100 - self.best_error:.2f}%")
        print(f"  Total Time: {elapsed/60:.1f} minutes")
        print(f"{'='*60}")


class MultiRunLogger:
    """Aggregates results across multiple training runs (paper: 30 runs)."""

    def __init__(self):
        self.best_errors: List[float] = []
        self.final_errors: List[float] = []

    def add_run(self, best_error: float, final_error: float):
        self.best_errors.append(best_error)
        self.final_errors.append(final_error)

    def summary(self) -> Dict[str, float]:
        import numpy as np
        best = np.array(self.best_errors)
        final = np.array(self.final_errors)
        return {
            "num_runs": len(self.best_errors),
            "avg_error": float(np.mean(final)),
            "std_error": float(np.std(final)),
            "best_error": float(np.min(best)),
            "avg_best_error": float(np.mean(best)),
        }

    def print_summary(self):
        s = self.summary()
        print(f"\n{'='*60}")
        print(f"  Multi-Run Summary ({s['num_runs']} runs)")
        print(f"  Avg Error: {s['avg_error']:.3f}% ± {s['std_error']:.5f}")
        print(f"  Best Error: {s['best_error']:.2f}%")
        print(f"{'='*60}")
