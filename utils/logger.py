from __future__ import annotations

import json
import os
import time
from collections import defaultdict
from typing import Dict, List, Optional


class TrainingLogger:
    def __init__(self, save_dir: str, run_name: str = "run"):
        self.save_dir = save_dir
        self.run_name = run_name
        os.makedirs(save_dir, exist_ok=True)
        self.history: Dict[str, List[float]] = defaultdict(list)
        self.best_error = float("inf")
        self.best_epoch = 0
        self.start_time = time.time()

    def log_epoch(self, epoch: int, train_metrics: Dict, val_metrics: Dict,
                  lr: float) -> bool:
        self.history["epoch"].append(epoch)
        self.history["train_loss"].append(float(train_metrics.get("loss", 0.0)))
        self.history["train_acc"].append(float(train_metrics.get("accuracy", 0.0)))
        self.history["val_loss"].append(float(val_metrics.get("loss", 0.0)))
        self.history["val_acc"].append(float(val_metrics.get("accuracy", 0.0)))
        self.history["val_error"].append(float(val_metrics.get("error_rate", 0.0)))
        self.history["lr"].append(float(lr))

        val_error = float(val_metrics.get("error_rate", float("inf")))
        is_best = val_error < self.best_error
        if is_best:
            self.best_error = val_error
            self.best_epoch = epoch

        elapsed = time.time() - self.start_time
        suffix = " *BEST*" if is_best else ""
        print(
            f"  Epoch [{epoch:3d}] "
            f"Train Loss: {train_metrics.get('loss', 0):.4f} | "
            f"Train Acc: {train_metrics.get('accuracy', 0):.2f}% | "
            f"Val Acc: {val_metrics.get('accuracy', 0):.2f}% | "
            f"Val Error: {val_error:.2f}% | "
            f"LR: {lr:.6f} | "
            f"Time: {elapsed:.0f}s"
            f"{suffix}"
        )
        return is_best

    def save_history(self) -> str:
        path = os.path.join(self.save_dir, f"{self.run_name}_history.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(dict(self.history), f, indent=2)
        print(f"  History saved to: {path}")
        return path

    def get_history(self) -> Dict[str, List[float]]:
        return dict(self.history)

    def print_summary(self) -> None:
        elapsed = time.time() - self.start_time
        print(f"\n{'=' * 60}")
        print(f"  Training Complete - {self.run_name}")
        print(f"  Best Val Error: {self.best_error:.2f}% (epoch {self.best_epoch})")
        print(f"  Best Val Accuracy: {100.0 - self.best_error:.2f}%")
        print(f"  Total Time: {elapsed / 60:.1f} minutes")
        print(f"{'=' * 60}")


class MultiRunLogger:
    def __init__(self):
        self.runs: List[Dict] = []

    def add_run(self, run_id: int, seed: int, best_val_error: float,
                final_val_error: float, test_error: Optional[float] = None) -> None:
        record = {
            "run_id": int(run_id),
            "seed": int(seed),
            "best_val_error": float(best_val_error),
            "final_val_error": float(final_val_error),
        }
        if test_error is not None:
            record["test_error"] = float(test_error)
        self.runs.append(record)

    def _bootstrap_ci(
        self,
        values: "np.ndarray",
        n_bootstrap: int = 10_000,
        confidence: float = 0.95,
        seed: int = 0,
    ) -> tuple:
        import numpy as np
        rng = np.random.default_rng(seed)
        boot = [rng.choice(values, size=len(values), replace=True).mean()
                for _ in range(n_bootstrap)]
        alpha = 1.0 - confidence
        lo = float(np.percentile(boot, 100 * alpha / 2))
        hi = float(np.percentile(boot, 100 * (1 - alpha / 2)))
        return lo, hi

    def summary(self) -> Dict:
        import numpy as np

        if not self.runs:
            return {"num_runs": 0}

        key = "test_error" if "test_error" in self.runs[0] else "final_val_error"
        values = np.array([run[key] for run in self.runs], dtype=float)
        accuracies = 100.0 - values
        best_vals = np.array([run["best_val_error"] for run in self.runs], dtype=float)
        ddof = 1 if len(values) > 1 else 0

        ci_lo, ci_hi = self._bootstrap_ci(values)
        acc_ci_lo, acc_ci_hi = self._bootstrap_ci(accuracies)

        result = {
            "num_runs": len(self.runs),
            "metric": key,
            "seeds": [int(run["seed"]) for run in self.runs],
            "mean_error": float(values.mean()),
            "std_error": float(values.std(ddof=ddof)),
            "ci95_error": [ci_lo, ci_hi],
            "mean_accuracy": float(accuracies.mean()),
            "std_accuracy": float(accuracies.std(ddof=ddof)),
            "ci95_accuracy": [acc_ci_lo, acc_ci_hi],
            "std_ddof": ddof,
            "avg_error": float(values.mean()),
            "min_error": float(values.min()),
            "avg_best_val_error": float(best_vals.mean()),
            "best_val_error": float(best_vals.min()),
        }

        if len(self.runs) >= 2:
            try:
                from scipy import stats
                t_stat, p_val = stats.ttest_1samp(values, popmean=values.mean())
                result["normality_shapiro_p"] = float(
                    stats.shapiro(values).pvalue if len(values) >= 3 else float("nan")
                )
            except ImportError:
                pass

        return result

    def save(self, path: str) -> str:
        payload = {"runs": self.runs, "summary": self.summary()}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        return path

    def print_summary(self) -> None:
        s = self.summary()
        if not s.get("num_runs"):
            return
        ci = s.get("ci95_error", [None, None])
        print(f"\n{'=' * 60}")
        print(f"  Multi-Run Summary ({s['num_runs']} runs, metric={s['metric']})")
        print(f"  Seeds: {s['seeds']}")
        print(f"  Mean Error: {s['mean_error']:.3f}% ± {s['std_error']:.5f}")
        if ci[0] is not None:
            print(f"  95% CI Error: [{ci[0]:.3f}%, {ci[1]:.3f}%]")
        print(
            f"  Mean Accuracy: {s['mean_accuracy']:.3f}% "
            f"± {s['std_accuracy']:.5f}"
        )
        print(f"  Min Error: {s['min_error']:.2f}%")
        print(f"  Best Val Error: {s['best_val_error']:.2f}%")
        print(f"{'=' * 60}")
