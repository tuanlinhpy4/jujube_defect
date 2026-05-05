from __future__ import annotations

import json
import os
import time
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np

import numpy as np


class TrainingLogger:
    def __init__(self, save_dir: str, seed: int):
        self.save_dir = save_dir
        self.seed = seed
        os.makedirs(save_dir, exist_ok=True)
        self.history: Dict[str, List[float]] = defaultdict(list)
        self.best_val_acc = 0.0
        self.best_epoch = 0
        self.start_time = time.time()
        self._epoch_start = time.time()      # per-epoch wall clock

    def training_summary(self) -> Dict:
        """Return training time stats for inclusion in test_results.json.

        Reported by jujube / agricultural Edge AI papers (JujubeNet,
        Ju-LiteMobileAtt, pre-trained jujube maturity paper) as evidence
        of training efficiency alongside inference metrics.
        """
        total_s = time.time() - self.start_time
        epoch_times = self.history.get("epoch_time_s", [])
        mean_epoch_s = float(np.mean(epoch_times)) if epoch_times else 0.0
        return {
            "best_epoch": self.best_epoch,
            "best_val_acc": round(self.best_val_acc, 4),
            "total_train_time_s": round(total_s, 1),
            "total_train_time_min": round(total_s / 60, 2),
            "mean_epoch_time_s": round(mean_epoch_s, 2),
        }

    def log_epoch(self, epoch: int, train_metrics: Dict, val_metrics: Dict, lr: float) -> bool:
        epoch_time = time.time() - self._epoch_start
        self._epoch_start = time.time()          # reset for next epoch

        self.history["epoch"].append(epoch)
        self.history["train_loss"].append(float(train_metrics.get("loss", 0.0)))
        self.history["train_acc"].append(float(train_metrics.get("accuracy", 0.0)))
        self.history["val_loss"].append(float(val_metrics.get("loss", 0.0)))
        self.history["val_acc"].append(float(val_metrics.get("accuracy", 0.0)))
        self.history["val_f1"].append(float(val_metrics.get("f1_macro", 0.0)))
        self.history["lr"].append(float(lr))
        self.history["epoch_time_s"].append(round(epoch_time, 2))

        val_acc = float(val_metrics.get("accuracy", 0.0))
        is_best = val_acc > self.best_val_acc
        if is_best:
            self.best_val_acc = val_acc
            self.best_epoch = epoch

        elapsed = time.time() - self.start_time
        marker = " *" if is_best else ""
        print(
            f"  [{epoch:3d}] "
            f"Loss={train_metrics.get('loss',0):.4f} "
            f"TrainAcc={train_metrics.get('accuracy',0):.2f}% "
            f"ValAcc={val_acc:.2f}% "
            f"ValF1={val_metrics.get('f1_macro',0):.2f}% "
            f"LR={lr:.2e} "
            f"t={elapsed:.0f}s{marker}"
        )
        return is_best

    def save_history(self) -> str:
        path = os.path.join(self.save_dir, "history.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(dict(self.history), f, indent=2)
        return path

    def get_history(self) -> Dict:
        return dict(self.history)


class MultiSeedLogger:
    def __init__(self):
        self.runs: List[Dict] = []

    def add(self, seed: int, test_metrics: Dict):
        self.runs.append({"seed": seed, **test_metrics})

    def summary(self) -> Dict:
        if not self.runs:
            return {}
        keys = ["accuracy", "f1_macro", "precision_macro", "recall_macro",
                "f1_weighted", "precision_weighted", "recall_weighted"]
        result = {"num_seeds": len(self.runs), "seeds": [r["seed"] for r in self.runs]}
        for k in keys:
            vals = np.array([r[k] for r in self.runs if k in r], dtype=float)
            if len(vals) == 0:
                continue
            ddof = 1 if len(vals) > 1 else 0
            result[f"{k}_mean"] = round(float(vals.mean()), 4)
            result[f"{k}_std"] = round(float(vals.std(ddof=ddof)), 4)
        return result

    def print_summary(self, model: str, loss: str, dataset: str):
        s = self.summary()
        if not s:
            return
        print(f"\n{'='*65}")
        print(f"  Multi-Seed Summary | dataset={dataset} model={model} loss={loss}")
        print(f"  Seeds: {s['seeds']}  (n={s['num_seeds']})")
        print(f"{'='*65}")
        for k in ["accuracy", "f1_macro", "precision_macro", "recall_macro"]:
            mean_k = f"{k}_mean"
            std_k = f"{k}_std"
            if mean_k in s:
                label = k.replace("_", " ").title()
                print(f"  {label:<22}: {s[mean_k]:.2f}% ± {s[std_k]:.2f}%")
        print(f"{'='*65}\n")

    def save(self, path: str, model: str, loss: str, dataset: str):
        payload = {
            "dataset": dataset, "model": model, "loss": loss,
            "runs": self.runs, "summary": self.summary()
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
