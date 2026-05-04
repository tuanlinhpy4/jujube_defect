import torch
import numpy as np
from typing import Dict, List, Optional, Tuple


class MetricTracker:
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
        self.all_preds.extend(preds.cpu().numpy())
        self.all_labels.extend(labels.cpu().numpy())
        self.running_loss += loss * batch_size
        self.num_samples += batch_size

    @property
    def avg_loss(self) -> float:
        return self.running_loss / max(self.num_samples, 1)

    @property
    def accuracy(self) -> float:
        if not self.all_labels:
            return 0.0
        preds = np.array(self.all_preds)
        labels = np.array(self.all_labels)
        return float((preds == labels).mean() * 100.0)

    @property
    def error_rate(self) -> float:
        return 100.0 - self.accuracy

    def confusion_matrix(self) -> np.ndarray:
        cm = np.zeros((self.num_classes, self.num_classes), dtype=int)
        for pred, label in zip(self.all_preds, self.all_labels):
            cm[label][pred] += 1
        return cm

    def per_class_metrics(self) -> Dict[str, Dict[str, float]]:
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
        per_class = self.per_class_metrics()
        avg_precision = float(np.mean([m["precision"] for m in per_class.values()]))
        avg_recall = float(np.mean([m["recall"] for m in per_class.values()]))
        avg_f1 = float(np.mean([m["f1"] for m in per_class.values()]))
        return {
            "loss": self.avg_loss,
            "accuracy": self.accuracy,
            "error_rate": self.error_rate,
            "avg_precision": avg_precision,
            "avg_recall": avg_recall,
            "avg_f1": avg_f1,
        }

    def bootstrap_ci(
        self,
        metric: str = "accuracy",
        n_bootstrap: int = 10_000,
        confidence: float = 0.95,
        seed: int = 0,
    ) -> Tuple[float, float]:
        rng = np.random.default_rng(seed)
        preds = np.array(self.all_preds)
        labels = np.array(self.all_labels)
        n = len(labels)
        correct = (preds == labels).astype(float)

        bootstrap_scores = []
        for _ in range(n_bootstrap):
            idx = rng.integers(0, n, size=n)
            if metric == "accuracy":
                bootstrap_scores.append(correct[idx].mean() * 100.0)
            elif metric == "error_rate":
                bootstrap_scores.append((1.0 - correct[idx].mean()) * 100.0)
            else:
                raise ValueError(f"Unsupported metric for bootstrap CI: {metric}")

        alpha = 1.0 - confidence
        lo = float(np.percentile(bootstrap_scores, 100 * alpha / 2))
        hi = float(np.percentile(bootstrap_scores, 100 * (1 - alpha / 2)))
        return lo, hi

    def print_report(self):
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
        print(f"\n  Confusion Matrix:")
        header = "  " + " " * 12 + "".join(f"{n:>8s}" for n in self.class_names)
        print(header)
        for i, name in enumerate(self.class_names):
            row = ("  " + f"{name:<12}"
                   + "".join(f"{cm[i, j]:>8d}" for j in range(self.num_classes)))
            print(row)
        print()


def mcnemar_test(
    preds_a: np.ndarray,
    preds_b: np.ndarray,
    labels: np.ndarray,
) -> Dict[str, float]:
    correct_a = (preds_a == labels)
    correct_b = (preds_b == labels)
    b = int(((correct_a) & (~correct_b)).sum())
    c = int(((~correct_a) & (correct_b)).sum())
    n_discordant = b + c

    if n_discordant == 0:
        return {"statistic": 0.0, "p_value": 1.0, "b": b, "c": c,
                "method": "mcnemar_exact"}

    try:
        from scipy.stats import binomtest
        result = binomtest(b, n=n_discordant, p=0.5, alternative="two-sided")
        p_value = float(result.pvalue)
        method = "mcnemar_exact"
    except ImportError:
        statistic = (abs(b - c) - 1.0) ** 2 / (b + c)
        from scipy.stats import chi2
        p_value = float(1.0 - chi2.cdf(statistic, df=1))
        method = "mcnemar_continuity_corrected"

    statistic = float((b - c) ** 2 / n_discordant)
    return {
        "statistic": statistic,
        "p_value": p_value,
        "b": b,
        "c": c,
        "n_discordant": n_discordant,
        "method": method,
        "significant_at_0.05": p_value < 0.05,
    }


def paired_ttest(
    errors_a: List[float],
    errors_b: List[float],
) -> Dict[str, float]:
    from scipy import stats
    a = np.array(errors_a, dtype=float)
    b = np.array(errors_b, dtype=float)
    result = stats.ttest_rel(a, b)
    return {
        "statistic": float(result.statistic),
        "p_value": float(result.pvalue),
        "mean_diff": float((a - b).mean()),
        "significant_at_0.05": float(result.pvalue) < 0.05,
    }
