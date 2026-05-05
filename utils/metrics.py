import numpy as np
import torch
from typing import Dict, List, Optional


class MetricTracker:
    def __init__(self, num_classes: int, class_names: Optional[List[str]] = None):
        self.num_classes = num_classes
        self.class_names = class_names or [str(i) for i in range(num_classes)]
        self.reset()

    def reset(self):
        self.all_preds = []
        self.all_labels = []
        self.all_probs = []          # softmax probabilities, for AUC-ROC
        self.running_loss = 0.0
        self.num_samples = 0

    def update(self, preds: torch.Tensor, labels: torch.Tensor,
               loss: float = 0.0, batch_size: int = 1,
               logits: Optional[torch.Tensor] = None):
        self.all_preds.extend(preds.cpu().numpy().tolist())
        self.all_labels.extend(labels.cpu().numpy().tolist())
        self.running_loss += loss * batch_size
        self.num_samples += batch_size
        if logits is not None:
            probs = torch.softmax(logits.detach(), dim=1)
            self.all_probs.extend(probs.cpu().numpy().tolist())

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
            cm[int(label)][int(pred)] += 1
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
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
                "support": int(cm[i, :].sum()),
            }
        return metrics

    def _macro(self, key: str) -> float:
        pc = self.per_class_metrics()
        return float(np.mean([m[key] for m in pc.values()]))

    def _weighted(self, key: str) -> float:
        pc = self.per_class_metrics()
        total = sum(m["support"] for m in pc.values())
        if total == 0:
            return 0.0
        return float(sum(m[key] * m["support"] for m in pc.values()) / total)

    # ------------------------------------------------------------------
    # Cohen's Kappa  (κ)
    # ------------------------------------------------------------------
    def cohen_kappa(self) -> float:
        if not self.all_labels:
            return 0.0
        cm = self.confusion_matrix().astype(float)
        n = cm.sum()
        if n == 0:
            return 0.0
        p_o = np.diag(cm).sum() / n
        row_sums = cm.sum(axis=1)
        col_sums = cm.sum(axis=0)
        p_e = (row_sums @ col_sums) / (n ** 2)
        if p_e >= 1.0:
            return 100.0
        kappa = (p_o - p_e) / (1.0 - p_e)
        return round(float(kappa * 100.0), 4)

    # ------------------------------------------------------------------
    # Matthews Correlation Coefficient  (MCC)
    # ------------------------------------------------------------------
    def mcc(self) -> float:
        if not self.all_labels:
            return 0.0
        cm = self.confusion_matrix().astype(float)
        n = cm.sum()
        if n == 0:
            return 0.0
        t_k = cm.sum(axis=1)
        p_k = cm.sum(axis=0)
        numerator = n * np.trace(cm) - np.dot(t_k, p_k)
        denom_a = n ** 2 - np.dot(p_k, p_k)
        denom_b = n ** 2 - np.dot(t_k, t_k)
        denom = np.sqrt(denom_a * denom_b)
        if denom == 0.0:
            return 0.0
        mcc_val = numerator / denom
        return round(float(mcc_val * 100.0), 4)

    # ------------------------------------------------------------------
    # AUC-ROC  (one-vs-rest, per class + macro)
    # Pure numpy — no sklearn dependency.
    # ------------------------------------------------------------------
    @staticmethod
    def _binary_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
        """AUC for one binary task via trapezoidal rule."""
        pos = y_true.sum()
        neg = len(y_true) - pos
        if pos == 0 or neg == 0:
            return float("nan")
        idx = np.argsort(-y_score)
        y_sorted = y_true[idx]
        tps = np.cumsum(y_sorted)
        fps = np.cumsum(1 - y_sorted)
        tpr = np.concatenate([[0.0], tps / pos])
        fpr = np.concatenate([[0.0], fps / neg])
        _trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz  # renamed in NumPy 2.0
        return float(_trapz(tpr, fpr))

    def auc_roc(self) -> Dict[str, float]:
        """Per-class AUC-ROC (one-vs-rest) + macro average.

        Returns empty dict when probabilities were not collected
        (i.e. logits were not passed to update()).
        """
        if not self.all_probs:
            return {}
        probs = np.array(self.all_probs)          # (N, C)
        labels = np.array(self.all_labels)         # (N,)
        result: Dict[str, float] = {}
        valid_aucs = []
        for i, name in enumerate(self.class_names):
            y_bin = (labels == i).astype(float)
            auc = self._binary_auc(y_bin, probs[:, i])
            result[name] = round(auc * 100.0, 4) if not np.isnan(auc) else float("nan")
            if not np.isnan(auc):
                valid_aucs.append(auc)
        result["macro"] = round(float(np.mean(valid_aucs)) * 100.0, 4) if valid_aucs else float("nan")
        return result

    def summary(self) -> Dict[str, float]:
        base = {
            "loss": round(self.avg_loss, 6),
            "accuracy": round(self.accuracy, 4),
            "error_rate": round(self.error_rate, 4),
            "precision_macro": round(self._macro("precision"), 4),
            "recall_macro": round(self._macro("recall"), 4),
            "f1_macro": round(self._macro("f1"), 4),
            "precision_weighted": round(self._weighted("precision"), 4),
            "recall_weighted": round(self._weighted("recall"), 4),
            "f1_weighted": round(self._weighted("f1"), 4),
            "kappa": self.cohen_kappa(),
            "mcc": self.mcc(),
        }
        auc = self.auc_roc()
        if auc:
            base["auc_macro"] = auc.get("macro", float("nan"))
            for name, val in auc.items():
                if name != "macro":
                    base[f"auc_{name}"] = val
        return base

    def print_report(self, split: str = "Test"):
        pc = self.per_class_metrics()
        cm = self.confusion_matrix()
        s = self.summary()
        w = max(len(n) for n in self.class_names) + 2
        print(f"\n{'='*65}")
        print(f"  {split} Results")
        print(f"{'='*65}")
        print(f"  Accuracy  : {s['accuracy']:.2f}%  (Error: {s['error_rate']:.2f}%)")
        print(f"  Loss      : {s['loss']:.4f}")
        print(f"  Macro     P={s['precision_macro']:.2f}%  "
              f"R={s['recall_macro']:.2f}%  F1={s['f1_macro']:.2f}%")
        print(f"  Weighted  P={s['precision_weighted']:.2f}%  "
              f"R={s['recall_weighted']:.2f}%  F1={s['f1_weighted']:.2f}%")
        print(f"  Kappa (κ) : {s['kappa']:.2f}%   MCC: {s['mcc']:.2f}%")
        if "auc_macro" in s:
            print(f"  AUC-ROC   : macro={s['auc_macro']:.2f}%")
        print(f"{'='*65}")
        print(f"  {'Class':<{w}} {'Recall':>8} {'Precision':>10} "
              f"{'F1':>8} {'AUC':>8} {'Support':>8}")
        print(f"  {'-'*(w+46)}")
        auc_dict = self.auc_roc()
        for name, m in pc.items():
            auc_val = auc_dict.get(name, float("nan"))
            auc_str = f"{auc_val:>7.2f}%" if not np.isnan(auc_val) else "    N/A"
            print(f"  {name:<{w}} {m['recall']:>7.2f}%  "
                  f"{m['precision']:>8.2f}%  {m['f1']:>6.2f}%  "
                  f"{auc_str}  {m['support']:>7d}")
        print(f"{'='*65}")
        print(f"\n  Confusion Matrix (rows=true, cols=pred):")
        header = "  " + " " * w + "".join(f"{n:>{w}s}" for n in self.class_names)
        print(header)
        for i, name in enumerate(self.class_names):
            row = (f"  {name:<{w}}"
                   + "".join(f"{cm[i,j]:>{w}d}" for j in range(self.num_classes)))
            print(row)
        print()
