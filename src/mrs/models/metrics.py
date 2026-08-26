"""Fraud-detection evaluation metrics (Dev Plan Sec 10, Sec 37).

Accuracy is deliberately not computed here as a headline number -- Sec 10 explicitly
rules it out given the dataset's ~0.84% base fraud rate, where a trivial "always genuine"
classifier already scores >99% accuracy.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass(frozen=True)
class ThresholdMetrics:
    threshold: float
    precision: float
    recall: float
    f1: float
    false_positive_rate: float
    true_negatives: int
    false_positives: int
    false_negatives: int
    true_positives: int

    def to_dict(self) -> dict:
        return {
            "threshold": self.threshold,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "false_positive_rate": self.false_positive_rate,
            "true_negatives": self.true_negatives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
            "true_positives": self.true_positives,
        }


def metrics_at_threshold(y_true, y_prob, threshold: float) -> ThresholdMetrics:
    """Precision/recall/F1/FPR/confusion-matrix counts at one fixed decision threshold."""
    y_true = np.asarray(y_true)
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else float("nan")

    return ThresholdMetrics(
        threshold=float(threshold),
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
        false_positive_rate=float(fpr),
        true_negatives=int(tn),
        false_positives=int(fp),
        false_negatives=int(fn),
        true_positives=int(tp),
    )


def threshold_independent_metrics(y_true, y_prob) -> dict:
    """PR-AUC and ROC-AUC: rank-based metrics that don't depend on a chosen threshold."""
    return {
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
    }


def threshold_sweep(y_true, y_prob, thresholds: np.ndarray | None = None) -> pd.DataFrame:
    """Metrics at each of a grid of thresholds, for inspecting the precision/recall tradeoff."""
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 99)
    rows = [metrics_at_threshold(y_true, y_prob, t).to_dict() for t in thresholds]
    return pd.DataFrame(rows)


def select_threshold_max_f1(
    y_true, y_prob, thresholds: np.ndarray | None = None
) -> tuple[float, pd.DataFrame]:
    """Pick the threshold maximizing F1 over a grid (Dev Plan Sec 37: validation only).

    Ties are broken by the lowest threshold (higher recall) among equally-best F1 scores.
    The full sweep is returned alongside so a caller can inspect the tradeoff rather than
    trust a single number blindly -- this default is documented as a starting point, not
    a frozen business decision (mirrors the placeholder framing of Dev Plan Sec 14).
    """
    sweep = threshold_sweep(y_true, y_prob, thresholds)
    best_f1 = sweep["f1"].max()
    best_row = sweep[sweep["f1"] == best_f1].iloc[0]
    return float(best_row["threshold"]), sweep
