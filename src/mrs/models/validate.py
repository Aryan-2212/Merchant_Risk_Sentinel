"""Phase 5 validation utilities -- post-hoc audit logic only, never imported by the
production training path (mrs.models.train_xgboost, scripts/07_train_xgboost.py). Used
by scripts/08_validate_phase5.py against real data, and by tests/test_model_validate.py
against small synthetic data.

Nothing here can influence model training, selection, or the persisted production
artifact -- it only reads an already-fitted pipeline or fits an independent audit-only
copy restricted to a feature subset / shuffled labels, for comparison purposes.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from mrs.features.registry import FEATURE_SPECS
from mrs.models.preprocessing import build_preprocessing_pipeline
from mrs.models.train_xgboost import build_model


def feature_groups() -> dict[str, tuple[str, ...]]:
    """Feature column names grouped by their registry `level` (Dev Plan Sec 12: the
    registry is the single source of truth for feature membership) -- never a
    hand-invented grouping.
    """
    groups: dict[str, list[str]] = defaultdict(list)
    for spec in FEATURE_SPECS:
        groups[spec.level].append(spec.name)
    return {level: tuple(names) for level, names in groups.items()}


def shuffle_labels(y: np.ndarray, seed: int) -> np.ndarray:
    """A random permutation of `y`: same class counts, different row assignment."""
    rng = np.random.default_rng(seed)
    return rng.permutation(np.asarray(y))


def random_ranking_scores(n: int, seed: int) -> np.ndarray:
    """Uniform random scores in [0, 1) -- a ranking baseline with zero real signal."""
    return np.random.default_rng(seed).random(n)


def majority_baseline_scores(n: int, positive_rate: float) -> np.ndarray:
    """A constant score for every row (the observed positive rate) -- the "always predict
    the base rate" baseline. Cannot rank at all: a constant score can never order two rows
    differently, so ROC-AUC collapses to 0.5 by construction regardless of the constant
    chosen.
    """
    return np.full(n, positive_rate, dtype=float)


def train_on_feature_subset(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    hyperparams: dict,
    scale_pos_weight: float,
    feature_subset: tuple[str, ...],
) -> Pipeline:
    """Fit a fresh preprocessing+XGBoost pipeline restricted to `feature_subset` columns.

    Reuses the exact production mrs.models.train_xgboost.build_model() and
    mrs.models.preprocessing.build_preprocessing_pipeline() -- this restricts the INPUT
    columns for audit purposes, it does not use a different model architecture or a
    different preprocessing strategy than production.
    """
    pipeline = Pipeline(
        [
            ("preprocess", build_preprocessing_pipeline()),
            ("classifier", build_model(hyperparams, scale_pos_weight)),
        ]
    )
    pipeline.fit(X_train[list(feature_subset)], y_train)
    return pipeline


def permutation_importance(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y_true: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    *,
    feature_names: list[str] | None = None,
    n_repeats: int = 1,
    seed: int = 0,
) -> pd.DataFrame:
    """Permutation feature importance: for each feature, shuffle only that column's
    values across rows (breaking its relationship with the label while preserving its
    marginal distribution and every other feature/row unchanged), re-score with the
    already-fitted `pipeline` (no retraining), and measure how much `metric_fn` degrades
    relative to the unpermuted baseline. A feature the model genuinely relies on should
    show real degradation; a feature with high gain-based importance but near-zero
    permutation degradation is a signal the gain metric is misleading (e.g. a feature that
    is highly correlated with a more useful one and gets "credit" without being load-
    bearing on its own).
    """
    columns = feature_names if feature_names is not None else list(X.columns)
    rng = np.random.default_rng(seed)

    baseline_prob = pipeline.predict_proba(X)[:, 1]
    baseline_score = metric_fn(y_true, baseline_prob)

    rows = []
    for col in columns:
        degradations = []
        for _ in range(n_repeats):
            permuted = X.copy()
            permuted[col] = rng.permutation(permuted[col].to_numpy())
            permuted_prob = pipeline.predict_proba(permuted)[:, 1]
            permuted_score = metric_fn(y_true, permuted_prob)
            degradations.append(baseline_score - permuted_score)
        rows.append(
            {
                "feature": col,
                "baseline_score": baseline_score,
                "mean_degradation": float(np.mean(degradations)),
                "std_degradation": float(np.std(degradations)) if n_repeats > 1 else 0.0,
            }
        )
    result = pd.DataFrame(rows)
    return result.sort_values("mean_degradation", ascending=False).reset_index(drop=True)
