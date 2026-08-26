"""Phase 5 main model: XGBoost trained on the same Phase 3 feature layer as the Phase 4
Logistic Regression baseline (Dev Plan Sec 9/27).

Reuses Phase 4 infrastructure wherever it is model-agnostic: mrs.models.dataset (feature
matrix / label join), mrs.models.preprocessing (impute+scale, fit on train only),
mrs.models.metrics (threshold metrics), mrs.models.persistence (save/load), and the small
split-lineage/error-analysis helpers from mrs.models.train (imported, not duplicated --
that module is Phase 4's working baseline and is not modified here).

Same temporal discipline as Phase 4: fit on train only; a small validation-scored
hyperparameter search picks the model (Dev Plan Sec 27 "validation tuning"); the decision
threshold is selected on validation only (Dev Plan Sec 37); test is scored exactly once,
at that frozen threshold, and never used to pick anything (Dev Plan Sec 6).
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from mrs import __version__ as PACKAGE_VERSION
from mrs.models import metrics as M
from mrs.models.dataset import FEATURE_COLUMNS, get_feature_matrix
from mrs.models.preprocessing import build_preprocessing_pipeline
from mrs.models.train import _amount_stats, _error_analysis, _split_lineage  # noqa: F401 (re-exported for tests)

MODEL_VERSION = "xgboost_v1"
RANDOM_SEED = 42

#: Hyperparameters shared across every candidate (Dev Plan Sec 40: hist is the fast,
#: memory-conscious tree method appropriate for the 1.75M-row dataset).
_FIXED_HYPERPARAMS: dict = {
    "tree_method": "hist",
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "min_child_weight": 1,
    "eval_metric": "aucpr",
}

#: A small, explicit validation-tuning search space (Dev Plan Sec 27 step 2), not a
#: general-purpose hyperparameter-search framework -- avoids unnecessary infrastructure
#: (Dev Plan Sec 14/29) while still selecting the model using validation data only.
HYPERPARAMETER_CANDIDATES: tuple[dict, ...] = (
    {"n_estimators": 200, "max_depth": 3, "learning_rate": 0.10},
    {"n_estimators": 300, "max_depth": 4, "learning_rate": 0.05},
    {"n_estimators": 400, "max_depth": 5, "learning_rate": 0.05},
    {"n_estimators": 300, "max_depth": 6, "learning_rate": 0.03},
)

#: The metric used to pick among HYPERPARAMETER_CANDIDATES. PR-AUC is threshold-
#: independent and is the metric Dev Plan Sec 10 calls out as "particularly important"
#: for this imbalanced problem, so it is used for model selection; F1 (threshold-
#: dependent) is reserved for the separate operating-threshold selection step below.
HYPERPARAMETER_SELECTION_METRIC = "pr_auc"

#: Dev Plan Sec 37: document what the model's output actually is, exactly as Phase 4 does
#: for Logistic Regression -- XGBoost's predict_proba() is not treated as calibrated here.
OUTPUT_TYPE = "uncalibrated_probability_estimate"
OUTPUT_TYPE_NOTES = (
    "sklearn-API XGBClassifier.predict_proba() output. NOT a calibrated probability "
    "against the true ~0.84% base fraud rate: scale_pos_weight reweights the training "
    "objective for class imbalance, which shifts predict_proba's scale away from the "
    "empirical prior -- the same effect class_weight='balanced' has for the Phase 4 "
    "Logistic Regression baseline. Treat values as a relative ranking/risk score, not as "
    "P(fraud) in absolute terms. No calibration step (Platt scaling, isotonic "
    "regression) has been applied."
)


@dataclass
class XGBoostResult:
    pipeline: Pipeline
    threshold: float
    validation_sweep: pd.DataFrame
    validation_metrics: dict
    test_metrics: dict
    feature_importance: pd.DataFrame
    error_analysis: dict
    hyperparameter_candidates: list[dict]
    metadata: dict


def build_model(hyperparams: dict, scale_pos_weight: float) -> XGBClassifier:
    return XGBClassifier(
        **hyperparams,
        **_FIXED_HYPERPARAMS,
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_SEED,
    )


def build_full_pipeline(hyperparams: dict, scale_pos_weight: float) -> Pipeline:
    return Pipeline(
        [
            ("preprocess", build_preprocessing_pipeline()),
            ("classifier", build_model(hyperparams, scale_pos_weight)),
        ]
    )


def _feature_importance_table(pipeline: Pipeline) -> pd.DataFrame:
    impute_step = pipeline.named_steps["preprocess"].named_steps["impute_flag"]
    names = impute_step.get_feature_names_out()
    importances = pipeline.named_steps["classifier"].feature_importances_
    table = pd.DataFrame({"feature": names, "importance": importances})
    return table.sort_values("importance", ascending=False).reset_index(drop=True)


def train_xgboost_model(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> XGBoostResult:
    """Fit each hyperparameter candidate on train, select the one with the best
    validation PR-AUC, select a decision threshold on validation only, evaluate once on
    test at that threshold, and run error analysis on the test-set misclassifications.
    """
    X_train = get_feature_matrix(train_df)
    y_train = train_df["TX_FRAUD"].to_numpy()

    X_val = get_feature_matrix(validation_df)
    y_val = validation_df["TX_FRAUD"].to_numpy()

    # scale_pos_weight computed from TRAIN labels only (Dev Plan Sec 5/33.6): validation
    # and test class balance must never influence how the model is fit.
    n_genuine = int((y_train == 0).sum())
    n_fraud = int((y_train == 1).sum())
    scale_pos_weight = n_genuine / n_fraud

    candidate_scores: list[dict] = []
    best_pipeline: Pipeline | None = None
    best_val_prob = None
    best_hyperparams: dict | None = None
    best_score = float("-inf")

    for hyperparams in HYPERPARAMETER_CANDIDATES:
        pipeline = build_full_pipeline(hyperparams, scale_pos_weight)
        pipeline.fit(X_train, y_train)
        val_prob = pipeline.predict_proba(X_val)[:, 1]
        score = M.threshold_independent_metrics(y_val, val_prob)[HYPERPARAMETER_SELECTION_METRIC]
        candidate_scores.append({"hyperparameters": hyperparams, "validation_pr_auc": score})

        if score > best_score:
            best_score = score
            best_pipeline = pipeline
            best_val_prob = val_prob
            best_hyperparams = hyperparams

    pipeline = best_pipeline
    val_prob = best_val_prob
    selected_hyperparams = best_hyperparams

    threshold, sweep = M.select_threshold_max_f1(y_val, val_prob)
    validation_metrics = {
        **M.metrics_at_threshold(y_val, val_prob, threshold).to_dict(),
        **M.threshold_independent_metrics(y_val, val_prob),
    }

    X_test = get_feature_matrix(test_df)
    y_test = test_df["TX_FRAUD"].to_numpy()
    test_prob = pipeline.predict_proba(X_test)[:, 1]
    test_metrics = {
        **M.metrics_at_threshold(y_test, test_prob, threshold).to_dict(),
        **M.threshold_independent_metrics(y_test, test_prob),
    }

    feature_importance = _feature_importance_table(pipeline)
    error_analysis = _error_analysis(test_df, y_test, test_prob, threshold)

    metadata = {
        "model_version": MODEL_VERSION,
        "model_type": "XGBClassifier",
        "random_seed": RANDOM_SEED,
        "hyperparameters": {
            **selected_hyperparams,
            **_FIXED_HYPERPARAMS,
            "scale_pos_weight": scale_pos_weight,
        },
        "hyperparameter_selection": {
            "criterion": f"max_validation_{HYPERPARAMETER_SELECTION_METRIC}",
            "evaluated_on": "validation",
            "candidates": candidate_scores,
            "selected": selected_hyperparams,
        },
        "output_type": OUTPUT_TYPE,
        "output_type_notes": OUTPUT_TYPE_NOTES,
        "threshold": threshold,
        "threshold_selection": {
            "criterion": "max_f1",
            "evaluated_on": "validation",
            "grid_min": float(sweep["threshold"].min()),
            "grid_max": float(sweep["threshold"].max()),
            "grid_size": int(len(sweep)),
        },
        "feature_lineage": {
            "package_version": PACKAGE_VERSION,
            "feature_count": len(FEATURE_COLUMNS),
            "feature_columns": list(FEATURE_COLUMNS),
        },
        "split_lineage": {
            "train": _split_lineage("train", train_df),
            "validation": _split_lineage("validation", validation_df),
            "test": _split_lineage("test", test_df),
        },
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "environment_notes": (
            "macOS development requires the OpenMP runtime (Homebrew 'libomp') for the "
            "xgboost wheel's native library to load; this is a host-machine dependency, "
            "not a project dependency-file entry (Phase 5 handoff Sec: important boundary)."
        ),
    }

    return XGBoostResult(
        pipeline=pipeline,
        threshold=threshold,
        validation_sweep=sweep,
        validation_metrics=validation_metrics,
        test_metrics=test_metrics,
        feature_importance=feature_importance,
        error_analysis=error_analysis,
        hyperparameter_candidates=candidate_scores,
        metadata=metadata,
    )
