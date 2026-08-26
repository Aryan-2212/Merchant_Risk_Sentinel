"""Phase 4 baseline: Logistic Regression trained on the Phase 3 feature layer.

Fit on train only; the decision threshold is selected on validation only (Dev Plan
Sec 37); test is scored exactly once, at that frozen threshold, and never used to pick
anything (Dev Plan Sec 6).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from mrs import __version__ as PACKAGE_VERSION
from mrs.data.splits import SPLIT_BOUNDARIES
from mrs.models import metrics as M
from mrs.models.dataset import FEATURE_COLUMNS, get_feature_matrix
from mrs.models.preprocessing import build_preprocessing_pipeline

MODEL_VERSION = "logreg_baseline_v1"
RANDOM_SEED = 42

#: Dev Plan Sec 37: document what the model's output actually is. LogisticRegression's
#: predict_proba() is not treated as a calibrated probability here -- see the note below,
#: which is carried into every saved model's metadata rather than asserted only in a
#: comment a reader might miss.
OUTPUT_TYPE = "uncalibrated_probability_estimate"
OUTPUT_TYPE_NOTES = (
    "sklearn LogisticRegression.predict_proba() output. NOT a calibrated probability "
    "against the true ~0.84% base fraud rate: class_weight='balanced' reweights the "
    "training loss, which shifts predict_proba's scale away from the empirical prior. "
    "Treat values as a relative ranking/risk score, not as P(fraud) in absolute terms. "
    "No calibration step (Platt scaling, isotonic regression) has been applied."
)


@dataclass
class BaselineResult:
    pipeline: Pipeline
    threshold: float
    validation_sweep: pd.DataFrame
    validation_metrics: dict
    test_metrics: dict
    coefficients: pd.DataFrame
    error_analysis: dict
    metadata: dict


def build_model() -> LogisticRegression:
    return LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=RANDOM_SEED,
    )


def build_full_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("preprocess", build_preprocessing_pipeline()),
            ("classifier", build_model()),
        ]
    )


def _coefficient_table(pipeline: Pipeline) -> pd.DataFrame:
    impute_step = pipeline.named_steps["preprocess"].named_steps["impute_flag"]
    names = impute_step.get_feature_names_out()
    coefs = pipeline.named_steps["classifier"].coef_.ravel()
    table = pd.DataFrame({"feature": names, "coefficient": coefs})
    table["abs_coefficient"] = table["coefficient"].abs()
    return table.sort_values("abs_coefficient", ascending=False).reset_index(drop=True)


def _split_lineage(split_name: str, df: pd.DataFrame) -> dict:
    # Configured range comes from mrs.data.splits.SPLIT_BOUNDARIES (Phase 2's existing
    # source of truth, Dev Plan Sec 6) -- not re-decided here. Observed dates/counts come
    # from the actual loaded data, not assumed to match the configuration exactly
    # (evidence-first, Dev Plan Sec 46): both are recorded so a mismatch would be visible.
    configured_start, configured_end = SPLIT_BOUNDARIES[split_name]
    fraud = df["TX_FRAUD"]
    return {
        "configured_range": {"start": configured_start, "end": configured_end},
        "observed_date_min": str(df["TX_DATETIME"].min()),
        "observed_date_max": str(df["TX_DATETIME"].max()),
        "row_count": int(len(df)),
        "fraud_count": int(fraud.sum()),
        "fraud_rate": float(fraud.mean()),
    }


def _amount_stats(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"mean": None, "median": None}
    return {"mean": float(df["tx_amount"].mean()), "median": float(df["tx_amount"].median())}


def _error_analysis(test_df: pd.DataFrame, y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    analysis_df = test_df[["TRANSACTION_ID", "TX_FRAUD", "TX_FRAUD_SCENARIO", "tx_amount"]].copy()
    analysis_df["y_prob"] = y_prob
    analysis_df["y_pred"] = y_pred

    actual_fraud = analysis_df[y_true == 1]
    false_negatives = analysis_df[(y_true == 1) & (y_pred == 0)]
    false_positives = analysis_df[(y_true == 0) & (y_pred == 1)]
    true_positives = analysis_df[(y_true == 1) & (y_pred == 1)]
    true_negative_count = int(((y_true == 0) & (y_pred == 0)).sum())

    def scenario_counts(df: pd.DataFrame) -> dict:
        return {int(k): int(v) for k, v in df["TX_FRAUD_SCENARIO"].value_counts().sort_index().items()}

    def recall_by_scenario() -> dict:
        # Static test-set snapshot only -- NOT the temporal detection-delay/recovery
        # analysis required for scenarios 2/3 (that is Phase 6's behavioral-risk scope,
        # Dev Plan Sec 11/Sec 6 Phase 6). This just answers "does the transaction-level
        # model see each scenario's fraud at all".
        result: dict = {}
        for scenario in sorted(int(s) for s in actual_fraud["TX_FRAUD_SCENARIO"].unique()):
            mask = actual_fraud["TX_FRAUD_SCENARIO"] == scenario
            total = int(mask.sum())
            detected = int((actual_fraud.loc[mask, "y_pred"] == 1).sum())
            result[scenario] = {
                "total_fraud": total,
                "detected": detected,
                "recall": detected / total if total else None,
            }
        return result

    return {
        "total_actual_fraud": int(len(actual_fraud)),
        "total_genuine": int(len(analysis_df) - len(actual_fraud)),
        "true_positive_count": int(len(true_positives)),
        "false_negative_count": int(len(false_negatives)),
        "false_positive_count": int(len(false_positives)),
        "true_negative_count": true_negative_count,
        "recall_by_scenario": recall_by_scenario(),
        "false_negatives_by_scenario": scenario_counts(false_negatives),
        "detected_fraud_by_scenario": scenario_counts(true_positives),
        "false_negative_amount": _amount_stats(false_negatives),
        "true_positive_amount": _amount_stats(true_positives),
        "false_positive_amount": _amount_stats(false_positives),
        "false_negative_score": (
            {"mean": float(false_negatives["y_prob"].mean()), "max": float(false_negatives["y_prob"].max())}
            if len(false_negatives)
            else {"mean": None, "max": None}
        ),
    }


def train_baseline(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> BaselineResult:
    """Fit the baseline pipeline on train, select a threshold on validation, evaluate once
    on test at that threshold, and run error analysis on the test-set misclassifications.
    """
    X_train = get_feature_matrix(train_df)
    y_train = train_df["TX_FRAUD"].to_numpy()

    pipeline = build_full_pipeline()
    pipeline.fit(X_train, y_train)

    X_val = get_feature_matrix(validation_df)
    y_val = validation_df["TX_FRAUD"].to_numpy()
    val_prob = pipeline.predict_proba(X_val)[:, 1]

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

    coefficients = _coefficient_table(pipeline)
    error_analysis = _error_analysis(test_df, y_test, test_prob, threshold)

    metadata = {
        "model_version": MODEL_VERSION,
        "model_type": "LogisticRegression",
        "random_seed": RANDOM_SEED,
        "hyperparameters": {
            "class_weight": "balanced",
            "max_iter": 1000,
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
            # mrs.__version__ is the repository's existing single source of truth for
            # code version (src/mrs/__init__.py); reused here rather than inventing a
            # separate feature-versioning scheme the project does not otherwise have.
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
    }

    return BaselineResult(
        pipeline=pipeline,
        threshold=threshold,
        validation_sweep=sweep,
        validation_metrics=validation_metrics,
        test_metrics=test_metrics,
        coefficients=coefficients,
        error_analysis=error_analysis,
        metadata=metadata,
    )
