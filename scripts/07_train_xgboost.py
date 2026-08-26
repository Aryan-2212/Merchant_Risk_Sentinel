#!/usr/bin/env python
"""Train and evaluate the Phase 5 XGBoost model, then compare it against the Phase 4
Logistic Regression baseline.

Pure orchestrator, mirroring scripts/06_train_baseline.py: loads the persisted Phase 3
feature splits (read-only), attaches labels via mrs.models.dataset, delegates all
model-selection/training/evaluation logic to mrs.models.train_xgboost.train_xgboost_model,
persists the results, and prints the comparison against the already-saved Phase 4
baseline (models/logreg_baseline_v1/, never modified here).

Run with: .venv/bin/python scripts/07_train_xgboost.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mrs import config  # noqa: E402
from mrs.models import compare  # noqa: E402
from mrs.models.dataset import load_processed_transactions, load_split  # noqa: E402
from mrs.models.persistence import METADATA_FILENAME, save_model  # noqa: E402
from mrs.models.train import MODEL_VERSION as LOGREG_MODEL_VERSION  # noqa: E402
from mrs.models.train_xgboost import MODEL_VERSION, train_xgboost_model  # noqa: E402

LOGREG_VERSION_DIR = config.MODELS_DIR / LOGREG_MODEL_VERSION


def _print_headline(label: str, m: dict) -> None:
    print(f"  [{label}]")
    for key in (
        "threshold", "precision", "recall", "f1", "pr_auc", "roc_auc",
        "false_positive_rate", "true_positives", "false_positives",
        "false_negatives", "true_negatives",
    ):
        print(f"    {key}: {m[key]}")


def _load_logreg_metadata_and_error_analysis() -> tuple[dict, dict]:
    if not LOGREG_VERSION_DIR.exists():
        raise SystemExit(
            f"Phase 4 baseline not found at {LOGREG_VERSION_DIR}. Run "
            "scripts/06_train_baseline.py first -- Phase 5 compares against it and "
            "cannot proceed without it."
        )
    with open(LOGREG_VERSION_DIR / METADATA_FILENAME) as f:
        metadata = json.load(f)
    with open(LOGREG_VERSION_DIR / "error_analysis.json") as f:
        error_analysis = json.load(f)
    return metadata, error_analysis


def main() -> None:
    version_dir = config.MODELS_DIR / MODEL_VERSION
    if version_dir.exists():
        raise SystemExit(
            f"Model version directory already exists: {version_dir}\n"
            "Phase 5 versioning requires an explicit new version rather than silently "
            "overwriting or auto-incrementing it (ambiguous experiment lineage, Dev Plan "
            "Sec 36). Remove it manually to retrain this exact version, or bump "
            "mrs.models.train_xgboost.MODEL_VERSION for a new experiment."
        )

    logreg_metadata, logreg_error_analysis = _load_logreg_metadata_and_error_analysis()

    print("Loading processed transactions (label source)...")
    labels_source = load_processed_transactions()

    print("Loading feature splits and attaching labels...")
    train_df = load_split("train", labels_source)
    validation_df = load_split("validation", labels_source)
    test_df = load_split("test", labels_source)
    print(f"  train={len(train_df):,} validation={len(validation_df):,} test={len(test_df):,}")

    print("Training XGBoost (validation-tuning over a small hyperparameter grid)...")
    start = time.time()
    result = train_xgboost_model(train_df, validation_df, test_df)
    print(f"  trained {len(result.hyperparameter_candidates)} candidates in {time.time() - start:.1f}s")

    print()
    print("Hyperparameter candidates (selected by validation PR-AUC):")
    for candidate in result.hyperparameter_candidates:
        marker = " <== selected" if candidate["hyperparameters"] == result.metadata["hyperparameter_selection"]["selected"] else ""
        print(f"    {candidate['hyperparameters']} -> pr_auc={candidate['validation_pr_auc']:.4f}{marker}")

    print()
    print(f"Selected threshold (max F1 on validation): {result.threshold:.3f}")
    _print_headline("validation", result.validation_metrics)
    _print_headline("test (frozen threshold)", result.test_metrics)

    print()
    print("Top 10 features by importance:")
    print(result.feature_importance.head(10).to_string(index=False))

    print()
    print("Error analysis (test set):")
    for key, value in result.error_analysis.items():
        print(f"  {key}: {value}")

    save_model(result.pipeline, result.metadata, version_dir)
    result.validation_sweep.to_csv(version_dir / "validation_threshold_sweep.csv", index=False)
    result.feature_importance.to_csv(version_dir / "feature_importance.csv", index=False)
    with open(version_dir / "error_analysis.json", "w") as f:
        json.dump(result.error_analysis, f, indent=2, sort_keys=True)
    with open(version_dir / "hyperparameter_candidates.json", "w") as f:
        json.dump(result.hyperparameter_candidates, f, indent=2, sort_keys=True)

    print()
    print(f"Saved model, metadata, threshold sweep, feature importance, hyperparameter "
          f"candidates, and error analysis to {version_dir}")

    print()
    print("=" * 70)
    print("Logistic Regression (Phase 4) vs XGBoost (Phase 5) -- TEST SET")
    print("=" * 70)
    metric_comparison = compare.build_metric_comparison(
        logreg_metadata["test_metrics"], result.test_metrics,
    )
    print(metric_comparison.to_string(index=False))
    print()
    scenario_comparison = compare.build_scenario_recall_comparison(
        logreg_error_analysis["recall_by_scenario"], result.error_analysis["recall_by_scenario"],
    )
    print(scenario_comparison.to_string(index=False))

    metric_comparison.to_csv(version_dir / "comparison_vs_logreg_metrics.csv", index=False)
    scenario_comparison.to_csv(version_dir / "comparison_vs_logreg_scenarios.csv", index=False)


if __name__ == "__main__":
    main()
