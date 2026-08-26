#!/usr/bin/env python
"""Train and evaluate the Phase 4 Logistic Regression baseline.

Pure orchestrator: loads the persisted Phase 3 feature splits (read-only), attaches
labels via mrs.models.dataset, delegates all model-selection/training/evaluation logic to
mrs.models.train.train_baseline, and persists the results. No metric computation, feature
engineering, or model logic lives here.

Run with: .venv/bin/python scripts/06_train_baseline.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mrs import config  # noqa: E402
from mrs.models.dataset import load_processed_transactions, load_split  # noqa: E402
from mrs.models.persistence import save_model  # noqa: E402
from mrs.models.train import MODEL_VERSION, train_baseline  # noqa: E402


def _print_headline(label: str, m: dict) -> None:
    print(f"  [{label}]")
    for key in (
        "threshold", "precision", "recall", "f1", "pr_auc", "roc_auc",
        "false_positive_rate", "true_positives", "false_positives",
        "false_negatives", "true_negatives",
    ):
        print(f"    {key}: {m[key]}")


def main() -> None:
    version_dir = config.MODELS_DIR / MODEL_VERSION
    if version_dir.exists():
        raise SystemExit(
            f"Model version directory already exists: {version_dir}\n"
            "Phase 4 baseline versioning requires an explicit new version rather than "
            "silently overwriting or auto-incrementing it (ambiguous experiment "
            "lineage, Dev Plan Sec 36). Remove it manually to retrain this exact "
            "version, or bump mrs.models.train.MODEL_VERSION for a new experiment."
        )

    print("Loading processed transactions (label source)...")
    labels_source = load_processed_transactions()

    print("Loading feature splits and attaching labels...")
    train_df = load_split("train", labels_source)
    validation_df = load_split("validation", labels_source)
    test_df = load_split("test", labels_source)
    print(f"  train={len(train_df):,} validation={len(validation_df):,} test={len(test_df):,}")

    print("Training Logistic Regression baseline...")
    start = time.time()
    result = train_baseline(train_df, validation_df, test_df)
    print(f"  trained in {time.time() - start:.1f}s")

    print()
    print(f"Selected threshold (max F1 on validation): {result.threshold:.3f}")
    _print_headline("validation", result.validation_metrics)
    _print_headline("test (frozen threshold)", result.test_metrics)

    print()
    print("Top 10 coefficients by |magnitude|:")
    print(result.coefficients.head(10).to_string(index=False))

    print()
    print("Error analysis (test set):")
    for key, value in result.error_analysis.items():
        print(f"  {key}: {value}")

    save_model(result.pipeline, result.metadata, version_dir)
    result.validation_sweep.to_csv(version_dir / "validation_threshold_sweep.csv", index=False)
    result.coefficients.to_csv(version_dir / "coefficients.csv", index=False)
    with open(version_dir / "error_analysis.json", "w") as f:
        json.dump(result.error_analysis, f, indent=2, sort_keys=True)

    print()
    print(f"Saved model, metadata, threshold sweep, coefficients, and error analysis to {version_dir}")


if __name__ == "__main__":
    main()
