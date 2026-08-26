"""Model comparison utilities (Dev Plan Sec 27 Phase 5 step 5: "Compare against
baseline"). Pure functions over already-computed metrics/error-analysis dicts -- no file
I/O, no model-specific logic -- so this works for comparing any two model runs that share
this project's metrics/error-analysis shape, not just Logistic Regression vs XGBoost.
"""

from __future__ import annotations

import pandas as pd

#: Metrics compared, in report order. Deliberately excludes accuracy (Dev Plan Sec 10).
COMPARISON_METRIC_KEYS: tuple[str, ...] = (
    "precision",
    "recall",
    "f1",
    "pr_auc",
    "roc_auc",
    "false_positive_rate",
)


def build_metric_comparison(
    baseline_metrics: dict,
    candidate_metrics: dict,
    baseline_label: str = "logistic_regression",
    candidate_label: str = "xgboost",
) -> pd.DataFrame:
    """Side-by-side table of COMPARISON_METRIC_KEYS with the candidate-minus-baseline delta."""
    rows = [
        {
            "metric": key,
            baseline_label: baseline_metrics[key],
            candidate_label: candidate_metrics[key],
            "delta": candidate_metrics[key] - baseline_metrics[key],
        }
        for key in COMPARISON_METRIC_KEYS
    ]
    return pd.DataFrame(rows)


def build_scenario_recall_comparison(
    baseline_recall_by_scenario: dict,
    candidate_recall_by_scenario: dict,
    baseline_label: str = "logistic_regression",
    candidate_label: str = "xgboost",
) -> pd.DataFrame:
    """Per-scenario recall comparison. Scenario keys are normalized to int so this works
    whether the caller passes in-memory dicts (int keys) or dicts reloaded from a saved
    error_analysis.json file (string keys, per JSON's object-key convention).
    """
    baseline_by_scenario = {int(k): v for k, v in baseline_recall_by_scenario.items()}
    candidate_by_scenario = {int(k): v for k, v in candidate_recall_by_scenario.items()}
    scenarios = sorted(set(baseline_by_scenario) | set(candidate_by_scenario))

    rows = []
    for scenario in scenarios:
        baseline_recall = baseline_by_scenario.get(scenario, {}).get("recall")
        candidate_recall = candidate_by_scenario.get(scenario, {}).get("recall")
        delta = (
            candidate_recall - baseline_recall
            if (baseline_recall is not None and candidate_recall is not None)
            else None
        )
        rows.append(
            {
                "scenario": scenario,
                baseline_label: baseline_recall,
                candidate_label: candidate_recall,
                "delta": delta,
            }
        )
    return pd.DataFrame(rows)
