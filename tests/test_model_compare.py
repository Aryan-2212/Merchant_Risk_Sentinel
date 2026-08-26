"""Tests for mrs.models.compare -- pure comparison-table logic, using small synthetic
dicts only (Dev Plan Sec 27 Phase 5 step 5: "Compare against baseline"). Deliberately does
not depend on any real trained model or the real 1.75M-row dataset.
"""

from __future__ import annotations

import pandas as pd
import pytest

from mrs.models.compare import (
    COMPARISON_METRIC_KEYS,
    build_metric_comparison,
    build_scenario_recall_comparison,
)


def _metrics(precision, recall, f1, pr_auc, roc_auc, fpr) -> dict:
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "false_positive_rate": fpr,
    }


# --- build_metric_comparison ---


def test_build_metric_comparison_includes_exactly_the_documented_metric_keys():
    baseline = _metrics(0.3, 0.7, 0.42, 0.35, 0.96, 0.01)
    candidate = _metrics(0.4, 0.75, 0.52, 0.45, 0.97, 0.008)

    table = build_metric_comparison(baseline, candidate)

    assert list(table["metric"]) == list(COMPARISON_METRIC_KEYS)
    assert set(table.columns) == {"metric", "logistic_regression", "xgboost", "delta"}


def test_build_metric_comparison_values_and_deltas_are_correct():
    baseline = _metrics(0.3480529159679523, 0.7334118570867687, 0.47207480414455394, 0.41200510158094866, 0.9623994937663668, 0.012263165642107623)
    candidate = _metrics(0.5, 0.6, 0.545, 0.5, 0.98, 0.005)

    table = build_metric_comparison(baseline, candidate)
    by_metric = table.set_index("metric")

    for key in COMPARISON_METRIC_KEYS:
        assert by_metric.loc[key, "logistic_regression"] == pytest.approx(baseline[key])
        assert by_metric.loc[key, "xgboost"] == pytest.approx(candidate[key])
        assert by_metric.loc[key, "delta"] == pytest.approx(candidate[key] - baseline[key])


def test_build_metric_comparison_respects_custom_labels():
    baseline = _metrics(0.3, 0.7, 0.42, 0.35, 0.96, 0.01)
    candidate = _metrics(0.4, 0.75, 0.52, 0.45, 0.97, 0.008)

    table = build_metric_comparison(baseline, candidate, baseline_label="lr", candidate_label="xgb")

    assert set(table.columns) == {"metric", "lr", "xgb", "delta"}


def test_build_metric_comparison_negative_delta_when_candidate_is_worse():
    baseline = _metrics(0.5, 0.5, 0.5, 0.5, 0.9, 0.02)
    candidate = _metrics(0.3, 0.4, 0.34, 0.3, 0.85, 0.03)

    table = build_metric_comparison(baseline, candidate)
    by_metric = table.set_index("metric")

    assert by_metric.loc["precision", "delta"] == pytest.approx(-0.2)
    assert by_metric.loc["false_positive_rate", "delta"] == pytest.approx(0.01)


# --- build_scenario_recall_comparison ---


def test_build_scenario_recall_comparison_matches_scenarios_by_int_key():
    baseline = {1: {"recall": 0.326}, 2: {"recall": 0.756}, 3: {"recall": 0.761}}
    candidate = {1: {"recall": 0.6}, 2: {"recall": 0.8}, 3: {"recall": 0.79}}

    table = build_scenario_recall_comparison(baseline, candidate)

    assert list(table["scenario"]) == [1, 2, 3]
    assert table.loc[table["scenario"] == 1, "logistic_regression"].iloc[0] == pytest.approx(0.326)
    assert table.loc[table["scenario"] == 1, "xgboost"].iloc[0] == pytest.approx(0.6)
    assert table.loc[table["scenario"] == 1, "delta"].iloc[0] == pytest.approx(0.6 - 0.326)


def test_build_scenario_recall_comparison_handles_string_keys_from_json_round_trip():
    # error_analysis.json stores scenario numbers as JSON object keys, which are always
    # strings once reloaded -- this must not break the comparison.
    baseline = {"1": {"recall": 0.326}, "2": {"recall": 0.756}}
    candidate = {"1": {"recall": 0.6}, "2": {"recall": 0.8}}

    table = build_scenario_recall_comparison(baseline, candidate)

    assert list(table["scenario"]) == [1, 2]
    assert table["scenario"].map(type).eq(int).all()


def test_build_scenario_recall_comparison_handles_a_scenario_missing_from_one_side():
    baseline = {1: {"recall": 0.5}}
    candidate = {1: {"recall": 0.6}, 2: {"recall": 0.7}}

    table = build_scenario_recall_comparison(baseline, candidate)
    row = table[table["scenario"] == 2].iloc[0]

    # A column mixing None and floats is coerced to float64 by pandas, so an absent side
    # surfaces as NaN here, not literally None -- check with pd.isna(), not `is None`.
    assert pd.isna(row["logistic_regression"])
    assert row["xgboost"] == pytest.approx(0.7)
    assert pd.isna(row["delta"])


def test_build_scenario_recall_comparison_handles_undefined_recall_as_none():
    # total_fraud == 0 for a scenario produces recall=None (Dev Plan Sec 34.1-style
    # cold-start convention: undefined, not silently 0).
    baseline = {1: {"recall": None}}
    candidate = {1: {"recall": 0.5}}

    table = build_scenario_recall_comparison(baseline, candidate)
    row = table.iloc[0]

    assert pd.isna(row["logistic_regression"])
    assert row["xgboost"] == pytest.approx(0.5)
    assert pd.isna(row["delta"])


def test_build_scenario_recall_comparison_respects_custom_labels():
    baseline = {1: {"recall": 0.5}}
    candidate = {1: {"recall": 0.6}}

    table = build_scenario_recall_comparison(baseline, candidate, baseline_label="lr", candidate_label="xgb")

    assert set(table.columns) == {"scenario", "lr", "xgb", "delta"}
