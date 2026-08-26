"""Tests for mrs.models.metrics -- cross-checked directly against sklearn's own metric
functions (Dev Plan Sec 33.8: deterministic transformation/unit tests). Accuracy is
deliberately never computed here (Dev Plan Sec 10).
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from mrs.models.metrics import (
    metrics_at_threshold,
    select_threshold_max_f1,
    threshold_independent_metrics,
    threshold_sweep,
)

# --- metrics_at_threshold ---


def test_metrics_at_threshold_matches_sklearn_reference():
    rng = np.random.default_rng(0)
    y_true = (rng.random(200) < 0.2).astype(int)
    y_prob = np.clip(y_true * 0.6 + rng.normal(0, 0.25, size=200) + 0.2, 0, 1)
    threshold = 0.5
    y_pred = (y_prob >= threshold).astype(int)

    result = metrics_at_threshold(y_true, y_prob, threshold)

    assert result.precision == pytest.approx(precision_score(y_true, y_pred, zero_division=0))
    assert result.recall == pytest.approx(recall_score(y_true, y_pred, zero_division=0))
    assert result.f1 == pytest.approx(f1_score(y_true, y_pred, zero_division=0))
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    assert (result.true_negatives, result.false_positives, result.false_negatives, result.true_positives) == (
        tn,
        fp,
        fn,
        tp,
    )
    assert result.false_positive_rate == pytest.approx(fp / (fp + tn))


def test_metrics_at_threshold_confusion_matrix_hand_computed():
    y_true = [0, 0, 0, 1, 1, 1]
    y_prob = [0.1, 0.6, 0.2, 0.9, 0.3, 0.8]

    result = metrics_at_threshold(y_true, y_prob, threshold=0.5)

    # predictions at >=0.5: [0, 1, 0, 1, 0, 1]
    # tn: idx0(0,0), idx2(0,0) -> 2 | fp: idx1(0->1) -> 1
    # fn: idx4(1->0) -> 1       | tp: idx3(1,1), idx5(1,1) -> 2
    assert (result.true_negatives, result.false_positives, result.false_negatives, result.true_positives) == (
        2,
        1,
        1,
        2,
    )
    assert result.precision == pytest.approx(2 / 3)
    assert result.recall == pytest.approx(2 / 3)
    assert result.false_positive_rate == pytest.approx(1 / 3)
    assert result.f1 == pytest.approx(2 / 3)


def test_metrics_at_threshold_handles_no_positive_predictions_without_raising():
    y_true = [0, 0, 1]
    y_prob = [0.1, 0.2, 0.3]

    result = metrics_at_threshold(y_true, y_prob, threshold=0.9)  # nothing crosses the threshold

    assert result.precision == 0.0  # zero_division=0, not an exception
    assert result.recall == 0.0
    assert result.f1 == 0.0
    assert result.false_positive_rate == 0.0
    assert result.true_positives == 0
    assert result.false_positives == 0


def test_metrics_at_threshold_false_positive_rate_is_nan_when_no_negatives_exist():
    y_true = [1, 1, 1]
    y_prob = [0.9, 0.1, 0.5]

    result = metrics_at_threshold(y_true, y_prob, threshold=0.5)

    assert result.true_negatives == 0
    assert result.false_positives == 0
    assert np.isnan(result.false_positive_rate)


def test_threshold_metrics_to_dict_contains_expected_keys():
    result = metrics_at_threshold([0, 1], [0.2, 0.8], threshold=0.5)

    d = result.to_dict()

    assert set(d.keys()) == {
        "threshold",
        "precision",
        "recall",
        "f1",
        "false_positive_rate",
        "true_negatives",
        "false_positives",
        "false_negatives",
        "true_positives",
    }
    assert d["threshold"] == pytest.approx(0.5)


# --- threshold_independent_metrics ---


def test_threshold_independent_metrics_matches_sklearn():
    rng = np.random.default_rng(1)
    y_true = (rng.random(100) < 0.3).astype(int)
    y_prob = rng.random(100)

    result = threshold_independent_metrics(y_true, y_prob)

    assert result["pr_auc"] == pytest.approx(average_precision_score(y_true, y_prob))
    assert result["roc_auc"] == pytest.approx(roc_auc_score(y_true, y_prob))


def test_threshold_independent_metrics_perfect_separation():
    y_true = [0, 0, 1, 1]
    y_prob = [0.1, 0.2, 0.8, 0.9]

    result = threshold_independent_metrics(y_true, y_prob)

    assert result["pr_auc"] == pytest.approx(1.0)
    assert result["roc_auc"] == pytest.approx(1.0)


def test_threshold_independent_metrics_random_guess_roc_auc_is_one_half():
    y_true = [0, 1]
    y_prob = [0.5, 0.5]

    result = threshold_independent_metrics(y_true, y_prob)

    assert result["roc_auc"] == pytest.approx(0.5)


# --- threshold_sweep ---


def test_threshold_sweep_default_grid_matches_documented_linspace():
    y_true = [0, 1] * 10
    y_prob = [0.3, 0.7] * 10

    sweep = threshold_sweep(y_true, y_prob)

    expected = np.linspace(0.01, 0.99, 99)
    assert len(sweep) == 99
    np.testing.assert_allclose(sweep["threshold"].to_numpy(), expected)


def test_threshold_sweep_respects_custom_grid():
    y_true = [0, 1, 0, 1]
    y_prob = [0.2, 0.8, 0.3, 0.7]
    custom = np.array([0.25, 0.5, 0.75])

    sweep = threshold_sweep(y_true, y_prob, thresholds=custom)

    np.testing.assert_allclose(sweep["threshold"].to_numpy(), custom)
    assert len(sweep) == 3


# --- select_threshold_max_f1 ---


def test_select_threshold_max_f1_matches_manual_argmax_with_tie_break_at_lowest_threshold():
    y_true = [0, 0, 0, 1, 1, 1, 1]
    y_prob = [0.05, 0.15, 0.35, 0.55, 0.65, 0.85, 0.95]

    threshold, sweep = select_threshold_max_f1(y_true, y_prob)

    best_f1 = sweep["f1"].max()
    expected_threshold = sweep.loc[sweep["f1"] == best_f1, "threshold"].min()
    assert threshold == pytest.approx(expected_threshold)
    # The returned sweep is the actual full grid, not a truncated/recomputed one.
    assert len(sweep) == 99


def test_select_threshold_max_f1_breaks_ties_at_lowest_threshold_explicit_example():
    # y_prob perfectly separates the two classes with a wide gap -> every threshold in
    # the gap (0.3 through 0.7) achieves the same perfect F1 -> the LOWEST such threshold
    # must be selected (Dev Plan Sec 37: documented default, not an arbitrary pick).
    y_true = [0, 0, 1, 1]
    y_prob = [0.1, 0.2, 0.8, 0.9]
    grid = np.array([0.3, 0.4, 0.5, 0.6, 0.7])

    threshold, sweep = select_threshold_max_f1(y_true, y_prob, thresholds=grid)

    assert threshold == pytest.approx(0.3)
    matched_row = sweep.loc[np.isclose(sweep["threshold"], 0.3), "f1"]
    assert matched_row.iloc[0] == pytest.approx(1.0)


def test_select_threshold_max_f1_uses_only_the_given_labels_and_probabilities():
    # Two independent calls with different (y_true, y_prob) pairs must not share state --
    # each selection depends only on its own arguments.
    y_true_a = [0, 0, 1, 1]
    y_prob_a = [0.1, 0.2, 0.8, 0.9]
    y_true_b = [0, 0, 1, 1]
    y_prob_b = [0.4, 0.45, 0.5, 0.55]

    threshold_a, _ = select_threshold_max_f1(y_true_a, y_prob_a)
    threshold_b, _ = select_threshold_max_f1(y_true_b, y_prob_b)

    assert threshold_a != pytest.approx(threshold_b)
