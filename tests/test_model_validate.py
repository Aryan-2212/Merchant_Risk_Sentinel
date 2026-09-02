"""Tests for mrs.models.validate -- Phase 5 audit utilities. Small synthetic data only,
never the real 1.75M-row dataset (that's exercised by scripts/08_validate_phase5.py).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score

from mrs.models.dataset import FEATURE_COLUMNS
from mrs.models.validate import (
    feature_groups,
    majority_baseline_scores,
    permutation_importance,
    random_ranking_scores,
    shuffle_labels,
    train_on_feature_subset,
)
from tests.model_test_helpers import make_synthetic_features

# --- feature_groups ---


def test_feature_groups_covers_every_registered_feature_exactly_once():
    groups = feature_groups()
    all_names = [name for names in groups.values() for name in names]

    assert set(all_names) == set(FEATURE_COLUMNS)
    assert len(all_names) == len(FEATURE_COLUMNS)  # no feature counted twice


def test_feature_groups_matches_known_registry_composition():
    groups = feature_groups()
    assert set(groups.keys()) == {"transaction", "customer", "terminal", "relationship"}
    assert len(groups["transaction"]) == 5
    assert len(groups["customer"]) == 12
    assert len(groups["terminal"]) == 14
    assert len(groups["relationship"]) == 2


def test_feature_groups_relationship_group_contains_expected_features():
    groups = feature_groups()
    assert set(groups["relationship"]) == {"pair_prior_interaction_count", "pair_is_new_relationship"}


# --- shuffle_labels ---


def test_shuffle_labels_preserves_class_counts():
    y = np.array([0] * 90 + [1] * 10)
    shuffled = shuffle_labels(y, seed=1)

    assert int(shuffled.sum()) == 10
    assert len(shuffled) == 100


def test_shuffle_labels_is_deterministic_given_seed():
    y = np.array([0, 1] * 50)
    a = shuffle_labels(y, seed=7)
    b = shuffle_labels(y, seed=7)
    np.testing.assert_array_equal(a, b)


def test_shuffle_labels_different_seeds_give_different_order():
    y = np.array([0] * 500 + [1] * 500)
    a = shuffle_labels(y, seed=1)
    b = shuffle_labels(y, seed=2)
    assert not np.array_equal(a, b)


def test_shuffle_labels_does_not_mutate_input():
    y = np.array([0, 0, 1, 1])
    original = y.copy()
    shuffle_labels(y, seed=3)
    np.testing.assert_array_equal(y, original)


# --- random_ranking_scores / majority_baseline_scores ---


def test_random_ranking_scores_in_unit_interval_and_seed_deterministic():
    a = random_ranking_scores(1000, seed=5)
    b = random_ranking_scores(1000, seed=5)
    assert np.all((a >= 0) & (a < 1))
    np.testing.assert_array_equal(a, b)


def test_random_ranking_scores_different_seeds_differ():
    a = random_ranking_scores(1000, seed=5)
    b = random_ranking_scores(1000, seed=6)
    assert not np.array_equal(a, b)


def test_majority_baseline_scores_is_constant_and_equals_positive_rate():
    scores = majority_baseline_scores(50, positive_rate=0.0084)
    assert len(scores) == 50
    assert np.all(scores == pytest.approx(0.0084))


def test_majority_baseline_scores_cannot_rank_anything():
    scores = majority_baseline_scores(10, positive_rate=0.5)
    assert len(set(scores.tolist())) == 1  # every row identical -> zero ranking power


# --- train_on_feature_subset ---


def test_train_on_feature_subset_restricts_pipeline_to_exactly_the_given_columns():
    features = make_synthetic_features(80, split_name="train", date_start="2018-04-01", nan_rate=0.1, seed=1)
    y = (np.random.default_rng(0).random(80) < 0.2).astype(int)
    subset = ("tx_amount", "tx_hour", "tx_is_weekend")

    pipeline = train_on_feature_subset(
        features, y, hyperparams={"n_estimators": 10, "max_depth": 2, "learning_rate": 0.3},
        scale_pos_weight=1.0, feature_subset=subset,
    )

    impute_step = pipeline.named_steps["preprocess"].named_steps["impute_flag"]
    assert impute_step.feature_names_in_ == list(subset)


def test_train_on_feature_subset_pipeline_predicts_without_error():
    features = make_synthetic_features(80, split_name="train", date_start="2018-04-01", nan_rate=0.1, seed=1)
    y = (np.random.default_rng(0).random(80) < 0.2).astype(int)
    subset = ("customer_tx_count_1h", "terminal_hist_fraud_rate")

    pipeline = train_on_feature_subset(
        features, y, hyperparams={"n_estimators": 10, "max_depth": 2, "learning_rate": 0.3},
        scale_pos_weight=1.0, feature_subset=subset,
    )
    proba = pipeline.predict_proba(features[list(subset)])
    assert proba.shape == (80, 2)
    assert np.isfinite(proba).all()


# --- permutation_importance ---


def _pr_auc(y_true, y_prob) -> float:
    return float(average_precision_score(y_true, y_prob))


def test_permutation_importance_ranks_an_informative_feature_above_pure_noise():
    rng = np.random.default_rng(0)
    n = 2000
    strong_signal = rng.normal(size=n)
    noise = rng.normal(size=n)
    # y depends only on strong_signal; noise has no relationship to y at all.
    y = (strong_signal + rng.normal(scale=0.3, size=n) > 0).astype(int)
    X = pd.DataFrame({"strong_signal": strong_signal, "noise": noise})

    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline as SkPipeline

    pipeline = SkPipeline([("clf", LogisticRegression())])
    pipeline.fit(X, y)

    result = permutation_importance(pipeline, X, y, _pr_auc, n_repeats=3, seed=1)

    strong_row = result[result["feature"] == "strong_signal"].iloc[0]
    noise_row = result[result["feature"] == "noise"].iloc[0]
    assert strong_row["mean_degradation"] > noise_row["mean_degradation"]
    assert strong_row["mean_degradation"] > 0.05  # meaningfully positive, not noise-level


def test_permutation_importance_output_is_sorted_descending_by_degradation():
    rng = np.random.default_rng(2)
    n = 500
    X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n), "c": rng.normal(size=n)})
    y = (X["a"] > 0).astype(int)

    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline as SkPipeline

    pipeline = SkPipeline([("clf", LogisticRegression())])
    pipeline.fit(X, y)

    result = permutation_importance(pipeline, X, y, _pr_auc, n_repeats=1, seed=0)

    degradations = result["mean_degradation"].to_numpy()
    assert np.all(degradations[:-1] >= degradations[1:])
    assert set(result["feature"]) == {"a", "b", "c"}


def test_permutation_importance_does_not_mutate_input_frame():
    rng = np.random.default_rng(3)
    X = pd.DataFrame({"a": rng.normal(size=200), "b": rng.normal(size=200)})
    y = (X["a"] > 0).astype(int)
    original = X.copy()

    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline as SkPipeline

    pipeline = SkPipeline([("clf", LogisticRegression())])
    pipeline.fit(X, y)

    permutation_importance(pipeline, X, y, _pr_auc, n_repeats=1, seed=0)

    pd.testing.assert_frame_equal(X, original)
