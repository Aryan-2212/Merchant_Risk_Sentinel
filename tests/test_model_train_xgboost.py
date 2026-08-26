"""Tests for mrs.models.train_xgboost.train_xgboost_model -- small, fast, fully-synthetic
data (NOT the real 1.75M-row dataset; that's covered separately once we run the real
Phase 5 training).

Mirrors tests/test_model_train.py's structure and focus: model correctness, chronological
train/validation/test separation, preprocessing fit only on train, threshold selection
only on validation, no label leakage, plus XGBoost-specific concerns (hyperparameter
selection, feature importance, scale_pos_weight) and a persistence round-trip (Dev Plan
Sec 5/6/33.6/36/37).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mrs import __version__ as PACKAGE_VERSION
from mrs.data.splits import SPLIT_BOUNDARIES
from mrs.models.dataset import FEATURE_COLUMNS, get_feature_matrix
from mrs.models.persistence import load_model, save_model
from mrs.models.train_xgboost import (
    HYPERPARAMETER_CANDIDATES,
    HYPERPARAMETER_SELECTION_METRIC,
    MODEL_VERSION,
    OUTPUT_TYPE,
    train_xgboost_model,
)
from tests.model_test_helpers import make_synthetic_labeled_frame

# --- fixtures: small synthetic splits, genuinely chronologically ordered ---


@pytest.fixture(scope="module")
def synthetic_splits():
    train_df = make_synthetic_labeled_frame(
        300,
        split_name="train",
        date_start="2018-04-05",
        fraud_rate=0.15,
        nan_rate=0.1,
        seed=10,
        start_transaction_id=0,
    )
    validation_df = make_synthetic_labeled_frame(
        150,
        split_name="validation",
        date_start="2018-08-05",
        fraud_rate=0.15,
        nan_rate=0.1,
        seed=20,
        start_transaction_id=10_000,
    )
    test_df = make_synthetic_labeled_frame(
        150,
        split_name="test",
        date_start="2018-09-05",
        fraud_rate=0.15,
        nan_rate=0.1,
        seed=30,
        start_transaction_id=20_000,
    )
    return train_df, validation_df, test_df


def test_synthetic_fixture_is_genuinely_chronologically_ordered(synthetic_splits):
    train_df, validation_df, test_df = synthetic_splits
    assert train_df["TX_DATETIME"].max() < validation_df["TX_DATETIME"].min()
    assert validation_df["TX_DATETIME"].max() < test_df["TX_DATETIME"].min()


@pytest.fixture(scope="module")
def xgboost_result(synthetic_splits):
    train_df, validation_df, test_df = synthetic_splits
    return train_xgboost_model(train_df, validation_df, test_df)


# --- model correctness ---


def test_train_xgboost_model_returns_fitted_pipeline_that_predicts_probabilities(xgboost_result, synthetic_splits):
    _, _, test_df = synthetic_splits
    proba = xgboost_result.pipeline.predict_proba(get_feature_matrix(test_df))
    assert proba.shape == (len(test_df), 2)
    assert np.isfinite(proba).all()
    assert np.all((proba >= 0) & (proba <= 1))
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, rtol=1e-5)


def test_pipeline_feature_order_matches_registered_feature_columns(xgboost_result):
    impute_step = xgboost_result.pipeline.named_steps["preprocess"].named_steps["impute_flag"]
    assert impute_step.feature_names_in_ == list(FEATURE_COLUMNS)


def test_train_xgboost_model_is_deterministic_given_fixed_seed(synthetic_splits):
    train_df, validation_df, test_df = synthetic_splits
    result_a = train_xgboost_model(train_df, validation_df, test_df)
    result_b = train_xgboost_model(train_df, validation_df, test_df)

    assert result_a.threshold == pytest.approx(result_b.threshold)
    for key in result_a.validation_metrics:
        assert result_a.validation_metrics[key] == pytest.approx(result_b.validation_metrics[key], nan_ok=True)
    for key in result_a.test_metrics:
        assert result_a.test_metrics[key] == pytest.approx(result_b.test_metrics[key], nan_ok=True)
    pd.testing.assert_frame_equal(
        result_a.feature_importance, result_b.feature_importance, check_exact=False, rtol=1e-5
    )


# --- preprocessing is fit only on train data ---


def test_preprocessing_medians_are_computed_only_from_train_data(xgboost_result, synthetic_splits):
    train_df, _, _ = synthetic_splits
    impute_step = xgboost_result.pipeline.named_steps["preprocess"].named_steps["impute_flag"]

    X_train = get_feature_matrix(train_df)
    expected_medians = X_train.median(axis=0, skipna=True).fillna(0.0)

    pd.testing.assert_series_equal(impute_step.medians_, expected_medians, check_names=False)


def test_preprocessing_scaler_statistics_come_from_train_not_validation_or_test(xgboost_result, synthetic_splits):
    train_df, _, _ = synthetic_splits
    impute_step = xgboost_result.pipeline.named_steps["preprocess"].named_steps["impute_flag"]
    scale_step = xgboost_result.pipeline.named_steps["preprocess"].named_steps["scale"]

    X_train = get_feature_matrix(train_df)
    train_transformed = impute_step.transform(X_train)
    expected_mean = train_transformed.mean(axis=0).to_numpy()

    np.testing.assert_allclose(scale_step.mean_, expected_mean, rtol=1e-8)


# --- no label leakage into the model ---


def test_no_label_or_identifier_columns_reach_the_model_as_features(xgboost_result):
    impute_step = xgboost_result.pipeline.named_steps["preprocess"].named_steps["impute_flag"]
    feature_names_seen_by_model = set(impute_step.feature_names_in_)

    assert feature_names_seen_by_model == set(FEATURE_COLUMNS)
    assert "TX_FRAUD" not in feature_names_seen_by_model
    assert "TX_FRAUD_SCENARIO" not in feature_names_seen_by_model
    assert "TRANSACTION_ID" not in feature_names_seen_by_model
    assert "TX_DATETIME" not in feature_names_seen_by_model


def test_feature_importance_table_covers_every_preprocessed_feature(xgboost_result):
    impute_step = xgboost_result.pipeline.named_steps["preprocess"].named_steps["impute_flag"]
    expected_names = set(impute_step.get_feature_names_out())

    assert set(xgboost_result.feature_importance["feature"]) == expected_names
    assert "TX_FRAUD" not in set(xgboost_result.feature_importance["feature"])
    assert list(xgboost_result.feature_importance.columns) == ["feature", "importance"]

    importances = xgboost_result.feature_importance["importance"].to_numpy()
    assert np.all(importances[:-1] >= importances[1:])
    assert np.all(importances >= 0)


# --- threshold selection is validation-only ---


def test_threshold_is_selected_only_from_validation_not_test(synthetic_splits):
    train_df, validation_df, test_df = synthetic_splits
    baseline = train_xgboost_model(train_df, validation_df, test_df)

    perturbed_test_df = test_df.copy()
    perturbed_test_df["TX_FRAUD"] = 1 - perturbed_test_df["TX_FRAUD"]
    perturbed = train_xgboost_model(train_df, validation_df, perturbed_test_df)

    assert perturbed.threshold == pytest.approx(baseline.threshold)
    for key in baseline.validation_metrics:
        assert perturbed.validation_metrics[key] == pytest.approx(baseline.validation_metrics[key], nan_ok=True)


def test_threshold_selection_changes_when_validation_labels_change(synthetic_splits):
    train_df, validation_df, test_df = synthetic_splits
    baseline = train_xgboost_model(train_df, validation_df, test_df)

    perturbed_validation_df = validation_df.copy()
    perturbed_validation_df["TX_FRAUD"] = 1 - perturbed_validation_df["TX_FRAUD"]
    perturbed = train_xgboost_model(train_df, perturbed_validation_df, test_df)

    assert perturbed.threshold != pytest.approx(baseline.threshold)


# --- hyperparameter selection (validation-tuning) ---


def test_hyperparameter_candidates_all_evaluated_and_recorded(xgboost_result):
    assert len(xgboost_result.hyperparameter_candidates) == len(HYPERPARAMETER_CANDIDATES)
    recorded_hyperparams = [c["hyperparameters"] for c in xgboost_result.hyperparameter_candidates]
    assert recorded_hyperparams == list(HYPERPARAMETER_CANDIDATES)


def test_selected_hyperparameters_have_the_best_validation_pr_auc(xgboost_result):
    candidates = xgboost_result.metadata["hyperparameter_selection"]["candidates"]
    selected = xgboost_result.metadata["hyperparameter_selection"]["selected"]

    best_score = max(c["validation_pr_auc"] for c in candidates)
    selected_score = next(c["validation_pr_auc"] for c in candidates if c["hyperparameters"] == selected)

    assert selected_score == pytest.approx(best_score)


def test_selected_hyperparameters_match_the_fitted_classifier(xgboost_result):
    selected = xgboost_result.metadata["hyperparameter_selection"]["selected"]
    classifier = xgboost_result.pipeline.named_steps["classifier"]

    assert classifier.n_estimators == selected["n_estimators"]
    assert classifier.max_depth == selected["max_depth"]
    assert classifier.learning_rate == pytest.approx(selected["learning_rate"])


def test_hyperparameter_selection_metric_is_pr_auc(xgboost_result):
    assert HYPERPARAMETER_SELECTION_METRIC == "pr_auc"
    assert xgboost_result.metadata["hyperparameter_selection"]["criterion"] == "max_validation_pr_auc"


# --- scale_pos_weight is computed from TRAIN labels only ---


def test_scale_pos_weight_computed_from_train_labels_only(xgboost_result, synthetic_splits):
    train_df, _, _ = synthetic_splits
    n_genuine = int((train_df["TX_FRAUD"] == 0).sum())
    n_fraud = int((train_df["TX_FRAUD"] == 1).sum())
    expected = n_genuine / n_fraud

    assert xgboost_result.metadata["hyperparameters"]["scale_pos_weight"] == pytest.approx(expected)
    assert xgboost_result.pipeline.named_steps["classifier"].scale_pos_weight == pytest.approx(expected)


def test_scale_pos_weight_is_unaffected_by_validation_or_test_class_balance(synthetic_splits):
    train_df, validation_df, test_df = synthetic_splits
    baseline = train_xgboost_model(train_df, validation_df, test_df)

    skewed_validation_df = validation_df.copy()
    skewed_validation_df["TX_FRAUD"] = 1
    skewed_test_df = test_df.copy()
    skewed_test_df["TX_FRAUD"] = 0

    skewed = train_xgboost_model(train_df, skewed_validation_df, skewed_test_df)

    assert skewed.metadata["hyperparameters"]["scale_pos_weight"] == pytest.approx(
        baseline.metadata["hyperparameters"]["scale_pos_weight"]
    )


# --- metadata: output type / threshold selection / lineage ---


def test_metadata_output_type_documents_uncalibrated_score(xgboost_result):
    assert xgboost_result.metadata["output_type"] == OUTPUT_TYPE
    assert OUTPUT_TYPE == "uncalibrated_probability_estimate"
    assert "scale_pos_weight" in xgboost_result.metadata["output_type_notes"]


def test_metadata_threshold_selection_records_criterion_and_grid(xgboost_result):
    selection = xgboost_result.metadata["threshold_selection"]
    assert selection["criterion"] == "max_f1"
    assert selection["evaluated_on"] == "validation"
    assert selection["grid_min"] == pytest.approx(0.01)
    assert selection["grid_max"] == pytest.approx(0.99)
    assert selection["grid_size"] == 99
    assert xgboost_result.metadata["threshold"] == pytest.approx(xgboost_result.threshold)


def test_metadata_feature_lineage_matches_registry_and_package_version(xgboost_result):
    lineage = xgboost_result.metadata["feature_lineage"]
    assert lineage["package_version"] == PACKAGE_VERSION
    assert lineage["feature_count"] == len(FEATURE_COLUMNS)
    assert lineage["feature_columns"] == list(FEATURE_COLUMNS)


def test_metadata_split_lineage_matches_configured_boundaries_and_observed_data(xgboost_result, synthetic_splits):
    train_df, validation_df, test_df = synthetic_splits
    lineage = xgboost_result.metadata["split_lineage"]

    for name, df in (("train", train_df), ("validation", validation_df), ("test", test_df)):
        entry = lineage[name]
        configured_start, configured_end = SPLIT_BOUNDARIES[name]
        assert entry["configured_range"] == {"start": configured_start, "end": configured_end}
        assert entry["observed_date_min"] == str(df["TX_DATETIME"].min())
        assert entry["observed_date_max"] == str(df["TX_DATETIME"].max())
        assert entry["row_count"] == len(df)
        assert entry["fraud_count"] == int(df["TX_FRAUD"].sum())
        assert entry["fraud_rate"] == pytest.approx(df["TX_FRAUD"].mean())

    assert lineage["train"]["observed_date_max"] < lineage["validation"]["observed_date_min"]
    assert lineage["validation"]["observed_date_max"] < lineage["test"]["observed_date_min"]


def test_metadata_contains_validation_and_test_metrics_matching_result_fields(xgboost_result):
    assert xgboost_result.metadata["validation_metrics"] == xgboost_result.validation_metrics
    assert xgboost_result.metadata["test_metrics"] == xgboost_result.test_metrics


def test_metadata_model_version_and_type(xgboost_result):
    assert xgboost_result.metadata["model_version"] == MODEL_VERSION
    assert xgboost_result.metadata["model_type"] == "XGBClassifier"
    assert xgboost_result.metadata["random_seed"] == 42


# --- error analysis ---


def test_error_analysis_counts_are_internally_consistent(xgboost_result, synthetic_splits):
    _, _, test_df = synthetic_splits
    ea = xgboost_result.error_analysis

    assert ea["true_positive_count"] + ea["false_negative_count"] == ea["total_actual_fraud"]
    assert ea["true_negative_count"] + ea["false_positive_count"] == ea["total_genuine"]
    assert ea["total_actual_fraud"] + ea["total_genuine"] == len(test_df)
    assert ea["total_actual_fraud"] == int(test_df["TX_FRAUD"].sum())

    for stats in ea["recall_by_scenario"].values():
        assert stats["detected"] <= stats["total_fraud"]
        if stats["total_fraud"] > 0:
            assert stats["recall"] == pytest.approx(stats["detected"] / stats["total_fraud"])
        else:
            assert stats["recall"] is None


# --- persistence: save -> load -> identical predictions, metadata/lineage preserved ---


def test_real_xgboost_result_metadata_and_predictions_round_trip_exactly(xgboost_result, synthetic_splits, tmp_path):
    _, _, test_df = synthetic_splits
    version_dir = tmp_path / MODEL_VERSION

    save_model(xgboost_result.pipeline, xgboost_result.metadata, version_dir)
    loaded_pipeline, loaded_metadata = load_model(version_dir)

    assert loaded_metadata == xgboost_result.metadata
    assert loaded_metadata["threshold"] == pytest.approx(xgboost_result.threshold)
    assert loaded_metadata["hyperparameter_selection"] == xgboost_result.metadata["hyperparameter_selection"]
    assert loaded_metadata["split_lineage"] == xgboost_result.metadata["split_lineage"]

    X_test = get_feature_matrix(test_df)
    loaded_proba = loaded_pipeline.predict_proba(X_test)
    original_proba = xgboost_result.pipeline.predict_proba(X_test)
    np.testing.assert_array_equal(loaded_proba, original_proba)
