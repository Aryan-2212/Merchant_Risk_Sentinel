"""Tests for mrs.models.train.train_baseline -- small, fast, fully-synthetic data (NOT
the real 1.75M-row dataset; that's covered separately once we run the real baseline).

Focus: chronological train/validation/test separation, preprocessing fit only on train,
threshold selection only on validation, and no label leakage into the model (Dev Plan
Sec 5/6/33.6/37).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mrs import __version__ as PACKAGE_VERSION
from mrs.data.splits import SPLIT_BOUNDARIES
from mrs.models.dataset import FEATURE_COLUMNS, get_feature_matrix
from mrs.models.train import MODEL_VERSION, OUTPUT_TYPE, train_baseline
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
    # Sanity on the fixture itself: these tests only mean something if train really
    # precedes validation really precedes test in time.
    train_df, validation_df, test_df = synthetic_splits
    assert train_df["TX_DATETIME"].max() < validation_df["TX_DATETIME"].min()
    assert validation_df["TX_DATETIME"].max() < test_df["TX_DATETIME"].min()


@pytest.fixture(scope="module")
def baseline_result(synthetic_splits):
    train_df, validation_df, test_df = synthetic_splits
    return train_baseline(train_df, validation_df, test_df)


# --- basic pipeline correctness ---


def test_train_baseline_returns_fitted_pipeline_that_predicts_probabilities(baseline_result, synthetic_splits):
    _, _, test_df = synthetic_splits
    proba = baseline_result.pipeline.predict_proba(get_feature_matrix(test_df))
    assert proba.shape == (len(test_df), 2)
    assert np.all((proba >= 0) & (proba <= 1))
    np.testing.assert_allclose(proba.sum(axis=1), 1.0)


def test_pipeline_feature_order_matches_registered_feature_columns(baseline_result):
    impute_step = baseline_result.pipeline.named_steps["preprocess"].named_steps["impute_flag"]
    assert impute_step.feature_names_in_ == list(FEATURE_COLUMNS)


# --- preprocessing is fit only on train data ---


def test_preprocessing_medians_are_computed_only_from_train_data(baseline_result, synthetic_splits):
    train_df, _, _ = synthetic_splits
    impute_step = baseline_result.pipeline.named_steps["preprocess"].named_steps["impute_flag"]

    X_train = get_feature_matrix(train_df)
    expected_medians = X_train.median(axis=0, skipna=True).fillna(0.0)

    pd.testing.assert_series_equal(impute_step.medians_, expected_medians, check_names=False)


def test_preprocessing_scaler_statistics_come_from_train_not_validation_or_test(baseline_result, synthetic_splits):
    train_df, validation_df, test_df = synthetic_splits
    impute_step = baseline_result.pipeline.named_steps["preprocess"].named_steps["impute_flag"]
    scale_step = baseline_result.pipeline.named_steps["preprocess"].named_steps["scale"]

    # Recompute what the scaler SHOULD have been fit on: the imputer's own transform of
    # train (never validation/test), by construction of a fresh, identically-fit imputer.
    X_train = get_feature_matrix(train_df)
    train_transformed = impute_step.transform(X_train)
    expected_mean = train_transformed.mean(axis=0).to_numpy()

    np.testing.assert_allclose(scale_step.mean_, expected_mean, rtol=1e-8)


# --- no label leakage into the model ---


def test_no_label_or_identifier_columns_reach_the_model_as_features(baseline_result):
    impute_step = baseline_result.pipeline.named_steps["preprocess"].named_steps["impute_flag"]
    feature_names_seen_by_model = set(impute_step.feature_names_in_)

    assert feature_names_seen_by_model == set(FEATURE_COLUMNS)
    assert "TX_FRAUD" not in feature_names_seen_by_model
    assert "TX_FRAUD_SCENARIO" not in feature_names_seen_by_model
    assert "TRANSACTION_ID" not in feature_names_seen_by_model
    assert "TX_DATETIME" not in feature_names_seen_by_model


def test_coefficient_table_covers_every_preprocessed_feature_and_is_sorted(baseline_result):
    impute_step = baseline_result.pipeline.named_steps["preprocess"].named_steps["impute_flag"]
    expected_names = set(impute_step.get_feature_names_out())

    assert set(baseline_result.coefficients["feature"]) == expected_names
    assert "TX_FRAUD" not in set(baseline_result.coefficients["feature"])
    assert list(baseline_result.coefficients.columns) == ["feature", "coefficient", "abs_coefficient"]

    abs_coefs = baseline_result.coefficients["abs_coefficient"].to_numpy()
    assert np.all(abs_coefs[:-1] >= abs_coefs[1:])  # sorted descending


# --- threshold selection is validation-only ---


def test_threshold_is_selected_only_from_validation_not_test(synthetic_splits):
    train_df, validation_df, test_df = synthetic_splits
    baseline = train_baseline(train_df, validation_df, test_df)

    # Drastically perturb the test set's labels. If threshold selection depended on test
    # in any way, this would change it. It must not (Dev Plan Sec 6/37).
    perturbed_test_df = test_df.copy()
    perturbed_test_df["TX_FRAUD"] = 1 - perturbed_test_df["TX_FRAUD"]
    perturbed = train_baseline(train_df, validation_df, perturbed_test_df)

    assert perturbed.threshold == pytest.approx(baseline.threshold)
    for key in baseline.validation_metrics:
        assert perturbed.validation_metrics[key] == pytest.approx(baseline.validation_metrics[key], nan_ok=True)


def test_threshold_selection_changes_when_validation_labels_change(synthetic_splits):
    # Converse of the above: threshold selection MUST be sensitive to validation, or the
    # previous test would be vacuously true (e.g. if the threshold were simply hardcoded).
    train_df, validation_df, test_df = synthetic_splits
    baseline = train_baseline(train_df, validation_df, test_df)

    perturbed_validation_df = validation_df.copy()
    perturbed_validation_df["TX_FRAUD"] = 1 - perturbed_validation_df["TX_FRAUD"]
    perturbed = train_baseline(train_df, perturbed_validation_df, test_df)

    assert perturbed.threshold != pytest.approx(baseline.threshold)


# --- determinism ---


def test_train_baseline_is_deterministic_given_fixed_seed(synthetic_splits):
    train_df, validation_df, test_df = synthetic_splits
    result_a = train_baseline(train_df, validation_df, test_df)
    result_b = train_baseline(train_df, validation_df, test_df)

    assert result_a.threshold == pytest.approx(result_b.threshold)
    for key in result_a.validation_metrics:
        assert result_a.validation_metrics[key] == pytest.approx(result_b.validation_metrics[key], nan_ok=True)
    for key in result_a.test_metrics:
        assert result_a.test_metrics[key] == pytest.approx(result_b.test_metrics[key], nan_ok=True)
    pd.testing.assert_frame_equal(result_a.coefficients, result_b.coefficients, check_exact=False, rtol=1e-6)


# --- metadata: output type / threshold selection / lineage ---


def test_metadata_output_type_documents_uncalibrated_score(baseline_result):
    assert baseline_result.metadata["output_type"] == OUTPUT_TYPE == "uncalibrated_probability_estimate"
    assert "class_weight='balanced'" in baseline_result.metadata["output_type_notes"]


def test_metadata_threshold_selection_records_criterion_and_grid(baseline_result):
    selection = baseline_result.metadata["threshold_selection"]
    assert selection["criterion"] == "max_f1"
    assert selection["evaluated_on"] == "validation"
    assert selection["grid_min"] == pytest.approx(0.01)
    assert selection["grid_max"] == pytest.approx(0.99)
    assert selection["grid_size"] == 99
    assert baseline_result.metadata["threshold"] == pytest.approx(baseline_result.threshold)


def test_metadata_feature_lineage_matches_registry_and_package_version(baseline_result):
    lineage = baseline_result.metadata["feature_lineage"]
    assert lineage["package_version"] == PACKAGE_VERSION
    assert lineage["feature_count"] == len(FEATURE_COLUMNS)
    assert lineage["feature_columns"] == list(FEATURE_COLUMNS)


def test_metadata_split_lineage_matches_configured_boundaries_and_observed_data(baseline_result, synthetic_splits):
    train_df, validation_df, test_df = synthetic_splits
    lineage = baseline_result.metadata["split_lineage"]

    for name, df in (("train", train_df), ("validation", validation_df), ("test", test_df)):
        entry = lineage[name]
        configured_start, configured_end = SPLIT_BOUNDARIES[name]
        assert entry["configured_range"] == {"start": configured_start, "end": configured_end}
        assert entry["observed_date_min"] == str(df["TX_DATETIME"].min())
        assert entry["observed_date_max"] == str(df["TX_DATETIME"].max())
        assert entry["row_count"] == len(df)
        assert entry["fraud_count"] == int(df["TX_FRAUD"].sum())
        assert entry["fraud_rate"] == pytest.approx(df["TX_FRAUD"].mean())

    # The recorded lineage itself must reflect genuine chronological separation.
    assert lineage["train"]["observed_date_max"] < lineage["validation"]["observed_date_min"]
    assert lineage["validation"]["observed_date_max"] < lineage["test"]["observed_date_min"]


def test_metadata_contains_validation_and_test_metrics_matching_result_fields(baseline_result):
    assert baseline_result.metadata["validation_metrics"] == baseline_result.validation_metrics
    assert baseline_result.metadata["test_metrics"] == baseline_result.test_metrics


def test_metadata_model_version_and_hyperparameters(baseline_result):
    assert baseline_result.metadata["model_version"] == MODEL_VERSION
    assert baseline_result.metadata["model_type"] == "LogisticRegression"
    assert baseline_result.metadata["hyperparameters"]["class_weight"] == "balanced"
    assert baseline_result.metadata["random_seed"] == 42


# --- error analysis ---


def test_error_analysis_counts_are_internally_consistent(baseline_result, synthetic_splits):
    _, _, test_df = synthetic_splits
    ea = baseline_result.error_analysis

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


def test_error_analysis_recall_by_scenario_matches_independent_manual_computation(baseline_result, synthetic_splits):
    _, _, test_df = synthetic_splits
    pipeline = baseline_result.pipeline
    threshold = baseline_result.threshold

    X_test = get_feature_matrix(test_df)
    y_pred = (pipeline.predict_proba(X_test)[:, 1] >= threshold).astype(int)

    manual = pd.DataFrame(
        {"TX_FRAUD_SCENARIO": test_df["TX_FRAUD_SCENARIO"].to_numpy(), "TX_FRAUD": test_df["TX_FRAUD"].to_numpy(), "y_pred": y_pred}
    )
    fraud_rows = manual[manual["TX_FRAUD"] == 1]

    for scenario, stats in baseline_result.error_analysis["recall_by_scenario"].items():
        scenario_rows = fraud_rows[fraud_rows["TX_FRAUD_SCENARIO"] == scenario]
        assert stats["total_fraud"] == len(scenario_rows)
        assert stats["detected"] == int((scenario_rows["y_pred"] == 1).sum())
