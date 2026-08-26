"""Tests for mrs.models.preprocessing -- median imputation with missingness flags, then
standard scaling, fit on train only (Dev Plan Sec 5/33.6/33.7).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from mrs.models.dataset import FEATURE_COLUMNS
from mrs.models.preprocessing import MedianImputerWithFlags, build_preprocessing_pipeline
from tests.model_test_helpers import FEATURE_COLUMNS_FOR_TESTS, make_synthetic_features

# --- median statistics fit from training data only ---


def test_fit_computes_medians_from_train_only_and_flags_columns_with_missing_train_values():
    train = pd.DataFrame(
        {
            "always_present": [1.0, 2.0, 3.0, 4.0],
            "sometimes_missing": [10.0, np.nan, 30.0, np.nan],
            "all_missing": [np.nan, np.nan, np.nan, np.nan],
        }
    )
    imputer = MedianImputerWithFlags().fit(train)

    assert imputer.feature_names_in_ == ["always_present", "sometimes_missing", "all_missing"]
    assert imputer.medians_["always_present"] == pytest.approx(2.5)
    assert imputer.medians_["sometimes_missing"] == pytest.approx(20.0)
    assert imputer.medians_["all_missing"] == pytest.approx(0.0)  # fallback: no non-null train values
    assert set(imputer.flagged_columns_) == {"sometimes_missing", "all_missing"}
    assert "always_present" not in imputer.flagged_columns_


# --- NaNs are imputed ---


def test_transform_imputes_nans_with_the_fitted_train_medians():
    train = pd.DataFrame({"sometimes_missing": [10.0, np.nan, 30.0, np.nan]})
    imputer = MedianImputerWithFlags().fit(train)

    out = imputer.transform(train)

    assert list(out["sometimes_missing"]) == [10.0, 20.0, 30.0, 20.0]
    assert not out["sometimes_missing"].isna().any()


# --- missingness flags are generated correctly ---


def test_missingness_flags_are_binary_and_match_which_rows_were_actually_nan():
    train = pd.DataFrame({"x": [1.0, np.nan, 3.0, np.nan, 5.0]})
    imputer = MedianImputerWithFlags().fit(train)

    out = imputer.transform(train)

    assert list(out["x_was_missing"]) == [0.0, 1.0, 0.0, 1.0, 0.0]
    assert set(out["x_was_missing"].unique()) <= {0.0, 1.0}


# --- validation/test transformation does not recompute medians ---


def test_transform_on_new_data_does_not_recompute_medians_or_flagged_columns():
    train = pd.DataFrame({"a": [1.0, 2.0, np.nan, 4.0], "b": [10.0, 20.0, 30.0, 40.0]})
    imputer = MedianImputerWithFlags().fit(train)
    medians_before = imputer.medians_.copy()
    flagged_before = list(imputer.flagged_columns_)

    wildly_different = pd.DataFrame({"a": [9000.0, np.nan, 8000.0], "b": [-500.0, -600.0, np.nan]})
    imputer.transform(wildly_different)

    pd.testing.assert_series_equal(imputer.medians_, medians_before)
    assert imputer.flagged_columns_ == flagged_before


# --- a feature that becomes missing only at inference is handled correctly ---


def test_feature_missing_only_at_inference_time_is_still_safely_imputed_without_a_flag_column():
    train = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": [10.0, np.nan, 30.0, 40.0]})
    imputer = MedianImputerWithFlags().fit(train)
    assert "a" not in imputer.flagged_columns_  # "a" was never missing in train

    inference = pd.DataFrame({"a": [np.nan, 2.0], "b": [10.0, 20.0]})
    out = imputer.transform(inference)

    assert "a_was_missing" not in out.columns  # no flag column exists for "a" at all
    assert out["a"].iloc[0] == pytest.approx(imputer.medians_["a"])  # still safely filled
    assert not out["a"].isna().any()


# --- columns remain deterministic ---


def test_get_feature_names_out_and_transform_column_order_are_deterministic_across_calls():
    train = pd.DataFrame({"z": [1.0, np.nan], "a": [2.0, 3.0], "m": [np.nan, 5.0]})
    imputer = MedianImputerWithFlags().fit(train)

    names_1 = list(imputer.get_feature_names_out())
    names_2 = list(imputer.get_feature_names_out())
    assert names_1 == names_2 == ["z", "a", "m", "z_was_missing", "m_was_missing"]

    out_1 = imputer.transform(train)
    out_2 = imputer.transform(train)
    assert list(out_1.columns) == list(out_2.columns) == names_1
    pd.testing.assert_frame_equal(out_1, out_2)


# --- all output values are finite ---


def test_toy_pipeline_output_contains_no_nans_or_infs():
    train = pd.DataFrame({"a": [1.0, np.nan, 3.0, 4.0], "b": [np.nan, np.nan, np.nan, np.nan]})
    pipeline = build_preprocessing_pipeline()

    transformed = pipeline.fit_transform(train)

    assert np.isfinite(transformed).all()
    assert transformed.shape == (4, 4)  # a, b, a_was_missing, b_was_missing


# --- StandardScaler is part of the pipeline ---


def test_pipeline_contains_imputer_then_standard_scaler_in_order():
    pipeline = build_preprocessing_pipeline()

    assert list(pipeline.named_steps.keys()) == ["impute_flag", "scale"]
    assert isinstance(pipeline.named_steps["impute_flag"], MedianImputerWithFlags)
    assert isinstance(pipeline.named_steps["scale"], StandardScaler)


# --- the preprocessing pipeline can handle the real registered 33-feature schema ---


def test_pipeline_handles_the_real_registered_33_feature_schema():
    assert FEATURE_COLUMNS_FOR_TESTS == FEATURE_COLUMNS
    features = make_synthetic_features(50, split_name="train", date_start="2018-04-01", nan_rate=0.15, seed=11)
    X = features[list(FEATURE_COLUMNS)]

    pipeline = build_preprocessing_pipeline()
    transformed = pipeline.fit_transform(X)

    assert transformed.shape[0] == len(X)
    assert transformed.shape[1] >= len(FEATURE_COLUMNS)  # widened by missingness flag columns
    assert np.isfinite(transformed).all()


# --- cold-start NaNs do not reach Logistic Regression ---


def test_cold_start_nans_do_not_reach_logistic_regression():
    features = make_synthetic_features(60, split_name="train", date_start="2018-04-01", nan_rate=0.2, seed=42)
    X = features[list(FEATURE_COLUMNS_FOR_TESTS)]
    rng = np.random.default_rng(0)
    y = rng.permutation(np.tile([0, 1], 30))

    # Sanity check: raw NaN-containing input really would break LogisticRegression
    # directly -- proves the pipeline's preprocessing step is doing necessary work.
    with pytest.raises(ValueError):
        LogisticRegression(max_iter=200).fit(X, y)

    pipeline = Pipeline(
        [
            ("preprocess", build_preprocessing_pipeline()),
            ("clf", LogisticRegression(max_iter=200)),
        ]
    )
    pipeline.fit(X, y)  # must not raise

    predictions = pipeline.predict_proba(X)
    assert predictions.shape == (len(X), 2)
    assert np.isfinite(predictions).all()


# --- the actual leakage property: transform must use TRAIN statistics, not validation's ---


def test_scaling_and_imputation_reflect_train_statistics_not_validation_statistics():
    train = pd.DataFrame(
        {
            "a": [10.0, 12.0, 11.0, 9.0, np.nan],
            "b": [100.0, 102.0, 98.0, 101.0, 99.0],
        }
    )
    # Validation has a radically different distribution for column "a" -- if the
    # imputer/scaler ever recomputed statistics from validation, this would be detected.
    validation = pd.DataFrame(
        {
            "a": [5000.0, np.nan, 4800.0],
            "b": [100.0, 100.0, 100.0],
        }
    )

    pipeline = build_preprocessing_pipeline()
    pipeline.fit(train)

    impute_step = pipeline.named_steps["impute_flag"]
    scale_step = pipeline.named_steps["scale"]

    train_median_a = train["a"].median()
    assert impute_step.medians_["a"] == pytest.approx(train_median_a)

    output_columns = list(impute_step.get_feature_names_out())
    a_index = output_columns.index("a")

    transformed_validation = pipeline.transform(validation)
    transformed_df = pd.DataFrame(transformed_validation, columns=output_columns)

    # Manually reproduce the TRAIN-fitted pipeline's math end to end: impute validation's
    # NaN with the TRAIN median, then scale with the TRAIN-fitted mean_/scale_.
    expected_imputed_a = validation["a"].fillna(train_median_a)
    expected_scaled_a = (expected_imputed_a - scale_step.mean_[a_index]) / scale_step.scale_[a_index]

    np.testing.assert_allclose(transformed_df["a"].to_numpy(), expected_scaled_a.to_numpy())

    # And the leakage check itself: this must NOT match what you'd get by (incorrectly)
    # standardizing using validation's own wildly different statistics. ddof=0 matches
    # StandardScaler's own population-variance convention, so this is an exact simulation
    # of "the scaler was wrongly fit on validation", not just a loose approximation.
    wrongly_leaked_mean = expected_imputed_a.mean()
    wrongly_leaked_std = expected_imputed_a.std(ddof=0)
    wrongly_leaked = (expected_imputed_a - wrongly_leaked_mean) / wrongly_leaked_std

    assert not np.allclose(transformed_df["a"].to_numpy(), wrongly_leaked.to_numpy())
