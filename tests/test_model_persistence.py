"""Tests for mrs.models.persistence -- save/load round-trip, prediction equivalence,
metadata/threshold/lineage preservation, and failure behavior on invalid/missing
artifacts (Dev Plan Sec 36: a saved model version must be exactly reconstructable later).

Uses only tiny synthetic data -- never the real 1.75M-row dataset.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from mrs.models.dataset import get_feature_matrix
from mrs.models.persistence import METADATA_FILENAME, MODEL_FILENAME, load_model, save_model
from mrs.models.train import train_baseline
from tests.model_test_helpers import make_synthetic_labeled_frame


def _toy_pipeline() -> Pipeline:
    pipeline = Pipeline([("classifier", LogisticRegression())])
    X = np.array([[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]])
    y = np.array([0, 0, 0, 1, 1, 1])
    pipeline.fit(X, y)
    return pipeline


# --- save/load round-trip ---


def test_save_and_load_round_trip_preserves_metadata(tmp_path):
    pipeline = _toy_pipeline()
    metadata = {"model_version": "test_v1", "threshold": 0.42, "notes": "toy"}
    version_dir = tmp_path / "test_v1"

    save_model(pipeline, metadata, version_dir)
    _, loaded_metadata = load_model(version_dir)

    assert loaded_metadata == metadata


def test_save_and_load_round_trip_preserves_predictions_exactly(tmp_path):
    pipeline = _toy_pipeline()
    version_dir = tmp_path / "test_v1"
    save_model(pipeline, {"a": 1}, version_dir)

    loaded_pipeline, _ = load_model(version_dir)

    X_query = np.array([[0.5], [2.5], [4.5]])
    np.testing.assert_array_equal(loaded_pipeline.predict(X_query), pipeline.predict(X_query))
    np.testing.assert_array_equal(loaded_pipeline.predict_proba(X_query), pipeline.predict_proba(X_query))


def test_save_model_creates_the_expected_files(tmp_path):
    pipeline = _toy_pipeline()
    version_dir = tmp_path / "v1"

    save_model(pipeline, {"x": 1}, version_dir)

    assert (version_dir / MODEL_FILENAME).exists()
    assert (version_dir / METADATA_FILENAME).exists()
    assert MODEL_FILENAME == "model.joblib"
    assert METADATA_FILENAME == "metadata.json"


def test_save_model_metadata_json_is_sorted_and_readable_on_disk(tmp_path):
    pipeline = _toy_pipeline()
    version_dir = tmp_path / "v1"
    metadata = {"zeta": 1, "alpha": 2, "middle": {"b": 1, "a": 2}}

    save_model(pipeline, metadata, version_dir)

    raw_text = (version_dir / METADATA_FILENAME).read_text()
    reparsed = json.loads(raw_text)
    assert reparsed == metadata
    # sort_keys=True: top-level keys must appear in the file in alphabetical order.
    assert raw_text.index('"alpha"') < raw_text.index('"middle"') < raw_text.index('"zeta"')


# --- refusing to overwrite an existing version ---


def test_save_model_refuses_to_overwrite_existing_version_dir(tmp_path):
    pipeline = _toy_pipeline()
    version_dir = tmp_path / "test_v1"
    save_model(pipeline, {"a": 1}, version_dir)

    with pytest.raises(FileExistsError):
        save_model(pipeline, {"a": 2}, version_dir)

    _, metadata = load_model(version_dir)
    assert metadata == {"a": 1}  # original artifacts untouched by the failed second save


# --- failure behavior for invalid/missing artifacts ---


def test_load_model_raises_when_version_dir_does_not_exist(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_model(tmp_path / "never_created")


def test_load_model_raises_when_model_file_is_missing(tmp_path):
    version_dir = tmp_path / "v1"
    version_dir.mkdir()
    (version_dir / METADATA_FILENAME).write_text(json.dumps({"a": 1}))
    # No model.joblib written.

    with pytest.raises(FileNotFoundError):
        load_model(version_dir)


def test_load_model_raises_when_metadata_file_is_missing(tmp_path):
    import joblib

    version_dir = tmp_path / "v1"
    version_dir.mkdir()
    joblib.dump(_toy_pipeline(), version_dir / MODEL_FILENAME)
    # No metadata.json written.

    with pytest.raises(FileNotFoundError):
        load_model(version_dir)


# --- realistic Phase 4 metadata/threshold/lineage preservation, via a genuine trained baseline ---


def test_real_baseline_metadata_and_predictions_round_trip_exactly(tmp_path):
    train_df = make_synthetic_labeled_frame(
        60, split_name="train", date_start="2018-04-05", fraud_rate=0.2, nan_rate=0.1, seed=1,
        start_transaction_id=0,
    )
    validation_df = make_synthetic_labeled_frame(
        30, split_name="validation", date_start="2018-08-05", fraud_rate=0.2, nan_rate=0.1, seed=2,
        start_transaction_id=1_000,
    )
    test_df = make_synthetic_labeled_frame(
        30, split_name="test", date_start="2018-09-05", fraud_rate=0.2, nan_rate=0.1, seed=3,
        start_transaction_id=2_000,
    )
    result = train_baseline(train_df, validation_df, test_df)

    version_dir = tmp_path / "logreg_baseline_v1"
    save_model(result.pipeline, result.metadata, version_dir)
    loaded_pipeline, loaded_metadata = load_model(version_dir)

    assert loaded_metadata == result.metadata
    # Threshold and lineage are nested inside metadata -- confirm they survived intact.
    assert loaded_metadata["threshold"] == pytest.approx(result.threshold)
    assert loaded_metadata["threshold_selection"] == result.metadata["threshold_selection"]
    assert loaded_metadata["feature_lineage"]["feature_columns"] == result.metadata["feature_lineage"]["feature_columns"]
    assert loaded_metadata["split_lineage"] == result.metadata["split_lineage"]

    X_test = get_feature_matrix(test_df)
    np.testing.assert_array_equal(
        loaded_pipeline.predict_proba(X_test), result.pipeline.predict_proba(X_test)
    )
