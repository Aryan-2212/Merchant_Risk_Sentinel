"""Tests for mrs.models.dataset -- the Phase 4 boundary where TX_FRAUD/TX_FRAUD_SCENARIO
are joined onto the Phase 3 feature layer (Dev Plan Sec 34.1: labels never live inside
mrs.features, only here, and only for training/evaluation, never as model inputs).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mrs import config
from mrs.data.schema import LABEL_COLUMNS
from mrs.features.registry import FEATURE_NAMES
from mrs.models.dataset import (
    FEATURE_COLUMNS,
    attach_labels,
    get_feature_matrix,
    load_processed_transactions,
    load_split,
)
from tests.model_test_helpers import make_synthetic_features, make_synthetic_labels

# --- FEATURE_COLUMNS ---


def test_feature_columns_is_sorted_registry_names():
    assert FEATURE_COLUMNS == tuple(sorted(FEATURE_NAMES))
    assert set(FEATURE_COLUMNS) == set(FEATURE_NAMES)


def test_feature_columns_excludes_label_columns():
    assert not (LABEL_COLUMNS & set(FEATURE_COLUMNS))


def test_feature_columns_has_the_expected_registry_count():
    # 5 transaction + 12 customer + 14 terminal + 2 relationship (Phase 3 registry).
    # Pinned so a silent registry drift shows up here, not just in mrs.features' own tests.
    assert len(FEATURE_COLUMNS) == 33


# --- attach_labels ---


def test_attach_labels_joins_by_transaction_id_regardless_of_row_order():
    features = make_synthetic_features(5, split_name="train", date_start="2018-04-01", nan_rate=0.0, seed=1)
    labels = make_synthetic_labels(features["TRANSACTION_ID"], fraud_rate=0.4, seed=2)
    shuffled_labels = labels.sample(frac=1, random_state=0).reset_index(drop=True)

    joined = attach_labels(features, shuffled_labels)

    assert len(joined) == len(features)
    expected = labels.set_index("TRANSACTION_ID")["TX_FRAUD"].sort_index()
    actual = joined.set_index("TRANSACTION_ID")["TX_FRAUD"].sort_index()
    pd.testing.assert_series_equal(actual, expected, check_names=False)


def test_attach_labels_preserves_row_count_and_transaction_id_set():
    features = make_synthetic_features(8, split_name="train", date_start="2018-04-01", nan_rate=0.1, seed=7)
    labels = make_synthetic_labels(features["TRANSACTION_ID"], fraud_rate=0.3, seed=8)

    joined = attach_labels(features, labels)

    assert len(joined) == len(features)
    assert set(joined["TRANSACTION_ID"]) == set(features["TRANSACTION_ID"])
    assert not joined["TRANSACTION_ID"].duplicated().any()


def test_attach_labels_attaches_exact_tx_fraud_and_scenario_values():
    features = pd.DataFrame({"TRANSACTION_ID": [10, 11, 12], "some_feature": [1.0, 2.0, 3.0]})
    labels = pd.DataFrame(
        {
            "TRANSACTION_ID": [12, 10, 11],
            "TX_FRAUD": [1, 0, 0],
            "TX_FRAUD_SCENARIO": [2, 0, 0],
        }
    )

    joined = attach_labels(features, labels)
    by_id = joined.set_index("TRANSACTION_ID")

    assert by_id.loc[10, "TX_FRAUD"] == 0
    assert by_id.loc[10, "TX_FRAUD_SCENARIO"] == 0
    assert by_id.loc[11, "TX_FRAUD"] == 0
    assert by_id.loc[12, "TX_FRAUD"] == 1
    assert by_id.loc[12, "TX_FRAUD_SCENARIO"] == 2
    assert list(joined["TRANSACTION_ID"]) == [10, 11, 12]


def test_attach_labels_raises_on_missing_label_columns():
    features = make_synthetic_features(3, split_name="train", date_start="2018-04-01", nan_rate=0.0, seed=1)
    bad_labels = pd.DataFrame({"TRANSACTION_ID": features["TRANSACTION_ID"]})

    with pytest.raises(ValueError, match="missing columns"):
        attach_labels(features, bad_labels)


def test_attach_labels_raises_on_non_one_to_one_duplicate_label_rows():
    features = make_synthetic_features(3, split_name="train", date_start="2018-04-01", nan_rate=0.0, seed=1)
    labels = make_synthetic_labels(features["TRANSACTION_ID"], fraud_rate=0.4, seed=2)
    duplicated = pd.concat([labels, labels.iloc[[0]]], ignore_index=True)

    with pytest.raises(ValueError):
        attach_labels(features, duplicated)


def test_attach_labels_raises_when_a_transaction_id_has_no_label():
    features = make_synthetic_features(5, split_name="train", date_start="2018-04-01", nan_rate=0.0, seed=1)
    labels = make_synthetic_labels(features["TRANSACTION_ID"][:-1], fraud_rate=0.4, seed=2)

    with pytest.raises(ValueError, match="row count changed"):
        attach_labels(features, labels)


def test_attach_labels_succeeds_when_labels_source_has_extra_unrelated_ids():
    features = make_synthetic_features(5, split_name="train", date_start="2018-04-01", nan_rate=0.0, seed=1)
    extra_ids = features["TRANSACTION_ID"].max() + np.arange(1, 6)
    all_ids = pd.concat([features["TRANSACTION_ID"], pd.Series(extra_ids)], ignore_index=True)
    labels = make_synthetic_labels(all_ids, fraud_rate=0.4, seed=2)

    joined = attach_labels(features, labels)

    assert len(joined) == len(features)
    assert set(joined["TRANSACTION_ID"]) == set(features["TRANSACTION_ID"])


# --- get_feature_matrix ---


def test_get_feature_matrix_selects_only_registered_columns_in_canonical_order():
    features = make_synthetic_features(4, split_name="train", date_start="2018-04-01", nan_rate=0.0, seed=1)
    labels = make_synthetic_labels(features["TRANSACTION_ID"], fraud_rate=0.5, seed=2)
    joined = attach_labels(features, labels)

    X = get_feature_matrix(joined)

    assert list(X.columns) == list(FEATURE_COLUMNS)
    assert len(X) == len(joined)


def test_get_feature_matrix_never_leaks_labels_or_identifiers():
    features = make_synthetic_features(4, split_name="train", date_start="2018-04-01", nan_rate=0.0, seed=1)
    labels = make_synthetic_labels(features["TRANSACTION_ID"], fraud_rate=0.5, seed=2)
    joined = attach_labels(features, labels)

    X = get_feature_matrix(joined)

    assert "TX_FRAUD" not in X.columns
    assert "TX_FRAUD_SCENARIO" not in X.columns
    assert "TRANSACTION_ID" not in X.columns
    assert "TX_DATETIME" not in X.columns
    assert "split" not in X.columns


def test_get_feature_matrix_column_order_is_independent_of_input_order():
    features = make_synthetic_features(4, split_name="train", date_start="2018-04-01", nan_rate=0.0, seed=1)
    labels = make_synthetic_labels(features["TRANSACTION_ID"], fraud_rate=0.5, seed=2)
    joined = attach_labels(features, labels)
    reversed_columns = list(joined.columns)[::-1]
    reordered = joined[reversed_columns]

    X = get_feature_matrix(reordered)

    assert list(X.columns) == list(FEATURE_COLUMNS)


def test_get_feature_matrix_raises_when_a_feature_column_is_missing():
    features = make_synthetic_features(4, split_name="train", date_start="2018-04-01", nan_rate=0.0, seed=1)
    incomplete = features.drop(columns=[FEATURE_COLUMNS[0]])

    with pytest.raises(ValueError, match="missing feature columns"):
        get_feature_matrix(incomplete)


# --- load_processed_transactions / load_split (temp paths, never the real 1.75M dataset) ---


@pytest.fixture
def fake_data_layout(tmp_path, monkeypatch):
    processed_dir = tmp_path / "processed" / "transactions"
    features_dir = tmp_path / "features"
    processed_dir.mkdir(parents=True)
    features_dir.mkdir(parents=True)
    monkeypatch.setattr(config, "PROCESSED_TRANSACTIONS_DIR", processed_dir)
    monkeypatch.setattr(config, "FEATURES_DIR", features_dir)
    return processed_dir, features_dir


def test_load_split_reads_persisted_parquet_and_attaches_labels(fake_data_layout):
    processed_dir, features_dir = fake_data_layout

    features = make_synthetic_features(10, split_name="train", date_start="2018-04-01", nan_rate=0.1, seed=3)
    features.to_parquet(features_dir / "features_train.parquet", index=False)

    labels = make_synthetic_labels(features["TRANSACTION_ID"], fraud_rate=0.3, seed=4)
    processed = labels.copy()
    processed["TX_DATETIME"] = features["TX_DATETIME"].to_numpy()
    processed.to_parquet(processed_dir / "transactions_2018-04.parquet", index=False)

    result = load_split("train")

    assert len(result) == len(features)
    assert "TX_FRAUD" in result.columns
    assert "TX_FRAUD_SCENARIO" in result.columns
    for col in FEATURE_COLUMNS:
        assert col in result.columns
    pd.testing.assert_series_equal(
        result.set_index("TRANSACTION_ID")["TX_FRAUD"].sort_index(),
        labels.set_index("TRANSACTION_ID")["TX_FRAUD"].sort_index(),
        check_names=False,
    )


def test_load_split_raises_file_not_found_when_split_file_absent(fake_data_layout):
    with pytest.raises(FileNotFoundError):
        load_split("validation")


def test_load_processed_transactions_raises_file_not_found_when_empty(fake_data_layout):
    with pytest.raises(FileNotFoundError):
        load_processed_transactions()


def test_load_processed_transactions_concatenates_multiple_monthly_files(fake_data_layout):
    processed_dir, _ = fake_data_layout

    part_a = make_synthetic_labels(np.arange(0, 5), fraud_rate=0.2, seed=1)
    part_a["TX_DATETIME"] = pd.to_datetime("2018-04-01")
    part_b = make_synthetic_labels(np.arange(5, 10), fraud_rate=0.2, seed=2)
    part_b["TX_DATETIME"] = pd.to_datetime("2018-05-01")
    part_a.to_parquet(processed_dir / "transactions_2018-04.parquet", index=False)
    part_b.to_parquet(processed_dir / "transactions_2018-05.parquet", index=False)

    result = load_processed_transactions()

    assert len(result) == 10
    assert set(result["TRANSACTION_ID"]) == set(range(10))


def test_load_split_uses_provided_labels_source_without_reloading(fake_data_layout, monkeypatch):
    _, features_dir = fake_data_layout
    features = make_synthetic_features(5, split_name="validation", date_start="2018-08-01", nan_rate=0.0, seed=5)
    features.to_parquet(features_dir / "features_validation.parquet", index=False)
    labels = make_synthetic_labels(features["TRANSACTION_ID"], fraud_rate=0.2, seed=6)

    import mrs.models.dataset as dataset_module

    def _boom():
        raise AssertionError("load_processed_transactions should not be called when labels_source is provided")

    monkeypatch.setattr(dataset_module, "load_processed_transactions", _boom)

    result = load_split("validation", labels_source=labels)

    assert len(result) == len(features)
