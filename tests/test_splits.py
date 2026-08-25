"""Split-boundary and leakage-guard tests for mrs.data.splits."""

from __future__ import annotations

import pandas as pd
import pytest

from mrs import config
from mrs.data.splits import (
    SPLIT_BOUNDARIES,
    SPLIT_ORDER,
    SplitError,
    assign_split,
    split_date_range,
    validate_split_boundaries,
)


def test_boundaries_are_chronological_and_non_overlapping():
    validate_split_boundaries()


def test_train_ends_before_validation_starts():
    _, train_end = split_date_range("train")
    validation_start, _ = split_date_range("validation")
    assert train_end < validation_start


def test_validation_ends_before_test_starts():
    _, validation_end = split_date_range("validation")
    test_start, _ = split_date_range("test")
    assert validation_end < test_start


def test_split_order_matches_boundary_keys():
    assert set(SPLIT_ORDER) == set(SPLIT_BOUNDARIES.keys())


def test_overlapping_boundaries_are_rejected(monkeypatch):
    import mrs.data.splits as splits_module

    bad_boundaries = {
        "train": ("2018-04-01", "2018-08-05"),  # overlaps validation start
        "validation": ("2018-08-01", "2018-08-31"),
        "test": ("2018-09-01", "2018-09-30"),
    }
    monkeypatch.setattr(splits_module, "SPLIT_BOUNDARIES", bad_boundaries)
    with pytest.raises(SplitError, match="does not come strictly after"):
        splits_module.validate_split_boundaries()


def test_assign_split_labels_boundary_dates_correctly():
    timestamps = pd.Series(
        pd.to_datetime(
            [
                "2018-04-01 00:00:00",  # first train day
                "2018-07-31 23:59:59",  # last train day
                "2018-08-01 00:00:00",  # first validation day
                "2018-08-31 23:59:59",  # last validation day
                "2018-09-01 00:00:00",  # first test day
                "2018-09-30 23:59:59",  # last test day
            ]
        )
    )
    labels = assign_split(timestamps)
    assert list(labels) == [
        "train",
        "train",
        "validation",
        "validation",
        "test",
        "test",
    ]


def test_assign_split_rejects_timestamps_outside_all_ranges():
    timestamps = pd.Series(pd.to_datetime(["2018-03-31 12:00:00"]))
    with pytest.raises(SplitError, match="outside all configured split boundaries"):
        assign_split(timestamps)


def test_covers_full_expected_dataset_range():
    """The three splits together must cover Phase 1's verified date range with no gap."""
    train_start, _ = split_date_range("train")
    _, test_end = split_date_range("test")
    assert train_start == pd.Timestamp(config.EXPECTED_START_DATE)
    assert test_end == pd.Timestamp(config.EXPECTED_END_DATE)


@pytest.mark.data
def test_split_assignment_on_real_processed_data_has_no_leftover_rows(require_processed_dataset):
    parts = sorted(config.PROCESSED_TRANSACTIONS_DIR.glob("*.parquet"))
    df = pd.concat(
        (pd.read_parquet(p, columns=["TX_DATETIME"]) for p in parts), ignore_index=True
    )
    labels = assign_split(df["TX_DATETIME"])
    assert labels.isna().sum() == 0
    assert set(labels.unique()) == set(SPLIT_ORDER)


@pytest.mark.data
def test_split_row_counts_match_measured_values(require_processed_dataset):
    """Regression pin against docs/DATASET_REPORT.md measured figures."""
    parts = sorted(config.PROCESSED_TRANSACTIONS_DIR.glob("*.parquet"))
    df = pd.concat(
        (pd.read_parquet(p, columns=["TX_DATETIME", "TX_FRAUD"]) for p in parts),
        ignore_index=True,
    )
    df = df.assign(split=assign_split(df["TX_DATETIME"]))

    counts = df.groupby("split").size()
    assert counts["train"] == 1_169_723
    assert counts["validation"] == 296_559
    assert counts["test"] == 287_873

    fraud_counts = df[df["TX_FRAUD"] == 1].groupby("split").size()
    assert fraud_counts["train"] == 9_465
    assert fraud_counts["validation"] == 2_669
    assert fraud_counts["test"] == 2_547
