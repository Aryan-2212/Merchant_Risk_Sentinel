"""Processed-layer integrity tests against the persisted Parquet partitions."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from mrs import config
from mrs.data.schema import PROCESSED_DTYPES, validate_processed_frame

pytestmark = pytest.mark.data


def _load_all_processed() -> pd.DataFrame:
    parts = sorted(config.PROCESSED_TRANSACTIONS_DIR.glob("*.parquet"))
    return pd.concat((pd.read_parquet(p) for p in parts), ignore_index=True)


def test_row_count_matches_manifest_expectation(require_processed_dataset):
    manifest = json.loads(config.RAW_MANIFEST_PATH.read_text())
    df = _load_all_processed()
    assert manifest["file_count"] == config.EXPECTED_FILE_COUNT
    assert len(df) == 1_754_155  # measured total; see docs/PHASE1_REPORT.md


def test_declared_dtypes_hold(require_processed_dataset):
    df = _load_all_processed()
    for column, dtype in PROCESSED_DTYPES.items():
        assert str(df[column].dtype) == dtype, f"{column}: {df[column].dtype} != {dtype}"


def test_chronological_order_and_uniqueness(require_processed_dataset):
    df = _load_all_processed()
    validate_processed_frame(df, source="processed (concatenated partitions)")


def test_labels_never_appear_among_declared_feature_candidates(require_processed_dataset):
    from mrs.data.schema import feature_candidate_columns

    df = _load_all_processed()
    candidates = feature_candidate_columns(list(df.columns))
    assert "TX_FRAUD" not in candidates
    assert "TX_FRAUD_SCENARIO" not in candidates
