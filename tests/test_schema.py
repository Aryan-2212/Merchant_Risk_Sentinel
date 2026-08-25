"""Unit tests for mrs.data.schema against synthetic frames (no dataset required)."""

from __future__ import annotations

import pandas as pd
import pytest

from mrs.data.schema import (
    LABEL_COLUMNS,
    RAW_COLUMNS,
    SchemaError,
    feature_candidate_columns,
    normalize_dtypes,
    validate_processed_frame,
    validate_raw_frame,
)


def _valid_raw_frame(n: int = 3) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "TRANSACTION_ID": list(range(n)),
            "TX_DATETIME": pd.to_datetime(
                [
                    "2018-04-01 00:00:31",
                    "2018-04-01 00:02:10",
                    "2018-04-01 00:07:56",
                ][:n]
            ),
            "CUSTOMER_ID": pd.array([596, 4961, 2][:n], dtype=object),
            "TERMINAL_ID": pd.array([3156, 3412, 1365][:n], dtype=object),
            "TX_AMOUNT": [57.16, 81.51, 146.00][:n],
            "TX_TIME_SECONDS": pd.array([31, 130, 476][:n], dtype=object),
            "TX_TIME_DAYS": pd.array([0, 0, 0][:n], dtype=object),
            "TX_FRAUD": [0, 0, 0][:n],
            "TX_FRAUD_SCENARIO": [0, 0, 0][:n],
        }
    )


def test_valid_frame_passes():
    validate_raw_frame(_valid_raw_frame(), source="synthetic")


def test_missing_column_rejected():
    df = _valid_raw_frame().drop(columns=["TX_AMOUNT"])
    with pytest.raises(SchemaError, match="column mismatch"):
        validate_raw_frame(df, source="synthetic")


def test_extra_column_rejected():
    df = _valid_raw_frame()
    df["EXTRA"] = 1
    with pytest.raises(SchemaError, match="column mismatch"):
        validate_raw_frame(df, source="synthetic")


def test_null_values_rejected():
    df = _valid_raw_frame()
    df.loc[0, "TX_AMOUNT"] = None
    with pytest.raises(SchemaError, match="nulls"):
        validate_raw_frame(df, source="synthetic")


def test_negative_amount_rejected():
    df = _valid_raw_frame()
    df.loc[0, "TX_AMOUNT"] = -1.0
    with pytest.raises(SchemaError, match="negative"):
        validate_raw_frame(df, source="synthetic")


def test_out_of_domain_fraud_label_rejected():
    df = _valid_raw_frame()
    df.loc[0, "TX_FRAUD"] = 2
    with pytest.raises(SchemaError, match="TX_FRAUD"):
        validate_raw_frame(df, source="synthetic")


def test_out_of_domain_scenario_rejected():
    df = _valid_raw_frame()
    df.loc[0, "TX_FRAUD_SCENARIO"] = 9
    with pytest.raises(SchemaError, match="TX_FRAUD_SCENARIO"):
        validate_raw_frame(df, source="synthetic")


def test_fraud_zero_with_nonzero_scenario_rejected():
    df = _valid_raw_frame()
    df.loc[0, "TX_FRAUD_SCENARIO"] = 1  # TX_FRAUD stays 0: inconsistent
    with pytest.raises(SchemaError, match="TX_FRAUD=0"):
        validate_raw_frame(df, source="synthetic")


def test_fraud_one_with_zero_scenario_rejected():
    df = _valid_raw_frame()
    df.loc[0, "TX_FRAUD"] = 1  # TX_FRAUD_SCENARIO stays 0: inconsistent
    with pytest.raises(SchemaError, match="TX_FRAUD=1"):
        validate_raw_frame(df, source="synthetic")


def test_unsorted_time_seconds_rejected():
    df = _valid_raw_frame()
    df.loc[0, "TX_TIME_SECONDS"] = 9999
    with pytest.raises(SchemaError, match="TX_TIME_SECONDS"):
        validate_raw_frame(df, source="synthetic")


def test_multiple_days_in_one_file_rejected():
    df = _valid_raw_frame()
    df.loc[df.index[-1], "TX_TIME_DAYS"] = 1
    with pytest.raises(SchemaError, match="TX_TIME_DAYS"):
        validate_raw_frame(df, source="synthetic")


def test_non_integer_castable_id_rejected():
    df = _valid_raw_frame()
    df.loc[0, "CUSTOMER_ID"] = "not-a-number"
    with pytest.raises(SchemaError, match="CUSTOMER_ID"):
        validate_raw_frame(df, source="synthetic")


def test_empty_frame_rejected():
    df = _valid_raw_frame(0)
    with pytest.raises(SchemaError, match="zero transactions"):
        validate_raw_frame(df, source="synthetic")


def test_label_columns_never_treated_as_features():
    assert LABEL_COLUMNS == {"TX_FRAUD", "TX_FRAUD_SCENARIO"}
    candidates = feature_candidate_columns(list(RAW_COLUMNS))
    assert "TX_FRAUD" not in candidates
    assert "TX_FRAUD_SCENARIO" not in candidates
    assert set(candidates) == set(RAW_COLUMNS) - LABEL_COLUMNS


def test_normalize_dtypes_casts_object_id_columns():
    df = _valid_raw_frame()
    normalized = normalize_dtypes(df)
    assert str(normalized["CUSTOMER_ID"].dtype) == "int32"
    assert str(normalized["TERMINAL_ID"].dtype) == "int32"
    assert str(normalized["TX_TIME_SECONDS"].dtype) == "int64"
    assert str(normalized["TX_TIME_DAYS"].dtype) == "int16"
    # original untouched
    assert df["CUSTOMER_ID"].dtype == object


def test_normalize_then_validate_processed_frame_passes():
    df = normalize_dtypes(_valid_raw_frame())
    validate_processed_frame(df, source="synthetic")


def test_validate_processed_rejects_duplicate_transaction_id():
    df = normalize_dtypes(_valid_raw_frame())
    df.loc[1, "TRANSACTION_ID"] = df.loc[0, "TRANSACTION_ID"]
    with pytest.raises(SchemaError, match="duplicate"):
        validate_processed_frame(df, source="synthetic")


def test_validate_processed_rejects_unsorted_datetime():
    df = normalize_dtypes(_valid_raw_frame())
    df = df.iloc[::-1].reset_index(drop=True)
    with pytest.raises(SchemaError, match="chronological"):
        validate_processed_frame(df, source="synthetic")
