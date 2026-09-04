"""Tests for the simulated recent operational stream."""

from __future__ import annotations

import pandas as pd

from mrs.data.recent_stream import RecentStreamConfig, build_recent_feature_frame, generate_recent_transactions
from mrs.data.schema import RAW_COLUMNS, normalize_dtypes, validate_processed_frame


def _profiles() -> tuple[pd.DataFrame, pd.DataFrame]:
    customers = pd.DataFrame(
        [
            {
                "CUSTOMER_ID": 1,
                "x_customer_id": 1.0,
                "y_customer_id": 1.0,
                "mean_amount": 50.0,
                "std_amount": 10.0,
                "mean_nb_tx_per_day": 2.0,
                "nb_terminals": 2,
                "available_terminals": [10, 11],
            },
            {
                "CUSTOMER_ID": 2,
                "x_customer_id": 2.0,
                "y_customer_id": 2.0,
                "mean_amount": 70.0,
                "std_amount": 12.0,
                "mean_nb_tx_per_day": 2.0,
                "nb_terminals": 2,
                "available_terminals": [10, 11],
            },
            {
                "CUSTOMER_ID": 3,
                "x_customer_id": 3.0,
                "y_customer_id": 3.0,
                "mean_amount": 90.0,
                "std_amount": 15.0,
                "mean_nb_tx_per_day": 2.0,
                "nb_terminals": 2,
                "available_terminals": [10, 11],
            },
        ]
    )
    terminals = pd.DataFrame(
        [
            {"TERMINAL_ID": 10, "x_terminal_id": 10.0, "y_terminal_id": 10.0},
            {"TERMINAL_ID": 11, "x_terminal_id": 11.0, "y_terminal_id": 11.0},
        ]
    )
    return customers, terminals


def test_recent_stream_is_deterministic_and_schema_valid():
    customers, terminals = _profiles()
    cfg = RecentStreamConfig(days=21, transactions_per_day=20, seed=123, transaction_id_start=9_000_000)
    first = normalize_dtypes(generate_recent_transactions(customers, terminals, config=cfg))
    second = normalize_dtypes(generate_recent_transactions(customers, terminals, config=cfg))

    pd.testing.assert_frame_equal(first, second)
    assert tuple(first.columns) == RAW_COLUMNS
    assert len(first) == 420
    assert first["TRANSACTION_ID"].is_unique
    assert first["TX_DATETIME"].is_monotonic_increasing
    validate_processed_frame(first, source="test recent stream")


def test_recent_feature_frame_excludes_ground_truth_labels():
    customers, terminals = _profiles()
    cfg = RecentStreamConfig(days=21, transactions_per_day=20, seed=456, transaction_id_start=9_100_000)
    recent = normalize_dtypes(generate_recent_transactions(customers, terminals, config=cfg))
    features = build_recent_feature_frame(recent)

    assert "TX_FRAUD" not in features.columns
    assert "TX_FRAUD_SCENARIO" not in features.columns
    assert len(features) == len(recent)
    assert features["TRANSACTION_ID"].is_unique
    assert features["TX_DATETIME"].is_monotonic_increasing


def test_recent_window_is_exactly_three_weeks():
    cfg = RecentStreamConfig()
    assert cfg.days == 21
    assert cfg.end == cfg.start + pd.Timedelta(days=21) - pd.Timedelta(seconds=1)
