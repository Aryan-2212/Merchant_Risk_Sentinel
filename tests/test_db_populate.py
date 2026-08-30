"""Tests for mrs.db.populate -- Phase 8 Step 2 database population.

Two groups, matching tests/test_db_models.py's split:

1. Pure row-builder tests (no database): small synthetic frames, exact dict output,
   including NaN -> None conversion (cold-start features / unavailable behavioral
   components must become SQL NULL, never a fabricated zero/calm value).
2. A small live-Postgres integration test (require_database fixture) that runs the
   real population functions end-to-end against a tiny synthetic dataset (not the
   full 1.75M-row real dataset -- that is exercised directly by running
   scripts/12_populate_db.py, not inside the test suite) and verifies the rows land
   correctly linked across all five populated tables.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import select

from mrs.db.models import Customer, RiskScore, Terminal, Transaction, TransactionFeatures
from mrs.db.populate import (
    FEATURE_VERSION,
    MODEL_VERSION,
    _nan_safe,
    assert_transactions_table_empty,
    customer_profile_rows,
    populate_customers_and_terminals,
    populate_risk_scores,
    populate_transaction_features,
    populate_transactions,
    risk_score_rows,
    terminal_profile_rows,
    transaction_features_rows,
    transaction_rows,
)

# --------------------------------------------------------------------- row builders


def test_nan_safe_converts_nan_to_none_and_preserves_zero():
    assert _nan_safe(float("nan")) is None
    assert _nan_safe(0.0) == 0.0
    assert _nan_safe(0) == 0
    assert _nan_safe(False) is False
    assert _nan_safe("NORMAL") == "NORMAL"
    assert _nan_safe(None) is None


def test_customer_profile_rows():
    df = pd.DataFrame(
        {
            "CUSTOMER_ID": [1],
            "x_customer_id": [10.5],
            "y_customer_id": [20.5],
            "mean_amount": [50.0],
            "std_amount": [12.0],
            "mean_nb_tx_per_day": [3.5],
            "nb_terminals": [7],
            "available_terminals": [np.array([2, 5, 9])],
        }
    )
    rows = customer_profile_rows(df)
    assert rows == [
        {
            "customer_id": 1,
            "x_customer_id": 10.5,
            "y_customer_id": 20.5,
            "mean_amount": 50.0,
            "std_amount": 12.0,
            "mean_nb_tx_per_day": 3.5,
            "nb_terminals": 7,
            "available_terminals": [2, 5, 9],
        }
    ]


def test_terminal_profile_rows():
    df = pd.DataFrame({"TERMINAL_ID": [3], "x_terminal_id": [1.0], "y_terminal_id": [2.0]})
    assert terminal_profile_rows(df) == [{"terminal_id": 3, "x_terminal_id": 1.0, "y_terminal_id": 2.0}]


def test_transaction_rows():
    df = pd.DataFrame(
        {
            "TRANSACTION_ID": [100],
            "TX_DATETIME": [pd.Timestamp("2018-04-01 12:00:00")],
            "CUSTOMER_ID": [1],
            "TERMINAL_ID": [2],
            "tx_amount": [42.5],
            "TX_TIME_SECONDS": [43200],
            "TX_TIME_DAYS": [0],
            "TX_FRAUD": [0],
            "TX_FRAUD_SCENARIO": [0],
            "split": ["train"],
        }
    )
    rows = transaction_rows(df)
    assert len(rows) == 1
    row = rows[0]
    assert row["transaction_id"] == 100
    assert row["customer_id"] == 1
    assert row["terminal_id"] == 2
    assert row["tx_amount"] == 42.5
    assert row["tx_time_seconds"] == 43200
    assert row["tx_time_days"] == 0
    assert row["tx_fraud"] == 0
    assert row["tx_fraud_scenario"] == 0
    assert row["split"] == "train"
    assert row["tx_datetime"].isoformat() == "2018-04-01T12:00:00"


def test_transaction_features_rows_converts_nan_to_none():
    ids = [1, 2]
    feature_chunk = pd.DataFrame(
        {
            "tx_amount": [10.0, 20.0],
            "customer_hist_amount_mean": [15.0, float("nan")],  # cold-start on row 2
            "customer_new_terminal_flag": [0, 1],
        }
    )
    rows = transaction_features_rows(ids, feature_chunk)
    assert rows[0] == {
        "transaction_id": 1,
        "feature_version": FEATURE_VERSION,
        "features": {"tx_amount": 10.0, "customer_hist_amount_mean": 15.0, "customer_new_terminal_flag": 0},
    }
    assert rows[1]["features"]["customer_hist_amount_mean"] is None
    assert rows[1]["features"]["tx_amount"] == 20.0


def test_risk_score_rows_preserves_unavailable_as_none():
    chunk = pd.DataFrame(
        {
            "TRANSACTION_ID": [1, 2],
            "CUSTOMER_ID": [10, 11],
            "TERMINAL_ID": [20, 21],
            "transaction_risk": [0.99, float("nan")],
            "transaction_risk_severity": [2, float("nan")],
            "terminal_risk_state": ["HIGH_RISK", float("nan")],
            "terminal_risk_severity": [2, float("nan")],
            "customer_risk_state": ["NORMAL", "INSUFFICIENT_HISTORY"],
            "customer_risk_severity": [0, float("nan")],
            "unified_risk_level": ["CRITICAL", "INSUFFICIENT_EVIDENCE"],
            "contributing_signals": [["transaction_ml_risk >= 0.97", "terminal_behavioral_risk: HIGH_RISK"], []],
        }
    )
    rows = risk_score_rows(chunk, transaction_risk_threshold=0.97)

    row0 = rows[0]
    assert row0["transaction_id"] == 1
    assert row0["customer_id"] == 10
    assert row0["terminal_id"] == 20
    assert row0["transaction_risk"] == 0.99
    assert row0["transaction_risk_severity"] == 2
    assert row0["terminal_risk_state"] == "HIGH_RISK"
    assert row0["unified_risk_level"] == "CRITICAL"
    assert row0["contributing_signals"] == ["transaction_ml_risk >= 0.97", "terminal_behavioral_risk: HIGH_RISK"]
    assert row0["model_version"] == MODEL_VERSION
    assert row0["transaction_risk_threshold"] == 0.97
    assert row0["feature_version"] == FEATURE_VERSION

    row1 = rows[1]
    assert row1["transaction_risk"] is None
    assert row1["transaction_risk_severity"] is None
    assert row1["terminal_risk_state"] is None
    assert row1["terminal_risk_severity"] is None
    assert row1["customer_risk_severity"] is None
    assert row1["unified_risk_level"] == "INSUFFICIENT_EVIDENCE"
    assert row1["contributing_signals"] == []


# ------------------------------------------------------------------ live-Postgres


def test_assert_transactions_table_empty_passes_on_fresh_schema(db_engine):
    assert_transactions_table_empty(db_engine)  # must not raise


def test_full_population_pipeline_small_dataset(db_engine):
    customer_profiles = pd.DataFrame(
        {
            "CUSTOMER_ID": [1, 2],
            "x_customer_id": [1.0, 2.0],
            "y_customer_id": [1.0, 2.0],
            "mean_amount": [50.0, 60.0],
            "std_amount": [10.0, 15.0],
            "mean_nb_tx_per_day": [2.0, 3.0],
            "nb_terminals": [1, 1],
            "available_terminals": [np.array([1]), np.array([1])],
        }
    )
    terminal_profiles = pd.DataFrame({"TERMINAL_ID": [1], "x_terminal_id": [5.0], "y_terminal_id": [5.0]})
    populate_customers_and_terminals(db_engine, customer_profiles, terminal_profiles)

    full_df = pd.DataFrame(
        {
            "TRANSACTION_ID": [1, 2],
            "TX_DATETIME": [pd.Timestamp("2018-04-01 00:00:00"), pd.Timestamp("2018-04-01 00:01:00")],
            "CUSTOMER_ID": [1, 2],
            "TERMINAL_ID": [1, 1],
            "tx_amount": [10.0, 500.0],
            "TX_TIME_SECONDS": [0, 60],
            "TX_TIME_DAYS": [0, 0],
            "TX_FRAUD": [0, 1],
            "TX_FRAUD_SCENARIO": [0, 1],
            "split": ["train", "train"],
        }
    )
    populate_transactions(db_engine, full_df)

    feature_matrix = pd.DataFrame({"tx_amount": [10.0, 500.0], "customer_hist_amount_mean": [float("nan"), 12.0]})
    populate_transaction_features(db_engine, full_df, feature_matrix)

    risk_full = pd.DataFrame(
        {
            "TRANSACTION_ID": [1, 2],
            "CUSTOMER_ID": [1, 2],
            "TERMINAL_ID": [1, 1],
            "transaction_risk": [0.01, 0.99],
            "transaction_risk_severity": [0, 2],
            "terminal_risk_state": ["NORMAL", "NORMAL"],
            "terminal_risk_severity": [0, 0],
            "customer_risk_state": ["INSUFFICIENT_HISTORY", "NORMAL"],
            "customer_risk_severity": [float("nan"), 0],
            "unified_risk_level": ["LOW", "HIGH"],
            "contributing_signals": [[], ["transaction_ml_risk >= 0.97"]],
        }
    )
    populate_risk_scores(db_engine, risk_full, transaction_risk_threshold=0.97)

    with db_engine.connect() as conn:
        assert conn.execute(select(Customer)).fetchall().__len__() == 2
        assert conn.execute(select(Terminal)).fetchall().__len__() == 1
        assert conn.execute(select(Transaction)).fetchall().__len__() == 2
        assert conn.execute(select(TransactionFeatures)).fetchall().__len__() == 2
        risk_rows = conn.execute(select(RiskScore).order_by(RiskScore.transaction_id)).fetchall()

    assert risk_rows[0].unified_risk_level == "LOW"
    assert risk_rows[0].customer_risk_severity is None  # INSUFFICIENT_HISTORY -> unavailable, stored as NULL
    assert risk_rows[1].unified_risk_level == "HIGH"
    assert risk_rows[1].contributing_signals == ["transaction_ml_risk >= 0.97"]

    # A second population attempt on top of existing rows must refuse, not corrupt data.
    with pytest.raises(RuntimeError):
        assert_transactions_table_empty(db_engine)
