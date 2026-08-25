"""Tests for mrs.features.terminal, including the fraud-history features -- the most
sensitive in the whole feature layer (Dev Plan Sec 10, Sec 34.2)."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from mrs.features.terminal import build_terminal_features


def _by_id(out, column):
    return dict(zip(out["TRANSACTION_ID"], out[column]))


def test_cold_start_first_transaction_ever():
    df = pd.DataFrame({
        "TRANSACTION_ID": [1],
        "TERMINAL_ID": [100],
        "CUSTOMER_ID": [1],
        "TX_DATETIME": pd.to_datetime(["2018-04-01 00:00:00"]),
        "TX_AMOUNT": [10.0],
        "TX_FRAUD": [0],
    })
    out = build_terminal_features(df)
    row = out.iloc[0]
    assert row["terminal_prior_tx_count"] == 0
    assert np.isnan(row["terminal_hist_amount_mean"])
    assert np.isnan(row["terminal_hist_amount_std"])
    assert np.isnan(row["terminal_time_since_prev_tx_seconds"])
    assert row["terminal_unique_customers_count"] == 0
    assert row["terminal_tx_count_10min"] == 0
    assert row["terminal_tx_count_1h"] == 0
    assert row["terminal_tx_count_24h"] == 0
    assert row["terminal_recent_fraud_count_24h"] == 0
    assert np.isnan(row["terminal_recent_fraud_rate_24h"])
    assert row["terminal_hist_fraud_count"] == 0
    assert np.isnan(row["terminal_hist_fraud_rate"])
    assert np.isnan(row["terminal_fraud_rate_deviation"])
    assert np.isnan(row["terminal_volume_deviation"])


def test_prior_tx_count_increments():
    df = pd.DataFrame({
        "TRANSACTION_ID": [1, 2, 3, 4],
        "TERMINAL_ID": [100, 100, 100, 100],
        "CUSTOMER_ID": [1, 2, 3, 4],
        "TX_DATETIME": pd.to_datetime(["2018-04-01 00:00:00", "2018-04-01 00:05:00", "2018-04-01 01:00:00", "2018-04-02 00:00:00"]),
        "TX_AMOUNT": [10.0, 20.0, 30.0, 40.0],
        "TX_FRAUD": [0, 0, 0, 0],
    })
    out = build_terminal_features(df)
    assert _by_id(out, "terminal_prior_tx_count") == {1: 0, 2: 1, 3: 2, 4: 3}


def test_velocity_windows_10min_1h_24h_with_exact_boundaries():
    df = pd.DataFrame({
        "TRANSACTION_ID": [1, 2, 3, 4],
        "TERMINAL_ID": [100, 100, 100, 100],
        "CUSTOMER_ID": [1, 2, 3, 4],
        "TX_DATETIME": pd.to_datetime(["2018-04-01 00:00:00", "2018-04-01 00:05:00", "2018-04-01 01:00:00", "2018-04-02 00:00:00"]),
        "TX_AMOUNT": [10.0, 20.0, 30.0, 40.0],
        "TX_FRAUD": [0, 0, 0, 0],
    })
    out = build_terminal_features(df)
    assert _by_id(out, "terminal_tx_count_10min") == {1: 0, 2: 1, 3: 0, 4: 0}
    assert _by_id(out, "terminal_tx_count_1h") == {1: 0, 2: 1, 3: 2, 4: 0}
    assert _by_id(out, "terminal_tx_count_24h") == {1: 0, 2: 1, 3: 2, 4: 3}


def test_hist_amount_mean_and_std():
    df = pd.DataFrame({
        "TRANSACTION_ID": [1, 2, 3],
        "TERMINAL_ID": [100, 100, 100],
        "CUSTOMER_ID": [1, 2, 3],
        "TX_DATETIME": pd.to_datetime(["2018-04-01 00:00:00", "2018-04-01 01:00:00", "2018-04-01 02:00:00"]),
        "TX_AMOUNT": [10.0, 20.0, 30.0],
        "TX_FRAUD": [0, 0, 0],
    })
    out = build_terminal_features(df)
    mean = _by_id(out, "terminal_hist_amount_mean")
    std = _by_id(out, "terminal_hist_amount_std")
    assert np.isnan(mean[1])
    assert mean[2] == 10.0
    assert mean[3] == pytest.approx(15.0)
    assert np.isnan(std[2])
    assert std[3] == pytest.approx(math.sqrt(50.0))


def test_time_since_prev_tx():
    df = pd.DataFrame({
        "TRANSACTION_ID": [1, 2, 3, 4],
        "TERMINAL_ID": [100, 100, 100, 100],
        "CUSTOMER_ID": [1, 2, 3, 4],
        "TX_DATETIME": pd.to_datetime(["2018-04-01 00:00:00", "2018-04-01 00:05:00", "2018-04-01 01:00:00", "2018-04-02 00:00:00"]),
        "TX_AMOUNT": [10.0, 20.0, 30.0, 40.0],
        "TX_FRAUD": [0, 0, 0, 0],
    })
    out = build_terminal_features(df)
    delta = _by_id(out, "terminal_time_since_prev_tx_seconds")
    assert np.isnan(delta[1])
    assert delta[2] == pytest.approx(300.0)
    assert delta[3] == pytest.approx(3300.0)
    assert delta[4] == pytest.approx(82800.0)


def test_unique_customers_count():
    df = pd.DataFrame({
        "TRANSACTION_ID": [1, 2, 3, 4, 5],
        "TERMINAL_ID": [100, 100, 100, 100, 100],
        "CUSTOMER_ID": [1, 1, 1, 1, 2],
        "TX_DATETIME": pd.to_datetime(["2018-04-01 00:00:00", "2018-04-01 01:00:00", "2018-04-01 02:00:00", "2018-04-02 00:00:00", "2018-04-03 00:00:00"]),
        "TX_AMOUNT": [10.0, 20.0, 30.0, 40.0, 50.0],
        "TX_FRAUD": [0, 0, 0, 0, 0],
    })
    out = build_terminal_features(df)
    unique_count = _by_id(out, "terminal_unique_customers_count")
    assert unique_count == {1: 0, 2: 1, 3: 1, 4: 1, 5: 1}


def test_current_row_own_fraud_label_never_leaks_into_own_historical_fraud_features():
    df = pd.DataFrame({
        "TRANSACTION_ID": [1],
        "TERMINAL_ID": [100],
        "CUSTOMER_ID": [1],
        "TX_DATETIME": pd.to_datetime(["2018-04-01 00:00:00"]),
        "TX_AMOUNT": [10.0],
        "TX_FRAUD": [1],
    })
    out = build_terminal_features(df)
    row = out.iloc[0]
    assert row["terminal_hist_fraud_count"] == 0
    assert np.isnan(row["terminal_hist_fraud_rate"])
    assert row["terminal_recent_fraud_count_24h"] == 0
    assert np.isnan(row["terminal_recent_fraud_rate_24h"])


def test_hist_and_recent_fraud_rate_progression():
    df = pd.DataFrame({
        "TRANSACTION_ID": [1, 2, 3],
        "TERMINAL_ID": [100, 100, 100],
        "CUSTOMER_ID": [1, 2, 3],
        "TX_DATETIME": pd.to_datetime(["2018-04-01 00:00:00", "2018-04-01 01:00:00", "2018-04-01 02:00:00"]),
        "TX_AMOUNT": [10.0, 20.0, 30.0],
        "TX_FRAUD": [1, 0, 1],
    })
    out = build_terminal_features(df)
    hist_count = _by_id(out, "terminal_hist_fraud_count")
    hist_rate = _by_id(out, "terminal_hist_fraud_rate")
    recent_count = _by_id(out, "terminal_recent_fraud_count_24h")
    recent_rate = _by_id(out, "terminal_recent_fraud_rate_24h")
    deviation = _by_id(out, "terminal_fraud_rate_deviation")

    assert hist_count == {1: 0, 2: 1, 3: 1}
    assert hist_rate[2] == pytest.approx(1.0)
    assert hist_rate[3] == pytest.approx(0.5)
    assert recent_count == {1: 0, 2: 1, 3: 1}
    assert recent_rate[2] == pytest.approx(1.0)
    assert recent_rate[3] == pytest.approx(0.5)
    assert deviation[2] == pytest.approx(0.0)
    assert deviation[3] == pytest.approx(0.0)


def test_recent_fraud_rate_nan_not_zero_when_no_recent_activity():
    df = pd.DataFrame({
        "TRANSACTION_ID": [1, 2, 3, 4],
        "TERMINAL_ID": [100, 100, 100, 100],
        "CUSTOMER_ID": [1, 2, 3, 4],
        "TX_DATETIME": pd.to_datetime(["2018-04-01 00:00:00", "2018-04-01 01:00:00", "2018-04-01 02:00:00", "2018-04-05 00:00:00"]),
        "TX_AMOUNT": [10.0, 20.0, 30.0, 40.0],
        "TX_FRAUD": [1, 0, 1, 0],
    })
    out = build_terminal_features(df)
    row4 = out.iloc[3]
    assert row4["terminal_recent_fraud_count_24h"] == 0
    assert np.isnan(row4["terminal_recent_fraud_rate_24h"])
    assert row4["terminal_hist_fraud_count"] == 2
    assert row4["terminal_hist_fraud_rate"] == pytest.approx(2.0 / 3.0)
    assert np.isnan(row4["terminal_fraud_rate_deviation"])


def test_future_row_fraud_mutation_does_not_affect_earlier_terminal_history():
    # TXN1=fraud1 gives TXN2 GENUINE non-zero established history (hist_fraud_count=1,
    # hist_fraud_rate=1.0, recent_fraud_count_24h=1, recent_fraud_rate_24h=1.0) -- not a
    # value that was already zero. Only TXN3 (strictly after TXN2) is then mutated.
    df = pd.DataFrame({
        "TRANSACTION_ID": [1, 2, 3],
        "TERMINAL_ID": [500, 500, 500],
        "CUSTOMER_ID": [1, 2, 3],
        "TX_DATETIME": pd.to_datetime(["2018-04-01 00:00:00", "2018-04-01 01:00:00", "2018-04-01 02:00:00"]),
        "TX_AMOUNT": [10.0, 20.0, 30.0],
        "TX_FRAUD": [1, 0, 0],
    })
    before = build_terminal_features(df)
    row2_before = before.iloc[1]
    assert row2_before["terminal_hist_fraud_count"] == 1
    assert row2_before["terminal_hist_fraud_rate"] == pytest.approx(1.0)
    assert row2_before["terminal_recent_fraud_count_24h"] == 1
    assert row2_before["terminal_recent_fraud_rate_24h"] == pytest.approx(1.0)

    mutated = df.copy()
    mutated.loc[2, "TX_FRAUD"] = 1  # TXN3 (strictly after TXN2) becomes fraudulent
    after = build_terminal_features(mutated)
    row2_after = after.iloc[1]

    assert row2_after["terminal_hist_fraud_count"] == row2_before["terminal_hist_fraud_count"]
    assert row2_after["terminal_hist_fraud_rate"] == row2_before["terminal_hist_fraud_rate"]
    assert row2_after["terminal_recent_fraud_count_24h"] == row2_before["terminal_recent_fraud_count_24h"]
    assert row2_after["terminal_recent_fraud_rate_24h"] == row2_before["terminal_recent_fraud_rate_24h"]


def test_volume_deviation_nan_when_insufficient_elapsed_history():
    df = pd.DataFrame({
        "TRANSACTION_ID": [1, 2],
        "TERMINAL_ID": [700, 700],
        "CUSTOMER_ID": [1, 2],
        "TX_DATETIME": pd.to_datetime(["2018-04-01 00:00:00", "2018-04-01 00:30:00"]),
        "TX_AMOUNT": [10.0, 20.0],
        "TX_FRAUD": [0, 0],
    })
    out = build_terminal_features(df)
    row2 = out.iloc[1]
    assert np.isnan(row2["terminal_volume_deviation"])


def test_volume_deviation_defined_at_exact_one_hour_boundary():
    df = pd.DataFrame({
        "TRANSACTION_ID": [1, 2],
        "TERMINAL_ID": [800, 800],
        "CUSTOMER_ID": [1, 2],
        "TX_DATETIME": pd.to_datetime(["2018-04-01 00:00:00", "2018-04-01 01:00:00"]),
        "TX_AMOUNT": [10.0, 20.0],
        "TX_FRAUD": [0, 0],
    })
    out = build_terminal_features(df)
    row2 = out.iloc[1]
    assert row2["terminal_volume_deviation"] == pytest.approx(0.0)


def test_entity_isolation_across_terminals():
    df = pd.DataFrame({
        "TRANSACTION_ID": [1, 2, 3],
        "TERMINAL_ID": [100, 200, 100],
        "CUSTOMER_ID": [1, 2, 3],
        "TX_DATETIME": pd.to_datetime(["2018-04-01 00:00:00", "2018-04-01 00:01:00", "2018-04-01 00:02:00"]),
        "TX_AMOUNT": [10.0, 999999.0, 20.0],
        "TX_FRAUD": [0, 1, 0],
    })
    out = build_terminal_features(df)
    row3 = out.iloc[2]
    assert row3["terminal_hist_amount_mean"] == 10.0
    assert row3["terminal_hist_fraud_count"] == 0
    assert row3["terminal_tx_count_10min"] == 1
    assert row3["terminal_prior_tx_count"] == 1


def test_no_label_columns_in_output():
    df = pd.DataFrame({
        "TRANSACTION_ID": [1],
        "TERMINAL_ID": [100],
        "CUSTOMER_ID": [1],
        "TX_DATETIME": pd.to_datetime(["2018-04-01 00:00:00"]),
        "TX_AMOUNT": [10.0],
        "TX_FRAUD": [1],
        "TX_FRAUD_SCENARIO": [1],
    })
    out = build_terminal_features(df)
    assert "TX_FRAUD" not in out.columns
    assert "TX_FRAUD_SCENARIO" not in out.columns


def test_shuffled_input_invariance():
    df = pd.DataFrame({
        "TRANSACTION_ID": [1, 2, 3, 4, 5, 6],
        "TERMINAL_ID": [100, 200, 100, 201, 101, 200],
        "CUSTOMER_ID": [1, 2, 3, 4, 5, 6],
        "TX_DATETIME": pd.to_datetime(["2018-04-01 00:00", "2018-04-01 00:10", "2018-04-01 00:20", "2018-04-01 00:30", "2018-04-01 00:40", "2018-04-01 00:50"]),
        "TX_AMOUNT": [10.0, 15.0, 20.0, 25.0, 30.0, 35.0],
        "TX_FRAUD": [0, 1, 0, 0, 1, 0],
    })
    baseline = build_terminal_features(df).set_index("TRANSACTION_ID").sort_index()

    shuffled = df.sample(frac=1.0, random_state=13).reset_index(drop=True)
    shuffled_result = build_terminal_features(shuffled).set_index("TRANSACTION_ID").sort_index()

    pd.testing.assert_frame_equal(baseline, shuffled_result)
