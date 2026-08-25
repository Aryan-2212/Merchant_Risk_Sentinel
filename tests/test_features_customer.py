"""Tests for mrs.features.customer -- the temporal primitives exercised through an
actual behavioral feature family."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from mrs.features.customer import build_customer_features


def _by_id(out, column):
    return dict(zip(out["TRANSACTION_ID"], out[column]))


def test_cold_start_first_transaction_ever():
    df = pd.DataFrame({
        "TRANSACTION_ID": [1],
        "CUSTOMER_ID": [1],
        "TERMINAL_ID": [100],
        "TX_DATETIME": pd.to_datetime(["2018-04-01 00:00:00"]),
        "TX_AMOUNT": [10.0],
    })
    out = build_customer_features(df)
    row = out.iloc[0]
    assert row["customer_prior_tx_count"] == 0
    assert np.isnan(row["customer_hist_amount_mean"])
    assert np.isnan(row["customer_hist_amount_std"])
    assert np.isnan(row["customer_amount_deviation"])
    assert np.isnan(row["customer_amount_zscore"])
    assert np.isnan(row["customer_time_since_prev_tx_seconds"])
    assert row["customer_new_terminal_flag"] == 1
    assert row["customer_unique_terminals_count"] == 0
    assert np.isnan(row["customer_hour_deviation"])
    assert row["customer_tx_count_10min"] == 0
    assert row["customer_tx_count_1h"] == 0
    assert row["customer_tx_count_24h"] == 0


def test_prior_tx_count_increments():
    df = pd.DataFrame({
        "TRANSACTION_ID": [1, 2, 3, 4],
        "CUSTOMER_ID": [1, 1, 1, 1],
        "TERMINAL_ID": [100, 100, 100, 100],
        "TX_DATETIME": pd.to_datetime(["2018-04-01 00:00:00", "2018-04-01 00:05:00", "2018-04-01 01:00:00", "2018-04-02 00:00:00"]),
        "TX_AMOUNT": [10.0, 20.0, 30.0, 40.0],
    })
    out = build_customer_features(df)
    assert _by_id(out, "customer_prior_tx_count") == {1: 0, 2: 1, 3: 2, 4: 3}


def test_velocity_windows_10min_1h_24h_with_exact_boundaries():
    # TXN1 04-01 00:00, TXN2 04-01 00:05, TXN3 04-01 01:00, TXN4 04-02 00:00 (exactly
    # 24h after TXN1 -- must be included per the closed='left' boundary rule).
    df = pd.DataFrame({
        "TRANSACTION_ID": [1, 2, 3, 4],
        "CUSTOMER_ID": [1, 1, 1, 1],
        "TERMINAL_ID": [100, 100, 100, 100],
        "TX_DATETIME": pd.to_datetime(["2018-04-01 00:00:00", "2018-04-01 00:05:00", "2018-04-01 01:00:00", "2018-04-02 00:00:00"]),
        "TX_AMOUNT": [10.0, 20.0, 30.0, 40.0],
    })
    out = build_customer_features(df)
    assert _by_id(out, "customer_tx_count_10min") == {1: 0, 2: 1, 3: 0, 4: 0}
    assert _by_id(out, "customer_tx_count_1h") == {1: 0, 2: 1, 3: 2, 4: 0}
    assert _by_id(out, "customer_tx_count_24h") == {1: 0, 2: 1, 3: 2, 4: 3}


def test_hist_amount_mean_and_std():
    df = pd.DataFrame({
        "TRANSACTION_ID": [1, 2, 3],
        "CUSTOMER_ID": [1, 1, 1],
        "TERMINAL_ID": [100, 100, 100],
        "TX_DATETIME": pd.to_datetime(["2018-04-01 00:00:00", "2018-04-01 01:00:00", "2018-04-01 02:00:00"]),
        "TX_AMOUNT": [10.0, 20.0, 30.0],
    })
    out = build_customer_features(df)
    mean = _by_id(out, "customer_hist_amount_mean")
    std = _by_id(out, "customer_hist_amount_std")
    assert np.isnan(mean[1])
    assert mean[2] == 10.0
    assert mean[3] == pytest.approx(15.0)
    assert np.isnan(std[2])  # n=1, undefined
    assert std[3] == pytest.approx(math.sqrt(50.0))  # sample std of [10,20]


def test_amount_deviation_and_zscore():
    df = pd.DataFrame({
        "TRANSACTION_ID": [1, 2, 3],
        "CUSTOMER_ID": [1, 1, 1],
        "TERMINAL_ID": [100, 100, 100],
        "TX_DATETIME": pd.to_datetime(["2018-04-01 00:00:00", "2018-04-01 01:00:00", "2018-04-01 02:00:00"]),
        "TX_AMOUNT": [10.0, 20.0, 30.0],
    })
    out = build_customer_features(df)
    deviation = _by_id(out, "customer_amount_deviation")
    zscore = _by_id(out, "customer_amount_zscore")
    assert np.isnan(deviation[1])
    assert deviation[2] == pytest.approx(10.0)  # 20 - hist_mean(10)
    assert deviation[3] == pytest.approx(15.0)  # 30 - hist_mean(15)
    assert np.isnan(zscore[2])  # std undefined (n=1)
    assert zscore[3] == pytest.approx(15.0 / math.sqrt(50.0))


def test_zscore_nan_when_history_has_zero_variance():
    df = pd.DataFrame({
        "TRANSACTION_ID": [1, 2, 3],
        "CUSTOMER_ID": [1, 1, 1],
        "TERMINAL_ID": [100, 100, 100],
        "TX_DATETIME": pd.to_datetime(["2018-04-01 00:00:00", "2018-04-01 01:00:00", "2018-04-01 02:00:00"]),
        "TX_AMOUNT": [10.0, 10.0, 999.0],
    })
    out = build_customer_features(df)
    std = _by_id(out, "customer_hist_amount_std")
    zscore = _by_id(out, "customer_amount_zscore")
    assert std[3] == 0.0  # history=[10.0, 10.0]
    assert np.isnan(zscore[3])  # never divides by zero


def test_time_since_prev_tx():
    df = pd.DataFrame({
        "TRANSACTION_ID": [1, 2, 3, 4],
        "CUSTOMER_ID": [1, 1, 1, 1],
        "TERMINAL_ID": [100, 100, 100, 100],
        "TX_DATETIME": pd.to_datetime(["2018-04-01 00:00:00", "2018-04-01 00:05:00", "2018-04-01 01:00:00", "2018-04-02 00:00:00"]),
        "TX_AMOUNT": [10.0, 20.0, 30.0, 40.0],
    })
    out = build_customer_features(df)
    delta = _by_id(out, "customer_time_since_prev_tx_seconds")
    assert np.isnan(delta[1])
    assert delta[2] == pytest.approx(300.0)
    assert delta[3] == pytest.approx(3300.0)
    assert delta[4] == pytest.approx(82800.0)


def test_new_terminal_flag_and_unique_terminals_count():
    df = pd.DataFrame({
        "TRANSACTION_ID": [1, 2, 3, 4, 5],
        "CUSTOMER_ID": [1, 1, 1, 1, 1],
        "TERMINAL_ID": [100, 100, 100, 100, 200],
        "TX_DATETIME": pd.to_datetime(["2018-04-01 00:00:00", "2018-04-01 01:00:00", "2018-04-01 02:00:00", "2018-04-02 00:00:00", "2018-04-03 00:00:00"]),
        "TX_AMOUNT": [10.0, 20.0, 30.0, 40.0, 50.0],
    })
    out = build_customer_features(df)
    new_flag = _by_id(out, "customer_new_terminal_flag")
    unique_count = _by_id(out, "customer_unique_terminals_count")
    assert new_flag == {1: 1, 2: 0, 3: 0, 4: 0, 5: 1}
    assert unique_count == {1: 0, 2: 1, 3: 1, 4: 1, 5: 1}


def test_circular_hour_deviation_wraparound_avoids_linear_mean_error():
    # Prior hours 23 and 1 must average to ~0 (midnight), not 12 (a linear-mean bug).
    df = pd.DataFrame({
        "TRANSACTION_ID": [1, 2, 3],
        "CUSTOMER_ID": [1, 1, 1],
        "TERMINAL_ID": [100, 100, 100],
        "TX_DATETIME": pd.to_datetime(["2018-04-01 23:00:00", "2018-04-02 01:00:00", "2018-04-03 00:00:00"]),
        "TX_AMOUNT": [10.0, 20.0, 30.0],
    })
    out = build_customer_features(df)
    deviation = _by_id(out, "customer_hour_deviation")
    # row3's own hour is 0; historical circular mean of [23, 1] is ~0 -> deviation ~0.
    assert deviation[3] == pytest.approx(0.0, abs=1e-6)


def test_circular_hour_deviation_simple_match_is_zero():
    df = pd.DataFrame({
        "TRANSACTION_ID": [1, 2],
        "CUSTOMER_ID": [1, 1],
        "TERMINAL_ID": [100, 100],
        "TX_DATETIME": pd.to_datetime(["2018-04-01 12:00:00", "2018-04-02 12:00:00"]),
        "TX_AMOUNT": [10.0, 20.0],
    })
    out = build_customer_features(df)
    deviation = _by_id(out, "customer_hour_deviation")
    assert deviation[2] == pytest.approx(0.0, abs=1e-6)


def test_current_row_exclusion_amount_and_velocity():
    df = pd.DataFrame({
        "TRANSACTION_ID": [1, 2],
        "CUSTOMER_ID": [1, 1],
        "TERMINAL_ID": [100, 100],
        "TX_DATETIME": pd.to_datetime(["2018-04-01 00:00:00", "2018-04-01 00:01:00"]),
        "TX_AMOUNT": [10.0, 999999.0],
    })
    out = build_customer_features(df)
    row2 = out.iloc[1]
    # row2's own huge amount must not appear in its own historical mean.
    assert row2["customer_hist_amount_mean"] == 10.0
    # row2's own presence must not inflate its own velocity count.
    assert row2["customer_tx_count_10min"] == 1  # sees only row1


def test_future_row_mutation_does_not_affect_earlier_customer_history():
    df = pd.DataFrame({
        "TRANSACTION_ID": [1, 2, 3],
        "CUSTOMER_ID": [1, 1, 1],
        "TERMINAL_ID": [100, 100, 100],
        "TX_DATETIME": pd.to_datetime(["2018-04-01 00:00:00", "2018-04-01 01:00:00", "2018-04-01 02:00:00"]),
        "TX_AMOUNT": [10.0, 20.0, 30.0],
    })
    before = build_customer_features(df)
    row2_before = before.iloc[1]
    assert row2_before["customer_hist_amount_mean"] == 10.0  # real, established history

    mutated = df.copy()
    mutated.loc[2, "TX_AMOUNT"] = 999999.0
    mutated.loc[2, "TERMINAL_ID"] = 999
    mutated.loc[2, "TX_DATETIME"] = pd.Timestamp("2018-04-01 02:30:00")
    after = build_customer_features(mutated)
    row2_after = after.iloc[1]

    assert row2_after["customer_hist_amount_mean"] == row2_before["customer_hist_amount_mean"]
    assert row2_after["customer_tx_count_1h"] == row2_before["customer_tx_count_1h"]
    assert row2_after["customer_unique_terminals_count"] == row2_before["customer_unique_terminals_count"]


def test_entity_isolation_across_customers():
    df = pd.DataFrame({
        "TRANSACTION_ID": [1, 2, 3],
        "CUSTOMER_ID": [1, 2, 1],
        "TERMINAL_ID": [100, 200, 100],
        "TX_DATETIME": pd.to_datetime(["2018-04-01 00:00:00", "2018-04-01 00:01:00", "2018-04-01 00:02:00"]),
        "TX_AMOUNT": [10.0, 999999.0, 20.0],
    })
    out = build_customer_features(df)
    row3 = out.iloc[2]
    # customer 1's row3 must not see customer 2's row2 amount/velocity at all.
    assert row3["customer_hist_amount_mean"] == 10.0
    assert row3["customer_tx_count_10min"] == 1
    assert row3["customer_prior_tx_count"] == 1


def test_shuffled_input_invariance():
    df = pd.DataFrame({
        "TRANSACTION_ID": [1, 2, 3, 4, 5, 6],
        "CUSTOMER_ID": [1, 2, 1, 2, 1, 2],
        "TERMINAL_ID": [100, 200, 100, 201, 101, 200],
        "TX_DATETIME": pd.to_datetime(["2018-04-01 00:00", "2018-04-01 00:10", "2018-04-01 00:20", "2018-04-01 00:30", "2018-04-01 00:40", "2018-04-01 00:50"]),
        "TX_AMOUNT": [10.0, 15.0, 20.0, 25.0, 30.0, 35.0],
    })
    baseline = build_customer_features(df).set_index("TRANSACTION_ID").sort_index()

    shuffled = df.sample(frac=1.0, random_state=13).reset_index(drop=True)
    shuffled_result = build_customer_features(shuffled).set_index("TRANSACTION_ID").sort_index()

    pd.testing.assert_frame_equal(baseline, shuffled_result)
