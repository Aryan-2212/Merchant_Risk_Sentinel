"""Leakage-critical tests for mrs.features._temporal.

These are the permanent regression form of the synthetic probes used to validate the
temporal primitives before any feature module was built on top of them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mrs.features import _temporal as T


def test_rolling_count_boundary_and_duplicate_timestamp_isolation():
    df = pd.DataFrame({
        "TRANSACTION_ID": [100, 101, 102, 103, 104, 105, 106],
        "CUSTOMER_ID": [1, 2, 1, 2, 1, 2, 1],
        "TX_DATETIME": pd.to_datetime(["2018-04-01 00:00:00", "2018-04-01 00:00:00", "2018-04-01 00:05:00", "2018-04-01 00:05:00", "2018-04-01 00:09:00", "2018-04-01 00:20:00", "2018-04-01 00:09:00"]),
    })
    counts = T.rolling_count(df, "CUSTOMER_ID", "10min")
    expected = {100: 0, 101: 0, 102: 1, 103: 1, 104: 2, 105: 0, 106: 2}
    assert dict(zip(df["TRANSACTION_ID"], counts)) == expected


def test_rolling_count_exact_left_boundary_included():
    # A row exactly `window` before t must be included (closed='left' semantics).
    df = pd.DataFrame({
        "TRANSACTION_ID": [1, 2],
        "CUSTOMER_ID": [1, 1],
        "TX_DATETIME": pd.to_datetime(["2018-04-01 00:00:00", "2018-04-01 00:10:00"]),
    })
    counts = T.rolling_count(df, "CUSTOMER_ID", "10min")
    assert counts[1] == 1


def test_rolling_count_future_row_never_affects_earlier_row():
    df = pd.DataFrame({
        "TRANSACTION_ID": [1, 2, 3],
        "CUSTOMER_ID": [1, 1, 1],
        "TX_DATETIME": pd.to_datetime(["2018-04-01 00:00:00", "2018-04-01 00:05:00", "2018-04-01 00:08:00"]),
    })
    before = T.rolling_count(df, "CUSTOMER_ID", "10min")
    assert before[1] == 1  # row1 has real history: sees row0

    mutated = df.copy()
    mutated.loc[2, "TX_DATETIME"] = pd.Timestamp("2018-04-01 00:06:00")  # row2 stays in the future relative to row1, just moved
    after = T.rolling_count(mutated, "CUSTOMER_ID", "10min")
    assert after[1] == before[1] == 1


def test_rolling_count_entity_isolation():
    df = pd.DataFrame({
        "TRANSACTION_ID": [1, 2, 3],
        "CUSTOMER_ID": [1, 2, 1],
        "TX_DATETIME": pd.to_datetime(["2018-04-01 00:00:00", "2018-04-01 00:01:00", "2018-04-01 00:02:00"]),
    })
    counts = T.rolling_count(df, "CUSTOMER_ID", "10min")
    assert counts[2] == 1  # row2 (customer 1) sees only row0, not row1 (customer 2)


def test_rolling_sum_recent_fraud_count_empty_window_is_zero_not_nan():
    df = pd.DataFrame({
        "TRANSACTION_ID": [1],
        "TERMINAL_ID": [1],
        "TX_DATETIME": pd.to_datetime(["2018-04-01 00:00:00"]),
        "TX_FRAUD": [0],
    })
    result = T.rolling_sum(df, "TERMINAL_ID", "TX_FRAUD", "24h")
    assert result[0] == 0.0
    assert not np.isnan(result[0])


def test_expanding_prior_cold_start_nan_for_mean_std_zero_for_count():
    df = pd.DataFrame({
        "TRANSACTION_ID": [1],
        "CUSTOMER_ID": [1],
        "TX_DATETIME": pd.to_datetime(["2018-04-01 00:00:00"]),
        "TX_AMOUNT": [10.0],
    })
    assert np.isnan(T.expanding_prior(df, "CUSTOMER_ID", "TX_AMOUNT", "mean")[0])
    assert np.isnan(T.expanding_prior(df, "CUSTOMER_ID", "TX_AMOUNT", "std")[0])
    assert T.expanding_prior(df, "CUSTOMER_ID", "TX_AMOUNT", "count")[0] == 0.0


def test_expanding_prior_current_row_excluded():
    # The current row's own amount must never appear in its own historical mean.
    df = pd.DataFrame({
        "TRANSACTION_ID": [1, 2],
        "CUSTOMER_ID": [1, 1],
        "TX_DATETIME": pd.to_datetime(["2018-04-01 00:00:00", "2018-04-01 01:00:00"]),
        "TX_AMOUNT": [10.0, 999999.0],
    })
    mean = T.expanding_prior(df, "CUSTOMER_ID", "TX_AMOUNT", "mean")
    assert mean[1] == 10.0


def test_expanding_prior_zero_variance_is_true_zero_not_nan():
    df = pd.DataFrame({
        "TRANSACTION_ID": [1, 2, 3],
        "CUSTOMER_ID": [1, 1, 1],
        "TX_DATETIME": pd.to_datetime(["2018-04-01 00:00:00", "2018-04-01 01:00:00", "2018-04-01 02:00:00"]),
        "TX_AMOUNT": [10.0, 10.0, 30.0],
    })
    std = T.expanding_prior(df, "CUSTOMER_ID", "TX_AMOUNT", "std")
    assert std[2] == 0.0  # history=[10.0,10.0] -> true zero variance
    assert np.isnan(std[1])  # history=[10.0] (n=1) -> variance undefined


def test_expanding_prior_future_row_never_affects_earlier_established_history():
    # row1 (TRANSACTION_ID=2) has REAL prior history (row0's amount=10.0), not a
    # cold-start. Mutating row2 -- a genuinely future transaction relative to row1 --
    # must not change row1's already-established historical mean.
    df = pd.DataFrame({
        "TRANSACTION_ID": [1, 2, 3],
        "CUSTOMER_ID": [1, 1, 1],
        "TX_DATETIME": pd.to_datetime(["2018-04-01 00:00:00", "2018-04-01 01:00:00", "2018-04-01 02:00:00"]),
        "TX_AMOUNT": [10.0, 20.0, 30.0],
    })
    before = T.expanding_prior(df, "CUSTOMER_ID", "TX_AMOUNT", "mean")
    assert before[1] == 10.0  # sanity check: row1's history is established, not cold-start

    mutated = df.copy()
    mutated.loc[2, "TX_AMOUNT"] = 999999.0  # row2 is strictly after row1 in time
    after = T.expanding_prior(mutated, "CUSTOMER_ID", "TX_AMOUNT", "mean")

    assert after[1] == 10.0
    assert before[1] == after[1]


def test_expanding_prior_duplicate_timestamp_tie_break_both_directions():
    # TRANSACTION_ID breaks ties: a later-ID row sees an earlier-ID row at the same
    # timestamp, but not vice versa -- no leakage in either direction.
    df = pd.DataFrame({
        "TRANSACTION_ID": [10, 11],
        "CUSTOMER_ID": [1, 1],
        "TX_DATETIME": pd.to_datetime(["2018-04-01 03:00:00", "2018-04-01 03:00:00"]),
        "TX_AMOUNT": [100.0, 200.0],
    })
    count = T.expanding_prior(df, "CUSTOMER_ID", "TX_AMOUNT", "count")
    assert count[0] == 0.0
    assert count[1] == 1.0


def test_expanding_prior_never_reads_fraud_label_of_current_row():
    df = pd.DataFrame({
        "TRANSACTION_ID": [1, 2],
        "TERMINAL_ID": [1, 1],
        "TX_DATETIME": pd.to_datetime(["2018-04-01 00:00:00", "2018-04-01 01:00:00"]),
        "TX_FRAUD": [1, 0],
    })
    fraud_count = T.expanding_prior(df, "TERMINAL_ID", "TX_FRAUD", "sum")
    assert fraud_count[0] == 0.0
    assert fraud_count[1] == 1.0


def test_time_since_previous_cold_start_is_nan():
    df = pd.DataFrame({
        "TRANSACTION_ID": [1],
        "CUSTOMER_ID": [1],
        "TX_DATETIME": pd.to_datetime(["2018-04-01 00:00:00"]),
    })
    assert np.isnan(T.time_since_previous(df, "CUSTOMER_ID")[0])


def test_time_since_previous_duplicate_timestamp_gives_zero():
    df = pd.DataFrame({
        "TRANSACTION_ID": [1, 2],
        "CUSTOMER_ID": [1, 1],
        "TX_DATETIME": pd.to_datetime(["2018-04-01 03:00:00", "2018-04-01 03:00:00"]),
    })
    delta = T.time_since_previous(df, "CUSTOMER_ID")
    assert delta[1] == 0.0


def test_first_occurrence_and_prior_unique_count():
    df = pd.DataFrame({
        "TRANSACTION_ID": [1, 2, 3, 4, 5, 6],
        "TERMINAL_ID": [50, 50, 50, 50, 50, 50],
        "CUSTOMER_ID": [1, 2, 1, 3, 2, 1],
        "TX_DATETIME": pd.to_datetime(["2018-04-01 00:00", "2018-04-01 01:00", "2018-04-01 02:00", "2018-04-01 03:00", "2018-04-01 04:00", "2018-04-01 05:00"]),
    })
    is_first, prior_unique = T.first_occurrence_and_prior_unique_count(df, "TERMINAL_ID", "CUSTOMER_ID")
    expected_unique = {1: 0, 2: 1, 3: 2, 4: 2, 5: 3, 6: 3}
    assert dict(zip(df["TRANSACTION_ID"], prior_unique)) == expected_unique
    expected_first = {1: 1, 2: 1, 3: 0, 4: 1, 5: 0, 6: 0}
    assert dict(zip(df["TRANSACTION_ID"], is_first)) == expected_first


def test_first_occurrence_current_row_not_counted_in_its_own_prior_unique_count():
    df = pd.DataFrame({
        "TRANSACTION_ID": [1],
        "TERMINAL_ID": [50],
        "CUSTOMER_ID": [1],
        "TX_DATETIME": pd.to_datetime(["2018-04-01 00:00:00"]),
    })
    is_first, prior_unique = T.first_occurrence_and_prior_unique_count(df, "TERMINAL_ID", "CUSTOMER_ID")
    assert is_first[0] == 1
    assert prior_unique[0] == 0


def test_first_seen_timestamp_unaffected_by_later_row_mutation():
    df = pd.DataFrame({
        "TRANSACTION_ID": [1, 2, 3],
        "CUSTOMER_ID": [1, 1, 1],
        "TX_DATETIME": pd.to_datetime(["2018-04-01 00:00", "2018-04-01 01:00", "2018-04-01 02:00"]),
    })
    before = T.first_seen_timestamp(df, "CUSTOMER_ID")[0]

    mutated = df.copy()
    mutated.loc[2, "TX_DATETIME"] = pd.Timestamp("2018-12-31 23:59:59")
    after = T.first_seen_timestamp(mutated, "CUSTOMER_ID")[0]

    assert before == after


def test_all_primitives_are_shuffled_input_invariant():
    # Functions must not depend on the caller having pre-sorted the input.
    df = pd.DataFrame({
        "TRANSACTION_ID": [1, 2, 3, 4, 5],
        "CUSTOMER_ID": [1, 1, 1, 1, 1],
        "TX_DATETIME": pd.to_datetime(["2018-04-01 00:00", "2018-04-01 01:00", "2018-04-01 02:00", "2018-04-01 03:00", "2018-04-01 03:00"]),
        "TX_AMOUNT": [10.0, 10.0, 30.0, 100.0, 999.0],
    })
    baseline = dict(zip(df["TRANSACTION_ID"], T.expanding_prior(df, "CUSTOMER_ID", "TX_AMOUNT", "mean")))

    shuffled = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    shuffled_result = dict(zip(shuffled["TRANSACTION_ID"], T.expanding_prior(shuffled, "CUSTOMER_ID", "TX_AMOUNT", "mean")))

    for tx_id, value in baseline.items():
        other = shuffled_result[tx_id]
        if np.isnan(value):
            assert np.isnan(other)
        else:
            assert value == other


def test_sort_canonical_breaks_ties_by_transaction_id():
    df = pd.DataFrame({
        "TRANSACTION_ID": [5, 3, 4],
        "TX_DATETIME": pd.to_datetime(["2018-04-01 00:00:00", "2018-04-01 00:00:00", "2018-04-01 00:00:00"]),
    })
    ordered = T.sort_canonical(df)
    assert list(ordered["TRANSACTION_ID"]) == [3, 4, 5]


def test_expanding_prior_rejects_unknown_stat():
    df = pd.DataFrame({
        "TRANSACTION_ID": [1],
        "CUSTOMER_ID": [1],
        "TX_DATETIME": pd.to_datetime(["2018-04-01 00:00:00"]),
        "TX_AMOUNT": [10.0],
    })
    with pytest.raises(ValueError, match="unsupported stat"):
        T.expanding_prior(df, "CUSTOMER_ID", "TX_AMOUNT", "median")
