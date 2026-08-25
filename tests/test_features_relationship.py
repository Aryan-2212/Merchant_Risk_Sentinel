"""Tests for mrs.features.relationship.

relationship.py has only two features: pair_prior_interaction_count and
pair_is_new_relationship, both keyed on the (CUSTOMER_ID, TERMINAL_ID) pair identity. It
has no rolling time-window feature and no pair-level amount/fraud statistic, unlike
customer.py/terminal.py -- tests below are scoped to what actually exists.
"""

from __future__ import annotations

import pandas as pd

from mrs.features.relationship import build_relationship_features


def _by_id(out, column):
    return dict(zip(out["TRANSACTION_ID"], out[column]))


def test_cold_start_first_interaction_ever():
    df = pd.DataFrame({
        "TRANSACTION_ID": [1],
        "CUSTOMER_ID": [1],
        "TERMINAL_ID": [100],
        "TX_DATETIME": pd.to_datetime(["2018-04-01 00:00:00"]),
        "TX_AMOUNT": [10.0],
    })
    out = build_relationship_features(df)
    row = out.iloc[0]
    assert row["pair_prior_interaction_count"] == 0
    assert row["pair_is_new_relationship"] == 1


def test_prior_interaction_count_increments_for_repeated_pair():
    df = pd.DataFrame({
        "TRANSACTION_ID": [1, 2, 3, 4],
        "CUSTOMER_ID": [1, 1, 1, 1],
        "TERMINAL_ID": [100, 100, 100, 100],
        "TX_DATETIME": pd.to_datetime(["2018-04-01 00:00:00", "2018-04-01 01:00:00", "2018-04-01 02:00:00", "2018-04-02 00:00:00"]),
        "TX_AMOUNT": [10.0, 20.0, 30.0, 40.0],
    })
    out = build_relationship_features(df)
    assert _by_id(out, "pair_prior_interaction_count") == {1: 0, 2: 1, 3: 2, 4: 3}
    assert _by_id(out, "pair_is_new_relationship") == {1: 1, 2: 0, 3: 0, 4: 0}


def test_duplicate_timestamp_tie_break_boundary():
    # Two transactions for the SAME pair at the exact same instant -- TRANSACTION_ID
    # breaks the tie deterministically; no leakage in either direction.
    df = pd.DataFrame({
        "TRANSACTION_ID": [10, 11],
        "CUSTOMER_ID": [1, 1],
        "TERMINAL_ID": [100, 100],
        "TX_DATETIME": pd.to_datetime(["2018-04-01 03:00:00", "2018-04-01 03:00:00"]),
        "TX_AMOUNT": [10.0, 20.0],
    })
    out = build_relationship_features(df)
    counts = _by_id(out, "pair_prior_interaction_count")
    assert counts[10] == 0  # earlier TRANSACTION_ID (tie-break) sees nothing
    assert counts[11] == 1  # later TRANSACTION_ID sees TRANSACTION_ID 10


def test_current_row_not_counted_in_its_own_interaction_count():
    df = pd.DataFrame({
        "TRANSACTION_ID": [1, 2],
        "CUSTOMER_ID": [1, 1],
        "TERMINAL_ID": [100, 100],
        "TX_DATETIME": pd.to_datetime(["2018-04-01 00:00:00", "2018-04-01 00:01:00"]),
        "TX_AMOUNT": [10.0, 20.0],
    })
    out = build_relationship_features(df)
    row2 = out.iloc[1]
    assert row2["pair_prior_interaction_count"] == 1  # sees only row1, not itself


def test_future_row_pair_reassignment_does_not_affect_earlier_established_history():
    # TXN2 has REAL established history (pair_prior_interaction_count=1 from TXN1), not
    # a cold-start zero. TXN3 (strictly after TXN2, same pair initially) is then
    # reassigned to a DIFFERENT customer -- changing its own pair membership entirely --
    # and TXN2's already-established value must not change either way.
    df = pd.DataFrame({
        "TRANSACTION_ID": [1, 2, 3],
        "CUSTOMER_ID": [1, 1, 1],
        "TERMINAL_ID": [100, 100, 100],
        "TX_DATETIME": pd.to_datetime(["2018-04-01 00:00:00", "2018-04-01 01:00:00", "2018-04-01 02:00:00"]),
        "TX_AMOUNT": [10.0, 20.0, 30.0],
    })
    before = build_relationship_features(df)
    row2_before = before.iloc[1]
    assert row2_before["pair_prior_interaction_count"] == 1
    assert row2_before["pair_is_new_relationship"] == 0

    mutated = df.copy()
    mutated.loc[2, "CUSTOMER_ID"] = 999  # TXN3 now belongs to pair (999, 100)
    after = build_relationship_features(mutated)
    row2_after = after.iloc[1]

    assert row2_after["pair_prior_interaction_count"] == row2_before["pair_prior_interaction_count"]
    assert row2_after["pair_is_new_relationship"] == row2_before["pair_is_new_relationship"]


def test_pair_isolation_same_customer_different_terminal_and_vice_versa():
    # TXN1: (1,100) first.  TXN2: (1,200) -- same customer, different terminal, must not
    # count toward (1,100).  TXN3: (2,100) -- different customer, same terminal, must not
    # count toward (1,100) either.  TXN4: (1,100) again -- should see ONLY TXN1.
    df = pd.DataFrame({
        "TRANSACTION_ID": [1, 2, 3, 4],
        "CUSTOMER_ID": [1, 1, 2, 1],
        "TERMINAL_ID": [100, 200, 100, 100],
        "TX_DATETIME": pd.to_datetime(["2018-04-01 00:00:00", "2018-04-01 01:00:00", "2018-04-01 02:00:00", "2018-04-01 03:00:00"]),
        "TX_AMOUNT": [10.0, 20.0, 30.0, 40.0],
    })
    out = build_relationship_features(df)
    row4 = out.iloc[3]
    assert row4["pair_prior_interaction_count"] == 1  # sees only TXN1, not TXN2 or TXN3
    assert row4["pair_is_new_relationship"] == 0


def test_no_historical_relationship_statistics_beyond_count_and_flag():
    # relationship.py intentionally has no pair-level amount/fraud aggregate -- confirm
    # the output is exactly the two documented features plus identifiers.
    df = pd.DataFrame({
        "TRANSACTION_ID": [1],
        "CUSTOMER_ID": [1],
        "TERMINAL_ID": [100],
        "TX_DATETIME": pd.to_datetime(["2018-04-01 00:00:00"]),
        "TX_AMOUNT": [10.0],
    })
    out = build_relationship_features(df)
    expected = {"TRANSACTION_ID", "CUSTOMER_ID", "TERMINAL_ID", "pair_prior_interaction_count", "pair_is_new_relationship"}
    assert set(out.columns) == expected


def test_no_label_columns_read_or_produced():
    df = pd.DataFrame({
        "TRANSACTION_ID": [1],
        "CUSTOMER_ID": [1],
        "TERMINAL_ID": [100],
        "TX_DATETIME": pd.to_datetime(["2018-04-01 00:00:00"]),
        "TX_AMOUNT": [10.0],
        "TX_FRAUD": [1],
        "TX_FRAUD_SCENARIO": [1],
    })
    out = build_relationship_features(df)
    assert "TX_FRAUD" not in out.columns
    assert "TX_FRAUD_SCENARIO" not in out.columns


def test_shuffled_input_invariance():
    df = pd.DataFrame({
        "TRANSACTION_ID": [1, 2, 3, 4, 5, 6],
        "CUSTOMER_ID": [1, 2, 1, 2, 1, 2],
        "TERMINAL_ID": [100, 200, 100, 201, 101, 200],
        "TX_DATETIME": pd.to_datetime(["2018-04-01 00:00", "2018-04-01 00:10", "2018-04-01 00:20", "2018-04-01 00:30", "2018-04-01 00:40", "2018-04-01 00:50"]),
        "TX_AMOUNT": [10.0, 15.0, 20.0, 25.0, 30.0, 35.0],
    })
    baseline = build_relationship_features(df).set_index("TRANSACTION_ID").sort_index()

    shuffled = df.sample(frac=1.0, random_state=13).reset_index(drop=True)
    shuffled_result = build_relationship_features(shuffled).set_index("TRANSACTION_ID").sort_index()

    pd.testing.assert_frame_equal(baseline, shuffled_result)
