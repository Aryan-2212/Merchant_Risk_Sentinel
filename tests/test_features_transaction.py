"""Tests for mrs.features.transaction (pure row-wise, no history involved)."""

from __future__ import annotations

import pandas as pd

from mrs.features.transaction import build_transaction_features


def _frame(datetimes, amounts=None):
    n = len(datetimes)
    return pd.DataFrame({
        "TRANSACTION_ID": list(range(n)),
        "TX_DATETIME": pd.to_datetime(datetimes),
        "TX_AMOUNT": amounts if amounts is not None else [1.0] * n,
    })


def test_row_count_and_order_preserved():
    df = _frame(["2018-04-01 08:00:00", "2018-04-02 09:00:00", "2018-04-03 10:00:00"])
    out = build_transaction_features(df)
    assert len(out) == len(df)
    assert list(out["TRANSACTION_ID"]) == list(df["TRANSACTION_ID"])


def test_tx_amount_passthrough():
    df = _frame(["2018-04-01 08:00:00", "2018-04-01 09:00:00"], amounts=[12.5, 300.0])
    out = build_transaction_features(df)
    assert list(out["tx_amount"]) == [12.5, 300.0]


def test_tx_hour_extraction():
    df = _frame(["2018-04-01 00:00:00", "2018-04-01 13:45:00", "2018-04-01 23:59:59"])
    out = build_transaction_features(df)
    assert list(out["tx_hour"]) == [0, 13, 23]


def test_tx_day_of_week_known_dates():
    # 2018-04-01 is a Sunday (dow=6); 2018-04-02 is a Monday (dow=0).
    df = _frame(["2018-04-01 12:00:00", "2018-04-02 12:00:00"])
    out = build_transaction_features(df)
    assert list(out["tx_day_of_week"]) == [6, 0]


def test_tx_is_weekend_saturday_and_sunday_true_weekday_false():
    df = _frame([
        "2018-04-01 12:00:00",  # Sunday
        "2018-04-02 12:00:00",  # Monday
        "2018-04-06 12:00:00",  # Friday
        "2018-04-07 12:00:00",  # Saturday
    ])
    out = build_transaction_features(df)
    assert list(out["tx_is_weekend"]) == [1, 0, 0, 1]


def test_tx_is_night_boundaries():
    # Night = hour >= 22 or hour < 6. Boundary hours: 21 (day), 22 (night),
    # 5 (night), 6 (day).
    df = _frame([
        "2018-04-01 21:00:00",
        "2018-04-01 22:00:00",
        "2018-04-01 05:00:00",
        "2018-04-01 06:00:00",
        "2018-04-01 00:00:00",
        "2018-04-01 23:59:59",
    ])
    out = build_transaction_features(df)
    assert list(out["tx_is_night"]) == [0, 1, 1, 0, 1, 1]


def test_output_has_exactly_the_expected_columns():
    df = _frame(["2018-04-01 08:00:00"])
    out = build_transaction_features(df)
    expected = {"TRANSACTION_ID", "tx_amount", "tx_hour", "tx_day_of_week", "tx_is_weekend", "tx_is_night"}
    assert set(out.columns) == expected


def test_no_label_columns_read_or_produced():
    df = pd.DataFrame({
        "TRANSACTION_ID": [0],
        "TX_DATETIME": pd.to_datetime(["2018-04-01 08:00:00"]),
        "TX_AMOUNT": [1.0],
        "TX_FRAUD": [1],
        "TX_FRAUD_SCENARIO": [1],
    })
    out = build_transaction_features(df)
    assert "TX_FRAUD" not in out.columns
    assert "TX_FRAUD_SCENARIO" not in out.columns
