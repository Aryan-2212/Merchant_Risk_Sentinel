"""Unit tests for mrs.data.dataset_stats against small synthetic frames."""

from __future__ import annotations

import pandas as pd

from mrs.data import dataset_stats


def _frame():
    return pd.DataFrame(
        {
            "TRANSACTION_ID": [0, 1, 2, 3, 4, 5],
            "TX_DATETIME": pd.to_datetime(
                [
                    "2018-04-01 08:00:00",  # Sunday
                    "2018-04-01 09:00:00",  # Sunday
                    "2018-04-02 08:00:00",  # Monday
                    "2018-05-01 08:00:00",
                    "2018-05-01 09:00:00",
                    "2018-05-02 08:00:00",
                ]
            ),
            "CUSTOMER_ID": [1, 1, 2, 1, 3, 3],
            "TERMINAL_ID": [10, 10, 20, 10, 30, 30],
            "TX_AMOUNT": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
            "TX_FRAUD": [0, 1, 0, 0, 1, 0],
            "TX_FRAUD_SCENARIO": [0, 2, 0, 0, 3, 0],
        }
    )


def test_overall_summary():
    summary = dataset_stats.overall_summary(_frame())
    assert summary["n_transactions"] == 6
    assert summary["n_customers"] == 3
    assert summary["n_terminals"] == 3
    assert summary["n_fraud"] == 2
    assert summary["fraud_rate"] == 2 / 6


def test_daily_summary_counts_and_rates():
    daily = dataset_stats.daily_summary(_frame())
    assert daily.loc[pd.Timestamp("2018-04-01").date(), "n_tx"] == 2
    assert daily.loc[pd.Timestamp("2018-04-01").date(), "n_fraud"] == 1
    assert daily.loc[pd.Timestamp("2018-04-01").date(), "fraud_rate"] == 0.5


def test_monthly_summary_aggregates_across_days():
    monthly = dataset_stats.monthly_summary(_frame())
    april = pd.Period("2018-04", freq="M")
    may = pd.Period("2018-05", freq="M")
    assert monthly.loc[april, "n_tx"] == 3
    assert monthly.loc[may, "n_tx"] == 3
    assert monthly.loc[april, "n_fraud"] == 1
    assert monthly.loc[may, "n_fraud"] == 1


def test_hourly_summary():
    hourly = dataset_stats.hourly_summary(_frame())
    assert hourly.loc[8, "n_tx"] == 4  # four rows at 08:00
    assert hourly.loc[9, "n_tx"] == 2  # two rows at 09:00


def test_day_of_week_summary_matches_known_weekdays():
    dow = dataset_stats.day_of_week_summary(_frame())
    # 2018-04-01 was a Sunday (dayofweek == 6); 2018-04-02 a Monday (0).
    assert dow.loc[6, "n_tx"] == 2
    assert dow.loc[0, "n_tx"] == 1


def test_scenario_counts_includes_genuine_zero():
    counts = dataset_stats.scenario_counts(_frame())
    assert counts.loc[0] == 4
    assert counts.loc[2] == 1
    assert counts.loc[3] == 1


def test_scenario_counts_by_month_only_includes_fraud_rows():
    by_month = dataset_stats.scenario_counts_by_month(_frame())
    april = pd.Period("2018-04", freq="M")
    may = pd.Period("2018-05", freq="M")
    assert by_month.loc[april, 2] == 1
    assert by_month.loc[may, 3] == 1
    # genuine scenario 0 must never appear as a column here
    assert 0 not in by_month.columns


def test_entity_activity_summary_counts_per_customer():
    summary = dataset_stats.entity_activity_summary(_frame(), "CUSTOMER_ID")
    assert summary["count"] == 3  # three distinct customers
    assert summary["max"] == 3  # customer 1 has 3 rows


def test_compromised_entity_counts():
    counts = dataset_stats.compromised_entity_counts(_frame())
    assert counts["n_terminals_ever_scenario2"] == 1
    assert counts["n_customers_ever_scenario3"] == 1


def test_compromise_episode_lengths_single_day_episodes():
    # Both scenario-2 and scenario-3 rows in _frame() are isolated single days.
    terminal_episodes = dataset_stats.compromise_episode_lengths(_frame(), 2, "TERMINAL_ID")
    assert list(terminal_episodes) == [1]

    customer_episodes = dataset_stats.compromise_episode_lengths(_frame(), 3, "CUSTOMER_ID")
    assert list(customer_episodes) == [1]


def test_compromise_episode_lengths_merges_contiguous_days():
    df = pd.DataFrame(
        {
            "TRANSACTION_ID": [0, 1, 2, 3],
            "TX_DATETIME": pd.to_datetime(
                [
                    "2018-04-01 08:00:00",
                    "2018-04-02 08:00:00",
                    "2018-04-03 08:00:00",
                    "2018-04-10 08:00:00",  # separate episode: gap > 1 day
                ]
            ),
            "CUSTOMER_ID": [1, 1, 1, 1],
            "TERMINAL_ID": [10, 10, 10, 10],
            "TX_AMOUNT": [10.0, 10.0, 10.0, 10.0],
            "TX_FRAUD": [1, 1, 1, 1],
            "TX_FRAUD_SCENARIO": [2, 2, 2, 2],
        }
    )
    episodes = dataset_stats.compromise_episode_lengths(df, 2, "TERMINAL_ID")
    assert sorted(episodes) == [1, 3]
