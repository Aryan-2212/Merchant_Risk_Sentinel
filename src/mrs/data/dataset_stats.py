"""Descriptive statistics over the processed transaction dataset (Phase 2).

Pure functions over an already-loaded processed DataFrame. Nothing here writes
anything or mutates its input; :mod:`scripts.04_dataset_report` is the only caller that
turns this into a written report. Kept separate from ``mrs.data.build_processed`` because
"understand the data" and "build the data" are different concerns (Dev Plan §33.3).
"""

from __future__ import annotations

import pandas as pd


def overall_summary(df: pd.DataFrame) -> dict:
    """Headline counts: rows, date range, entity counts, fraud count/rate."""
    return {
        "n_transactions": int(len(df)),
        "date_min": df["TX_DATETIME"].min(),
        "date_max": df["TX_DATETIME"].max(),
        "n_customers": int(df["CUSTOMER_ID"].nunique()),
        "n_terminals": int(df["TERMINAL_ID"].nunique()),
        "n_fraud": int(df["TX_FRAUD"].sum()),
        "fraud_rate": float(df["TX_FRAUD"].mean()),
    }


def _rate_table(df: pd.DataFrame, key: pd.Series | str) -> pd.DataFrame:
    grouped = df.groupby(key)
    out = grouped["TX_FRAUD"].agg(n_tx="count", n_fraud="sum")
    out["fraud_rate"] = out["n_fraud"] / out["n_tx"]
    return out


def daily_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Transaction count, fraud count, and fraud rate per calendar date."""
    return _rate_table(df, df["TX_DATETIME"].dt.date)


def monthly_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Transaction count, fraud count, and fraud rate per calendar month."""
    return _rate_table(df, df["TX_DATETIME"].dt.to_period("M"))


def hourly_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Transaction count, fraud count, and fraud rate per hour-of-day (0-23)."""
    return _rate_table(df, df["TX_DATETIME"].dt.hour)


def day_of_week_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Transaction count, fraud count, and fraud rate per weekday (0=Mon..6=Sun)."""
    return _rate_table(df, df["TX_DATETIME"].dt.dayofweek)


def scenario_counts(df: pd.DataFrame) -> pd.Series:
    """Row count for each TX_FRAUD_SCENARIO value (0 = genuine)."""
    return df["TX_FRAUD_SCENARIO"].value_counts().sort_index()


def scenario_counts_by_month(df: pd.DataFrame) -> pd.DataFrame:
    """Fraud-only (scenario != 0) counts, months x scenario."""
    fraud = df[df["TX_FRAUD"] == 1]
    month = fraud["TX_DATETIME"].dt.to_period("M")
    return fraud.groupby([month, "TX_FRAUD_SCENARIO"]).size().unstack(fill_value=0)


def entity_activity_summary(df: pd.DataFrame, id_column: str) -> pd.Series:
    """describe()-style distribution of transaction count per entity (customer/terminal)."""
    return df.groupby(id_column).size().describe()


def compromised_entity_counts(df: pd.DataFrame) -> dict:
    """How many distinct terminals/customers were ever labeled compromised."""
    return {
        "n_terminals_ever_scenario2": int(
            df.loc[df["TX_FRAUD_SCENARIO"] == 2, "TERMINAL_ID"].nunique()
        ),
        "n_customers_ever_scenario3": int(
            df.loc[df["TX_FRAUD_SCENARIO"] == 3, "CUSTOMER_ID"].nunique()
        ),
    }


def compromise_episode_lengths(df: pd.DataFrame, scenario: int, id_column: str) -> pd.Series:
    """Length (in days) of each contiguous run of fraud-labeled days per entity.

    This is a proxy for 'how long a compromise stays visible in the data', not the
    simulator's internal compromise-window length: the window can be longer than what is
    observed here if, on some day inside the true window, none of that entity's
    transactions happened to be sampled as fraudulent (Dev Plan §33.11 — this
    approximation is stated explicitly rather than silently assumed to be exact).
    """
    subset = df[df["TX_FRAUD_SCENARIO"] == scenario]
    lengths: list[int] = []
    for _, group in subset.groupby(id_column):
        days = sorted(group["TX_DATETIME"].dt.date.unique())
        run_start = run_prev = days[0]
        for day in days[1:]:
            if (day - run_prev).days > 1:
                lengths.append((run_prev - run_start).days + 1)
                run_start = day
            run_prev = day
        lengths.append((run_prev - run_start).days + 1)
    return pd.Series(lengths, name="episode_length_days")
