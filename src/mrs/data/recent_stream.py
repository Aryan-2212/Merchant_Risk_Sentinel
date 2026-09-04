"""Deterministic simulated recent transaction stream for the final demo.

This module adds a clearly separate 21-day *simulated operational window* on top of the
frozen Fraud Detection Handbook benchmark. It never changes the benchmark split or model
training/evaluation data. The generator reuses Handbook customer/terminal profiles and
creates controlled temporal scenarios so the existing causal feature + behavioral + risk
pipeline can demonstrate NORMAL -> RISK_RISING -> HIGH_RISK -> RECOVERY patterns.

The generated TX_FRAUD/TX_FRAUD_SCENARIO fields are synthetic scenario annotations only.
They are never passed to the transaction model as features and are never included in the
official benchmark metrics.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import numpy as np
import pandas as pd

from mrs.data.schema import RAW_COLUMNS
from mrs.features.customer import build_customer_features
from mrs.features.relationship import build_relationship_features
from mrs.features.terminal import build_terminal_features
from mrs.features.transaction import build_transaction_features
from mrs.features import _temporal as T

RECENT_DAYS = 21
DEFAULT_TX_PER_DAY = 1_800
DEFAULT_SEED = 20260904
RECENT_START = pd.Timestamp("2026-08-15")
RECENT_END = pd.Timestamp("2026-09-04 23:59:59")
RECENT_SPLIT = "recent"
RECENT_ID_START = 2_000_000


@dataclass(frozen=True)
class RecentStreamConfig:
    start: pd.Timestamp = RECENT_START
    days: int = RECENT_DAYS
    transactions_per_day: int = DEFAULT_TX_PER_DAY
    seed: int = DEFAULT_SEED
    transaction_id_start: int = RECENT_ID_START

    @property
    def end(self) -> pd.Timestamp:
        return self.start + pd.Timedelta(days=self.days) - pd.Timedelta(seconds=1)


def _profile_maps(customer_profiles: pd.DataFrame, terminal_profiles: pd.DataFrame):
    customers = customer_profiles.copy()
    terminals = terminal_profiles.copy()
    customers["CUSTOMER_ID"] = customers["CUSTOMER_ID"].astype(int)
    terminals["TERMINAL_ID"] = terminals["TERMINAL_ID"].astype(int)
    customer_rows = {int(r.CUSTOMER_ID): r for r in customers.itertuples(index=False)}
    terminal_ids = terminals["TERMINAL_ID"].astype(int).to_numpy()

    terminal_customers: dict[int, list[int]] = {int(t): [] for t in terminal_ids}
    for row in customers.itertuples(index=False):
        for terminal_id in row.available_terminals:
            if int(terminal_id) in terminal_customers:
                terminal_customers[int(terminal_id)].append(int(row.CUSTOMER_ID))
    return customer_rows, terminal_ids, terminal_customers


def _sample_amount(rng: np.random.Generator, profile, multiplier: float = 1.0) -> float:
    mean = max(float(profile.mean_amount), 1.0)
    std = max(float(profile.std_amount), mean * 0.05)
    amount = rng.normal(mean * multiplier, std * (1.0 if multiplier <= 1.5 else 1.25))
    return float(max(0.50, amount))


def generate_recent_transactions(
    customer_profiles: pd.DataFrame,
    terminal_profiles: pd.DataFrame,
    *,
    config: RecentStreamConfig | None = None,
) -> pd.DataFrame:
    """Generate a reproducible 21-day recent stream with controlled risk episodes.

    Week 1 is mostly baseline activity. Week 2 introduces rising activity for selected
    customers/terminals. Week 3 contains a stronger incident followed by two recovery days.
    This creates explainable temporal evidence without changing the trained model.
    """
    cfg = config or RecentStreamConfig()
    rng = np.random.default_rng(cfg.seed)
    customer_rows, terminal_ids, terminal_customers = _profile_maps(customer_profiles, terminal_profiles)
    customer_ids = np.array(sorted(customer_rows), dtype=int)

    # Fixed cohorts make the demo stable across runs and ensure the same investigations
    # can be revisited from the UI.
    target_customers = customer_ids[:12]
    eligible_terminals = [t for t in terminal_ids if terminal_customers.get(int(t))]
    target_terminals = np.array(eligible_terminals[:12], dtype=int)
    target_terminal_set = set(map(int, target_terminals))
    target_customer_set = set(map(int, target_customers))

    rows: list[dict] = []
    next_id = cfg.transaction_id_start

    for day in range(cfg.days):
        day_start = cfg.start + pd.Timedelta(days=day)
        phase = 0 if day < 7 else 1 if day < 14 else 2
        recovery = day >= 19

        for _ in range(cfg.transactions_per_day):
            # Concentrate a controlled fraction of traffic on the target cohorts only
            # after the baseline week has established enough history.
            target_terminal = phase >= 1 and rng.random() < (0.14 if phase == 1 else 0.22)
            target_customer = phase >= 1 and rng.random() < (0.10 if phase == 1 else 0.16)

            if target_terminal and not recovery:
                terminal_id = int(rng.choice(target_terminals))
                possible_customers = terminal_customers.get(terminal_id) or customer_ids.tolist()
                customer_id = int(rng.choice(possible_customers))
            else:
                customer_id = int(rng.choice(customer_ids))
                profile = customer_rows[customer_id]
                available = np.array([int(t) for t in profile.available_terminals], dtype=int)
                terminal_id = int(rng.choice(available))

            # Explicit customer cohort selection makes the customer behavioral scenario
            # reproducible rather than relying on a lucky random sample.
            if target_customer and not recovery:
                customer_id = int(rng.choice(target_customers))
                profile = customer_rows[customer_id]
                available = np.array([int(t) for t in profile.available_terminals], dtype=int)
                terminal_id = int(rng.choice(available))

            profile = customer_rows[customer_id]
            multiplier = 1.0
            if customer_id in target_customer_set and phase >= 1 and not recovery:
                multiplier = 2.5 if phase == 1 else 4.8
            if recovery and customer_id in target_customer_set:
                multiplier = 1.0

            amount = _sample_amount(rng, profile, multiplier)

            # Synthetic annotations are deliberately sparse in the baseline and increase
            # around the controlled terminal/customer incidents. They are retained only as
            # post-hoc simulation labels; downstream features still enforce label exclusion.
            fraud_probability = 0.004
            scenario = 0
            if terminal_id in target_terminal_set and phase >= 1 and not recovery:
                fraud_probability = 0.18 if phase == 1 else 0.32
                scenario = 2
            elif customer_id in target_customer_set and phase >= 1 and not recovery:
                fraud_probability = 0.10 if phase == 1 else 0.18
                scenario = 3

            is_fraud = int(rng.random() < fraud_probability)
            if not is_fraud:
                scenario = 0

            # Spread events through the day with realistic bursts around several hours.
            hour = int(rng.choice([0, 1, 2, 3, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]))
            minute = int(rng.integers(0, 60))
            second = int(rng.integers(0, 60))
            timestamp = day_start + pd.Timedelta(hours=hour, minutes=minute, seconds=second)
            rows.append(
                {
                    "TRANSACTION_ID": next_id,
                    "TX_DATETIME": timestamp,
                    "CUSTOMER_ID": customer_id,
                    "TERMINAL_ID": terminal_id,
                    "TX_AMOUNT": amount,
                    "TX_TIME_SECONDS": int(day * 86400 + hour * 3600 + minute * 60 + second),
                    "TX_TIME_DAYS": day,
                    "TX_FRAUD": is_fraud,
                    "TX_FRAUD_SCENARIO": scenario,
                }
            )
            next_id += 1

    df = pd.DataFrame(rows, columns=list(RAW_COLUMNS))
    df["TX_DATETIME"] = pd.to_datetime(df["TX_DATETIME"])
    df = df.sort_values(["TX_DATETIME", "TRANSACTION_ID"]).reset_index(drop=True)
    if df.empty or len(df) != cfg.days * cfg.transactions_per_day:
        raise AssertionError("recent stream generator produced an unexpected row count")
    if df["TRANSACTION_ID"].duplicated().any():
        raise AssertionError("recent stream generator produced duplicate transaction IDs")
    return df


def build_recent_feature_frame(transactions: pd.DataFrame) -> pd.DataFrame:
    """Run the existing Phase-3 feature components on the recent window only.

    This intentionally does not call ``build_feature_frame`` because the frozen train /
    validation / test split boundaries are 2018 dates. The component feature builders are
    unchanged; this orchestration simply omits the benchmark split assignment and labels.
    """
    ordered = T.sort_canonical(transactions)
    tx = build_transaction_features(ordered)
    cust = build_customer_features(ordered)
    term = build_terminal_features(ordered)
    rel = build_relationship_features(ordered)
    result = tx.merge(cust, on="TRANSACTION_ID", validate="one_to_one")
    result = result.merge(term, on="TRANSACTION_ID", validate="one_to_one")
    result = result.merge(rel, on="TRANSACTION_ID", validate="one_to_one")
    result = result.merge(ordered[["TRANSACTION_ID", "TX_DATETIME"]], on="TRANSACTION_ID", validate="one_to_one")
    result = result.sort_values(["TX_DATETIME", "TRANSACTION_ID"]).reset_index(drop=True)
    return result
