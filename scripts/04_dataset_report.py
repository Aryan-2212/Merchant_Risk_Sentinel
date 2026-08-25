#!/usr/bin/env python
"""Compute Phase 2 dataset-understanding statistics and print them for the report.

Run with: .venv/bin/python scripts/04_dataset_report.py

This script only prints measured numbers; it does not write docs/DATASET_REPORT.md
itself. The report is authored narratively from this output (the same convention Phase 1
used for docs/PHASE1_REPORT.md). Every figure quoted in docs/DATASET_REPORT.md must be
reproducible by re-running this script against the processed dataset.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from mrs import config  # noqa: E402
from mrs.data import dataset_stats  # noqa: E402
from mrs.data.splits import SPLIT_BOUNDARIES  # noqa: E402
from mrs.data.splits import SPLIT_ORDER  # noqa: E402
from mrs.data.splits import assign_split  # noqa: E402


def load_processed():
    parts = sorted(config.PROCESSED_TRANSACTIONS_DIR.glob("*.parquet"))
    if not parts:
        raise SystemExit("No processed data found. Run scripts/02_build_processed.py first.")
    frames = [pd.read_parquet(p) for p in parts]
    return pd.concat(frames, ignore_index=True)


def print_split_report(df):
    print()
    print("=== Chronological split ===")
    labeled = df.assign(split=assign_split(df["TX_DATETIME"]))

    for name in SPLIT_ORDER:
        start, end = SPLIT_BOUNDARIES[name]
        part = labeled[labeled["split"] == name]
        fraud_rows = part[part["TX_FRAUD"] == 1]
        scenario_breakdown = fraud_rows["TX_FRAUD_SCENARIO"].value_counts()

        n_tx = len(part)
        n_fraud = int(part["TX_FRAUD"].sum())
        fraud_rate = part["TX_FRAUD"].mean()
        n_s1 = int(scenario_breakdown.get(1, 0))
        n_s2 = int(scenario_breakdown.get(2, 0))
        n_s3 = int(scenario_breakdown.get(3, 0))

        line = (
            "  " + name.ljust(10) + " " + start + ".." + end
            + "  n_tx=" + format(n_tx, ",")
            + "  n_fraud=" + format(n_fraud, ",")
            + "  fraud_rate=" + format(fraud_rate, ".4%")
            + "  scenario1/2/3=" + str(n_s1) + "/" + str(n_s2) + "/" + str(n_s3)
        )
        print(line)


def main():
    df = load_processed()

    print("=== Overall summary ===")
    overall = dataset_stats.overall_summary(df)
    for key in overall:
        print("  " + key + ": " + str(overall[key]))

    print()
    print("=== Monthly summary ===")
    print(dataset_stats.monthly_summary(df).to_string())

    print()
    print("=== First 15 days (ramp-up check) ===")
    print(dataset_stats.daily_summary(df).head(15).to_string())

    print()
    print("=== Scenario counts (overall) ===")
    print(dataset_stats.scenario_counts(df).to_string())

    print()
    print("=== Scenario counts by month (fraud rows only) ===")
    print(dataset_stats.scenario_counts_by_month(df).to_string())

    print()
    print("=== Hourly summary ===")
    print(dataset_stats.hourly_summary(df).to_string())

    print()
    print("=== Day-of-week summary (0=Mon..6=Sun) ===")
    print(dataset_stats.day_of_week_summary(df).to_string())

    print()
    print("=== Transactions per customer ===")
    print(dataset_stats.entity_activity_summary(df, "CUSTOMER_ID").to_string())

    print()
    print("=== Transactions per terminal ===")
    print(dataset_stats.entity_activity_summary(df, "TERMINAL_ID").to_string())

    print()
    print("=== Compromised entity counts ===")
    compromised = dataset_stats.compromised_entity_counts(df)
    for key in compromised:
        print("  " + key + ": " + str(compromised[key]))

    print()
    print("=== Compromise episode length (days), scenario 2 (terminal) ===")
    terminal_episodes = dataset_stats.compromise_episode_lengths(df, 2, "TERMINAL_ID")
    print(terminal_episodes.describe().to_string())

    print()
    print("=== Compromise episode length (days), scenario 3 (customer) ===")
    customer_episodes = dataset_stats.compromise_episode_lengths(df, 3, "CUSTOMER_ID")
    print(customer_episodes.describe().to_string())

    print_split_report(df)


if __name__ == "__main__":
    main()
