#!/usr/bin/env python
"""Phase 6 real-data run: terminal behavioral states over the full frozen dataset.

Pure orchestrator, mirroring scripts/06/07's pattern: loads the persisted Phase 3
feature splits (read-only), runs mrs.behavioral.terminal.compute_terminal_behavioral_states
ONCE over the full concatenated train+validation+test set (never per-split -- a
terminal's history and state genuinely continues across split boundaries, exactly as
Phase 3's build_feature_frame is itself built once over the complete dataset before ever
being split, Dev Plan Sec 13/39), and prints/persists a quantitative report.

TX_FRAUD_SCENARIO is used ONLY here, after the fact, for validation/reporting (comparing
detected behavioral episodes against known compromised-terminal windows) -- it is never
read by mrs.behavioral.terminal itself, which never sees a label column at all.

Run with: .venv/bin/python scripts/09_terminal_behavioral_report.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from mrs import config  # noqa: E402
from mrs.behavioral.terminal import (  # noqa: E402
    HIGH_RISK,
    INSUFFICIENT_HISTORY,
    NORMAL,
    RECOVERY,
    RISK_RISING,
    compute_terminal_behavioral_states,
)
from mrs.models.dataset import load_processed_transactions, load_split  # noqa: E402

OUTPUT_PATH = config.MODELS_DIR.parent / "docs" / "terminal_behavioral_report_data.json"

ELEVATED_STATES = (RISK_RISING, HIGH_RISK, RECOVERY)


def _episodes(group: pd.DataFrame) -> list[dict]:
    """Maximal contiguous runs of an elevated state (RISK_RISING/HIGH_RISK/RECOVERY) for
    one terminal's transactions, already in chronological order. An episode ends when the
    state returns to NORMAL (or the terminal's data ends). Returns one dict per episode.
    """
    episodes = []
    in_episode = False
    start_idx = None
    reached_high_risk = False

    states = group["terminal_risk_state"].to_numpy()
    times = group["TX_DATETIME"].to_numpy()

    for i, state in enumerate(states):
        elevated = state in ELEVATED_STATES
        if elevated and not in_episode:
            in_episode = True
            start_idx = i
            reached_high_risk = state == HIGH_RISK
        elif elevated and in_episode:
            reached_high_risk = reached_high_risk or (state == HIGH_RISK)
        elif not elevated and in_episode:
            episodes.append(
                {
                    "n_transactions": i - start_idx,
                    "duration_hours": (times[i - 1] - times[start_idx]) / np.timedelta64(1, "h"),
                    "reached_high_risk": bool(reached_high_risk),
                }
            )
            in_episode = False
            reached_high_risk = False

    if in_episode:
        episodes.append(
            {
                "n_transactions": len(states) - start_idx,
                "duration_hours": (times[-1] - times[start_idx]) / np.timedelta64(1, "h"),
                "reached_high_risk": bool(reached_high_risk),
                "still_active_at_end_of_data": True,
            }
        )
    return episodes


def main() -> None:
    print("Loading processed transactions (for TX_FRAUD_SCENARIO validation only)...")
    labels_source = load_processed_transactions()

    print("Loading all three Phase 3 feature splits and concatenating into one full,")
    print("chronologically-continuous dataset (never processed per-split)...")
    train_df = load_split("train", labels_source)
    validation_df = load_split("validation", labels_source)
    test_df = load_split("test", labels_source)
    full_df = pd.concat([train_df, validation_df, test_df], ignore_index=True)
    print(f"  total rows: {len(full_df):,}")
    print(f"  distinct terminals: {full_df['TERMINAL_ID'].nunique():,}")
    print(f"  date range: {full_df['TX_DATETIME'].min()} -> {full_df['TX_DATETIME'].max()}")

    print("\nRunning compute_terminal_behavioral_states over the full dataset...")
    start = time.time()
    result = compute_terminal_behavioral_states(full_df)
    elapsed = time.time() - start
    print(f"  completed in {elapsed:.2f}s ({len(result) / elapsed:,.0f} rows/sec)")

    assert len(result) == len(full_df)
    assert set(result["TRANSACTION_ID"]) == set(full_df["TRANSACTION_ID"])

    # Merge back TERMINAL_ID/TX_DATETIME/TX_FRAUD_SCENARIO for reporting only -- never
    # fed back into the state machine, which has already finished running.
    merged = result.merge(
        full_df[["TRANSACTION_ID", "TERMINAL_ID", "TX_DATETIME", "TX_FRAUD", "TX_FRAUD_SCENARIO"]],
        on="TRANSACTION_ID", how="inner", validate="one_to_one",
    ).sort_values(["TERMINAL_ID", "TX_DATETIME", "TRANSACTION_ID"]).reset_index(drop=True)
    assert len(merged) == len(full_df)

    print("\n=== State distribution (all transactions) ===")
    state_counts = merged["terminal_risk_state"].value_counts()
    state_pcts = (state_counts / len(merged) * 100).round(4)
    for state in [INSUFFICIENT_HISTORY, NORMAL, RISK_RISING, HIGH_RISK, RECOVERY]:
        count = int(state_counts.get(state, 0))
        pct = float(state_pcts.get(state, 0.0))
        print(f"  {state:<22} {count:>10,}  ({pct:.4f}%)")

    print("\n=== Terminal-level summary ===")
    per_terminal_states = merged.groupby("TERMINAL_ID")["terminal_risk_state"].apply(set)
    n_terminals = len(per_terminal_states)
    n_ever_risk_rising = int(per_terminal_states.apply(lambda s: RISK_RISING in s).sum())
    n_ever_high_risk = int(per_terminal_states.apply(lambda s: HIGH_RISK in s).sum())
    n_ever_recovery = int(per_terminal_states.apply(lambda s: RECOVERY in s).sum())
    n_only_insufficient = int(per_terminal_states.apply(lambda s: s == {INSUFFICIENT_HISTORY}).sum())
    n_pure_normal = int(per_terminal_states.apply(lambda s: s <= {NORMAL, INSUFFICIENT_HISTORY} and NORMAL in s).sum())
    print(f"  total distinct terminals: {n_terminals:,}")
    print(f"  terminals that ever reached RISK_RISING: {n_ever_risk_rising:,}")
    print(f"  terminals that ever reached HIGH_RISK: {n_ever_high_risk:,}")
    print(f"  terminals that ever reached RECOVERY: {n_ever_recovery:,}")
    print(f"  terminals that never left INSUFFICIENT_HISTORY (too low-volume): {n_only_insufficient:,}")
    print(f"  terminals that stayed purely NORMAL (never elevated): {n_pure_normal:,}")

    print("\n=== Behavioral episodes (contiguous elevated runs: RISK_RISING/HIGH_RISK/RECOVERY) ===")
    start = time.time()
    all_episodes: list[dict] = []
    for terminal_id, group in merged.groupby("TERMINAL_ID", sort=False):
        all_episodes.extend(_episodes(group))
    episode_time = time.time() - start
    print(f"  episode extraction completed in {episode_time:.2f}s")

    episodes_df = pd.DataFrame(all_episodes)
    n_episodes = len(episodes_df)
    n_high_risk_episodes = int(episodes_df["reached_high_risk"].sum()) if n_episodes else 0
    print(f"  total elevated episodes: {n_episodes:,}")
    print(f"  episodes that reached HIGH_RISK: {n_high_risk_episodes:,}")
    print(f"  episodes that stayed at RISK_RISING only (resolved without confirmed high): {n_episodes - n_high_risk_episodes:,}")
    if n_episodes:
        print(f"  episode length (transactions): median={episodes_df['n_transactions'].median():.1f} "
              f"mean={episodes_df['n_transactions'].mean():.1f} max={episodes_df['n_transactions'].max()}")
        print(f"  episode duration (hours): median={episodes_df['duration_hours'].median():.2f} "
              f"mean={episodes_df['duration_hours'].mean():.2f} max={episodes_df['duration_hours'].max():.2f}")
        if n_high_risk_episodes:
            hr = episodes_df[episodes_df["reached_high_risk"]]
            print(f"  HIGH_RISK-reaching episode duration (hours): median={hr['duration_hours'].median():.2f} "
                  f"mean={hr['duration_hours'].mean():.2f} max={hr['duration_hours'].max():.2f}")

    print("\n=== Cross-check against known Scenario 2 (compromised terminal) labels ===")
    print("(TX_FRAUD_SCENARIO used here ONLY for this post-hoc validation printout --")
    print(" never read by the state machine itself.)")
    scenario2 = merged[merged["TX_FRAUD_SCENARIO"] == 2]
    n_scenario2 = len(scenario2)
    scenario2_elevated = scenario2["terminal_risk_state"].isin(ELEVATED_STATES).sum()
    scenario2_high_risk = (scenario2["terminal_risk_state"] == HIGH_RISK).sum()
    print(f"  total Scenario-2 fraud transactions: {n_scenario2:,}")
    print(f"  ... flagged elevated (RISK_RISING/HIGH_RISK/RECOVERY) at that moment: "
          f"{scenario2_elevated:,} ({scenario2_elevated / n_scenario2 * 100:.2f}%)")
    print(f"  ... flagged specifically HIGH_RISK at that moment: "
          f"{scenario2_high_risk:,} ({scenario2_high_risk / n_scenario2 * 100:.2f}%)")

    high_risk_rows = merged[merged["terminal_risk_state"] == HIGH_RISK]
    n_high_risk_rows = len(high_risk_rows)
    high_risk_fraud_breakdown = high_risk_rows["TX_FRAUD_SCENARIO"].value_counts().sort_index()
    print(f"\n  Of {n_high_risk_rows:,} transactions flagged HIGH_RISK, breakdown by TX_FRAUD_SCENARIO:")
    for scenario, count in high_risk_fraud_breakdown.items():
        label = "genuine (non-fraud)" if scenario == 0 else f"scenario {scenario}"
        print(f"    {label}: {count:,} ({count / n_high_risk_rows * 100:.2f}%)")

    report_data = {
        "total_rows": len(merged),
        "distinct_terminals": n_terminals,
        "runtime_seconds": elapsed,
        "rows_per_second": len(result) / elapsed,
        "state_distribution": {k: int(v) for k, v in state_counts.to_dict().items()},
        "terminals_ever_risk_rising": n_ever_risk_rising,
        "terminals_ever_high_risk": n_ever_high_risk,
        "terminals_ever_recovery": n_ever_recovery,
        "terminals_only_insufficient_history": n_only_insufficient,
        "terminals_pure_normal": n_pure_normal,
        "n_episodes": n_episodes,
        "n_high_risk_episodes": n_high_risk_episodes,
        "episode_length_transactions_median": float(episodes_df["n_transactions"].median()) if n_episodes else None,
        "episode_length_transactions_max": int(episodes_df["n_transactions"].max()) if n_episodes else None,
        "episode_duration_hours_median": float(episodes_df["duration_hours"].median()) if n_episodes else None,
        "episode_duration_hours_max": float(episodes_df["duration_hours"].max()) if n_episodes else None,
        "scenario2_total": int(n_scenario2),
        "scenario2_flagged_elevated": int(scenario2_elevated),
        "scenario2_flagged_high_risk": int(scenario2_high_risk),
        "high_risk_rows_total": n_high_risk_rows,
        "high_risk_rows_by_scenario": {str(k): int(v) for k, v in high_risk_fraud_breakdown.to_dict().items()},
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(report_data, f, indent=2, sort_keys=True)
    print(f"\nReport data written to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
