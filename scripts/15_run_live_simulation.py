#!/usr/bin/env python
"""Simulated Live Transaction Ingestion -- progressive, chronological playback of the
deterministic Simulated Recent Operational Stream (see mrs.live.simulate,
mrs.data.recent_stream, docs/RECENT_STREAM.md).

NOT real payment traffic. NOT a live production feed. A deterministic simulated live
demonstration: the same 21-day recent stream scripts/14_ingest_recent_stream.py
batch-ingests, released one (or a few) transactions at a time through the exact same
reused pipeline, so the dashboard's Entity Network and Overview can show it "arriving"
for a demo rather than already fully materialized.

Idempotent / resumable: transactions already persisted (by a prior live run, or by the
batch script) are skipped -- this run picks up wherever the recent stream's own
deterministic order left off. If nothing is pending, run
scripts/16_reset_recent_stream.py first (scoped only to recent-stream rows; the frozen
benchmark is never touched).

Examples:

    .venv/bin/python scripts/15_run_live_simulation.py --interval 2
        ~1 transaction every 2 seconds (the default).

    .venv/bin/python scripts/15_run_live_simulation.py --interval 0.2 --limit 300
        A faster demo mode: ~5/second, stopping after 300 transactions.

    .venv/bin/python scripts/15_run_live_simulation.py --batch-size 5 --interval 3
        5 transactions released together every 3 seconds (still one deterministic,
        chronologically-ordered pipeline call per tick).

No Kafka, no queues, no streaming infrastructure -- a single deterministic Python
process, exactly as much as a demo needs.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from mrs.db.engine import get_database_url, get_engine  # noqa: E402
from mrs.live.simulate import ingest_batch, load_model_and_threshold, load_stream_and_pending  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--interval", type=float, default=2.0, help="Seconds to wait between releases (default: 2.0).")
    parser.add_argument("--batch-size", type=int, default=1, help="Transactions released per tick (default: 1).")
    parser.add_argument("--limit", type=int, default=None, help="Stop after releasing this many transactions (default: run to the end of the stream).")
    args = parser.parse_args()

    engine = get_engine()
    print(f"Database: {get_database_url()}")
    print(f"interval={args.interval}s  batch_size={args.batch_size}  limit={args.limit or 'end of stream'}")

    print()
    print("=" * 70)
    print("Loading the deterministic recent stream and finding where it left off")
    print("=" * 70)
    released_so_far, pending = load_stream_and_pending(engine)
    print(f"Already persisted: {len(released_so_far):,} transactions")
    print(f"Pending: {len(pending):,} transactions")

    if len(pending) == 0:
        print()
        print("Nothing pending -- the recent stream is already fully ingested.")
        print("Run scripts/16_reset_recent_stream.py first to demo live playback from the start.")
        return

    if args.limit is not None:
        pending = pending.head(args.limit)
        print(f"Limited to the first {len(pending):,} pending transactions this run.")

    print()
    print("=" * 70)
    print("Loading the frozen Phase 5 XGBoost model (inference only, no retraining)")
    print("=" * 70)
    pipeline, threshold = load_model_and_threshold()
    print(f"Model: xgboost_v1  threshold: {threshold}")

    print()
    print("=" * 70)
    print(f"Releasing {len(pending):,} transactions ({args.interval}s apart, batch size {args.batch_size})")
    print("=" * 70)

    n_released = 0
    n_alerts = 0
    t_start = time.time()
    try:
        for start in range(0, len(pending), args.batch_size):
            batch = pending.iloc[start : start + args.batch_size]
            result = ingest_batch(engine, pipeline, threshold, released_so_far, batch)

            for i, tid in enumerate(result["transaction_ids"]):
                level = result["unified_risk_levels"][tid]
                row = batch[batch["TRANSACTION_ID"] == tid].iloc[0]
                print(
                    f"  [{n_released + i + 1:>6}/{len(pending)}] "
                    f"TX_{tid} {row['TX_DATETIME']} CUST_{row['CUSTOMER_ID']} -> TERM_{row['TERMINAL_ID']} "
                    f"amount={row['TX_AMOUNT']:.2f} risk={level}"
                )

            n_released += len(result["transaction_ids"])
            n_alerts += result["n_alerts_written"]
            released_so_far = pd.concat([released_so_far, batch], ignore_index=True)

            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nStopped by user.")

    elapsed = time.time() - t_start
    print()
    print("=" * 70)
    print(f"DONE -- released {n_released:,} transactions, wrote {n_alerts:,} alerts, in {elapsed:.1f}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
