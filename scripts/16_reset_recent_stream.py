#!/usr/bin/env python
"""Safe, scoped reset of the Simulated Recent Operational Stream (and nothing else).

Deletes ONLY rows with transaction_id >= mrs.config.RECENT_STREAM_TX_ID_OFFSET, across
alerts / audit_logs / risk_scores / transaction_features / transactions -- the exact
id range mrs.data.recent_stream generates into (2,000,000,000+), far above the frozen
benchmark's own maximum id (1,754,154). The frozen benchmark (train/validation/test)
is never touched by this script; it has no code path that could reach a benchmark row.

Use this before a Simulated Live Stream demo run (scripts/15_run_live_simulation.py)
to clear a previously batch- or live-ingested recent stream so the live producer has
pending (not-yet-persisted) transactions to progressively release again -- otherwise
every transaction in the deterministic stream already exists and the live run has
nothing left to do (a correct, but demo-uninteresting, outcome).

Does NOT re-populate customers/terminals (never touched by this reset either -- the
recent stream only ever reused existing rows there, never inserted new ones).

Run with: .venv/bin/python scripts/16_reset_recent_stream.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sqlalchemy import func, select, text  # noqa: E402

from mrs import config  # noqa: E402
from mrs.db.engine import get_database_url, get_engine  # noqa: E402
from mrs.db.models import Alert, AuditLog, RiskScore, Transaction, TransactionFeatures  # noqa: E402

_TABLES = ("alerts", "audit_logs", "risk_scores", "transaction_features", "transactions")


def main() -> None:
    engine = get_engine()
    print(f"Database: {get_database_url()}")
    print(f"Scope: transaction_id >= {config.RECENT_STREAM_TX_ID_OFFSET:,} only")

    with engine.connect() as conn:
        benchmark_before = conn.execute(
            select(func.count()).select_from(Transaction.__table__).where(
                Transaction.transaction_id < config.RECENT_STREAM_TX_ID_OFFSET
            )
        ).scalar_one()
        recent_before = {
            model.__tablename__: conn.execute(
                select(func.count()).select_from(model.__table__).where(
                    model.transaction_id >= config.RECENT_STREAM_TX_ID_OFFSET
                )
            ).scalar_one()
            for model in (Transaction, TransactionFeatures, RiskScore)
        }
        recent_before["alerts"] = conn.execute(
            select(func.count()).select_from(Alert.__table__).where(Alert.transaction_id >= config.RECENT_STREAM_TX_ID_OFFSET)
        ).scalar_one()
        recent_before["audit_logs"] = conn.execute(
            select(func.count()).select_from(AuditLog.__table__).where(
                AuditLog.transaction_id >= config.RECENT_STREAM_TX_ID_OFFSET
            )
        ).scalar_one()

    print("\nBefore reset:")
    print(f"  benchmark transactions (untouched by this script): {benchmark_before:,}")
    for table in _TABLES:
        print(f"  {table}: {recent_before[table]:,}")

    with engine.begin() as conn:
        for table in _TABLES:
            conn.execute(text(f"DELETE FROM {table} WHERE transaction_id >= :offset"), {"offset": config.RECENT_STREAM_TX_ID_OFFSET})

    with engine.connect() as conn:
        benchmark_after = conn.execute(
            select(func.count()).select_from(Transaction.__table__).where(
                Transaction.transaction_id < config.RECENT_STREAM_TX_ID_OFFSET
            )
        ).scalar_one()

    print("\nAfter reset:")
    print(f"  benchmark transactions: {benchmark_after:,} (must equal the before-count above)")
    for table in _TABLES:
        print(f"  {table}: 0")

    if benchmark_after != benchmark_before:
        raise AssertionError(
            f"scripts/16_reset_recent_stream.py: benchmark transaction count changed "
            f"({benchmark_before:,} -> {benchmark_after:,}) -- this must never happen; "
            "investigate before trusting this database further."
        )
    print("\nDone. Benchmark row count verified unchanged.")


if __name__ == "__main__":
    main()
