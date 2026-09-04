#!/usr/bin/env python
"""Ingest the Simulated Recent Operational Stream (see mrs.data.recent_stream).

Generates the deterministic 21-day recent stream and pushes it through the SAME,
UNMODIFIED pipeline scripts/12_populate_db.py already validated for the frozen
benchmark: feature build -> frozen Phase 5 XGBoost inference (no retraining) ->
Phase 6/7 behavioral engines -> Phase 7 risk aggregation -> Phase 8 persistence ->
Phase 8 policy engine. Computes nothing new of its own; every step below is a call
into an existing, already-tested module.

Does NOT call mrs.db.populate.populate_customers_and_terminals -- the recent stream
reuses existing customer/terminal IDs already present in those tables (Dev Plan
addendum: "use existing customer IDs", "use existing terminal IDs"), so inserting
reference rows again would violate their primary keys.

Idempotent / rerun-safe: refuses to run if any transaction_id in the recent stream's
id range (>= mrs.config.RECENT_STREAM_TX_ID_OFFSET) already exists (mirrors
mrs.db.populate.assert_transactions_table_empty's guard, scoped to this id range only
-- the frozen benchmark's own rows are untouched either way). Re-running
mrs.policy.engine.apply_policy afterward is separately idempotent (unchanged).

Run with: .venv/bin/python scripts/14_ingest_recent_stream.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sqlalchemy import func, select  # noqa: E402

from mrs import config  # noqa: E402
from mrs.behavioral.customer import compute_customer_behavioral_states  # noqa: E402
from mrs.behavioral.terminal import compute_terminal_behavioral_states  # noqa: E402
from mrs.data.recent_stream import generate_recent_stream  # noqa: E402
from mrs.db import populate  # noqa: E402
from mrs.db.engine import get_database_url, get_engine  # noqa: E402
from mrs.db.models import Transaction  # noqa: E402
from mrs.features.build import build_feature_frame  # noqa: E402
from mrs.models.dataset import attach_labels, get_feature_matrix  # noqa: E402
from mrs.models.persistence import load_model  # noqa: E402
from mrs.policy.engine import apply_policy  # noqa: E402
from mrs.risk.aggregate import aggregate_risk  # noqa: E402

XGBOOST_VERSION_DIR = config.MODELS_DIR / "xgboost_v1"


def assert_recent_stream_not_loaded(engine) -> None:
    with engine.connect() as conn:
        count = conn.execute(
            select(func.count())
            .select_from(Transaction.__table__)
            .where(Transaction.transaction_id >= config.RECENT_STREAM_TX_ID_OFFSET)
        ).scalar_one()
    if count > 0:
        raise RuntimeError(
            f"scripts/14_ingest_recent_stream.py: {count:,} recent-stream row(s) "
            f"(transaction_id >= {config.RECENT_STREAM_TX_ID_OFFSET:,}) already exist -- "
            "refusing to insert again. DELETE FROM transactions WHERE transaction_id >= "
            f"{config.RECENT_STREAM_TX_ID_OFFSET} (cascading through transaction_features/"
            "risk_scores/alerts/audit_logs referencing those ids) first if you intend to "
            "regenerate it."
        )


def main() -> None:
    engine = get_engine()
    print(f"Database: {get_database_url()}")

    print()
    print("=" * 70)
    print("STEP 0: Refuse to run on top of an already-loaded recent stream")
    print("=" * 70)
    assert_recent_stream_not_loaded(engine)
    print("  no existing recent-stream rows -- proceeding.")

    print()
    print("=" * 70)
    print("STEP 1: Generate the deterministic Simulated Recent Operational Stream")
    print("=" * 70)
    t0 = time.time()
    recent = generate_recent_stream()
    print(
        f"Generated {len(recent):,} rows over {recent['TX_TIME_DAYS'].nunique()} days "
        f"({recent['TX_DATETIME'].min()} .. {recent['TX_DATETIME'].max()}) in {time.time() - t0:.2f}s"
    )
    print("  (already schema-validated inside generate_recent_stream via validate_processed_frame)")

    print()
    print("=" * 70)
    print("STEP 2: Build features -- same mrs.features.build_feature_frame, split_override='recent'")
    print("=" * 70)
    t0 = time.time()
    features = build_feature_frame(recent, split_override=config.RECENT_STREAM_SPLIT_LABEL)
    full_df = attach_labels(features, recent)
    full_df = full_df.merge(
        recent[["TRANSACTION_ID", "TX_TIME_SECONDS", "TX_TIME_DAYS"]],
        on="TRANSACTION_ID",
        how="left",
        validate="one_to_one",
    )
    assert full_df["TX_TIME_SECONDS"].notna().all()
    print(f"Built {len(full_df):,} feature rows in {time.time() - t0:.2f}s")

    print()
    print("=" * 70)
    print("STEP 3: Score transaction ML risk with the FROZEN Phase 5 model (no retraining)")
    print("=" * 70)
    pipeline, metadata = load_model(XGBOOST_VERSION_DIR)
    threshold = metadata["threshold"]
    print(f"Model: {XGBOOST_VERSION_DIR.name}  threshold: {threshold}")
    X = get_feature_matrix(full_df)
    t0 = time.time()
    transaction_risk = pipeline.predict_proba(X)[:, 1]
    print(f"Scored {len(X):,} rows in {time.time() - t0:.2f}s")
    import pandas as pd  # local import: only needed for this small frame assembly

    transaction_df = pd.DataFrame(
        {"TRANSACTION_ID": full_df["TRANSACTION_ID"].to_numpy(), "transaction_risk": transaction_risk}
    )

    print()
    print("=" * 70)
    print("STEP 4: Run the unmodified Phase 6/7 behavioral engines")
    print("=" * 70)
    terminal_full = compute_terminal_behavioral_states(full_df)
    customer_full = compute_customer_behavioral_states(full_df)
    print("Terminal states:", terminal_full["terminal_risk_state"].value_counts().to_dict())
    print("Customer states:", customer_full["customer_risk_state"].value_counts().to_dict())
    terminal_df = terminal_full[["TRANSACTION_ID", "terminal_risk_state"]]
    customer_df = customer_full[["TRANSACTION_ID", "customer_risk_state"]]

    print()
    print("=" * 70)
    print("STEP 5: Run aggregate_risk() -- unmodified, no tuning")
    print("=" * 70)
    result = aggregate_risk(transaction_df, terminal_df, customer_df, threshold)
    print("Unified risk levels:", result["unified_risk_level"].value_counts().to_dict())
    risk_full = result.merge(
        full_df[["TRANSACTION_ID", "CUSTOMER_ID", "TERMINAL_ID"]], on="TRANSACTION_ID", how="left", validate="one_to_one"
    )
    assert risk_full["CUSTOMER_ID"].notna().all() and risk_full["TERMINAL_ID"].notna().all()

    print()
    print("=" * 70)
    print("STEP 6: Persist (customers/terminals already exist -- not re-populated)")
    print("=" * 70)
    t0 = time.time()
    n = populate.populate_transactions(engine, full_df)
    print(f"transactions: {n:,} rows in {time.time() - t0:.1f}s")
    t0 = time.time()
    n = populate.populate_transaction_features(engine, full_df, X)
    print(f"transaction_features: {n:,} rows in {time.time() - t0:.1f}s")
    t0 = time.time()
    n = populate.populate_risk_scores(engine, risk_full, transaction_risk_threshold=threshold)
    print(f"risk_scores: {n:,} rows in {time.time() - t0:.1f}s")

    print()
    print("=" * 70)
    print("STEP 7: Apply the deterministic policy engine (idempotent, unmodified)")
    print("=" * 70)
    summary = apply_policy(engine)
    print(f"newly decided this run: {summary['n_newly_decided']:,}")
    print(f"alerts written this run: {summary['n_alerts_written']:,}")
    print(f"audit_log rows written this run: {summary['n_audit_written']:,}")

    print()
    print("=" * 70)
    print("DONE -- Simulated Recent Operational Stream ingested.")
    print("=" * 70)


if __name__ == "__main__":
    main()
