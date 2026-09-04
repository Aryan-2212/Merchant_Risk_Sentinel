#!/usr/bin/env python
"""Generate, score, and ingest the simulated three-week operational stream.

This is deliberately separate from scripts/12_populate_db.py. The frozen Handbook
benchmark remains the official training/evaluation source; this script only appends a
reproducible 21-day simulated recent window to the existing operational database.

Pipeline:
    generator -> schema validation -> Phase-3 features -> frozen XGBoost inference
    -> customer/terminal behavioral states -> risk aggregation -> deterministic policy

Synthetic TX_FRAUD/TX_FRAUD_SCENARIO values are scenario annotations used to exercise the
existing terminal behavioral fraud-rate features. They are never model inputs and are
never used in official benchmark metrics.

Run after the Phase 8 database has been initialized/populated/policy-applied:
    .venv/bin/python scripts/14_ingest_recent_stream.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402
from sqlalchemy import func, select  # noqa: E402

from mrs import config  # noqa: E402
from mrs.behavioral.customer import compute_customer_behavioral_states  # noqa: E402
from mrs.behavioral.terminal import compute_terminal_behavioral_states  # noqa: E402
from mrs.data.recent_stream import RecentStreamConfig, build_recent_feature_frame, generate_recent_transactions  # noqa: E402
from mrs.data.schema import validate_processed_frame  # noqa: E402
from mrs.db import populate  # noqa: E402
from mrs.db.engine import get_database_url, get_engine  # noqa: E402
from mrs.db.models import RiskScore, Transaction, TransactionFeatures  # noqa: E402
from mrs.models.dataset import get_feature_matrix  # noqa: E402
from mrs.models.persistence import load_model  # noqa: E402
from mrs.policy.engine import apply_policy  # noqa: E402
from mrs.risk.aggregate import aggregate_risk  # noqa: E402

XGBOOST_VERSION_DIR = config.MODELS_DIR / "xgboost_v1"
RECENT_OUTPUT_DIR = config.DATA_DIR / "recent"
RECENT_TRANSACTIONS_PATH = RECENT_OUTPUT_DIR / "transactions_21d.parquet"
RECENT_FEATURES_PATH = RECENT_OUTPUT_DIR / "features_21d.parquet"


def _assert_recent_window_absent(engine, transaction_id_start: int) -> None:
    with engine.connect() as conn:
        count = conn.execute(
            select(func.count())
            .select_from(Transaction.__table__)
            .where(Transaction.transaction_id >= transaction_id_start)
        ).scalar_one()
    if count:
        raise RuntimeError(
            f"recent stream already contains {count:,} transaction(s) with id >= "
            f"{transaction_id_start:,}; refusing to duplicate the stream."
        )


def main() -> None:
    cfg = RecentStreamConfig()
    engine = get_engine()
    print(f"Database: {get_database_url()}")
    _assert_recent_window_absent(engine, cfg.transaction_id_start)

    customer_profiles = pd.read_parquet(config.REFERENCE_DIR / "customer_profiles.parquet")
    terminal_profiles = pd.read_parquet(config.REFERENCE_DIR / "terminal_profiles.parquet")

    print("=" * 70)
    print("STEP 1: Generate deterministic 21-day recent stream")
    print("=" * 70)
    t0 = time.time()
    recent = generate_recent_transactions(customer_profiles, terminal_profiles, config=cfg)
    validate_processed_frame(recent, source="recent simulated stream")
    # The generic processed validator expects TX_TIME_DAYS to be globally ordered only
    # through TX_DATETIME; it is intentionally not used as a split/evaluation label.
    recent = recent.copy()
    RECENT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    recent.to_parquet(RECENT_TRANSACTIONS_PATH, index=False)
    print(f"Generated {len(recent):,} rows: {recent.TX_DATETIME.min()} -> {recent.TX_DATETIME.max()}")
    print(f"Saved {RECENT_TRANSACTIONS_PATH} in {time.time() - t0:.2f}s")

    print("=" * 70)
    print("STEP 2: Build recent causal features")
    print("=" * 70)
    t0 = time.time()
    recent_features = build_recent_feature_frame(recent)
    recent_features.to_parquet(RECENT_FEATURES_PATH, index=False)
    X = get_feature_matrix(recent_features)
    print(f"Built {len(recent_features):,} feature rows in {time.time() - t0:.2f}s")

    print("=" * 70)
    print("STEP 3: Apply the frozen XGBoost model (inference only)")
    print("=" * 70)
    pipeline, metadata = load_model(XGBOOST_VERSION_DIR)
    threshold = float(metadata["threshold"])
    transaction_risk = pipeline.predict_proba(X)[:, 1]
    transaction_df = pd.DataFrame(
        {"TRANSACTION_ID": recent["TRANSACTION_ID"].to_numpy(), "transaction_risk": transaction_risk}
    )
    print(f"Scored {len(transaction_df):,} recent transactions; threshold={threshold}")

    print("=" * 70)
    print("STEP 4: Compute causal customer + terminal behavioral states")
    print("=" * 70)
    terminal_full = compute_terminal_behavioral_states(recent)
    customer_full = compute_customer_behavioral_states(recent)
    terminal_df = terminal_full[["TRANSACTION_ID", "terminal_risk_state"]]
    customer_df = customer_full[["TRANSACTION_ID", "customer_risk_state"]]

    print("=" * 70)
    print("STEP 5: Aggregate unified risk")
    print("=" * 70)
    result = aggregate_risk(transaction_df, terminal_df, customer_df, threshold)
    risk_full = result.merge(
        recent[["TRANSACTION_ID", "CUSTOMER_ID", "TERMINAL_ID"]],
        on="TRANSACTION_ID",
        how="left",
        validate="one_to_one",
    )

    print("=" * 70)
    print("STEP 6: Persist recent transactions, features, and risk")
    print("=" * 70)
    n = populate.populate_transactions(engine, recent.assign(split="recent"))
    print(f"transactions: {n:,}")
    n = populate.populate_transaction_features(engine, recent, X)
    print(f"transaction_features: {n:,}")
    n = populate.populate_risk_scores(engine, risk_full, transaction_risk_threshold=threshold)
    print(f"risk_scores: {n:,}")

    print("=" * 70)
    print("STEP 7: Apply deterministic policy only to the new recent window")
    print("=" * 70)
    policy_summary = apply_policy(engine, min_transaction_id=cfg.transaction_id_start)
    print(policy_summary)

    with engine.connect() as conn:
        recent_risk_count = conn.execute(
            select(func.count()).select_from(RiskScore.__table__).where(
                RiskScore.transaction_id >= cfg.transaction_id_start
            )
        ).scalar_one()
    if recent_risk_count != len(recent):
        raise AssertionError(
            f"recent risk row count mismatch: db={recent_risk_count} generated={len(recent)}"
        )

    print("=" * 70)
    print("RECENT STREAM INGESTION: PASS")
    print(f"Window: {cfg.start.date()} -> {cfg.end.date()}")
    print(f"Rows: {len(recent):,}")
    print(f"Recent artifacts: {RECENT_OUTPUT_DIR}")
    print("Official benchmark metrics remain unchanged.")
    print("=" * 70)


if __name__ == "__main__":
    main()
