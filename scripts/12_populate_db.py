#!/usr/bin/env python
"""Phase 8 Step 2: populate PostgreSQL with the materialized Phase 5+6+7 pipeline output.

Reuses the identical real-data assembly scripts/10_run_risk_aggregation.py already
validated (mrs.models.dataset, mrs.models.persistence, mrs.behavioral.terminal/
customer, mrs.risk.aggregate) -- computes nothing new and duplicates no Phase 3/5/6/7
logic. This script's only added work is the persistence step (mrs.db.populate): bulk
inserts into the Phase 8 Step 1 schema, chunked to bound memory.

Populates, in FK-safe order: customers, terminals, transactions, transaction_features,
risk_scores. alerts/audit_logs stay empty (the policy engine, a later Phase 8 step,
will write those). Refuses to run if `transactions` already has rows (see
mrs.db.populate.assert_transactions_table_empty) -- re-run scripts/11_init_db_schema.py
after a manual TRUNCATE/drop_all+create_all to reload intentionally.

Ends with a real-data validation pass: queries the freshly-populated database and
cross-checks row counts and the unified_risk_level distribution against the numbers
already validated in docs/risk_aggregation_report_data.json (Phase 7's real-data run),
to confirm what landed in Postgres matches the already-validated in-memory result bit
for bit, not a second independent computation.

Run with: .venv/bin/python scripts/12_populate_db.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402
from sqlalchemy import func, select  # noqa: E402

from mrs import config  # noqa: E402
from mrs.behavioral.customer import compute_customer_behavioral_states  # noqa: E402
from mrs.behavioral.terminal import compute_terminal_behavioral_states  # noqa: E402
from mrs.db import populate  # noqa: E402
from mrs.db.engine import get_database_url, get_engine  # noqa: E402
from mrs.db.models import Alert, AuditLog, Customer, RiskScore, Terminal, Transaction, TransactionFeatures  # noqa: E402
from mrs.models.dataset import get_feature_matrix, load_processed_transactions, load_split  # noqa: E402
from mrs.models.persistence import load_model  # noqa: E402
from mrs.risk.aggregate import aggregate_risk  # noqa: E402

XGBOOST_VERSION_DIR = config.MODELS_DIR / "xgboost_v1"
KNOWN_GOOD_PATH = config.MODELS_DIR.parent / "docs" / "risk_aggregation_report_data.json"


def main() -> None:
    engine = get_engine()
    print(f"Database: {get_database_url()}")

    print()
    print("=" * 70)
    print("STEP 0: Refuse to run on top of existing data")
    print("=" * 70)
    populate.assert_transactions_table_empty(engine)
    print("  transactions table is empty -- proceeding.")

    print()
    print("=" * 70)
    print("STEP 1: Load the frozen dataset via the established pipeline")
    print("=" * 70)
    labels_source = load_processed_transactions()
    train_df = load_split("train", labels_source)
    validation_df = load_split("validation", labels_source)
    test_df = load_split("test", labels_source)
    full_df = pd.concat([train_df, validation_df, test_df], ignore_index=True)
    assert len(full_df) == 1_754_155, "frozen dataset row count mismatch"
    assert not full_df["TRANSACTION_ID"].duplicated().any(), "duplicate TRANSACTION_ID in loaded dataset"

    # The features layer does not carry TX_TIME_SECONDS/TX_TIME_DAYS (only the
    # processed layer does) -- merge them in once, here, for the transactions table.
    full_df = full_df.merge(
        labels_source[["TRANSACTION_ID", "TX_TIME_SECONDS", "TX_TIME_DAYS"]],
        on="TRANSACTION_ID",
        how="left",
        validate="one_to_one",
    )
    assert full_df["TX_TIME_SECONDS"].notna().all()
    print(f"Total rows: {len(full_df):,}")

    print()
    print("=" * 70)
    print("STEP 2: Load the frozen Phase 5 XGBoost model + validated threshold")
    print("=" * 70)
    pipeline, metadata = load_model(XGBOOST_VERSION_DIR)
    threshold = metadata["threshold"]
    print(f"Threshold: {threshold} (source: {XGBOOST_VERSION_DIR / 'metadata.json'})")

    print()
    print("=" * 70)
    print("STEP 3: Score transaction ML risk (inference only, no retraining)")
    print("=" * 70)
    X = get_feature_matrix(full_df)
    t0 = time.time()
    transaction_risk = pipeline.predict_proba(X)[:, 1]
    print(f"Scored {len(X):,} rows in {time.time() - t0:.2f}s")
    transaction_df = pd.DataFrame(
        {"TRANSACTION_ID": full_df["TRANSACTION_ID"].to_numpy(), "transaction_risk": transaction_risk}
    )

    print()
    print("=" * 70)
    print("STEP 4: Run the unmodified Phase 6/7 behavioral engines")
    print("=" * 70)
    t0 = time.time()
    terminal_full = compute_terminal_behavioral_states(full_df)
    print(f"Terminal engine: {len(terminal_full):,} rows in {time.time() - t0:.2f}s")
    terminal_df = terminal_full[["TRANSACTION_ID", "terminal_risk_state"]]

    t0 = time.time()
    customer_full = compute_customer_behavioral_states(full_df)
    print(f"Customer engine: {len(customer_full):,} rows in {time.time() - t0:.2f}s")
    customer_df = customer_full[["TRANSACTION_ID", "customer_risk_state"]]

    print()
    print("=" * 70)
    print("STEP 5: Run aggregate_risk() -- unmodified, no tuning")
    print("=" * 70)
    t0 = time.time()
    result = aggregate_risk(transaction_df, terminal_df, customer_df, threshold)
    print(f"Aggregated {len(result):,} rows in {time.time() - t0:.2f}s")
    # aggregate_risk's own output only carries TRANSACTION_ID -- attach CUSTOMER_ID/
    # TERMINAL_ID (risk_scores denormalizes them, mrs/db/models.py) via the one frame
    # that already has them.
    risk_full = result.merge(
        full_df[["TRANSACTION_ID", "CUSTOMER_ID", "TERMINAL_ID"]], on="TRANSACTION_ID", how="left", validate="one_to_one"
    )
    assert risk_full["CUSTOMER_ID"].notna().all() and risk_full["TERMINAL_ID"].notna().all()

    print()
    print("=" * 70)
    print("STEP 6: Populate customers / terminals")
    print("=" * 70)
    customer_profiles = pd.read_parquet(config.REFERENCE_DIR / "customer_profiles.parquet")
    terminal_profiles = pd.read_parquet(config.REFERENCE_DIR / "terminal_profiles.parquet")
    t0 = time.time()
    populate.populate_customers_and_terminals(engine, customer_profiles, terminal_profiles)
    print(f"customers={len(customer_profiles):,} terminals={len(terminal_profiles):,} in {time.time() - t0:.2f}s")

    print()
    print("=" * 70)
    print("STEP 7: Populate transactions")
    print("=" * 70)
    t0 = time.time()
    n = populate.populate_transactions(engine, full_df)
    print(f"transactions: {n:,} rows in {time.time() - t0:.1f}s")

    print()
    print("=" * 70)
    print("STEP 8: Populate transaction_features")
    print("=" * 70)
    t0 = time.time()
    n = populate.populate_transaction_features(engine, full_df, X)
    print(f"transaction_features: {n:,} rows in {time.time() - t0:.1f}s")

    print()
    print("=" * 70)
    print("STEP 9: Populate risk_scores")
    print("=" * 70)
    t0 = time.time()
    n = populate.populate_risk_scores(engine, risk_full, transaction_risk_threshold=threshold)
    print(f"risk_scores: {n:,} rows in {time.time() - t0:.1f}s")

    print()
    print("=" * 70)
    print("STEP 10: Real-data validation -- cross-check Postgres against the already-")
    print("         validated Phase 7 in-memory result (docs/risk_aggregation_report_data.json)")
    print("=" * 70)
    _validate_against_known_good(engine)


def _validate_against_known_good(engine) -> None:
    with open(KNOWN_GOOD_PATH) as f:
        known_good = json.load(f)

    with engine.connect() as conn:
        counts = {
            "customers": conn.execute(select(func.count()).select_from(Customer.__table__)).scalar_one(),
            "terminals": conn.execute(select(func.count()).select_from(Terminal.__table__)).scalar_one(),
            "transactions": conn.execute(select(func.count()).select_from(Transaction.__table__)).scalar_one(),
            "transaction_features": conn.execute(
                select(func.count()).select_from(TransactionFeatures.__table__)
            ).scalar_one(),
            "risk_scores": conn.execute(select(func.count()).select_from(RiskScore.__table__)).scalar_one(),
            "alerts": conn.execute(select(func.count()).select_from(Alert.__table__)).scalar_one(),
            "audit_logs": conn.execute(select(func.count()).select_from(AuditLog.__table__)).scalar_one(),
        }
        level_rows = conn.execute(
            select(RiskScore.unified_risk_level, func.count()).group_by(RiskScore.unified_risk_level)
        ).all()

    level_distribution = {level: count for level, count in level_rows}

    print("Row counts:")
    for table, count in counts.items():
        print(f"  {table}: {count:,}")

    print("unified_risk_level distribution (Postgres vs. known-good):")
    all_pass = True
    for level in ("LOW", "MEDIUM", "HIGH", "CRITICAL"):
        actual = level_distribution.get(level, 0)
        expected = known_good["level_distribution"][level]
        ok = actual == expected
        all_pass &= ok
        print(f"  {level:<10} db={actual:>10,}  expected={expected:>10,}  {'PASS' if ok else 'FAIL'}")

    expected_total = known_good["output_rows"]
    total_ok = counts["risk_scores"] == expected_total == counts["transactions"] == counts["transaction_features"]
    all_pass &= total_ok
    print(f"  total risk_scores == transactions == transaction_features == {expected_total:,}: "
          f"{'PASS' if total_ok else 'FAIL'}")

    print()
    print("VALIDATION: " + ("PASS -- Postgres matches the already-validated Phase 7 result." if all_pass else "FAIL"))
    if not all_pass:
        raise AssertionError("scripts/12_populate_db.py: post-load validation against known-good numbers failed")


if __name__ == "__main__":
    main()
