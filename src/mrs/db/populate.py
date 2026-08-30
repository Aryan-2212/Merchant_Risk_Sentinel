"""Phase 8 Step 2: materialize the already-computed Phase 3/5/6/7 pipeline output into
the Phase 8 Step 1 schema (Dev Plan §20; approved Replay decision -- precomputed
history, read back chronologically, never recomputed live).

This module performs no feature engineering, no model inference, and no behavioral/
aggregation computation of its own. Every value it writes was already produced by
mrs.features / mrs.models / mrs.behavioral / mrs.risk (unmodified). Its only job is
converting those already-validated pandas outputs into bulk inserts against
mrs.db.models, chunked to bound memory (Dev Plan §40). The orchestration (loading the
real dataset, running the Phase 5/6/7 pipeline, calling these functions) lives in
scripts/12_populate_db.py, mirroring scripts/10_run_risk_aggregation.py's own
in-memory assembly exactly -- this module only adds the persistence step.

alerts/audit_logs are intentionally not populated here: they are the policy engine's
output, a later Phase 8 step.
"""

from __future__ import annotations

import math

import pandas as pd
from sqlalchemy import func, insert, select
from sqlalchemy.engine import Engine

from mrs.db.models import Customer, RiskScore, Terminal, Transaction, TransactionFeatures

#: Feature-schema lineage tag (Dev Plan §36). Distinct from mrs.__version__ (whole
#: package version) -- identifies specifically the Phase 3 feature contract, which has
#: not changed since Phase 3 was frozen.
FEATURE_VERSION = "phase3_v1"

#: Matches the saved model directory name (models/xgboost_v1/), the existing lineage
#: convention used by mrs.models.persistence.
MODEL_VERSION = "xgboost_v1"

DEFAULT_CHUNK_SIZE = 50_000


# --------------------------------------------------------------------- row builders
# Pure functions: pandas in, list[dict] out. No I/O. Each is independently testable
# against small synthetic frames without a database or the real dataset.


def customer_profile_rows(customer_profiles: pd.DataFrame) -> list[dict]:
    """customer_profiles: data/reference/customer_profiles.parquet, unmodified."""
    return [
        {
            "customer_id": int(r.CUSTOMER_ID),
            "x_customer_id": float(r.x_customer_id),
            "y_customer_id": float(r.y_customer_id),
            "mean_amount": float(r.mean_amount),
            "std_amount": float(r.std_amount),
            "mean_nb_tx_per_day": float(r.mean_nb_tx_per_day),
            "nb_terminals": int(r.nb_terminals),
            "available_terminals": [int(t) for t in r.available_terminals],
        }
        for r in customer_profiles.itertuples(index=False)
    ]


def terminal_profile_rows(terminal_profiles: pd.DataFrame) -> list[dict]:
    """terminal_profiles: data/reference/terminal_profiles.parquet, unmodified."""
    return [
        {
            "terminal_id": int(r.TERMINAL_ID),
            "x_terminal_id": float(r.x_terminal_id),
            "y_terminal_id": float(r.y_terminal_id),
        }
        for r in terminal_profiles.itertuples(index=False)
    ]


def transaction_rows(chunk: pd.DataFrame) -> list[dict]:
    """chunk must have TRANSACTION_ID, TX_DATETIME, CUSTOMER_ID, TERMINAL_ID, tx_amount,
    TX_TIME_SECONDS, TX_TIME_DAYS, TX_FRAUD, TX_FRAUD_SCENARIO, split -- exactly what
    mrs.models.dataset.load_split's output has (feature columns) plus TX_TIME_SECONDS/
    TX_TIME_DAYS merged in from mrs.models.dataset.load_processed_transactions (the
    features layer does not itself carry those two, only the processed layer does).
    """
    return [
        {
            "transaction_id": int(r.TRANSACTION_ID),
            "tx_datetime": r.TX_DATETIME.to_pydatetime()
            if hasattr(r.TX_DATETIME, "to_pydatetime")
            else r.TX_DATETIME,
            "customer_id": int(r.CUSTOMER_ID),
            "terminal_id": int(r.TERMINAL_ID),
            "tx_amount": float(r.tx_amount),
            "tx_time_seconds": int(r.TX_TIME_SECONDS),
            "tx_time_days": int(r.TX_TIME_DAYS),
            "tx_fraud": int(r.TX_FRAUD),
            "tx_fraud_scenario": int(r.TX_FRAUD_SCENARIO),
            "split": r.split,
        }
        for r in chunk.itertuples(index=False)
    ]


def _nan_safe(value):
    """NaN (float) -> None (JSON null). Every other value (including 0/0.0/False)
    passes through unchanged -- only genuine missingness is converted, never a
    legitimate zero (Dev Plan §28: absence of evidence is never treated as calm)."""
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def transaction_features_rows(ids: list[int], feature_chunk: pd.DataFrame) -> list[dict]:
    """ids aligned 1:1 with feature_chunk's rows (same order). feature_chunk's columns
    are exactly mrs.models.dataset.get_feature_matrix's 33 registered feature columns
    -- this function does not decide what a feature is, only serializes what it is
    given. NaN entries (cold-start features, Dev Plan §7.2/§7.3) become JSON null, not
    a fabricated 0.
    """
    records = feature_chunk.to_dict(orient="records")
    return [
        {
            "transaction_id": int(tid),
            "feature_version": FEATURE_VERSION,
            "features": {k: _nan_safe(v) for k, v in rec.items()},
        }
        for tid, rec in zip(ids, records)
    ]


def risk_score_rows(chunk: pd.DataFrame, *, transaction_risk_threshold: float) -> list[dict]:
    """chunk must be mrs.risk.aggregate.aggregate_risk's output merged with each row's
    CUSTOMER_ID/TERMINAL_ID (aggregate_risk's own output only carries TRANSACTION_ID).
    An "unavailable" component (Dev Plan §28) is stored as SQL NULL, not a fabricated
    calm value -- exactly mirroring aggregate_risk's own None/NaN semantics.
    """
    rows = []
    for r in chunk.itertuples(index=False):
        rows.append(
            {
                "transaction_id": int(r.TRANSACTION_ID),
                "customer_id": int(r.CUSTOMER_ID),
                "terminal_id": int(r.TERMINAL_ID),
                "transaction_risk": None if pd.isna(r.transaction_risk) else float(r.transaction_risk),
                "transaction_risk_severity": None
                if pd.isna(r.transaction_risk_severity)
                else int(r.transaction_risk_severity),
                "terminal_risk_state": None if pd.isna(r.terminal_risk_state) else r.terminal_risk_state,
                "terminal_risk_severity": None
                if pd.isna(r.terminal_risk_severity)
                else int(r.terminal_risk_severity),
                "customer_risk_state": None if pd.isna(r.customer_risk_state) else r.customer_risk_state,
                "customer_risk_severity": None
                if pd.isna(r.customer_risk_severity)
                else int(r.customer_risk_severity),
                "unified_risk_level": r.unified_risk_level,
                "contributing_signals": list(r.contributing_signals),
                "model_version": MODEL_VERSION,
                "transaction_risk_threshold": float(transaction_risk_threshold),
                "feature_version": FEATURE_VERSION,
            }
        )
    return rows


# ------------------------------------------------------------------------- loaders


def assert_transactions_table_empty(engine: Engine) -> None:
    """Refuse to populate on top of existing rows -- a second run would otherwise fail
    partway through on a primary-key violation, leaving a partially-loaded table. Run
    mrs.db.base.drop_all + create_all (or TRUNCATE) first to reload intentionally.
    """
    with engine.connect() as conn:
        count = conn.execute(select(func.count()).select_from(Transaction.__table__)).scalar_one()
    if count > 0:
        raise RuntimeError(
            f"mrs.db.populate: transactions already has {count:,} row(s) -- refusing to "
            "populate again. Drop and recreate the schema first if you intend to reload "
            "(mrs.db.base.drop_all/create_all, or scripts/11_init_db_schema.py after a "
            "manual TRUNCATE)."
        )


def populate_customers_and_terminals(
    engine: Engine, customer_profiles: pd.DataFrame, terminal_profiles: pd.DataFrame
) -> None:
    with engine.begin() as conn:
        conn.execute(insert(Customer.__table__), customer_profile_rows(customer_profiles))
        conn.execute(insert(Terminal.__table__), terminal_profile_rows(terminal_profiles))


def populate_transactions(engine: Engine, full_df: pd.DataFrame, *, chunk_size: int = DEFAULT_CHUNK_SIZE) -> int:
    n = len(full_df)
    with engine.begin() as conn:
        for start in range(0, n, chunk_size):
            chunk = full_df.iloc[start : start + chunk_size]
            conn.execute(insert(Transaction.__table__), transaction_rows(chunk))
    return n


def populate_transaction_features(
    engine: Engine,
    full_df: pd.DataFrame,
    feature_matrix: pd.DataFrame,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> int:
    """full_df and feature_matrix must be row-aligned (same index/order) -- exactly the
    relationship between a Phase 3 feature frame and
    mrs.models.dataset.get_feature_matrix's output on it."""
    n = len(full_df)
    ids = full_df["TRANSACTION_ID"].to_numpy()
    with engine.begin() as conn:
        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            rows = transaction_features_rows(ids[start:end].tolist(), feature_matrix.iloc[start:end])
            conn.execute(insert(TransactionFeatures.__table__), rows)
    return n


def populate_risk_scores(
    engine: Engine,
    risk_full: pd.DataFrame,
    *,
    transaction_risk_threshold: float,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> int:
    n = len(risk_full)
    with engine.begin() as conn:
        for start in range(0, n, chunk_size):
            chunk = risk_full.iloc[start : start + chunk_size]
            conn.execute(
                insert(RiskScore.__table__),
                risk_score_rows(chunk, transaction_risk_threshold=transaction_risk_threshold),
            )
    return n
