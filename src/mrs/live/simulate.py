"""Simulated Live Transaction Ingestion.

Progressive, chronological playback of the SAME deterministic Simulated Recent
Operational Stream scripts/14_ingest_recent_stream.py already batch-ingests (see
mrs.data.recent_stream, docs/RECENT_STREAM.md) -- not a second dataset, and not a
second risk engine. Every transaction released here goes through the exact same
reused pipeline batch ingestion uses:

    mrs.features.build.build_feature_frame          (feature engineering)
    the frozen Phase 5 xgboost_v1 model, inference only
    mrs.behavioral.customer / mrs.behavioral.terminal
    mrs.risk.aggregate.aggregate_risk
    mrs.db.populate                                  (persistence)
    mrs.policy.engine.decide_and_persist             (deterministic policy)

This is simulated demo playback, not real payment traffic and not a live production
feed -- see mrs.data.recent_stream's own module docstring for the same caveat that
applies to every row this module ever releases.

Temporal correctness: a released transaction's features/behavioral state depend only
on the recent-stream rows released strictly before it (`released_so_far`, passed in by
the caller) -- never on anything not yet released, and never on TX_FRAUD/
TX_FRAUD_SCENARIO as a predictive input (mrs.models.dataset.get_feature_matrix
structurally cannot include them, exactly as it already can't for batch ingestion).
Recomputing mrs.features.build_feature_frame / the behavioral engines over the growing
`released_so_far + new_batch` slice each call reproduces BIT-IDENTICAL values to what
a full-batch run over the whole stream would have produced for those same rows --
tests/test_recent_stream.py's temporal-leakage tests already establish why (every
rolling/expanding aggregate in mrs.features._temporal depends only on strictly-earlier
rows of the SAME entity, so appending later rows can never change an earlier row's
own feature values). Only `new_batch`'s own rows are ever written to the database;
`released_so_far`'s risk was already persisted on an earlier call/tick.
"""

from __future__ import annotations

import pandas as pd
from sqlalchemy import select
from sqlalchemy.engine import Engine

from mrs import config
from mrs.behavioral.customer import compute_customer_behavioral_states
from mrs.behavioral.terminal import compute_terminal_behavioral_states
from mrs.data.recent_stream import generate_recent_stream
from mrs.db import populate
from mrs.db.models import RiskScore, Transaction
from mrs.features.build import build_feature_frame
from mrs.models.dataset import attach_labels, get_feature_matrix
from mrs.models.persistence import load_model
from mrs.policy.engine import decide_and_persist
from mrs.risk.aggregate import aggregate_risk

XGBOOST_VERSION_DIR = config.MODELS_DIR / "xgboost_v1"

#: Same shape mrs.policy.rules.evaluate needs -- see mrs.policy.engine's own
#: (private) _RISK_SCORE_COLUMNS, which this intentionally does not import (avoids
#: reaching into another module's private members); this is a small, declarative
#: column list, not decision logic, so a second copy carries no real duplication risk.
_RISK_SCORE_READBACK_COLUMNS = (
    RiskScore.transaction_id,
    RiskScore.customer_id,
    RiskScore.terminal_id,
    RiskScore.transaction_risk,
    RiskScore.transaction_risk_severity,
    RiskScore.terminal_risk_state,
    RiskScore.terminal_risk_severity,
    RiskScore.customer_risk_state,
    RiskScore.customer_risk_severity,
    RiskScore.unified_risk_level,
    RiskScore.contributing_signals,
    RiskScore.model_version,
    RiskScore.transaction_risk_threshold,
    RiskScore.feature_version,
)


def load_model_and_threshold():
    """The frozen Phase 5 model, used for inference only -- never retrained or
    re-thresholded here, exactly like scripts/14_ingest_recent_stream.py."""
    pipeline, metadata = load_model(XGBOOST_VERSION_DIR)
    return pipeline, metadata["threshold"]


def already_ingested_transaction_ids(engine: Engine) -> set[int]:
    """Recent-stream transaction_ids (>= RECENT_STREAM_TX_ID_OFFSET) already
    persisted -- the idempotency check. A live run resumes from wherever a previous
    run (live or the batch scripts/14) left off; it never re-inserts or re-decides a
    transaction twice."""
    stmt = select(Transaction.transaction_id).where(
        Transaction.transaction_id >= config.RECENT_STREAM_TX_ID_OFFSET
    )
    with engine.connect() as conn:
        return {row[0] for row in conn.execute(stmt)}


def load_stream_and_pending(engine: Engine) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(released_so_far, pending): two chronologically-ordered slices of the SAME
    deterministic generate_recent_stream() frame, split by what is already persisted
    in the database (by either a prior live run or the batch script). `released_so_far`
    seeds correct temporal history for scoring `pending`'s rows -- see module docstring.
    """
    recent = generate_recent_stream()
    ingested_ids = already_ingested_transaction_ids(engine)
    is_ingested = recent["TRANSACTION_ID"].isin(ingested_ids)
    released_so_far = recent[is_ingested].reset_index(drop=True)
    pending = recent[~is_ingested].reset_index(drop=True)
    return released_so_far, pending


def _read_back_risk_scores(engine: Engine, transaction_ids: list[int]) -> list[dict]:
    stmt = select(*_RISK_SCORE_READBACK_COLUMNS).where(RiskScore.transaction_id.in_(transaction_ids))
    with engine.connect() as conn:
        return [dict(r) for r in conn.execute(stmt).mappings().all()]


def ingest_batch(
    engine: Engine,
    pipeline,
    threshold: float,
    released_so_far: pd.DataFrame,
    new_batch: pd.DataFrame,
) -> dict:
    """Score and persist exactly `new_batch` (already known not to exist in the
    database), reusing `released_so_far` only to seed each entity's correct temporal
    history. Returns {"transaction_ids", "unified_risk_levels", "n_alerts_written",
    "action_counts"} -- real, computed values, nothing invented.
    """
    cumulative = pd.concat([released_so_far, new_batch], ignore_index=True)

    features = build_feature_frame(cumulative, split_override=config.RECENT_STREAM_SPLIT_LABEL)
    full_df = attach_labels(features, cumulative)
    full_df = full_df.merge(
        cumulative[["TRANSACTION_ID", "TX_TIME_SECONDS", "TX_TIME_DAYS"]],
        on="TRANSACTION_ID",
        how="left",
        validate="one_to_one",
    )

    new_ids = set(new_batch["TRANSACTION_ID"].tolist())
    new_mask = full_df["TRANSACTION_ID"].isin(new_ids)

    X_full = get_feature_matrix(full_df)
    transaction_risk_full = pipeline.predict_proba(X_full)[:, 1]

    terminal_full = compute_terminal_behavioral_states(full_df)
    customer_full = compute_customer_behavioral_states(full_df)

    transaction_df = pd.DataFrame(
        {"TRANSACTION_ID": full_df["TRANSACTION_ID"].to_numpy(), "transaction_risk": transaction_risk_full}
    )
    result = aggregate_risk(
        transaction_df,
        terminal_full[["TRANSACTION_ID", "terminal_risk_state"]],
        customer_full[["TRANSACTION_ID", "customer_risk_state"]],
        threshold,
    )

    new_full_df = full_df[new_mask].reset_index(drop=True)
    new_X = X_full[new_mask].reset_index(drop=True)
    new_result = result[result["TRANSACTION_ID"].isin(new_ids)].merge(
        new_full_df[["TRANSACTION_ID", "CUSTOMER_ID", "TERMINAL_ID"]],
        on="TRANSACTION_ID",
        how="left",
        validate="one_to_one",
    )

    populate.populate_transactions(engine, new_full_df)
    populate.populate_transaction_features(engine, new_full_df, new_X)
    populate.populate_risk_scores(engine, new_result, transaction_risk_threshold=threshold)

    risk_rows = _read_back_risk_scores(engine, sorted(new_ids))
    policy_summary = decide_and_persist(engine, risk_rows)

    return {
        "transaction_ids": sorted(new_ids),
        "unified_risk_levels": dict(zip(new_result["TRANSACTION_ID"].tolist(), new_result["unified_risk_level"].tolist())),
        "n_alerts_written": policy_summary["n_alerts_written"],
        "action_counts": policy_summary["action_counts"],
    }
