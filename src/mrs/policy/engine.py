"""Phase 8 Step 3: deterministic policy engine orchestration (Dev Plan §15/§20).

Reads already-computed risk_scores (never recomputes transaction ML / behavioral /
aggregation signals), evaluates mrs.policy.rules.evaluate for every row, and writes:

- audit_logs: one event_type="POLICY_DECISION" row per newly-evaluated transaction
  (Dev Plan §33.9) -- the append-only governance trail.
- alerts: one row per transaction whose decided action != ALLOW (Dev Plan §20).

Idempotent / rerun-safe (Step 3 requirement): a transaction that already has a
POLICY_DECISION audit_log entry is skipped on a later run (no duplicate audit rows),
and alerts are inserted via INSERT ... ON CONFLICT (transaction_id) DO NOTHING (backed
by the UNIQUE constraint mrs.db.models.Alert already defines), so a second run can
never duplicate an alert even if called concurrently with a partially-completed run.

No FastAPI, no LLM, no Replay -- this module only turns already-computed evidence into
policy decisions and their persisted trail.
"""

from __future__ import annotations

import time
from collections.abc import Iterator

from sqlalchemy import insert, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.engine import Engine

from mrs.db.models import Alert, AuditLog, RiskScore
from mrs.policy.rules import POLICY_VERSION, evaluate

DEFAULT_CHUNK_SIZE = 50_000

_RISK_SCORE_COLUMNS = (
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


def already_decided_transaction_ids(engine: Engine) -> set[int]:
    """Transaction IDs that already have a POLICY_DECISION audit_log entry -- the
    idempotency check a rerun uses to skip re-writing alerts/audit_logs. Deliberately
    keyed on event_type only (not policy_version): re-evaluating under a new
    POLICY_VERSION is an explicit, separate concern this function does not decide.
    """
    stmt = select(AuditLog.transaction_id).where(AuditLog.event_type == "POLICY_DECISION")
    with engine.connect() as conn:
        return {row[0] for row in conn.execute(stmt)}


def _iter_risk_score_chunks(engine: Engine, chunk_size: int) -> Iterator[list[dict]]:
    """Keyset pagination on the indexed transaction_id PK -- bounds memory to one
    chunk at a time regardless of table size (Dev Plan §40)."""
    last_id = -1
    while True:
        stmt = (
            select(*_RISK_SCORE_COLUMNS)
            .where(RiskScore.transaction_id > last_id)
            .order_by(RiskScore.transaction_id)
            .limit(chunk_size)
        )
        with engine.connect() as conn:
            rows = [dict(r) for r in conn.execute(stmt).mappings().all()]
        if not rows:
            return
        yield rows
        last_id = rows[-1]["transaction_id"]


def apply_policy(engine: Engine, *, chunk_size: int = DEFAULT_CHUNK_SIZE) -> dict:
    """Evaluate policy for every risk_scores row and persist alerts/audit_logs for any
    not already decided. Safe to call repeatedly (idempotent).

    Returns a summary dict: n_rows_read, n_newly_decided, n_skipped_already_decided,
    n_alerts_written, n_audit_written, action_counts (over ALL rows read, not just
    newly-decided -- a full-population view for reporting), level_counts (ditto),
    elapsed_seconds. Definitive persisted totals should be read back from the database
    directly (see scripts/13_run_policy_engine.py), the same pattern
    scripts/12_populate_db.py used for its own real-data validation.
    """
    t0 = time.time()
    already_decided = already_decided_transaction_ids(engine)

    action_counts: dict[str, int] = {}
    level_counts: dict[str, int] = {}
    n_rows_read = 0
    n_newly_decided = 0
    n_skipped = 0
    n_alerts_written = 0
    n_audit_written = 0

    for chunk in _iter_risk_score_chunks(engine, chunk_size):
        audit_batch = []
        alert_batch = []

        for row in chunk:
            n_rows_read += 1
            level_counts[row["unified_risk_level"]] = level_counts.get(row["unified_risk_level"], 0) + 1

            decision = evaluate(row)
            action_counts[decision.action] = action_counts.get(decision.action, 0) + 1

            tid = row["transaction_id"]
            if tid in already_decided:
                n_skipped += 1
                continue
            n_newly_decided += 1

            audit_batch.append(
                {
                    "transaction_id": decision.transaction_id,
                    "alert_id": None,
                    "event_type": "POLICY_DECISION",
                    "payload": {
                        "policy_version": decision.policy_version,
                        "action": decision.action,
                        "unified_risk_level": decision.unified_risk_level,
                        "reason": decision.reason,
                        "evidence": decision.evidence,
                    },
                    "model_version": row["model_version"],
                }
            )
            if decision.is_alert:
                alert_batch.append(
                    {
                        "transaction_id": decision.transaction_id,
                        "customer_id": row["customer_id"],
                        "terminal_id": row["terminal_id"],
                        "severity": decision.unified_risk_level,
                        "reason": decision.reason,
                        "evidence": decision.evidence,
                        "recommended_action": decision.action,
                        "status": "OPEN",
                    }
                )

        if audit_batch:
            with engine.begin() as conn:
                conn.execute(insert(AuditLog.__table__), audit_batch)
            n_audit_written += len(audit_batch)

        if alert_batch:
            with engine.begin() as conn:
                stmt = pg_insert(Alert.__table__).values(alert_batch).on_conflict_do_nothing(
                    index_elements=["transaction_id"]
                )
                conn.execute(stmt)
            n_alerts_written += len(alert_batch)

    return {
        "policy_version": POLICY_VERSION,
        "n_rows_read": n_rows_read,
        "n_newly_decided": n_newly_decided,
        "n_skipped_already_decided": n_skipped,
        "n_alerts_written": n_alerts_written,
        "n_audit_written": n_audit_written,
        "action_counts": action_counts,
        "level_counts": level_counts,
        "elapsed_seconds": time.time() - t0,
    }
