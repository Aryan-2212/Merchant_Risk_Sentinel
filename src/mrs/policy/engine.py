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
from mrs.policy.rules import POLICY_VERSION, PolicyDecision, evaluate

DEFAULT_CHUNK_SIZE = 50_000

#: Postgres caps a single query at 65,535 bound parameters. audit_logs/alerts rows have
#: enough columns that a naive one-INSERT-per-50k-row-chunk (this module's original
#: design) can exceed that on a chunk with unusually high alert density -- surfaced by
#: the Simulated Recent Operational Stream's much higher alert rate than the frozen
#: benchmark's, not previously exercised. Sub-batching the INSERT itself (independent of
#: DEFAULT_CHUNK_SIZE, which still bounds how much is read into memory at once) fixes
#: this for any future alert-density mix without changing what gets decided or persisted.
_MAX_INSERT_ROWS = 5_000


def _sub_batches(rows: list[dict], size: int = _MAX_INSERT_ROWS) -> Iterator[list[dict]]:
    for start in range(0, len(rows), size):
        yield rows[start : start + size]

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


def _persist_decisions(engine: Engine, decided: list[tuple[dict, PolicyDecision]]) -> dict:
    """Build and insert the audit_logs/alerts rows for an already-evaluated
    (row, PolicyDecision) list -- the write-only half shared by apply_policy (which
    evaluates every row it reads, for full-population reporting, but persists only the
    not-already-decided subset) and decide_and_persist (which evaluates and persists
    the same small, already-known-fresh list). No policy decision is made here -- the
    caller already ran mrs.policy.rules.evaluate; this only turns its result into rows.
    """
    audit_batch = []
    alert_batch = []
    for row, decision in decided:
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
            for sub_batch in _sub_batches(audit_batch):
                conn.execute(insert(AuditLog.__table__), sub_batch)

    if alert_batch:
        with engine.begin() as conn:
            for sub_batch in _sub_batches(alert_batch):
                stmt = pg_insert(Alert.__table__).values(sub_batch).on_conflict_do_nothing(
                    index_elements=["transaction_id"]
                )
                conn.execute(stmt)

    return {"n_alerts_written": len(alert_batch), "n_audit_written": len(audit_batch)}


def decide_and_persist(engine: Engine, rows: list[dict]) -> dict:
    """Evaluate mrs.policy.rules.evaluate for each row and persist alerts/audit_logs.

    `rows` must already be known to need deciding (e.g. freshly-inserted risk_scores
    rows a caller just wrote itself) -- this function does NOT check
    already_decided_transaction_ids, so it never pays that table-wide scan. It shares
    _persist_decisions with apply_policy's chunk loop (same write logic, no duplicated
    decision-persistence code), so a caller with a small, already-known-fresh row list
    -- see mrs.live.simulate, the Simulated Live Stream's per-transaction ingestion --
    can invoke it directly without apply_policy's much more expensive full-table
    idempotency scan, which is appropriate for a bulk/periodic run but not for a
    per-transaction live tick.

    Returns {n_decided, n_alerts_written, n_audit_written, action_counts, level_counts}.
    """
    action_counts: dict[str, int] = {}
    level_counts: dict[str, int] = {}
    decided: list[tuple[dict, PolicyDecision]] = []

    for row in rows:
        level_counts[row["unified_risk_level"]] = level_counts.get(row["unified_risk_level"], 0) + 1
        decision = evaluate(row)
        action_counts[decision.action] = action_counts.get(decision.action, 0) + 1
        decided.append((row, decision))

    persisted = _persist_decisions(engine, decided)

    return {
        "n_decided": len(rows),
        "n_alerts_written": persisted["n_alerts_written"],
        "n_audit_written": persisted["n_audit_written"],
        "action_counts": action_counts,
        "level_counts": level_counts,
    }


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
        to_persist: list[tuple[dict, PolicyDecision]] = []

        for row in chunk:
            n_rows_read += 1
            level_counts[row["unified_risk_level"]] = level_counts.get(row["unified_risk_level"], 0) + 1

            # Evaluated for EVERY row read, not just newly-decided ones -- action_counts/
            # level_counts are a full-population view for reporting (see docstring),
            # independent of whether this particular run persists anything for that row.
            decision = evaluate(row)
            action_counts[decision.action] = action_counts.get(decision.action, 0) + 1

            if row["transaction_id"] in already_decided:
                n_skipped += 1
                continue
            n_newly_decided += 1
            to_persist.append((row, decision))

        if to_persist:
            persisted = _persist_decisions(engine, to_persist)
            n_alerts_written += persisted["n_alerts_written"]
            n_audit_written += persisted["n_audit_written"]

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
