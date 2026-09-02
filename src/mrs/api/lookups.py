"""Shared read-only query helpers used by more than one router (Dev Plan §33.3: avoid
duplicating logic across modules). No computation -- straight lookups against
already-persisted rows.
"""

from __future__ import annotations

from sqlalchemy import select, text
from sqlalchemy.orm import Session

from mrs.db.models import AuditLog

#: Recent-vs-baseline behavioral deviation windows (Command Center's "High Deviation
#: Terminals" panel, and the per-entity deviation used on individual terminal/customer
#: pages). Shared here so both call sites use the exact same window definition.
RECENT_WINDOW_DAYS = 7
BASELINE_WINDOW_DAYS = 30


def policy_version_for_transaction(db: Session, transaction_id: int) -> str | None:
    """The policy_version recorded in that transaction's POLICY_DECISION audit_log
    entry (mrs.policy.engine.apply_policy), or None if policy has not been applied to
    it yet. Read verbatim from the stored payload -- never fabricated."""
    payload = db.execute(
        select(AuditLog.payload)
        .where(AuditLog.transaction_id == transaction_id, AuditLog.event_type == "POLICY_DECISION")
        .order_by(AuditLog.created_at.desc())
        .limit(1)
    ).scalar_one_or_none()
    return payload.get("policy_version") if payload else None


def entity_deviation_rates(
    db: Session,
    entity_type: str,
    entity_ids: list[int],
    recent_days: int = RECENT_WINDOW_DAYS,
    baseline_days: int = BASELINE_WINDOW_DAYS,
) -> dict[int, dict[str, float]]:
    """Each entity's OWN fraction of transactions at severity 2 (mrs.risk.aggregate's
    customer/terminal component), recent window vs. the window before it -- a real,
    computed behavioral deviation. Never uses tx_fraud (ground truth, never an
    operational signal). Shared by GET /stats/terminals-at-risk (bulk, currently
    at-risk cohort) and GET /terminals/{id}/deviation, GET /customers/{id}/deviation
    (single arbitrary entity, any state) so the two call sites can't drift.

    Returns {entity_id: {"current_rate":, "current_count":, "baseline_rate":,
    "baseline_count":}}; a bucket key is simply absent if that entity has no
    transactions in that window (never fabricated as a 0.0 rate).
    """
    if entity_type not in ("customer", "terminal"):
        raise ValueError(f"entity_deviation_rates: unknown entity_type {entity_type!r}")
    if not entity_ids:
        return {}
    id_col = f"{entity_type}_id"
    severity_col = f"{entity_type}_risk_severity"
    rows = db.execute(
        text(
            f"""
            WITH bounds AS (SELECT max(tx_datetime) AS max_dt FROM transactions),
            windowed AS (
                SELECT rs.{id_col} AS entity_id,
                    (t.tx_datetime > bounds.max_dt - interval '{recent_days} days') AS is_recent,
                    (rs.{severity_col} = 2)::int AS is_high
                FROM risk_scores rs
                JOIN transactions t ON t.transaction_id = rs.transaction_id, bounds
                WHERE rs.{id_col} = ANY(:ids)
                  AND t.tx_datetime > bounds.max_dt - interval '{recent_days + baseline_days} days'
            )
            SELECT entity_id, is_recent, avg(is_high)::float AS rate, count(*) AS n
            FROM windowed
            GROUP BY entity_id, is_recent
            """
        ),
        {"ids": entity_ids},
    ).all()

    result: dict[int, dict[str, float]] = {eid: {} for eid in entity_ids}
    for r in rows:
        bucket = "current" if r.is_recent else "baseline"
        result[r.entity_id][f"{bucket}_rate"] = r.rate
        result[r.entity_id][f"{bucket}_count"] = r.n
    return result
