"""Shared read-only query helpers used by more than one router (Dev Plan §33.3: avoid
duplicating logic across modules). No computation -- straight lookups against
already-persisted rows.
"""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from mrs.db.models import AuditLog


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
