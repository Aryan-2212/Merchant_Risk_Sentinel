"""Alert read endpoints (Dev Plan §19 api/alerts, §20, §21 View 4).

Lists/reads already-persisted policy decisions (mrs.policy.engine) -- never decides,
recomputes, or alters an alert.
"""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from mrs.api import schemas
from mrs.api.deps import get_db
from mrs.api.lookups import policy_version_for_transaction
from mrs.db.models import Alert
from mrs.risk.aggregate import CRITICAL, HIGH, INSUFFICIENT_EVIDENCE, MEDIUM

router = APIRouter(prefix="/alerts", tags=["alerts"])

#: Alerts are only ever created for a non-ALLOW action (mrs.policy.rules), so LOW never
#: appears as an alert severity -- the filter's accepted values reflect that exactly.
_ALERT_SEVERITIES = (MEDIUM, HIGH, CRITICAL, INSUFFICIENT_EVIDENCE)


@router.get("", response_model=schemas.PaginatedAlerts)
def list_alerts(
    status: str | None = Query(None),
    severity: Literal[MEDIUM, HIGH, CRITICAL, INSUFFICIENT_EVIDENCE] | None = Query(None),
    customer_id: int | None = Query(None),
    terminal_id: int | None = Query(None),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
) -> schemas.PaginatedAlerts:
    stmt = select(Alert)
    count_stmt = select(func.count()).select_from(Alert)
    conditions = []
    if status is not None:
        conditions.append(Alert.status == status)
    if severity is not None:
        conditions.append(Alert.severity == severity)
    if customer_id is not None:
        conditions.append(Alert.customer_id == customer_id)
    if terminal_id is not None:
        conditions.append(Alert.terminal_id == terminal_id)
    for cond in conditions:
        stmt = stmt.where(cond)
        count_stmt = count_stmt.where(cond)

    total = db.execute(count_stmt).scalar_one()
    rows = db.execute(stmt.order_by(Alert.created_at.desc(), Alert.alert_id.desc()).limit(limit).offset(offset)).scalars().all()
    return schemas.PaginatedAlerts(items=list(rows), total=total, limit=limit, offset=offset)


@router.get("/{alert_id}", response_model=schemas.AlertDetailOut)
def get_alert(alert_id: int, db: Session = Depends(get_db)) -> schemas.AlertDetailOut:
    alert = db.get(Alert, alert_id)
    if alert is None:
        raise HTTPException(status_code=404, detail=f"alert_id {alert_id} not found")
    decision = schemas.AlertDetailOut.model_validate(alert)
    return decision.model_copy(update={"policy_version": policy_version_for_transaction(db, alert.transaction_id)})
