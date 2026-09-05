"""Alert read endpoints (Dev Plan §19 api/alerts, §20, §21 View 4).

Lists/reads already-persisted policy decisions (mrs.policy.engine) -- never decides,
recomputes, or alters an alert.
"""

from __future__ import annotations

import datetime as dt
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from mrs.api import schemas
from mrs.api.deps import get_db
from mrs.api.lookups import policy_version_for_transaction
from mrs.db.models import Alert, Transaction
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
    start: dt.datetime | None = Query(
        None, description="Inclusive lower bound on the alerting transaction's tx_datetime."
    ),
    end: dt.datetime | None = Query(
        None, description="Exclusive upper bound on the alerting transaction's tx_datetime."
    ),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
) -> schemas.PaginatedAlerts:
    # Joined so each alert can carry the timestamp of the transaction that raised it.
    # Alert.created_at is the batch row-insertion time -- identical across every alert
    # loaded in the same run -- so ordering or dating the list by it made the whole
    # table read as a single day. tx_datetime is when the activity actually happened.
    #
    # The count query carries the same join, so a date-filtered total counts the same
    # rows the page does rather than every alert in the table.
    join_on = Transaction.transaction_id == Alert.transaction_id
    stmt = select(Alert, Transaction.tx_datetime).join(Transaction, join_on)
    count_stmt = select(func.count()).select_from(Alert).join(Transaction, join_on)
    conditions = []
    if start is not None:
        conditions.append(Transaction.tx_datetime >= start)
    if end is not None:
        conditions.append(Transaction.tx_datetime < end)
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
    rows = db.execute(
        stmt.order_by(Transaction.tx_datetime.desc(), Alert.alert_id.desc()).limit(limit).offset(offset)
    ).all()
    items = [
        schemas.AlertSummaryOut.model_validate(alert).model_copy(update={"tx_datetime": tx_datetime})
        for alert, tx_datetime in rows
    ]
    return schemas.PaginatedAlerts(items=items, total=total, limit=limit, offset=offset)


@router.get("/{alert_id}", response_model=schemas.AlertDetailOut)
def get_alert(alert_id: int, db: Session = Depends(get_db)) -> schemas.AlertDetailOut:
    alert = db.get(Alert, alert_id)
    if alert is None:
        raise HTTPException(status_code=404, detail=f"alert_id {alert_id} not found")
    decision = schemas.AlertDetailOut.model_validate(alert)
    tx_datetime = db.execute(
        select(Transaction.tx_datetime).where(Transaction.transaction_id == alert.transaction_id)
    ).scalar_one_or_none()
    return decision.model_copy(
        update={
            "policy_version": policy_version_for_transaction(db, alert.transaction_id),
            "tx_datetime": tx_datetime,
        }
    )
