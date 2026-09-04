"""Read-only endpoints for the simulated recent operational stream.

The ingestion itself is performed by scripts/14_ingest_recent_stream.py. These endpoints
only expose persisted recent rows to the dashboard; they never compute risk or policy.
"""

from __future__ import annotations

import datetime as dt

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import and_, func, or_, select
from sqlalchemy.orm import Session

from mrs.api import schemas
from mrs.api.deps import get_db
from mrs.db.models import Alert, RiskScore, Transaction

router = APIRouter(prefix="/recent", tags=["recent"])

RECENT_SPLIT = "recent"


@router.get("/bounds", response_model=schemas.ReplayBounds)
def recent_bounds(db: Session = Depends(get_db)) -> schemas.ReplayBounds:
    min_dt, max_dt, total = db.execute(
        select(
            func.min(Transaction.tx_datetime),
            func.max(Transaction.tx_datetime),
            func.count(),
        ).where(Transaction.split == RECENT_SPLIT)
    ).one()
    if not total:
        raise HTTPException(status_code=404, detail="recent stream is not ingested")
    return schemas.ReplayBounds(
        min_tx_datetime=min_dt,
        max_tx_datetime=max_dt,
        total_transactions=total,
    )


@router.get("/transactions", response_model=schemas.ReplayPage)
def recent_transactions(
    after_cursor: str | None = Query(None),
    limit: int = Query(100, ge=1, le=2000),
    db: Session = Depends(get_db),
) -> schemas.ReplayPage:
    stmt = (
        select(Transaction, RiskScore, Alert)
        .outerjoin(RiskScore, RiskScore.transaction_id == Transaction.transaction_id)
        .outerjoin(Alert, Alert.transaction_id == Transaction.transaction_id)
        .where(Transaction.split == RECENT_SPLIT)
    )

    if after_cursor is not None:
        try:
            dt_str, id_str = after_cursor.rsplit("|", 1)
            cursor_dt, cursor_id = dt.datetime.fromisoformat(dt_str), int(id_str)
        except (ValueError, TypeError) as exc:
            raise HTTPException(status_code=400, detail="invalid recent cursor") from exc
        stmt = stmt.where(
            or_(
                Transaction.tx_datetime > cursor_dt,
                and_(Transaction.tx_datetime == cursor_dt, Transaction.transaction_id > cursor_id),
            )
        )

    rows = db.execute(
        stmt.order_by(Transaction.tx_datetime, Transaction.transaction_id).limit(limit + 1)
    ).all()
    has_more = len(rows) > limit
    rows = rows[:limit]

    items = [
        schemas.ReplayItemOut(
            transaction=schemas.TransactionOut.model_validate(tx),
            risk_score=schemas.RiskScoreOut.model_validate(risk) if risk is not None else None,
            alert=schemas.AlertSummaryOut.model_validate(alert) if alert is not None else None,
        )
        for tx, risk, alert in rows
    ]

    next_cursor = None
    if has_more and rows:
        last_tx = rows[-1][0]
        next_cursor = f"{last_tx.tx_datetime.isoformat()}|{last_tx.transaction_id}"
    return schemas.ReplayPage(items=items, count=len(items), next_cursor=next_cursor)
