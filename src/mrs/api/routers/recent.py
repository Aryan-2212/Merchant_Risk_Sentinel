"""Simulated Recent Operational Stream read API (see mrs.data.recent_stream).

Deliberately a thin sibling of mrs.api.routers.replay, not a merge into it: reuses the
exact same response schemas (schemas.ReplayBounds/ReplayPage/ReplayItemOut) and the
same keyset-cursor pagination convention, but is permanently scoped to
Transaction.split == mrs.config.RECENT_STREAM_SPLIT_LABEL so it can never be confused
with -- or accidentally widen -- the historical benchmark Replay stream. Read-only,
same as every other router in this package: no route here recomputes anything.

This is simulated operational/demo data (Aug-Sep 2026 in this build), never real
Razorpay production traffic -- see mrs.data.recent_stream's module docstring.
"""

from __future__ import annotations

import datetime as dt

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import and_, func, or_, select
from sqlalchemy.orm import Session

from mrs import config
from mrs.api import schemas
from mrs.api.deps import get_db
from mrs.db.models import Alert, RiskScore, Transaction

router = APIRouter(prefix="/recent", tags=["recent"])

_RECENT_SPLIT = config.RECENT_STREAM_SPLIT_LABEL


def _format_cursor(tx_datetime: dt.datetime, transaction_id: int) -> str:
    return f"{tx_datetime.isoformat()}|{transaction_id}"


def _parse_cursor(cursor: str) -> tuple[dt.datetime, int]:
    try:
        dt_str, id_str = cursor.rsplit("|", 1)
        return dt.datetime.fromisoformat(dt_str), int(id_str)
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=400, detail=f"invalid cursor: {cursor!r}") from exc


@router.get("/bounds", response_model=schemas.ReplayBounds)
def get_recent_bounds(db: Session = Depends(get_db)) -> schemas.ReplayBounds:
    """The chronological range of the Simulated Recent Operational Stream only."""
    min_dt, max_dt, total = db.execute(
        select(func.min(Transaction.tx_datetime), func.max(Transaction.tx_datetime), func.count())
        .where(Transaction.split == _RECENT_SPLIT)
    ).one()
    if total == 0:
        raise HTTPException(status_code=404, detail="no recent-stream transactions available")
    return schemas.ReplayBounds(min_tx_datetime=min_dt, max_tx_datetime=max_dt, total_transactions=total)


@router.get("/transactions", response_model=schemas.ReplayPage)
def recent_transactions(
    after_cursor: str | None = Query(None, description="Opaque cursor from a previous page's next_cursor."),
    start: dt.datetime | None = Query(None, description="Inclusive lower bound on tx_datetime; ignored if after_cursor is given."),
    end: dt.datetime | None = Query(None, description="Exclusive upper bound on tx_datetime."),
    customer_id: int | None = Query(None),
    terminal_id: int | None = Query(None),
    desc: bool = Query(
        False,
        description="Most-recent-first instead of the default chronological-forward order -- same convention and "
        "use case as GET /replay/transactions' desc (e.g. 'this entity's N most recent recent-stream transactions' "
        "for the Network investigation panel when the focus entity's activity is in the recent stream). No cursor "
        "support in this direction, same as replay's.",
    ),
    limit: int = Query(100, ge=1, le=2000),
    db: Session = Depends(get_db),
) -> schemas.ReplayPage:
    """One chronological window of the recent-stream demo data, same shape and cursor
    semantics as GET /replay/transactions, but always restricted to split=="recent"."""
    stmt = (
        select(Transaction, RiskScore, Alert)
        .outerjoin(RiskScore, RiskScore.transaction_id == Transaction.transaction_id)
        .outerjoin(Alert, Alert.transaction_id == Transaction.transaction_id)
        .where(Transaction.split == _RECENT_SPLIT)
    )

    if after_cursor is not None:
        cursor_dt, cursor_id = _parse_cursor(after_cursor)
        stmt = stmt.where(
            or_(
                Transaction.tx_datetime > cursor_dt,
                and_(Transaction.tx_datetime == cursor_dt, Transaction.transaction_id > cursor_id),
            )
        )
    elif start is not None:
        stmt = stmt.where(Transaction.tx_datetime >= start)

    if end is not None:
        stmt = stmt.where(Transaction.tx_datetime < end)
    if customer_id is not None:
        stmt = stmt.where(Transaction.customer_id == customer_id)
    if terminal_id is not None:
        stmt = stmt.where(Transaction.terminal_id == terminal_id)

    if desc:
        stmt = stmt.order_by(Transaction.tx_datetime.desc(), Transaction.transaction_id.desc()).limit(limit)
        rows = db.execute(stmt).all()
        items = [
            schemas.ReplayItemOut(
                transaction=schemas.TransactionOut.model_validate(tx),
                risk_score=schemas.RiskScoreOut.model_validate(risk) if risk is not None else None,
                alert=schemas.AlertSummaryOut.model_validate(alert) if alert is not None else None,
            )
            for tx, risk, alert in rows
        ]
        return schemas.ReplayPage(items=items, count=len(items), next_cursor=None)

    stmt = stmt.order_by(Transaction.tx_datetime, Transaction.transaction_id).limit(limit + 1)
    rows = db.execute(stmt).all()

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
        next_cursor = _format_cursor(last_tx.tx_datetime, last_tx.transaction_id)

    return schemas.ReplayPage(items=items, count=len(items), next_cursor=next_cursor)
