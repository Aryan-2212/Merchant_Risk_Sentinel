"""Continuous Simulated Live Stream: control plane + read API (mrs.live.manager,
mrs.live.continuous).

Control plane (the one deliberate exception to this API package's otherwise strictly
read-only design -- see mrs.api.main's docstring): POST /live/start and
POST /live/stop start/stop an in-process background thread that generates and scores
new demo transactions -- they never touch risk scoring, aggregation, or policy logic
themselves (mrs.live.continuous.run_one_tick does, unchanged, once per tick), and
they execute no arbitrary input, so this remains safe to leave unauthenticated at this
project's demo scope (Dev Plan Sec 26: no auth infrastructure for this project).

Read API: GET /live/bounds and GET /live/transactions are a thin sibling of
mrs.api.routers.recent (itself a thin sibling of mrs.api.routers.replay) -- same
response schemas (schemas.ReplayBounds/ReplayPage/ReplayItemOut) and the same
keyset-cursor pagination convention, permanently scoped to
Transaction.split == mrs.config.LIVE_STREAM_SPLIT_LABEL. This exists so the Network
investigation panel's "Recent Transactions" list can show a live-only customer/
terminal's actual activity (previously empty for such an entity, since neither
GET /replay/transactions nor GET /recent/transactions can see split=="live" rows).
Deliberately duplicated rather than merged into recent.py, matching that router's own
documented reasoning: each stream stays permanently and independently scoped, so
extending one can never accidentally widen another.

NOT real payment traffic, NOT a live production feed -- see mrs.live.continuous's own
module docstring. The dashboard must always label this "SIMULATED LIVE STREAM".
"""

from __future__ import annotations

import datetime as dt

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import and_, func, or_, select
from sqlalchemy.orm import Session

from mrs import config
from mrs.api import schemas
from mrs.api.deps import get_db
from mrs.db.engine import get_engine
from mrs.db.models import Alert, RiskScore, Transaction
from mrs.live.manager import manager

router = APIRouter(prefix="/live", tags=["live"])

_LIVE_SPLIT = config.LIVE_STREAM_SPLIT_LABEL


def _format_cursor(tx_datetime: dt.datetime, transaction_id: int) -> str:
    return f"{tx_datetime.isoformat()}|{transaction_id}"


def _parse_cursor(cursor: str) -> tuple[dt.datetime, int]:
    try:
        dt_str, id_str = cursor.rsplit("|", 1)
        return dt.datetime.fromisoformat(dt_str), int(id_str)
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=400, detail=f"invalid cursor: {cursor!r}") from exc


@router.get("/status", response_model=schemas.LiveStreamStatus)
def get_live_status() -> schemas.LiveStreamStatus:
    return schemas.LiveStreamStatus(**manager.status())


@router.post("/start", response_model=schemas.LiveStreamStatus)
def start_live_stream(
    interval_seconds: float = Query(
        None,
        ge=0.2,
        le=60,
        description="Seconds between generated transactions (default: "
        "mrs.config.LIVE_STREAM_DEFAULT_INTERVAL_SECONDS, ~2s). A no-op if already running "
        "-- clicking Start twice never spawns a second producer or resets progress.",
    ),
) -> schemas.LiveStreamStatus:
    manager.start(get_engine(), interval_seconds)
    return schemas.LiveStreamStatus(**manager.status())


@router.post("/stop", response_model=schemas.LiveStreamStatus)
def stop_live_stream() -> schemas.LiveStreamStatus:
    manager.stop()
    return schemas.LiveStreamStatus(**manager.status())


@router.get("/bounds", response_model=schemas.ReplayBounds)
def get_live_bounds(db: Session = Depends(get_db)) -> schemas.ReplayBounds:
    """The chronological range of the Continuous Simulated Live Stream only."""
    min_dt, max_dt, total = db.execute(
        select(func.min(Transaction.tx_datetime), func.max(Transaction.tx_datetime), func.count())
        .where(Transaction.split == _LIVE_SPLIT)
    ).one()
    if total == 0:
        raise HTTPException(status_code=404, detail="no live-stream transactions available yet")
    return schemas.ReplayBounds(min_tx_datetime=min_dt, max_tx_datetime=max_dt, total_transactions=total)


@router.get("/transactions", response_model=schemas.ReplayPage)
def live_transactions(
    after_cursor: str | None = Query(None, description="Opaque cursor from a previous page's next_cursor."),
    start: dt.datetime | None = Query(None, description="Inclusive lower bound on tx_datetime; ignored if after_cursor is given."),
    end: dt.datetime | None = Query(None, description="Exclusive upper bound on tx_datetime."),
    customer_id: int | None = Query(None),
    terminal_id: int | None = Query(None),
    desc: bool = Query(
        False,
        description="Most-recent-first instead of the default chronological-forward order -- same convention as "
        "GET /replay/transactions' and GET /recent/transactions' own desc (e.g. 'this entity's N most recent "
        "live-stream transactions' for the Network investigation panel when the focus entity's activity is in the "
        "continuous live stream). No cursor support in this direction, same as the other two.",
    ),
    limit: int = Query(100, ge=1, le=2000),
    db: Session = Depends(get_db),
) -> schemas.ReplayPage:
    """One chronological window of the continuous live-stream demo data, same shape
    and cursor semantics as GET /replay/transactions and GET /recent/transactions,
    but always restricted to split=="live"."""
    stmt = (
        select(Transaction, RiskScore, Alert)
        .outerjoin(RiskScore, RiskScore.transaction_id == Transaction.transaction_id)
        .outerjoin(Alert, Alert.transaction_id == Transaction.transaction_id)
        .where(Transaction.split == _LIVE_SPLIT)
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
