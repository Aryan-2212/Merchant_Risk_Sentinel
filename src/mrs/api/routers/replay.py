"""Replay API (Dev Plan §22/§39; Phase 8 Step 5).

Approved architecture decision (Phase 8 kickoff, decision 1): Replay reads the
already-computed, already-validated Phase 5+6+7+8 pipeline output (transactions,
risk_scores, alerts) back out in strict chronological order. It does NOT recompute
features, does NOT re-score transactions live, and does NOT re-run the behavioral
engines or aggregation -- "live online incremental scoring" was explicitly ruled out
at Phase 8 kickoff in favor of materializing the pipeline into Postgres once (Steps
2/3) and replaying those records.

Dev Plan §39's replay semantics (build features from prior-state only, score, only
then reveal the outcome) are satisfied by construction, not by anything in this
module: every persisted feature/risk value was already computed using only
strictly-prior information at its own transaction's original scoring time (Phase 3/5
leakage-safety, independently re-audited in the Phase 5 validation report). This
module's only job is to reveal those already-correct results one chronological window
at a time, instead of returning all 1.75M rows at once (Dev Plan §22: "Do not attempt
to process 1.75M transactions live during the demo").

Pacing ("allow the demo to accelerate time", Dev Plan §22) is left to the client: this
API introduces no server-side delay, timer, or session state. A client drives its own
replay speed by choosing how large a window to request and how often to request the
next one -- keyset pagination (cursor = last row's (tx_datetime, transaction_id)) makes
each request stateless and independently resumable, avoiding a server-held "replay
session" that Dev Plan §26 would flag as infrastructure this project does not need.

Deliberately does not duplicate GET /customers/{id}/risk or /terminals/{id}/risk
(Step 4) -- those already serve one entity's chronological history; this module serves
the full historical transaction stream Dev Plan §22 describes.
"""

from __future__ import annotations

import datetime as dt

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import and_, func, or_, select
from sqlalchemy.orm import Session

from mrs.api import schemas
from mrs.api.deps import get_db
from mrs.db.models import Alert, RiskScore, Transaction

router = APIRouter(prefix="/replay", tags=["replay"])


def _format_cursor(tx_datetime: dt.datetime, transaction_id: int) -> str:
    return f"{tx_datetime.isoformat()}|{transaction_id}"


def _parse_cursor(cursor: str) -> tuple[dt.datetime, int]:
    try:
        dt_str, id_str = cursor.rsplit("|", 1)
        return dt.datetime.fromisoformat(dt_str), int(id_str)
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=400, detail=f"invalid cursor: {cursor!r}") from exc


@router.get("/bounds", response_model=schemas.ReplayBounds)
def get_replay_bounds(db: Session = Depends(get_db)) -> schemas.ReplayBounds:
    """The chronological range available to replay -- lets a client compute how far
    through the stream a given cursor/time is, e.g. for a progress indicator."""
    min_dt, max_dt, total = db.execute(
        select(func.min(Transaction.tx_datetime), func.max(Transaction.tx_datetime), func.count())
    ).one()
    if total == 0:
        raise HTTPException(status_code=404, detail="no transactions available to replay")
    return schemas.ReplayBounds(min_tx_datetime=min_dt, max_tx_datetime=max_dt, total_transactions=total)


@router.get("/transactions", response_model=schemas.ReplayPage)
def replay_transactions(
    after_cursor: str | None = Query(
        None, description="Opaque cursor from a previous page's next_cursor. Takes precedence over start."
    ),
    start: dt.datetime | None = Query(None, description="Inclusive lower bound on tx_datetime; ignored if after_cursor is given."),
    end: dt.datetime | None = Query(None, description="Exclusive upper bound on tx_datetime."),
    customer_id: int | None = Query(None, description="Restrict to one customer's transactions (same filter convention as GET /alerts)."),
    terminal_id: int | None = Query(None, description="Restrict to one terminal's transactions."),
    desc: bool = Query(
        False,
        description="Most-recent-first instead of the default chronological-forward replay order. Used by callers that "
        "want 'this entity's N most recent transactions' (e.g. the Network investigation panel) rather than a replay "
        "window; next_cursor is omitted when true since nothing consumes a descending cursor today.",
    ),
    limit: int = Query(100, ge=1, le=2000),
    db: Session = Depends(get_db),
) -> schemas.ReplayPage:
    """One chronological window of the historical transaction stream, each item
    carrying its already-computed risk_score and alert (if any). Keyset-paginated on
    (tx_datetime, transaction_id) -- stable regardless of how large the table is."""
    stmt = (
        select(Transaction, RiskScore, Alert)
        .outerjoin(RiskScore, RiskScore.transaction_id == Transaction.transaction_id)
        .outerjoin(Alert, Alert.transaction_id == Transaction.transaction_id)
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
        # No cursor support in this direction -- every current/planned caller of
        # desc=True wants a small bounded "N most recent" list, not a paginated
        # backward stream, so next_cursor is deliberately left None below.
        stmt = stmt.order_by(Transaction.tx_datetime.desc(), Transaction.transaction_id.desc()).limit(limit)
    else:
        # Fetch one extra row to know whether a next page exists, without a second query.
        stmt = stmt.order_by(Transaction.tx_datetime, Transaction.transaction_id).limit(limit + 1)
    rows = db.execute(stmt).all()

    if desc:
        items = [
            schemas.ReplayItemOut(
                transaction=schemas.TransactionOut.model_validate(tx),
                risk_score=schemas.RiskScoreOut.model_validate(risk) if risk is not None else None,
                alert=schemas.AlertSummaryOut.model_validate(alert) if alert is not None else None,
            )
            for tx, risk, alert in rows
        ]
        return schemas.ReplayPage(items=items, count=len(items), next_cursor=None)

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
