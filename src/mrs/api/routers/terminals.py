"""Terminal read endpoints (Dev Plan §19 api/terminals, §21 View 2)."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from mrs.api import schemas
from mrs.api.deps import get_db
from mrs.db.models import RiskScore, Terminal, Transaction

router = APIRouter(prefix="/terminals", tags=["terminals"])


@router.get("/{terminal_id}", response_model=schemas.TerminalOut)
def get_terminal(terminal_id: int, db: Session = Depends(get_db)) -> Terminal:
    terminal = db.get(Terminal, terminal_id)
    if terminal is None:
        raise HTTPException(status_code=404, detail=f"terminal_id {terminal_id} not found")
    return terminal


@router.get("/{terminal_id}/risk", response_model=schemas.PaginatedRiskHistory)
def get_terminal_risk_history(
    terminal_id: int,
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
) -> schemas.PaginatedRiskHistory:
    if db.get(Terminal, terminal_id) is None:
        raise HTTPException(status_code=404, detail=f"terminal_id {terminal_id} not found")

    total = db.execute(
        select(func.count()).select_from(RiskScore).where(RiskScore.terminal_id == terminal_id)
    ).scalar_one()
    rows = (
        db.execute(
            select(RiskScore)
            .join(Transaction, Transaction.transaction_id == RiskScore.transaction_id)
            .where(RiskScore.terminal_id == terminal_id)
            .order_by(Transaction.tx_datetime, Transaction.transaction_id)
            .limit(limit)
            .offset(offset)
        )
        .scalars()
        .all()
    )
    return schemas.PaginatedRiskHistory(items=list(rows), total=total, limit=limit, offset=offset)
