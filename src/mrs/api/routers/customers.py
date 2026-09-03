"""Customer read endpoints (Dev Plan §19 api/customers, §21 View 3)."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from mrs.api import schemas
from mrs.api.deps import get_db
from mrs.api.lookups import BASELINE_WINDOW_DAYS, RECENT_WINDOW_DAYS, entity_deviation_rates
from mrs.db.models import Customer, RiskScore, Transaction

router = APIRouter(prefix="/customers", tags=["customers"])


@router.get("/{customer_id}", response_model=schemas.CustomerOut)
def get_customer(customer_id: int, db: Session = Depends(get_db)) -> Customer:
    customer = db.get(Customer, customer_id)
    if customer is None:
        raise HTTPException(status_code=404, detail=f"customer_id {customer_id} not found")
    return customer


@router.get("/{customer_id}/risk", response_model=schemas.PaginatedRiskHistory)
def get_customer_risk_history(
    customer_id: int,
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
) -> schemas.PaginatedRiskHistory:
    if db.get(Customer, customer_id) is None:
        raise HTTPException(status_code=404, detail=f"customer_id {customer_id} not found")

    total = db.execute(
        select(func.count()).select_from(RiskScore).where(RiskScore.customer_id == customer_id)
    ).scalar_one()
    rows = (
        db.execute(
            select(RiskScore)
            .join(Transaction, Transaction.transaction_id == RiskScore.transaction_id)
            .where(RiskScore.customer_id == customer_id)
            .order_by(Transaction.tx_datetime, Transaction.transaction_id)
            .limit(limit)
            .offset(offset)
        )
        .scalars()
        .all()
    )
    return schemas.PaginatedRiskHistory(items=list(rows), total=total, limit=limit, offset=offset)


@router.get("/{customer_id}/deviation", response_model=schemas.EntityDeviation)
def get_customer_deviation(customer_id: int, db: Session = Depends(get_db)) -> schemas.EntityDeviation:
    """Real recent-vs-baseline severity-2 rate for this specific customer -- the
    customer-side counterpart to GET /terminals/{id}/deviation (identical computation,
    mrs.api.lookups.entity_deviation_rates, so the two entity types can never drift).
    Never a "fraud rate": this is this system's own customer_risk_severity, never the
    ground-truth tx_fraud label."""
    if db.get(Customer, customer_id) is None:
        raise HTTPException(status_code=404, detail=f"customer_id {customer_id} not found")

    rates = entity_deviation_rates(db, "customer", [customer_id])
    row = rates.get(customer_id, {})
    return schemas.EntityDeviation(
        entity_type="customer",
        entity_id=customer_id,
        current_rate=row.get("current_rate"),
        baseline_rate=row.get("baseline_rate"),
        current_transaction_count=int(row.get("current_count", 0)),
        baseline_transaction_count=int(row.get("baseline_count", 0)),
        recent_window_days=RECENT_WINDOW_DAYS,
        baseline_window_days=BASELINE_WINDOW_DAYS,
    )
