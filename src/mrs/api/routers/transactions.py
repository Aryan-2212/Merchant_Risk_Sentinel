"""Transaction read endpoints (Dev Plan §19 api/transactions, §21 View 4).

Serves already-persisted rows only -- never recomputes transaction ML risk, customer/
terminal behavioral risk, or aggregation (mrs.risk.aggregate / mrs.behavioral.* /
mrs.models.train_xgboost are never imported here).
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from mrs.api import schemas
from mrs.api.deps import get_db
from mrs.api.lookups import policy_version_for_transaction
from mrs.db.models import Alert, RiskScore, Transaction

router = APIRouter(prefix="/transactions", tags=["transactions"])


@router.get("/{transaction_id}", response_model=schemas.TransactionDetailOut)
def get_transaction(transaction_id: int, db: Session = Depends(get_db)) -> schemas.TransactionDetailOut:
    tx = db.get(Transaction, transaction_id)
    if tx is None:
        raise HTTPException(status_code=404, detail=f"transaction_id {transaction_id} not found")

    risk = db.get(RiskScore, transaction_id)
    alert = db.execute(select(Alert).where(Alert.transaction_id == transaction_id)).scalar_one_or_none()
    policy_version = policy_version_for_transaction(db, transaction_id)

    alert_out = None
    if alert is not None:
        alert_out = schemas.AlertDetailOut.model_validate(alert).model_copy(
            update={"policy_version": policy_version}
        )

    return schemas.TransactionDetailOut(
        transaction=schemas.TransactionOut.model_validate(tx),
        risk_score=schemas.RiskScoreOut.model_validate(risk) if risk is not None else None,
        alert=alert_out,
        policy_version=policy_version,
    )


@router.get("/{transaction_id}/risk", response_model=schemas.RiskScoreOut)
def get_transaction_risk(transaction_id: int, db: Session = Depends(get_db)) -> RiskScore:
    risk = db.get(RiskScore, transaction_id)
    if risk is None:
        raise HTTPException(status_code=404, detail=f"risk score for transaction_id {transaction_id} not found")
    return risk
