"""AI Risk Analyst endpoint (Dev Plan §16/§41; Phase 8 Step 6).

Assembles already-persisted rows into structured evidence (mrs.analyst.evidence),
calls the analyst (mrs.analyst.client -- one structured LLM call, or a deterministic
fallback on any failure), and returns the result. This route computes nothing about
risk and writes nothing back to the database -- fully read-only, like every other
Step 4/5 endpoint.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from mrs.analyst.client import ANALYST_MODEL, generate_explanation
from mrs.analyst.evidence import build_evidence
from mrs.api import schemas
from mrs.api.deps import get_db
from mrs.api.lookups import policy_version_for_transaction
from mrs.db.models import Alert, RiskScore, Transaction

router = APIRouter(prefix="/transactions", tags=["analyst"])


@router.get("/{transaction_id}/analyst", response_model=schemas.AnalystResponseOut)
def get_transaction_analyst_explanation(
    transaction_id: int, db: Session = Depends(get_db)
) -> schemas.AnalystResponseOut:
    tx = db.get(Transaction, transaction_id)
    if tx is None:
        raise HTTPException(status_code=404, detail=f"transaction_id {transaction_id} not found")

    risk = db.get(RiskScore, transaction_id)
    if risk is None:
        raise HTTPException(
            status_code=404,
            detail=f"no computed risk_score for transaction_id {transaction_id}; nothing for the analyst to explain",
        )

    alert = db.execute(select(Alert).where(Alert.transaction_id == transaction_id)).scalar_one_or_none()
    policy_version = policy_version_for_transaction(db, transaction_id)

    evidence = build_evidence(tx, risk, alert, policy_version)
    result = generate_explanation(evidence)

    return schemas.AnalystResponseOut(
        transaction_id=transaction_id,
        unified_risk_level=evidence.unified_risk_level,
        deterministic_action=evidence.policy_action,
        policy_version=policy_version,
        summary=result.explanation.summary,
        evidence_explanation=result.explanation.evidence_explanation,
        recommended_action=result.explanation.recommended_action,
        recommendation_rationale=result.explanation.recommendation_rationale,
        confidence=result.explanation.confidence,
        caveats=result.explanation.caveats,
        is_fallback=result.is_fallback,
        fallback_reason=result.fallback_reason,
        analyst_model=None if result.is_fallback else ANALYST_MODEL,
    )
