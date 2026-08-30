"""Builds the structured evidence handed to the AI Risk Analyst (Dev Plan §16/§41).

Pure function: takes already-fetched rows (mrs.db.models), returns an
AnalystEvidence. No database access, no risk computation, no policy decision --
everything here was already decided by mrs.models/mrs.behavioral/mrs.risk/
mrs.policy. This module's only job is assembling their already-computed outputs into
the one typed object the LLM is allowed to see (Dev Plan §41: "structured, computed
evidence", never invented).
"""

from __future__ import annotations

from mrs.analyst.schemas import AnalystEvidence
from mrs.db.models import Alert, RiskScore, Transaction
from mrs.policy.rules import ALLOW


def build_evidence(
    transaction: Transaction,
    risk_score: RiskScore,
    alert: Alert | None,
    policy_version: str | None,
) -> AnalystEvidence:
    """transaction/risk_score: required, the transaction being explained must already
    be scored. alert: None when the deterministic policy decided ALLOW (no alert row
    exists for that case -- mrs.policy.engine only writes alerts for non-ALLOW
    actions) -- policy_action then defaults to ALLOW, never fabricated as anything
    else. policy_version: from mrs.api.lookups.policy_version_for_transaction (or
    None if policy has not been applied to this transaction yet).
    """
    return AnalystEvidence(
        transaction_id=transaction.transaction_id,
        tx_amount=transaction.tx_amount,
        tx_datetime=transaction.tx_datetime,
        customer_id=transaction.customer_id,
        terminal_id=transaction.terminal_id,
        unified_risk_level=risk_score.unified_risk_level,
        transaction_risk=risk_score.transaction_risk,
        transaction_risk_severity=risk_score.transaction_risk_severity,
        terminal_risk_state=risk_score.terminal_risk_state,
        terminal_risk_severity=risk_score.terminal_risk_severity,
        customer_risk_state=risk_score.customer_risk_state,
        customer_risk_severity=risk_score.customer_risk_severity,
        contributing_signals=list(risk_score.contributing_signals or []),
        policy_action=alert.recommended_action if alert is not None else ALLOW,
        policy_reason=alert.reason if alert is not None else None,
        policy_version=policy_version,
        model_version=risk_score.model_version,
        feature_version=risk_score.feature_version,
        transaction_risk_threshold=risk_score.transaction_risk_threshold,
    )
