"""Structured input/output contracts for the AI Risk Analyst (Dev Plan §16/§41;
Phase 8 Step 6).

AnalystEvidence is the ONLY thing the LLM ever sees -- a small, fully-typed snapshot
of one already-scored transaction, built by mrs.analyst.evidence.build_evidence from
already-persisted rows. It is not a database handle, not raw SQL, not other
transactions' data -- Dev Plan §41 "receives only structured, computed evidence".

AnalystExplanation is the ONLY thing the LLM is allowed to produce -- validated via
Anthropic structured outputs (client.messages.parse(output_format=AnalystExplanation)),
never free-form text. recommended_action is a Literal over the same five bounded
actions mrs.policy.rules defines (imported, not duplicated), so a value outside that
set cannot even parse successfully -- Dev Plan §15/§41: "use a strict structured
response schema and validate it."
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, Field

from mrs.policy.rules import ALLOW, ESCALATE, MONITOR, STEP_UP_VERIFICATION, TEMPORARY_REVIEW


class AnalystEvidence(BaseModel):
    """Everything the LLM is given about one transaction -- copied verbatim from
    mrs.db.models rows by mrs.analyst.evidence.build_evidence, nothing computed here."""

    transaction_id: int
    tx_amount: float
    tx_datetime: dt.datetime
    customer_id: int
    terminal_id: int

    unified_risk_level: str
    transaction_risk: float | None
    transaction_risk_severity: int | None
    terminal_risk_state: str | None
    terminal_risk_severity: int | None
    customer_risk_state: str | None
    customer_risk_severity: int | None
    contributing_signals: list[str]

    #: The deterministic policy engine's own already-decided action (mrs.policy.rules),
    #: ALLOW when no alert exists. Authoritative -- the LLM may reference it but never
    #: changes it (Dev Plan §16: "A deterministic policy validator decides").
    policy_action: str
    policy_reason: str | None
    policy_version: str | None

    model_version: str
    feature_version: str
    transaction_risk_threshold: float


class AnalystExplanation(BaseModel):
    """The LLM's structured response. Every string field must be grounded only in the
    supplied AnalystEvidence -- enforced by the system prompt and, for the single most
    safety-critical class of violation (fraud-certainty claims), by
    mrs.analyst.client._check_evidence_grounding after parsing."""

    summary: str = Field(
        ..., description="1-3 sentence plain-language summary of the risk situation, grounded only in the supplied evidence."
    )
    evidence_explanation: str = Field(
        ...,
        description="Explanation of which contributing signals / behavioral states drove the risk level, referencing only supplied fields.",
    )
    recommended_action: Literal[ALLOW, MONITOR, STEP_UP_VERIFICATION, TEMPORARY_REVIEW, ESCALATE] = Field(
        ..., description="Your own advisory recommendation -- one of the five bounded actions. Advisory only."
    )
    recommendation_rationale: str = Field(..., description="Why you recommend that action, grounded only in the evidence.")
    confidence: Literal["low", "medium", "high"]
    caveats: list[str] = Field(default_factory=list, description="Any limitations or missing evidence worth flagging.")


@dataclass(frozen=True)
class AnalystResult:
    """Wraps an AnalystExplanation with provenance: whether it came from the LLM or
    the deterministic fallback, and why, if it fell back."""

    explanation: AnalystExplanation
    is_fallback: bool
    fallback_reason: str | None
