"""Pydantic response schemas for the Phase 8 Step 4 read API (Dev Plan §19).

Every schema here is a typed *view* over already-persisted rows (mrs.db.models) --
no schema here computes anything. Two deliberate omissions, both documented rather
than silent (Step 4 requirement: "avoid returning unnecessary raw database fields"):

- TransactionOut excludes tx_fraud/tx_fraud_scenario. This is a risk-manager API: a
  live system would never have the ground-truth fraud label at scoring time (Dev Plan
  §34.1/§34.5), so the API's response shape stays honest about what the system
  actually knew, even though this demo runs over a frozen historical/labeled dataset.
  (TX_FRAUD_SCENARIO remains usable for internal evaluation scripts, per Dev Plan
  §34.3 -- just not surfaced through this API.)
- AlertSummaryOut (list view) omits `evidence`/`reason` to keep paginated responses
  small; AlertDetailOut (single-alert view) includes them in full, verbatim from the
  stored row -- never re-derived or summarized.

policy_version is not a column on `alerts` (mrs.policy.engine records it only in the
audit_logs payload) -- AlertDetailOut/TransactionDetailOut source it via a read-only
join to the corresponding POLICY_DECISION audit_log entry, never invented.
"""

from __future__ import annotations

import datetime as dt

from pydantic import BaseModel, ConfigDict


class CustomerOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    customer_id: int
    x_customer_id: float
    y_customer_id: float
    mean_amount: float
    std_amount: float
    mean_nb_tx_per_day: float
    nb_terminals: int
    available_terminals: list[int]


class TerminalOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    terminal_id: int
    x_terminal_id: float
    y_terminal_id: float


class TransactionOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    transaction_id: int
    tx_datetime: dt.datetime
    customer_id: int
    terminal_id: int
    tx_amount: float
    split: str


class RiskScoreOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    transaction_id: int
    customer_id: int
    terminal_id: int
    transaction_risk: float | None
    transaction_risk_severity: int | None
    terminal_risk_state: str | None
    terminal_risk_severity: int | None
    customer_risk_state: str | None
    customer_risk_severity: int | None
    unified_risk_level: str
    contributing_signals: list[str]
    model_version: str
    transaction_risk_threshold: float
    feature_version: str
    computed_at: dt.datetime


class AlertSummaryOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    alert_id: int
    transaction_id: int
    customer_id: int
    terminal_id: int
    severity: str
    recommended_action: str | None
    status: str
    created_at: dt.datetime


class AlertDetailOut(AlertSummaryOut):
    reason: str
    evidence: dict
    #: Sourced from the linked audit_logs POLICY_DECISION entry; None only if that
    #: entry is somehow missing (never fabricated as a placeholder version string).
    policy_version: str | None = None


class TransactionDetailOut(BaseModel):
    """The single-transaction investigation view: raw transaction + its risk result +
    its alert (if any) + policy_version -- everything Dev Plan §21 View 4 (Alert
    Investigation) needs, assembled from already-persisted rows only."""

    transaction: TransactionOut
    risk_score: RiskScoreOut | None
    alert: AlertDetailOut | None
    policy_version: str | None = None


class PaginatedAlerts(BaseModel):
    items: list[AlertSummaryOut]
    total: int
    limit: int
    offset: int


class ReplayItemOut(BaseModel):
    """One chronological replay step (Dev Plan §22/§39; Phase 8 Step 5). alert here is
    the summary shape (not AlertDetailOut) -- a replay stream returns many items per
    request, and full evidence is one GET /alerts/{id} away when a client needs it."""

    transaction: TransactionOut
    risk_score: RiskScoreOut | None
    alert: AlertSummaryOut | None


class ReplayPage(BaseModel):
    items: list[ReplayItemOut]
    count: int
    #: Opaque token for the next GET .../replay/transactions?after_cursor=... call.
    #: None means this page reached the end of the chronological stream.
    next_cursor: str | None


class ReplayBounds(BaseModel):
    min_tx_datetime: dt.datetime
    max_tx_datetime: dt.datetime
    total_transactions: int


class AnalystResponseOut(BaseModel):
    """Response for GET /transactions/{id}/analyst (Dev Plan §16/§41; Phase 8 Step 6).

    deterministic_action is the policy engine's own already-decided action (mrs.policy,
    authoritative, unchanged by anything here). recommended_action is the AI Risk
    Analyst's own advisory opinion -- it may agree or disagree with
    deterministic_action; nothing in this API or elsewhere acts on it.
    """

    transaction_id: int
    unified_risk_level: str
    deterministic_action: str
    policy_version: str | None

    summary: str
    evidence_explanation: str
    recommended_action: str
    recommendation_rationale: str
    confidence: str
    caveats: list[str]

    is_fallback: bool
    fallback_reason: str | None
    #: The LLM model that produced this explanation, or None when is_fallback is True
    #: (the fallback path never calls the LLM).
    analyst_model: str | None


class PaginatedRiskHistory(BaseModel):
    items: list[RiskScoreOut]
    total: int
    limit: int
    offset: int
