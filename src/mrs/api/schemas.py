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
    #: Row-insertion time -- when the pipeline materialised this alert into Postgres,
    #: NOT when the underlying activity happened. Every alert loaded in the same batch
    #: shares one value, so it is useless for ordering or for reading as "when this
    #: happened"; tx_datetime is the analytically meaningful timestamp.
    created_at: dt.datetime
    #: When the alerting transaction actually occurred. Optional because the same
    #: schema is reused in replay pages, which are built straight from an Alert row
    #: and already carry the transaction alongside.
    tx_datetime: dt.datetime | None = None


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


class AuditLogOut(BaseModel):
    """One audit_logs row, verbatim (Dev Plan §20 'audit_logs', §33.9). Currently the
    only event_type the policy engine writes is POLICY_DECISION (mrs.policy.engine) --
    this schema does not assume that is the only one that will ever exist."""

    model_config = ConfigDict(from_attributes=True)

    audit_id: int
    transaction_id: int | None
    alert_id: int | None
    event_type: str
    payload: dict
    model_version: str | None
    created_at: dt.datetime


class OverviewStats(BaseModel):
    """Aggregate counts for the Command Center overview (Dev Plan §21 View 1).
    Every field is a COUNT/GROUP BY over already-persisted rows -- no risk, behavioral,
    or policy logic is evaluated here."""

    total_transactions: int
    total_customers: int
    total_terminals: int
    total_risk_scores: int
    total_alerts: int
    #: Keyed by mrs.risk.aggregate level (LOW/MEDIUM/HIGH/CRITICAL/INSUFFICIENT_EVIDENCE).
    #: A level with zero rows is simply absent, never filled in as 0.
    risk_level_counts: dict[str, int]
    #: Keyed by mrs.policy.rules.BOUNDED_ACTIONS. ALLOW never appears (Dev Plan Sec 15:
    #: only a non-ALLOW decision becomes an alerts row).
    alert_action_counts: dict[str, int]
    alert_status_counts: dict[str, int]
    #: Distinct customers/terminals whose MOST RECENT scored transaction has
    #: customer_risk_state/terminal_risk_state in (RISK_RISING, HIGH_RISK) -- a
    #: temporal snapshot ("at risk right now"), not a permanent label; RECOVERY is
    #: deliberately excluded (already improving, per Dev Plan Sec 8).
    customers_at_risk: int
    terminals_at_risk: int
    #: SUM(tx_amount) over transactions whose unified_risk_level is HIGH or CRITICAL --
    #: real money at elevated risk, never described as confirmed fraud (Dev Plan Sec 25).
    risk_exposure_amount: float


class RiskActivityPoint(BaseModel):
    """One day's worth of severity-2 ("elevated") component counts (Dev Plan Sec 16:
    temporal risk activity). Counts, not scores -- how many scored transactions that
    day had each component at its most severe tier."""

    date: dt.date
    transaction_high: int
    customer_high: int
    terminal_high: int
    #: Count of transactions that day whose unified_risk_level was HIGH or CRITICAL
    #: (mrs.risk.aggregate) -- the single "how much elevated risk activity" trend line.
    elevated_transactions: int
    total_scored: int


class NetworkNode(BaseModel):
    """One entity in the Command Center's Entity Risk Network (Dev Plan Sec 8: a
    behavioral STATE, never a permanent label). id is "customer:<id>" or
    "terminal:<id>" -- unique across both entity types."""

    id: str
    entity_type: str  # "customer" | "terminal"
    entity_id: int
    risk_state: str | None
    risk_severity: int | None
    is_focus: bool


class NetworkEdge(BaseModel):
    """A real customer<->terminal relationship, derived from actual shared
    transactions -- never a fabricated or inferred connection."""

    source: str
    target: str
    #: Number of transactions between this exact customer/terminal pair.
    weight: int


class NetworkGraph(BaseModel):
    nodes: list[NetworkNode]
    edges: list[NetworkEdge]
    #: node ids that were chosen as graph focus points (currently most-severe
    #: entities), vs. neighbors pulled in only because they transacted with one.
    focus_ids: list[str]
    #: Only set when the `live_window` query param was used (GET /stats/network):
    #: the single most recent transaction_id in that window, so the client can
    #: identify which node(s) it touches as "just arrived" without a second request.
    #: None for the default (unwindowed) graph -- there is no one meaningful "latest"
    #: transaction across the entity's entire history in that mode.
    latest_transaction_id: int | None = None


class EntityAtRiskRow(BaseModel):
    """One currently-elevated customer or terminal, with a real, computed behavioral
    deviation (Dev Plan Sec 8/14: risk as a temporal, moving state -- never a static
    label). current_rate/baseline_rate are each entity's OWN fraction of its
    transactions at severity 2 in the recent vs. prior window -- never the ground-truth
    tx_fraud label, which this system never surfaces as an operational signal."""

    entity_type: str  # "customer" | "terminal"
    entity_id: int
    risk_state: str
    risk_severity: int | None
    #: Fraction (0-1) of this entity's transactions at severity 2 in the most recent
    #: window (default 7 days of available data).
    current_rate: float
    #: Same fraction, prior window (default the 30 days before the recent window).
    #: None if the entity has no transactions in the prior window at all (too new to
    #: have a baseline -- never presented as 0, which would read as "no risk before").
    baseline_rate: float | None
    recent_transaction_count: int
    last_activity: dt.datetime


class EntityDeviation(BaseModel):
    """Real recent-vs-baseline behavioral deviation for ONE arbitrary entity (any
    state, not just currently-elevated ones) -- GET /terminals/{id}/deviation. Reuses
    the exact same computation as EntityAtRiskRow (mrs.api.lookups.entity_deviation_rates)
    so the two views can never disagree with each other."""

    entity_type: str
    entity_id: int
    #: Fraction (0-1) of this entity's transactions at severity 2, recent window.
    current_rate: float | None
    baseline_rate: float | None
    current_transaction_count: int
    baseline_transaction_count: int
    recent_window_days: int
    baseline_window_days: int
