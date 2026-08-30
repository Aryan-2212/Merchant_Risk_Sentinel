"""ORM models for the Phase 8 backend schema (Dev Plan §20; approved Phase 8 Step 1).

Mirrors the seven entities the Development Plan specifies for the backend database:
customers, terminals, transactions, transaction_features, risk_scores, alerts,
audit_logs. This module defines structure only -- no rows are written here (database
population is a later, separately-approved step) and no Phase 1-7 computation is
duplicated: transaction_features/risk_scores columns are storage targets for values
already computed by mrs.features / mrs.models / mrs.behavioral / mrs.risk, not
recomputed here.

Design notes -- decisions the Plan leaves open, made explicitly here rather than
silently:

- transaction_features.features is one JSONB column keyed by feature name, not 33
  individual typed columns. mrs.features.registry.FEATURE_SPECS is already the single
  source of truth for the feature contract (round-trip enforced by
  tests/test_feature_registry.py); mirroring all 33 names as SQLAlchemy columns would
  create a second contract that can silently drift from the first. JSONB keeps one
  source of truth while remaining independently queryable/indexable if ever needed.
- risk_scores denormalizes customer_id/terminal_id (in addition to its transaction_id
  FK/PK) so behavioral-risk queries ("this customer's risk history", "this terminal's
  risk history") do not require joining through transactions every time -- a read-path
  optimization, not new business logic; the values must always match the referenced
  transaction's own customer_id/terminal_id.
- No ON DELETE CASCADE anywhere. The Handbook dataset is a frozen, immutable historical
  record (Dev Plan §33.5); nothing in this system is designed to delete transactions,
  so foreign keys use the default RESTRICT rather than silently cascading deletes.
- alerts.transaction_id is UNIQUE: at most one alert per transaction, matching a
  policy engine invoked once per scored transaction (policy engine itself is a later
  step; this is a structural constraint, not an implementation of it).
- audit_logs is intentionally the most flexible table (a JSONB payload keyed by
  event_type) since Dev Plan §33.9 lists several distinct kinds of things it must be
  able to record (risk output, evidence, AI explanation, policy decision,
  errors/fallbacks) that later phases (9/10) will define -- schema-level flexibility,
  not an excuse to skip structure elsewhere.
"""

from __future__ import annotations

import datetime as dt

from sqlalchemy import (
    BigInteger,
    Float,
    ForeignKey,
    Index,
    Integer,
    SmallInteger,
    String,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from sqlalchemy.orm import Mapped, mapped_column

from mrs.db.base import Base


class Customer(Base):
    """Customer profile (Dev Plan §20 'customers'; §4.1).

    Source: data/reference/customer_profiles.parquet (Phase 1, unmodified).
    """

    __tablename__ = "customers"

    customer_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    x_customer_id: Mapped[float] = mapped_column(Float, nullable=False)
    y_customer_id: Mapped[float] = mapped_column(Float, nullable=False)
    mean_amount: Mapped[float] = mapped_column(Float, nullable=False)
    std_amount: Mapped[float] = mapped_column(Float, nullable=False)
    mean_nb_tx_per_day: Mapped[float] = mapped_column(Float, nullable=False)
    nb_terminals: Mapped[int] = mapped_column(Integer, nullable=False)
    available_terminals: Mapped[list[int]] = mapped_column(ARRAY(Integer), nullable=False)
    created_at: Mapped[dt.datetime] = mapped_column(server_default=func.now(), nullable=False)


class Terminal(Base):
    """Terminal / payment-acceptance entity (Dev Plan §20 'terminals'; §4.2).

    Source: data/reference/terminal_profiles.parquet (Phase 1, unmodified).
    """

    __tablename__ = "terminals"

    terminal_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    x_terminal_id: Mapped[float] = mapped_column(Float, nullable=False)
    y_terminal_id: Mapped[float] = mapped_column(Float, nullable=False)
    created_at: Mapped[dt.datetime] = mapped_column(server_default=func.now(), nullable=False)


class Transaction(Base):
    """Raw transaction record (Dev Plan §20 'transactions'; §4.3 column set).

    tx_fraud / tx_fraud_scenario are stored here as an immutable historical record only
    (Dev Plan §34.3: ground truth for evaluation, demo, and audit display). Nothing in
    mrs.models/mrs.features reads this table at all -- they operate on the Parquet
    feature layer -- so this column can never leak into a live scoring path; it exists
    for alert-investigation/replay display and post-hoc scenario evaluation only.
    """

    __tablename__ = "transactions"

    transaction_id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    tx_datetime: Mapped[dt.datetime] = mapped_column(nullable=False)
    customer_id: Mapped[int] = mapped_column(ForeignKey("customers.customer_id"), nullable=False)
    terminal_id: Mapped[int] = mapped_column(ForeignKey("terminals.terminal_id"), nullable=False)
    tx_amount: Mapped[float] = mapped_column(Float, nullable=False)
    tx_time_seconds: Mapped[int] = mapped_column(BigInteger, nullable=False)
    tx_time_days: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    tx_fraud: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    tx_fraud_scenario: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    split: Mapped[str] = mapped_column(String(16), nullable=False)
    created_at: Mapped[dt.datetime] = mapped_column(server_default=func.now(), nullable=False)

    __table_args__ = (
        # Chronological replay (Dev Plan §22/§39) reads this table ordered by
        # tx_datetime -- the index this ordering depends on.
        Index("ix_transactions_tx_datetime", "tx_datetime"),
        Index("ix_transactions_customer_id", "customer_id"),
        Index("ix_transactions_terminal_id", "terminal_id"),
    )


class TransactionFeatures(Base):
    """Persisted Phase 3 feature vector for one transaction (Dev Plan §20
    'transaction_features'). Storage only -- see module docstring for the JSONB choice.
    """

    __tablename__ = "transaction_features"

    transaction_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("transactions.transaction_id"), primary_key=True
    )
    feature_version: Mapped[str] = mapped_column(String(32), nullable=False)
    features: Mapped[dict] = mapped_column(JSONB, nullable=False)
    created_at: Mapped[dt.datetime] = mapped_column(server_default=func.now(), nullable=False)


class RiskScore(Base):
    """Persisted Phase 5+6+7 unified risk result for one transaction (Dev Plan §20
    'risk_scores'; §38 'keep component scores separately'; §36 governance/lineage).

    One row per transaction_id: the materialized output of
    mrs.risk.aggregate.aggregate_risk (OUTPUT_COLUMNS), plus the Phase 5
    model_version/threshold and Phase 3 feature_version it was produced from, so a
    score remains reconstructable later (Dev Plan §36).
    """

    __tablename__ = "risk_scores"

    transaction_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("transactions.transaction_id"), primary_key=True
    )
    customer_id: Mapped[int] = mapped_column(ForeignKey("customers.customer_id"), nullable=False)
    terminal_id: Mapped[int] = mapped_column(ForeignKey("terminals.terminal_id"), nullable=False)

    # transaction_risk / *_severity / *_state are nullable: an "unavailable" component
    # (Dev Plan §28 -- absence of evidence is never treated as evidence of calm) is a
    # real, expected value here, not a data-quality defect. See
    # mrs.risk.aggregate._is_missing.
    transaction_risk: Mapped[float | None] = mapped_column(Float, nullable=True)
    transaction_risk_severity: Mapped[int | None] = mapped_column(SmallInteger, nullable=True)
    terminal_risk_state: Mapped[str | None] = mapped_column(String(24), nullable=True)
    terminal_risk_severity: Mapped[int | None] = mapped_column(SmallInteger, nullable=True)
    customer_risk_state: Mapped[str | None] = mapped_column(String(24), nullable=True)
    customer_risk_severity: Mapped[int | None] = mapped_column(SmallInteger, nullable=True)

    unified_risk_level: Mapped[str] = mapped_column(String(24), nullable=False)
    contributing_signals: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)

    model_version: Mapped[str] = mapped_column(String(32), nullable=False)
    transaction_risk_threshold: Mapped[float] = mapped_column(Float, nullable=False)
    feature_version: Mapped[str] = mapped_column(String(32), nullable=False)

    computed_at: Mapped[dt.datetime] = mapped_column(server_default=func.now(), nullable=False)

    __table_args__ = (
        Index("ix_risk_scores_unified_risk_level", "unified_risk_level"),
        Index("ix_risk_scores_customer_id", "customer_id"),
        Index("ix_risk_scores_terminal_id", "terminal_id"),
    )


class Alert(Base):
    """Alert derived from a risk score by the policy engine (Dev Plan §15/§20 'alerts').

    The policy engine itself does not exist yet (a later Phase 8 step); this table is
    structure only. recommended_action is nullable for the same reason.
    """

    __tablename__ = "alerts"

    alert_id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    transaction_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("risk_scores.transaction_id"), nullable=False
    )
    customer_id: Mapped[int] = mapped_column(ForeignKey("customers.customer_id"), nullable=False)
    terminal_id: Mapped[int] = mapped_column(ForeignKey("terminals.terminal_id"), nullable=False)

    severity: Mapped[str] = mapped_column(String(24), nullable=False)
    reason: Mapped[str] = mapped_column(String(255), nullable=False)
    evidence: Mapped[dict] = mapped_column(JSONB, nullable=False)
    #: One of the Dev Plan §15 bounded actions (ALLOW/MONITOR/STEP_UP_VERIFICATION/
    #: TEMPORARY_REVIEW/ESCALATE) once the policy engine exists. Not a DB-level enum so
    #: the action set stays defined in one place (the future policy engine module), not
    #: duplicated as a Postgres type.
    recommended_action: Mapped[str | None] = mapped_column(String(32), nullable=True)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="OPEN")

    created_at: Mapped[dt.datetime] = mapped_column(server_default=func.now(), nullable=False)

    __table_args__ = (
        UniqueConstraint("transaction_id", name="uq_alerts_transaction_id"),
        Index("ix_alerts_status", "status"),
        Index("ix_alerts_severity", "severity"),
        Index("ix_alerts_customer_id", "customer_id"),
        Index("ix_alerts_terminal_id", "terminal_id"),
    )


class AuditLog(Base):
    """Append-only audit trail (Dev Plan §20 'audit_logs'; §33.9 observability).

    alert_id is nullable: not every audited event corresponds to an alert (e.g. a
    routine RISK_SCORED event for a LOW-risk transaction).
    """

    __tablename__ = "audit_logs"

    audit_id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    transaction_id: Mapped[int | None] = mapped_column(
        BigInteger, ForeignKey("transactions.transaction_id"), nullable=True
    )
    alert_id: Mapped[int | None] = mapped_column(BigInteger, ForeignKey("alerts.alert_id"), nullable=True)
    event_type: Mapped[str] = mapped_column(String(32), nullable=False)
    payload: Mapped[dict] = mapped_column(JSONB, nullable=False)
    model_version: Mapped[str | None] = mapped_column(String(32), nullable=True)

    created_at: Mapped[dt.datetime] = mapped_column(server_default=func.now(), nullable=False)

    __table_args__ = (
        Index("ix_audit_logs_created_at", "created_at"),
        Index("ix_audit_logs_transaction_id", "transaction_id"),
    )
