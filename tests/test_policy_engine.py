"""Tests for mrs.policy.engine -- Phase 8 Step 3 policy persistence (Dev Plan §15/§20).

Live-Postgres integration tests only (require_database fixture, the isolated
merchant_risk_sentinel_test database via tests/conftest.py's db_engine fixture --
never the real merchant_risk_sentinel database). A tiny synthetic dataset (a handful
of customers/terminals/transactions/risk_scores rows), not the real 1.75M-row dataset.
"""

from __future__ import annotations

import datetime as dt

import pytest
from sqlalchemy import insert, select

from mrs.db.models import Alert, AuditLog, Customer, RiskScore, Terminal, Transaction
from mrs.policy.engine import already_decided_transaction_ids, apply_policy
from mrs.policy.rules import ALLOW, CRITICAL, ESCALATE, HIGH, INSUFFICIENT_EVIDENCE, LOW, MEDIUM, MONITOR, POLICY_VERSION, STEP_UP_VERIFICATION, TEMPORARY_REVIEW  # noqa: E501


def _seed(session_conn, rows: list[dict]) -> None:
    """rows: list of dicts with customer_id/terminal_id/transaction_id plus the
    risk_scores fields to seed. Creates the minimal parent rows each risk_scores row's
    foreign keys require."""
    customer_ids = sorted({r["customer_id"] for r in rows})
    terminal_ids = sorted({r["terminal_id"] for r in rows})

    session_conn.execute(
        insert(Customer.__table__),
        [
            {
                "customer_id": cid,
                "x_customer_id": 0.0,
                "y_customer_id": 0.0,
                "mean_amount": 50.0,
                "std_amount": 10.0,
                "mean_nb_tx_per_day": 1.0,
                "nb_terminals": 1,
                "available_terminals": [terminal_ids[0]],
            }
            for cid in customer_ids
        ],
    )
    session_conn.execute(
        insert(Terminal.__table__),
        [{"terminal_id": tid, "x_terminal_id": 0.0, "y_terminal_id": 0.0} for tid in terminal_ids],
    )
    session_conn.execute(
        insert(Transaction.__table__),
        [
            {
                "transaction_id": r["transaction_id"],
                "tx_datetime": dt.datetime(2018, 4, 1, 0, 0, 0),
                "customer_id": r["customer_id"],
                "terminal_id": r["terminal_id"],
                "tx_amount": 10.0,
                "tx_time_seconds": 0,
                "tx_time_days": 0,
                "tx_fraud": 0,
                "tx_fraud_scenario": 0,
                "split": "train",
            }
            for r in rows
        ],
    )
    session_conn.execute(
        insert(RiskScore.__table__),
        [
            {
                "transaction_id": r["transaction_id"],
                "customer_id": r["customer_id"],
                "terminal_id": r["terminal_id"],
                "transaction_risk": r.get("transaction_risk"),
                "transaction_risk_severity": r.get("transaction_risk_severity"),
                "terminal_risk_state": r.get("terminal_risk_state"),
                "terminal_risk_severity": r.get("terminal_risk_severity"),
                "customer_risk_state": r.get("customer_risk_state"),
                "customer_risk_severity": r.get("customer_risk_severity"),
                "unified_risk_level": r["unified_risk_level"],
                "contributing_signals": r.get("contributing_signals", []),
                "model_version": "xgboost_v1",
                "transaction_risk_threshold": 0.97,
                "feature_version": "phase3_v1",
            }
            for r in rows
        ],
    )


FOUR_LEVEL_ROWS = [
    {
        "transaction_id": 1,
        "customer_id": 1,
        "terminal_id": 1,
        "unified_risk_level": LOW,
        "transaction_risk_severity": 0,
        "terminal_risk_severity": 0,
        "customer_risk_severity": 0,
        "contributing_signals": [],
    },
    {
        "transaction_id": 2,
        "customer_id": 2,
        "terminal_id": 1,
        "unified_risk_level": MEDIUM,
        "terminal_risk_state": "RISK_RISING",
        "terminal_risk_severity": 1,
        "contributing_signals": ["terminal_behavioral_risk: RISK_RISING"],
    },
    {
        "transaction_id": 3,
        "customer_id": 3,
        "terminal_id": 1,
        "unified_risk_level": HIGH,
        "transaction_risk": 0.99,
        "transaction_risk_severity": 2,
        "contributing_signals": ["transaction_ml_risk >= 0.97"],
    },
    {
        "transaction_id": 4,
        "customer_id": 4,
        "terminal_id": 1,
        "unified_risk_level": CRITICAL,
        "transaction_risk_severity": 2,
        "terminal_risk_state": "HIGH_RISK",
        "terminal_risk_severity": 2,
        "contributing_signals": ["transaction_ml_risk >= 0.97", "terminal_behavioral_risk: HIGH_RISK"],
    },
    {
        "transaction_id": 5,
        "customer_id": 5,
        "terminal_id": 1,
        "unified_risk_level": INSUFFICIENT_EVIDENCE,
        "customer_risk_state": "INSUFFICIENT_HISTORY",
        "contributing_signals": [],
    },
]


def test_apply_policy_writes_alerts_for_every_level_except_low(db_engine):
    with db_engine.begin() as conn:
        _seed(conn, FOUR_LEVEL_ROWS)

    summary = apply_policy(db_engine)

    assert summary["n_rows_read"] == 5
    assert summary["n_newly_decided"] == 5
    assert summary["n_skipped_already_decided"] == 0
    assert summary["policy_version"] == POLICY_VERSION
    assert summary["action_counts"] == {
        ALLOW: 1,
        MONITOR: 1,
        STEP_UP_VERIFICATION: 1,
        ESCALATE: 1,
        TEMPORARY_REVIEW: 1,
    }
    assert summary["level_counts"] == {LOW: 1, MEDIUM: 1, HIGH: 1, CRITICAL: 1, INSUFFICIENT_EVIDENCE: 1}

    with db_engine.connect() as conn:
        alert_tx_ids = {row[0] for row in conn.execute(select(Alert.transaction_id))}
        audit_tx_ids = {row[0] for row in conn.execute(select(AuditLog.transaction_id))}

    # LOW (transaction_id=1, action=ALLOW) must NOT get an alert.
    assert alert_tx_ids == {2, 3, 4, 5}
    # Every transaction gets an audit_log entry, including LOW/ALLOW.
    assert audit_tx_ids == {1, 2, 3, 4, 5}


def test_alert_fields_are_correct_for_critical(db_engine):
    with db_engine.begin() as conn:
        _seed(conn, FOUR_LEVEL_ROWS)
    apply_policy(db_engine)

    with db_engine.connect() as conn:
        alert = conn.execute(select(Alert).where(Alert.transaction_id == 4)).mappings().one()

    assert alert["severity"] == CRITICAL
    assert alert["recommended_action"] == ESCALATE
    assert alert["status"] == "OPEN"
    assert "transaction_ml_risk >= 0.97" in alert["reason"]
    assert "terminal_behavioral_risk: HIGH_RISK" in alert["reason"]
    assert alert["evidence"]["unified_risk_level"] == CRITICAL


def test_audit_log_payload_carries_policy_version_and_action(db_engine):
    with db_engine.begin() as conn:
        _seed(conn, FOUR_LEVEL_ROWS)
    apply_policy(db_engine)

    with db_engine.connect() as conn:
        entry = conn.execute(select(AuditLog).where(AuditLog.transaction_id == 3)).mappings().one()

    assert entry["event_type"] == "POLICY_DECISION"
    assert entry["payload"]["policy_version"] == POLICY_VERSION
    assert entry["payload"]["action"] == STEP_UP_VERIFICATION
    assert entry["payload"]["unified_risk_level"] == HIGH


def test_rerun_is_idempotent_no_duplicate_alerts_or_audit_rows(db_engine):
    with db_engine.begin() as conn:
        _seed(conn, FOUR_LEVEL_ROWS)

    first = apply_policy(db_engine)
    assert first["n_newly_decided"] == 5
    assert first["n_alerts_written"] == 4

    second = apply_policy(db_engine)
    assert second["n_newly_decided"] == 0
    assert second["n_skipped_already_decided"] == 5
    assert second["n_alerts_written"] == 0
    assert second["n_audit_written"] == 0

    with db_engine.connect() as conn:
        alert_count = len(conn.execute(select(Alert)).fetchall())
        audit_count = len(conn.execute(select(AuditLog)).fetchall())

    assert alert_count == 4  # not 8
    assert audit_count == 5  # not 10


def test_already_decided_transaction_ids_reflects_prior_run(db_engine):
    with db_engine.begin() as conn:
        _seed(conn, FOUR_LEVEL_ROWS[:2])
    assert already_decided_transaction_ids(db_engine) == set()

    apply_policy(db_engine)
    assert already_decided_transaction_ids(db_engine) == {1, 2}


def test_alert_unique_constraint_backs_idempotency_even_on_direct_conflict(db_engine):
    """Belt-and-suspenders: even bypassing the audit_log skip-check, the DB's own
    UNIQUE constraint on alerts.transaction_id prevents a duplicate row (ON CONFLICT DO
    NOTHING), so idempotency does not rely solely on the application-level check."""
    with db_engine.begin() as conn:
        _seed(conn, FOUR_LEVEL_ROWS[2:3])  # transaction_id=3, HIGH
    apply_policy(db_engine)

    from sqlalchemy.dialects.postgresql import insert as pg_insert

    with db_engine.begin() as conn:
        stmt = pg_insert(Alert.__table__).values(
            transaction_id=3,
            customer_id=3,
            terminal_id=1,
            severity=HIGH,
            reason="duplicate attempt",
            evidence={},
            recommended_action=STEP_UP_VERIFICATION,
            status="OPEN",
        ).on_conflict_do_nothing(index_elements=["transaction_id"])
        conn.execute(stmt)

    with db_engine.connect() as conn:
        rows = conn.execute(select(Alert).where(Alert.transaction_id == 3)).fetchall()
    assert len(rows) == 1
    assert rows[0].reason != "duplicate attempt"
