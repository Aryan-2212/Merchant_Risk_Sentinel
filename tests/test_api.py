"""Tests for mrs.api -- Phase 8 Step 4 read API.

Uses the isolated test database (tests/conftest.py's db_engine fixture) via FastAPI's
dependency_overrides -- the app under test never touches MRS_DATABASE_URL / the real
database. No test in this module writes to or reads from merchant_risk_sentinel.
"""

from __future__ import annotations

import datetime as dt

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import insert
from sqlalchemy.orm import sessionmaker

from mrs.api.deps import get_db
from mrs.api.main import app
from mrs.db.models import Alert, AuditLog, Customer, RiskScore, Terminal, Transaction


@pytest.fixture()
def client(db_engine):
    Session = sessionmaker(bind=db_engine)

    def override_get_db():
        session = Session()
        try:
            yield session
        finally:
            session.close()

    app.dependency_overrides[get_db] = override_get_db
    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.pop(get_db, None)


def _seed_full(engine) -> None:
    with engine.begin() as conn:
        conn.execute(
            insert(Customer.__table__),
            [
                {
                    "customer_id": 1,
                    "x_customer_id": 1.0,
                    "y_customer_id": 1.0,
                    "mean_amount": 50.0,
                    "std_amount": 10.0,
                    "mean_nb_tx_per_day": 2.0,
                    "nb_terminals": 1,
                    "available_terminals": [1],
                }
            ],
        )
        conn.execute(insert(Terminal.__table__), [{"terminal_id": 1, "x_terminal_id": 5.0, "y_terminal_id": 5.0}])
        conn.execute(
            insert(Transaction.__table__),
            [
                {
                    "transaction_id": 100,
                    "tx_datetime": dt.datetime(2018, 4, 1, 12, 0, 0),
                    "customer_id": 1,
                    "terminal_id": 1,
                    "tx_amount": 999.0,
                    "tx_time_seconds": 0,
                    "tx_time_days": 0,
                    "tx_fraud": 1,
                    "tx_fraud_scenario": 1,
                    "split": "train",
                }
            ],
        )
        conn.execute(
            insert(RiskScore.__table__),
            [
                {
                    "transaction_id": 100,
                    "customer_id": 1,
                    "terminal_id": 1,
                    "transaction_risk": 0.99,
                    "transaction_risk_severity": 2,
                    "terminal_risk_state": "HIGH_RISK",
                    "terminal_risk_severity": 2,
                    "customer_risk_state": "NORMAL",
                    "customer_risk_severity": 0,
                    "unified_risk_level": "CRITICAL",
                    "contributing_signals": ["transaction_ml_risk >= 0.97", "terminal_behavioral_risk: HIGH_RISK"],
                    "model_version": "xgboost_v1",
                    "transaction_risk_threshold": 0.97,
                    "feature_version": "phase3_v1",
                }
            ],
        )
        conn.execute(
            insert(Alert.__table__),
            [
                {
                    "transaction_id": 100,
                    "customer_id": 1,
                    "terminal_id": 1,
                    "severity": "CRITICAL",
                    "reason": "transaction_ml_risk >= 0.97; terminal_behavioral_risk: HIGH_RISK",
                    "evidence": {"unified_risk_level": "CRITICAL"},
                    "recommended_action": "ESCALATE",
                    "status": "OPEN",
                }
            ],
        )
        conn.execute(
            insert(AuditLog.__table__),
            [
                {
                    "transaction_id": 100,
                    "alert_id": None,
                    "event_type": "POLICY_DECISION",
                    "payload": {"policy_version": "policy_v1", "action": "ESCALATE"},
                    "model_version": "xgboost_v1",
                }
            ],
        )


# --------------------------------------------------------------------------- health


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


# ---------------------------------------------------------------------- transactions


def test_get_transaction_detail(db_engine, client):
    _seed_full(db_engine)
    resp = client.get("/transactions/100")
    assert resp.status_code == 200
    body = resp.json()
    assert body["transaction"]["transaction_id"] == 100
    assert body["transaction"]["tx_amount"] == 999.0
    assert "tx_fraud" not in body["transaction"]
    assert "tx_fraud_scenario" not in body["transaction"]
    assert body["risk_score"]["unified_risk_level"] == "CRITICAL"
    assert body["alert"]["severity"] == "CRITICAL"
    assert body["alert"]["recommended_action"] == "ESCALATE"
    assert body["policy_version"] == "policy_v1"
    assert body["alert"]["policy_version"] == "policy_v1"


def test_get_transaction_not_found(client):
    resp = client.get("/transactions/999999")
    assert resp.status_code == 404
    assert "999999" in resp.json()["detail"]


def test_get_transaction_invalid_id_type_returns_422(client):
    resp = client.get("/transactions/not-an-int")
    assert resp.status_code == 422


def test_get_transaction_risk(db_engine, client):
    _seed_full(db_engine)
    resp = client.get("/transactions/100/risk")
    assert resp.status_code == 200
    body = resp.json()
    assert body["transaction_risk"] == 0.99
    assert body["contributing_signals"] == ["transaction_ml_risk >= 0.97", "terminal_behavioral_risk: HIGH_RISK"]
    assert body["model_version"] == "xgboost_v1"
    assert body["feature_version"] == "phase3_v1"


def test_get_transaction_risk_not_found(client):
    resp = client.get("/transactions/999999/risk")
    assert resp.status_code == 404


def test_transaction_with_no_alert_has_null_alert_and_policy_version(db_engine, client):
    with db_engine.begin() as conn:
        conn.execute(
            insert(Customer.__table__),
            [
                {
                    "customer_id": 2,
                    "x_customer_id": 0.0,
                    "y_customer_id": 0.0,
                    "mean_amount": 10.0,
                    "std_amount": 2.0,
                    "mean_nb_tx_per_day": 1.0,
                    "nb_terminals": 1,
                    "available_terminals": [1],
                }
            ],
        )
        conn.execute(insert(Terminal.__table__), [{"terminal_id": 2, "x_terminal_id": 0.0, "y_terminal_id": 0.0}])
        conn.execute(
            insert(Transaction.__table__),
            [
                {
                    "transaction_id": 200,
                    "tx_datetime": dt.datetime(2018, 4, 1, 0, 0, 0),
                    "customer_id": 2,
                    "terminal_id": 2,
                    "tx_amount": 5.0,
                    "tx_time_seconds": 0,
                    "tx_time_days": 0,
                    "tx_fraud": 0,
                    "tx_fraud_scenario": 0,
                    "split": "train",
                }
            ],
        )
        conn.execute(
            insert(RiskScore.__table__),
            [
                {
                    "transaction_id": 200,
                    "customer_id": 2,
                    "terminal_id": 2,
                    "transaction_risk": 0.01,
                    "transaction_risk_severity": 0,
                    "terminal_risk_state": "NORMAL",
                    "terminal_risk_severity": 0,
                    "customer_risk_state": "NORMAL",
                    "customer_risk_severity": 0,
                    "unified_risk_level": "LOW",
                    "contributing_signals": [],
                    "model_version": "xgboost_v1",
                    "transaction_risk_threshold": 0.97,
                    "feature_version": "phase3_v1",
                }
            ],
        )

    resp = client.get("/transactions/200")
    assert resp.status_code == 200
    body = resp.json()
    assert body["alert"] is None
    assert body["risk_score"]["unified_risk_level"] == "LOW"
    # No POLICY_DECISION audit_log row was seeded for this transaction.
    assert body["policy_version"] is None


# -------------------------------------------------------------------------- customers


def test_get_customer(db_engine, client):
    _seed_full(db_engine)
    resp = client.get("/customers/1")
    assert resp.status_code == 200
    assert resp.json()["customer_id"] == 1
    assert resp.json()["available_terminals"] == [1]


def test_get_customer_not_found(client):
    resp = client.get("/customers/999999")
    assert resp.status_code == 404


def test_get_customer_risk_history(db_engine, client):
    _seed_full(db_engine)
    resp = client.get("/customers/1/risk")
    assert resp.status_code == 200
    body = resp.json()
    assert body["total"] == 1
    assert body["limit"] == 50
    assert body["offset"] == 0
    assert len(body["items"]) == 1
    assert body["items"][0]["unified_risk_level"] == "CRITICAL"


def test_get_customer_risk_history_not_found_customer(client):
    resp = client.get("/customers/999999/risk")
    assert resp.status_code == 404


# -------------------------------------------------------------------------- terminals


def test_get_terminal(db_engine, client):
    _seed_full(db_engine)
    resp = client.get("/terminals/1")
    assert resp.status_code == 200
    assert resp.json()["terminal_id"] == 1


def test_get_terminal_not_found(client):
    resp = client.get("/terminals/999999")
    assert resp.status_code == 404


def test_get_terminal_risk_history(db_engine, client):
    _seed_full(db_engine)
    resp = client.get("/terminals/1/risk")
    assert resp.status_code == 200
    assert resp.json()["total"] == 1


# ----------------------------------------------------------------------------- alerts


def test_list_alerts(db_engine, client):
    _seed_full(db_engine)
    resp = client.get("/alerts")
    assert resp.status_code == 200
    body = resp.json()
    assert body["total"] == 1
    assert body["items"][0]["severity"] == "CRITICAL"
    # Summary schema omits full evidence/reason.
    assert "evidence" not in body["items"][0]
    assert "reason" not in body["items"][0]


def test_list_alerts_filter_by_severity(db_engine, client):
    _seed_full(db_engine)
    resp = client.get("/alerts", params={"severity": "CRITICAL"})
    assert resp.json()["total"] == 1

    resp = client.get("/alerts", params={"severity": "MEDIUM"})
    assert resp.json()["total"] == 0


def test_list_alerts_invalid_severity_rejected(client):
    resp = client.get("/alerts", params={"severity": "LOW"})  # never a valid alert severity
    assert resp.status_code == 422


def test_list_alerts_filter_by_customer_and_terminal(db_engine, client):
    _seed_full(db_engine)
    assert client.get("/alerts", params={"customer_id": 1}).json()["total"] == 1
    assert client.get("/alerts", params={"customer_id": 999999}).json()["total"] == 0
    assert client.get("/alerts", params={"terminal_id": 1}).json()["total"] == 1


def test_get_alert_detail_includes_evidence_and_policy_version(db_engine, client):
    _seed_full(db_engine)
    alert_id = client.get("/alerts").json()["items"][0]["alert_id"]

    resp = client.get(f"/alerts/{alert_id}")
    assert resp.status_code == 200
    body = resp.json()
    assert body["evidence"] == {"unified_risk_level": "CRITICAL"}
    assert body["reason"] == "transaction_ml_risk >= 0.97; terminal_behavioral_risk: HIGH_RISK"
    assert body["policy_version"] == "policy_v1"


def test_get_alert_not_found(client):
    resp = client.get("/alerts/999999")
    assert resp.status_code == 404


def test_pagination_limit_and_offset_validated(client):
    assert client.get("/alerts", params={"limit": 0}).status_code == 422
    assert client.get("/alerts", params={"limit": 501}).status_code == 422
    assert client.get("/alerts", params={"offset": -1}).status_code == 422
