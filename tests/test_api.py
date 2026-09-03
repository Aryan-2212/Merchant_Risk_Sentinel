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
    body = resp.json()
    assert body["status"] == "ok"
    assert isinstance(body["ai_analyst_configured"], bool)


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


def test_get_transaction_audit(db_engine, client):
    _seed_full(db_engine)
    resp = client.get("/transactions/100/audit")
    assert resp.status_code == 200
    body = resp.json()
    assert len(body) == 1
    assert body[0]["event_type"] == "POLICY_DECISION"
    assert body[0]["payload"]["action"] == "ESCALATE"
    assert body[0]["transaction_id"] == 100


def test_get_transaction_audit_empty_when_no_policy_decision_yet(db_engine, client):
    with db_engine.begin() as conn:
        conn.execute(
            insert(Customer.__table__),
            [
                {
                    "customer_id": 3,
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
        conn.execute(insert(Terminal.__table__), [{"terminal_id": 3, "x_terminal_id": 0.0, "y_terminal_id": 0.0}])
        conn.execute(
            insert(Transaction.__table__),
            [
                {
                    "transaction_id": 300,
                    "tx_datetime": dt.datetime(2018, 4, 1, 0, 0, 0),
                    "customer_id": 3,
                    "terminal_id": 3,
                    "tx_amount": 5.0,
                    "tx_time_seconds": 0,
                    "tx_time_days": 0,
                    "tx_fraud": 0,
                    "tx_fraud_scenario": 0,
                    "split": "train",
                }
            ],
        )
    resp = client.get("/transactions/300/audit")
    assert resp.status_code == 200
    assert resp.json() == []


def test_get_transaction_audit_not_found(client):
    resp = client.get("/transactions/999999/audit")
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


def test_get_customer_deviation(db_engine, client):
    _seed_full(db_engine)
    resp = client.get("/customers/1/deviation")
    assert resp.status_code == 200
    body = resp.json()
    assert body["entity_type"] == "customer"
    assert body["entity_id"] == 1
    # The seeded transaction (customer_risk_severity=0) falls inside its own "recent"
    # window; no prior-window transactions exist, so baseline is None, never 0.0.
    assert body["current_rate"] == 0.0
    assert body["current_transaction_count"] == 1
    assert body["baseline_rate"] is None
    assert body["baseline_transaction_count"] == 0
    assert body["recent_window_days"] == 7
    assert body["baseline_window_days"] == 30


def test_get_customer_deviation_not_found(client):
    resp = client.get("/customers/999999/deviation")
    assert resp.status_code == 404


def test_get_customer_deviation_no_transactions(db_engine, client):
    with db_engine.begin() as conn:
        conn.execute(
            insert(Customer.__table__),
            [
                {
                    "customer_id": 5,
                    "x_customer_id": 0.0,
                    "y_customer_id": 0.0,
                    "mean_amount": 50.0,
                    "std_amount": 10.0,
                    "mean_nb_tx_per_day": 2.0,
                    "nb_terminals": 1,
                    "available_terminals": [1],
                }
            ],
        )
    resp = client.get("/customers/5/deviation")
    assert resp.status_code == 200
    body = resp.json()
    assert body["current_rate"] is None
    assert body["baseline_rate"] is None
    assert body["current_transaction_count"] == 0


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


def test_get_terminal_deviation(db_engine, client):
    _seed_full(db_engine)
    resp = client.get("/terminals/1/deviation")
    assert resp.status_code == 200
    body = resp.json()
    assert body["entity_type"] == "terminal"
    assert body["entity_id"] == 1
    # The seeded transaction (terminal_risk_severity=2) falls inside its own "recent"
    # window; no prior-window transactions exist, so baseline is None, never 0.0.
    assert body["current_rate"] == 1.0
    assert body["current_transaction_count"] == 1
    assert body["baseline_rate"] is None
    assert body["baseline_transaction_count"] == 0
    assert body["recent_window_days"] == 7
    assert body["baseline_window_days"] == 30


def test_get_terminal_deviation_not_found(client):
    resp = client.get("/terminals/999999/deviation")
    assert resp.status_code == 404


def test_get_terminal_deviation_no_transactions(db_engine, client):
    with db_engine.begin() as conn:
        conn.execute(insert(Terminal.__table__), [{"terminal_id": 5, "x_terminal_id": 0.0, "y_terminal_id": 0.0}])
    resp = client.get("/terminals/5/deviation")
    assert resp.status_code == 200
    body = resp.json()
    assert body["current_rate"] is None
    assert body["baseline_rate"] is None
    assert body["current_transaction_count"] == 0


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


# ------------------------------------------------------------------------------ stats


def test_get_overview_stats(db_engine, client):
    _seed_full(db_engine)
    resp = client.get("/stats/overview")
    assert resp.status_code == 200
    body = resp.json()
    assert body["total_transactions"] == 1
    assert body["total_customers"] == 1
    assert body["total_terminals"] == 1
    assert body["total_risk_scores"] == 1
    assert body["total_alerts"] == 1
    assert body["risk_level_counts"] == {"CRITICAL": 1}
    assert body["alert_action_counts"] == {"ESCALATE": 1}
    assert body["alert_status_counts"] == {"OPEN": 1}
    # terminal 1 is HIGH_RISK (at risk); customer 1 is NORMAL (not at risk).
    assert body["terminals_at_risk"] == 1
    assert body["customers_at_risk"] == 0
    # Only transaction 100 (CRITICAL, tx_amount 999.0) counts toward exposure.
    assert body["risk_exposure_amount"] == 999.0


def test_get_overview_stats_empty_database(client):
    resp = client.get("/stats/overview")
    assert resp.status_code == 200
    body = resp.json()
    assert body["total_transactions"] == 0
    assert body["risk_level_counts"] == {}
    assert body["alert_action_counts"] == {}
    assert body["customers_at_risk"] == 0
    assert body["terminals_at_risk"] == 0
    assert body["risk_exposure_amount"] == 0.0


def test_get_risk_activity_buckets_by_day(db_engine, client):
    _seed_full(db_engine)
    resp = client.get("/stats/risk-activity", params={"days": 30})
    assert resp.status_code == 200
    body = resp.json()
    assert len(body) == 1
    assert body[0]["date"] == "2018-04-01"
    assert body[0]["transaction_high"] == 1
    assert body[0]["terminal_high"] == 1
    assert body[0]["customer_high"] == 0
    assert body[0]["elevated_transactions"] == 1
    assert body[0]["total_scored"] == 1


def test_get_risk_activity_empty_database(client):
    resp = client.get("/stats/risk-activity")
    assert resp.status_code == 200
    assert resp.json() == []


def test_get_risk_activity_days_validated(client):
    assert client.get("/stats/risk-activity", params={"days": 0}).status_code == 422
    assert client.get("/stats/risk-activity", params={"days": 184}).status_code == 422


def test_get_recent_activity_includes_low_risk_not_just_alerts(db_engine, client):
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
                    "tx_datetime": dt.datetime(2018, 4, 2, 0, 0, 0),
                    "customer_id": 2,
                    "terminal_id": 2,
                    "tx_amount": 5.0,
                    "tx_time_seconds": 0,
                    "tx_time_days": 1,
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

    resp = client.get("/stats/recent-activity", params={"limit": 5})
    assert resp.status_code == 200
    body = resp.json()
    # Most recent (transaction 200) first -- and it has no alert, unlike GET /alerts.
    assert body[0]["transaction"]["transaction_id"] == 200
    assert body[0]["risk_score"]["unified_risk_level"] == "LOW"
    assert body[0]["alert"] is None


def test_get_recent_activity_limit_validated(client):
    assert client.get("/stats/recent-activity", params={"limit": 0}).status_code == 422
    assert client.get("/stats/recent-activity", params={"limit": 101}).status_code == 422


def test_get_recent_activity_filters_by_level(db_engine, client):
    _seed_full(db_engine)  # transaction 100, CRITICAL
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
                    "tx_datetime": dt.datetime(2018, 4, 2, 0, 0, 0),
                    "customer_id": 2,
                    "terminal_id": 2,
                    "tx_amount": 5.0,
                    "tx_time_seconds": 0,
                    "tx_time_days": 1,
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

    # Unfiltered: both transactions, most recent (200) first.
    resp = client.get("/stats/recent-activity")
    ids = [item["transaction"]["transaction_id"] for item in resp.json()]
    assert ids == [200, 100]

    # levels=CRITICAL: only transaction 100, even though it's not the most recent.
    resp = client.get("/stats/recent-activity", params={"levels": "CRITICAL"})
    body = resp.json()
    assert len(body) == 1
    assert body[0]["transaction"]["transaction_id"] == 100

    resp = client.get("/stats/recent-activity", params={"levels": "HIGH,CRITICAL"})
    assert [item["transaction"]["transaction_id"] for item in resp.json()] == [100]

    resp = client.get("/stats/recent-activity", params={"levels": "LOW"})
    assert [item["transaction"]["transaction_id"] for item in resp.json()] == [200]


def test_get_entity_network_default_focus(db_engine, client):
    _seed_full(db_engine)
    resp = client.get("/stats/network")
    assert resp.status_code == 200
    body = resp.json()
    assert "terminal:1" in body["focus_ids"]
    node_ids = {n["id"] for n in body["nodes"]}
    assert "terminal:1" in node_ids
    assert "customer:1" in node_ids  # real neighbor, via transaction 100
    terminal_node = next(n for n in body["nodes"] if n["id"] == "terminal:1")
    assert terminal_node["is_focus"] is True
    assert terminal_node["risk_state"] == "HIGH_RISK"
    customer_node = next(n for n in body["nodes"] if n["id"] == "customer:1")
    assert customer_node["is_focus"] is False
    assert any(e["source"] == "customer:1" and e["target"] == "terminal:1" for e in body["edges"])


def test_get_entity_network_explicit_focus(db_engine, client):
    _seed_full(db_engine)
    resp = client.get("/stats/network", params={"focus_type": "customer", "focus_id": 1})
    assert resp.status_code == 200
    body = resp.json()
    assert body["focus_ids"] == ["customer:1"]
    node_ids = {n["id"] for n in body["nodes"]}
    assert "customer:1" in node_ids
    assert "terminal:1" in node_ids


def test_get_entity_network_invalid_focus_type_rejected(client):
    resp = client.get("/stats/network", params={"focus_type": "merchant", "focus_id": 1})
    assert resp.status_code == 422


def test_get_entity_network_empty_database(client):
    resp = client.get("/stats/network")
    assert resp.status_code == 200
    body = resp.json()
    assert body["nodes"] == []
    assert body["edges"] == []
    assert body["focus_ids"] == []


def test_get_terminals_at_risk(db_engine, client):
    _seed_full(db_engine)
    resp = client.get("/stats/terminals-at-risk", params={"limit": 8})
    assert resp.status_code == 200
    body = resp.json()
    assert len(body) == 1
    row = body[0]
    assert row["entity_type"] == "terminal"
    assert row["entity_id"] == 1
    assert row["risk_state"] == "HIGH_RISK"
    assert row["risk_severity"] == 2
    # The seeded transaction falls inside its own "recent" window; no prior-window
    # transactions exist at all, so baseline_rate is None (never fabricated as 0).
    assert row["current_rate"] == 1.0
    assert row["baseline_rate"] is None
    assert row["recent_transaction_count"] == 1


def test_get_terminals_at_risk_empty_database(client):
    resp = client.get("/stats/terminals-at-risk")
    assert resp.status_code == 200
    assert resp.json() == []


def test_get_terminals_at_risk_limit_validated(client):
    assert client.get("/stats/terminals-at-risk", params={"limit": 0}).status_code == 422
    assert client.get("/stats/terminals-at-risk", params={"limit": 26}).status_code == 422
