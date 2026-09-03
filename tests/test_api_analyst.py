"""Tests for GET /transactions/{id}/analyst -- Phase 8 Step 6 (Dev Plan §16/§41).

No test in this module makes a real network call: mrs.api.routers.analyst.generate_explanation
is monkeypatched, exactly like tests/test_analyst.py does for the underlying module.
Uses the isolated test database (tests/conftest.py's db_engine fixture) via FastAPI's
dependency_overrides, exactly like tests/test_api.py.
"""

from __future__ import annotations

import datetime as dt

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import insert
from sqlalchemy.orm import sessionmaker

from mrs.analyst.schemas import AnalystExplanation, AnalystResult
from mrs.api.deps import get_db
from mrs.api.main import app
from mrs.db.models import Alert, Customer, RiskScore, Terminal, Transaction
from mrs.policy.rules import ALLOW, CRITICAL, ESCALATE


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


def _seed(engine, *, with_alert: bool) -> None:
    with engine.begin() as conn:
        conn.execute(
            insert(Customer.__table__),
            [
                {
                    "customer_id": 1,
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
        conn.execute(insert(Terminal.__table__), [{"terminal_id": 1, "x_terminal_id": 0.0, "y_terminal_id": 0.0}])
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
                    "unified_risk_level": CRITICAL,
                    "contributing_signals": ["transaction_ml_risk >= 0.97", "terminal_behavioral_risk: HIGH_RISK"],
                    "model_version": "xgboost_v1",
                    "transaction_risk_threshold": 0.97,
                    "feature_version": "phase3_v1",
                }
            ],
        )
        if with_alert:
            conn.execute(
                insert(Alert.__table__),
                [
                    {
                        "transaction_id": 100,
                        "customer_id": 1,
                        "terminal_id": 1,
                        "severity": CRITICAL,
                        "reason": "transaction_ml_risk >= 0.97; terminal_behavioral_risk: HIGH_RISK",
                        "evidence": {"unified_risk_level": CRITICAL},
                        "recommended_action": ESCALATE,
                        "status": "OPEN",
                    }
                ],
            )


def _stub_success(evidence):
    return AnalystResult(
        explanation=AnalystExplanation(
            summary="Elevated risk driven by both transaction-ML and terminal behavioral signals.",
            evidence_explanation="transaction_ml_risk >= 0.97; terminal_behavioral_risk: HIGH_RISK",
            recommended_action=ESCALATE,
            recommendation_rationale="Two independent severe signals corroborate each other.",
            confidence="high",
            caveats=[],
        ),
        is_fallback=False,
        fallback_reason=None,
    )


def _stub_fallback(evidence):
    return AnalystResult(
        explanation=AnalystExplanation(
            summary=f"Transaction {evidence.transaction_id}: unified_risk_level={evidence.unified_risk_level}.",
            evidence_explanation="; ".join(evidence.contributing_signals) or "no elevated component signals",
            recommended_action=evidence.policy_action,
            recommendation_rationale="Deterministic fallback (AI Risk Analyst unavailable).",
            confidence="low",
            caveats=["AI explanation unavailable; this is a deterministic fallback.", "simulated LLM outage"],
        ),
        is_fallback=True,
        fallback_reason="simulated LLM outage",
    )


def test_analyst_not_found_transaction(client):
    resp = client.get("/transactions/999999/analyst")
    assert resp.status_code == 404


def test_analyst_not_found_when_no_risk_score(db_engine, client):
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
        conn.execute(insert(Terminal.__table__), [{"terminal_id": 1, "x_terminal_id": 0.0, "y_terminal_id": 0.0}])
        conn.execute(
            insert(Transaction.__table__),
            [
                {
                    "transaction_id": 300,
                    "tx_datetime": dt.datetime(2018, 4, 1, 0, 0, 0),
                    "customer_id": 2,
                    "terminal_id": 1,
                    "tx_amount": 5.0,
                    "tx_time_seconds": 0,
                    "tx_time_days": 0,
                    "tx_fraud": 0,
                    "tx_fraud_scenario": 0,
                    "split": "train",
                }
            ],
        )

    resp = client.get("/transactions/300/analyst")
    assert resp.status_code == 404


def test_analyst_success_path(db_engine, client, monkeypatch):
    _seed(db_engine, with_alert=True)
    monkeypatch.setattr("mrs.api.routers.analyst.generate_explanation", _stub_success)

    resp = client.get("/transactions/100/analyst")
    assert resp.status_code == 200
    body = resp.json()

    assert body["transaction_id"] == 100
    assert body["unified_risk_level"] == CRITICAL
    assert body["deterministic_action"] == ESCALATE  # from the persisted Alert, authoritative
    assert body["recommended_action"] == ESCALATE  # AI's own advisory recommendation
    assert body["policy_version"] is None  # no audit_log POLICY_DECISION row was seeded in this test
    assert body["is_fallback"] is False
    assert body["fallback_reason"] is None
    assert body["analyst_model"] == "gemini-3.5-flash-lite"
    assert "confirmed fraud" not in body["summary"].lower()


def test_analyst_fallback_path_still_returns_200_with_useful_content(db_engine, client, monkeypatch):
    _seed(db_engine, with_alert=True)
    monkeypatch.setattr("mrs.api.routers.analyst.generate_explanation", _stub_fallback)

    resp = client.get("/transactions/100/analyst")
    assert resp.status_code == 200
    body = resp.json()

    assert body["is_fallback"] is True
    assert body["fallback_reason"] == "simulated LLM outage"
    assert body["analyst_model"] is None
    assert body["deterministic_action"] == ESCALATE  # policy decision still authoritative even on LLM failure
    assert body["recommended_action"] == ESCALATE  # fallback mirrors it, never invents a different one
    assert body["summary"] != ""


def test_analyst_deterministic_action_present_when_no_alert(db_engine, client, monkeypatch):
    _seed(db_engine, with_alert=False)
    monkeypatch.setattr("mrs.api.routers.analyst.generate_explanation", _stub_fallback)

    resp = client.get("/transactions/100/analyst")
    assert resp.status_code == 200
    body = resp.json()
    assert body["deterministic_action"] == ALLOW  # no Alert row -> defaults to ALLOW, never fabricated


def test_analyst_response_never_exposes_ground_truth_fraud_label(db_engine, client, monkeypatch):
    _seed(db_engine, with_alert=True)
    monkeypatch.setattr("mrs.api.routers.analyst.generate_explanation", _stub_success)

    resp = client.get("/transactions/100/analyst")
    body = resp.json()
    assert "tx_fraud" not in body
    assert "tx_fraud_scenario" not in body
