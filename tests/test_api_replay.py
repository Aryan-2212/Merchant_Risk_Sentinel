"""Tests for mrs.api.routers.replay -- Phase 8 Step 5 (Dev Plan §22/§39).

Uses the isolated test database (tests/conftest.py's db_engine fixture) via FastAPI's
dependency_overrides, exactly like tests/test_api.py. No test in this module writes to
or reads from the real merchant_risk_sentinel database.
"""

from __future__ import annotations

import datetime as dt

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import insert
from sqlalchemy.orm import sessionmaker

from mrs.api.deps import get_db
from mrs.api.main import app
from mrs.db.models import Alert, Customer, RiskScore, Terminal, Transaction


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


def _dt(minute: int) -> dt.datetime:
    return dt.datetime(2018, 4, 1, 0, minute, 0)


def _seed_chronological_stream(engine) -> None:
    """5 transactions spread over 5 minutes, chronologically ordered, with mixed
    risk/alert presence: tx 1 LOW/no-alert, tx 2 MEDIUM/alert, tx 3 no risk_score at
    all (simulates a not-yet-scored transaction), tx 4 HIGH/alert, tx 5 CRITICAL/alert.
    """
    with engine.begin() as conn:
        conn.execute(
            insert(Customer.__table__),
            [
                {
                    "customer_id": cid,
                    "x_customer_id": 0.0,
                    "y_customer_id": 0.0,
                    "mean_amount": 10.0,
                    "std_amount": 1.0,
                    "mean_nb_tx_per_day": 1.0,
                    "nb_terminals": 1,
                    "available_terminals": [1],
                }
                for cid in range(1, 6)
            ],
        )
        conn.execute(insert(Terminal.__table__), [{"terminal_id": 1, "x_terminal_id": 0.0, "y_terminal_id": 0.0}])
        conn.execute(
            insert(Transaction.__table__),
            [
                {
                    "transaction_id": i,
                    "tx_datetime": _dt(i),
                    "customer_id": i,
                    "terminal_id": 1,
                    "tx_amount": 10.0 * i,
                    "tx_time_seconds": i * 60,
                    "tx_time_days": 0,
                    "tx_fraud": 0,
                    "tx_fraud_scenario": 0,
                    "split": "train",
                }
                for i in range(1, 6)
            ],
        )
        conn.execute(
            insert(RiskScore.__table__),
            [
                {
                    "transaction_id": 1,
                    "customer_id": 1,
                    "terminal_id": 1,
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
                },
                {
                    "transaction_id": 2,
                    "customer_id": 2,
                    "terminal_id": 1,
                    "transaction_risk": 0.5,
                    "transaction_risk_severity": 0,
                    "terminal_risk_state": "RISK_RISING",
                    "terminal_risk_severity": 1,
                    "customer_risk_state": "NORMAL",
                    "customer_risk_severity": 0,
                    "unified_risk_level": "MEDIUM",
                    "contributing_signals": ["terminal_behavioral_risk: RISK_RISING"],
                    "model_version": "xgboost_v1",
                    "transaction_risk_threshold": 0.97,
                    "feature_version": "phase3_v1",
                },
                # transaction_id=3: no risk_score row at all.
                {
                    "transaction_id": 4,
                    "customer_id": 4,
                    "terminal_id": 1,
                    "transaction_risk": 0.99,
                    "transaction_risk_severity": 2,
                    "terminal_risk_state": "NORMAL",
                    "terminal_risk_severity": 0,
                    "customer_risk_state": "NORMAL",
                    "customer_risk_severity": 0,
                    "unified_risk_level": "HIGH",
                    "contributing_signals": ["transaction_ml_risk >= 0.97"],
                    "model_version": "xgboost_v1",
                    "transaction_risk_threshold": 0.97,
                    "feature_version": "phase3_v1",
                },
                {
                    "transaction_id": 5,
                    "customer_id": 5,
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
                },
            ],
        )
        conn.execute(
            insert(Alert.__table__),
            [
                {
                    "transaction_id": 2,
                    "customer_id": 2,
                    "terminal_id": 1,
                    "severity": "MEDIUM",
                    "reason": "terminal_behavioral_risk: RISK_RISING",
                    "evidence": {"unified_risk_level": "MEDIUM"},
                    "recommended_action": "MONITOR",
                    "status": "OPEN",
                },
                {
                    "transaction_id": 4,
                    "customer_id": 4,
                    "terminal_id": 1,
                    "severity": "HIGH",
                    "reason": "transaction_ml_risk >= 0.97",
                    "evidence": {"unified_risk_level": "HIGH"},
                    "recommended_action": "STEP_UP_VERIFICATION",
                    "status": "OPEN",
                },
                {
                    "transaction_id": 5,
                    "customer_id": 5,
                    "terminal_id": 1,
                    "severity": "CRITICAL",
                    "reason": "transaction_ml_risk >= 0.97; terminal_behavioral_risk: HIGH_RISK",
                    "evidence": {"unified_risk_level": "CRITICAL"},
                    "recommended_action": "ESCALATE",
                    "status": "OPEN",
                },
            ],
        )


# ------------------------------------------------------------------------------ bounds


def test_bounds_empty_database_returns_404(client):
    resp = client.get("/replay/bounds")
    assert resp.status_code == 404


def test_bounds_reflects_seeded_range(db_engine, client):
    _seed_chronological_stream(db_engine)
    resp = client.get("/replay/bounds")
    assert resp.status_code == 200
    body = resp.json()
    assert body["total_transactions"] == 5
    assert body["min_tx_datetime"] == _dt(1).isoformat()
    assert body["max_tx_datetime"] == _dt(5).isoformat()


# ------------------------------------------------------------------------- pagination


def test_first_page_returns_items_in_chronological_order(db_engine, client):
    _seed_chronological_stream(db_engine)
    resp = client.get("/replay/transactions", params={"limit": 3})
    assert resp.status_code == 200
    body = resp.json()
    assert body["count"] == 3
    ids = [item["transaction"]["transaction_id"] for item in body["items"]]
    assert ids == [1, 2, 3]
    assert body["next_cursor"] is not None


def test_second_page_via_cursor_continues_where_first_left_off(db_engine, client):
    _seed_chronological_stream(db_engine)
    first = client.get("/replay/transactions", params={"limit": 3}).json()
    second = client.get("/replay/transactions", params={"after_cursor": first["next_cursor"]}).json()

    ids = [item["transaction"]["transaction_id"] for item in second["items"]]
    assert ids == [4, 5]
    assert second["next_cursor"] is None  # end of stream


def test_full_stream_reconstructed_by_following_cursors_matches_bounds_total(db_engine, client):
    _seed_chronological_stream(db_engine)
    total_expected = client.get("/replay/bounds").json()["total_transactions"]

    seen_ids = []
    cursor = None
    for _ in range(10):  # generous upper bound on iterations, avoids an infinite loop on a bug
        params = {"limit": 2}
        if cursor is not None:
            params["after_cursor"] = cursor
        page = client.get("/replay/transactions", params=params).json()
        seen_ids.extend(item["transaction"]["transaction_id"] for item in page["items"])
        cursor = page["next_cursor"]
        if cursor is None:
            break

    assert seen_ids == [1, 2, 3, 4, 5]
    assert len(seen_ids) == total_expected


def test_invalid_cursor_returns_400(client):
    resp = client.get("/replay/transactions", params={"after_cursor": "not-a-valid-cursor"})
    assert resp.status_code == 400


def test_limit_bounds_validated(client):
    assert client.get("/replay/transactions", params={"limit": 0}).status_code == 422
    assert client.get("/replay/transactions", params={"limit": 2001}).status_code == 422


# ------------------------------------------------------------------ start/end filters


def test_start_filters_inclusive(db_engine, client):
    _seed_chronological_stream(db_engine)
    resp = client.get("/replay/transactions", params={"start": _dt(3).isoformat()})
    ids = [item["transaction"]["transaction_id"] for item in resp.json()["items"]]
    assert ids == [3, 4, 5]


def test_end_filters_exclusive(db_engine, client):
    _seed_chronological_stream(db_engine)
    resp = client.get("/replay/transactions", params={"end": _dt(3).isoformat()})
    ids = [item["transaction"]["transaction_id"] for item in resp.json()["items"]]
    assert ids == [1, 2]  # transaction 3 itself excluded (exclusive upper bound)


def test_start_and_end_combine_to_a_window(db_engine, client):
    _seed_chronological_stream(db_engine)
    resp = client.get("/replay/transactions", params={"start": _dt(2).isoformat(), "end": _dt(5).isoformat()})
    ids = [item["transaction"]["transaction_id"] for item in resp.json()["items"]]
    assert ids == [2, 3, 4]


def test_after_cursor_takes_precedence_over_start(db_engine, client):
    _seed_chronological_stream(db_engine)
    cursor = client.get("/replay/transactions", params={"limit": 1}).json()["next_cursor"]
    # start would otherwise re-include transaction_id=1 (its own tx_datetime); cursor wins.
    resp = client.get(
        "/replay/transactions", params={"after_cursor": cursor, "start": _dt(1).isoformat()}
    )
    ids = [item["transaction"]["transaction_id"] for item in resp.json()["items"]]
    assert 1 not in ids


# ---------------------------------------------------------------- evidence/content shape


def test_item_includes_risk_score_and_alert_when_present(db_engine, client):
    _seed_chronological_stream(db_engine)
    body = client.get("/replay/transactions", params={"limit": 5}).json()
    by_id = {item["transaction"]["transaction_id"]: item for item in body["items"]}

    critical_item = by_id[5]
    assert critical_item["risk_score"]["unified_risk_level"] == "CRITICAL"
    assert critical_item["alert"]["severity"] == "CRITICAL"
    assert critical_item["alert"]["recommended_action"] == "ESCALATE"
    # Replay's alert shape is the summary schema -- no full evidence/reason.
    assert "evidence" not in critical_item["alert"]
    assert "reason" not in critical_item["alert"]


def test_item_handles_missing_risk_score_gracefully(db_engine, client):
    _seed_chronological_stream(db_engine)
    body = client.get("/replay/transactions", params={"limit": 5}).json()
    by_id = {item["transaction"]["transaction_id"]: item for item in body["items"]}

    not_yet_scored = by_id[3]
    assert not_yet_scored["risk_score"] is None
    assert not_yet_scored["alert"] is None


def test_item_handles_no_alert_gracefully(db_engine, client):
    _seed_chronological_stream(db_engine)
    body = client.get("/replay/transactions", params={"limit": 5}).json()
    by_id = {item["transaction"]["transaction_id"]: item for item in body["items"]}

    low_no_alert = by_id[1]
    assert low_no_alert["risk_score"]["unified_risk_level"] == "LOW"
    assert low_no_alert["alert"] is None


def test_transaction_fields_exclude_ground_truth_labels(db_engine, client):
    _seed_chronological_stream(db_engine)
    body = client.get("/replay/transactions", params={"limit": 1}).json()
    tx = body["items"][0]["transaction"]
    assert "tx_fraud" not in tx
    assert "tx_fraud_scenario" not in tx
