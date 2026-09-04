"""Tests for GET /stats/network?live_window=N -- the Simulated Live Stream's rolling
Entity Network view (see mrs.api.routers.stats._live_window_network,
mrs.live.simulate). Isolated test database only (tests/conftest.py's db_engine
fixture via FastAPI dependency_overrides), same pattern as tests/test_api.py and
tests/test_api_replay.py.
"""

from __future__ import annotations

import datetime as dt

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import insert
from sqlalchemy.orm import sessionmaker

from mrs.api.deps import get_db
from mrs.api.main import app
from mrs.db.models import Customer, RiskScore, Terminal, Transaction


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
    return dt.datetime(2026, 8, 15, 0, minute, 0)


def _seed_terminal_centric_stream(engine) -> None:
    """8 transactions, chronologically ordered: transactions 1-5 are 5 DIFFERENT
    customers all transacting with the SAME terminal (10) -- the terminal-centric
    pattern the live network is meant to expose -- then 3 older/unrelated
    transactions (6,7,8, at EARLIER... no, LATER minute numbers so they'd fall outside
    a small window if window excludes them by recency) touching a different
    terminal/customer pair, all at strictly earlier timestamps than 1-5 so a small
    live_window naturally excludes them."""
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
                    "available_terminals": [10],
                }
                for cid in range(1, 9)
            ],
        )
        conn.execute(
            insert(Terminal.__table__),
            [{"terminal_id": tid, "x_terminal_id": 0.0, "y_terminal_id": 0.0} for tid in (10, 99)],
        )
        # Older rows (minute 1-3): customers 6,7,8 all at terminal 99.
        conn.execute(
            insert(Transaction.__table__),
            [
                {
                    "transaction_id": i,
                    "tx_datetime": _dt(i),
                    "customer_id": i + 5,
                    "terminal_id": 99,
                    "tx_amount": 10.0,
                    "tx_time_seconds": i * 60,
                    "tx_time_days": 0,
                    "tx_fraud": 0,
                    "tx_fraud_scenario": 0,
                    "split": "recent",
                }
                for i in range(1, 4)
            ]
            # Newer rows (minute 10-14): customers 1-5, all terminal 10 -- the
            # terminal-centric cluster.
            + [
                {
                    "transaction_id": 100 + i,
                    "tx_datetime": _dt(10 + i),
                    "customer_id": i,
                    "terminal_id": 10,
                    "tx_amount": 20.0 + i,
                    "tx_time_seconds": (10 + i) * 60,
                    "tx_time_days": 0,
                    "tx_fraud": 0,
                    "tx_fraud_scenario": 0,
                    "split": "recent",
                }
                for i in range(1, 6)
            ],
        )
        conn.execute(
            insert(RiskScore.__table__),
            [
                {
                    "transaction_id": tid,
                    "customer_id": cid,
                    "terminal_id": term,
                    "transaction_risk": 0.05,
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
                for tid, cid, term in (
                    [(i, i + 5, 99) for i in range(1, 4)] + [(100 + i, i, 10) for i in range(1, 6)]
                )
            ],
        )


def test_live_window_returns_only_transactions_inside_the_window(db_engine, client):
    _seed_terminal_centric_stream(db_engine)
    resp = client.get("/stats/network", params={"live_window": 5})
    assert resp.status_code == 200
    body = resp.json()

    # Only the 5 newest transactions (customers 1-5, terminal 10) should appear --
    # the 3 older ones (customers 6-8, terminal 99) must be entirely absent.
    node_ids = {n["id"] for n in body["nodes"]}
    assert node_ids == {"customer:1", "customer:2", "customer:3", "customer:4", "customer:5", "terminal:10"}
    assert "terminal:99" not in node_ids
    assert "customer:6" not in node_ids


def test_live_window_exposes_the_terminal_centric_relationship(db_engine, client):
    """Customer -> Transaction -> Terminal: 5 different customers sharing terminal
    10 must all show up as edges into that one terminal node."""
    _seed_terminal_centric_stream(db_engine)
    resp = client.get("/stats/network", params={"live_window": 5})
    body = resp.json()

    edges_to_terminal_10 = [e for e in body["edges"] if e["target"] == "terminal:10"]
    assert len(edges_to_terminal_10) == 5
    sources = {e["source"] for e in edges_to_terminal_10}
    assert sources == {f"customer:{i}" for i in range(1, 6)}
    assert all(e["weight"] == 1 for e in edges_to_terminal_10)


def test_live_window_names_the_newest_transaction(db_engine, client):
    _seed_terminal_centric_stream(db_engine)
    resp = client.get("/stats/network", params={"live_window": 5})
    body = resp.json()
    assert body["latest_transaction_id"] == 105  # customer 5 -> terminal 10, minute 15, the newest row
    newest_terminal = next(n for n in body["nodes"] if n["id"] == "terminal:10")
    newest_customer = next(n for n in body["nodes"] if n["id"] == "customer:5")
    assert newest_terminal["is_focus"] is True
    assert newest_customer["is_focus"] is True
    # A customer NOT part of the newest transaction must not be marked as focus.
    other_customer = next(n for n in body["nodes"] if n["id"] == "customer:1")
    assert other_customer["is_focus"] is False


def test_live_window_reflects_newly_ingested_transactions(db_engine, client):
    """New transactions become visible after ingestion: seed only the 3 older rows
    first, confirm the window is scoped to just them, then 'arrive' the 5 newer ones
    and confirm the window updates to include them and exclude the now-stale rows."""
    with db_engine.begin() as conn:
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
                    "available_terminals": [10],
                }
                for cid in range(1, 9)
            ],
        )
        conn.execute(insert(Terminal.__table__), [{"terminal_id": tid, "x_terminal_id": 0.0, "y_terminal_id": 0.0} for tid in (10, 99)])
        conn.execute(
            insert(Transaction.__table__),
            [
                {
                    "transaction_id": i,
                    "tx_datetime": _dt(i),
                    "customer_id": i + 5,
                    "terminal_id": 99,
                    "tx_amount": 10.0,
                    "tx_time_seconds": i * 60,
                    "tx_time_days": 0,
                    "tx_fraud": 0,
                    "tx_fraud_scenario": 0,
                    "split": "recent",
                }
                for i in range(1, 4)
            ],
        )

    first = client.get("/stats/network", params={"live_window": 5})
    assert {n["entity_id"] for n in first.json()["nodes"] if n["entity_type"] == "terminal"} == {99}

    with db_engine.begin() as conn:
        conn.execute(
            insert(Transaction.__table__),
            [
                {
                    "transaction_id": 100 + i,
                    "tx_datetime": _dt(10 + i),
                    "customer_id": i,
                    "terminal_id": 10,
                    "tx_amount": 20.0,
                    "tx_time_seconds": (10 + i) * 60,
                    "tx_time_days": 0,
                    "tx_fraud": 0,
                    "tx_fraud_scenario": 0,
                    "split": "recent",
                }
                for i in range(1, 6)
            ],
        )

    second = client.get("/stats/network", params={"live_window": 5})
    assert {n["entity_id"] for n in second.json()["nodes"] if n["entity_type"] == "terminal"} == {10}
    assert second.json()["latest_transaction_id"] == 105


def test_default_network_mode_is_unaffected_by_live_window_existing(db_engine, client):
    """Historical isolation: the default (no live_window param) GET /stats/network
    behavior -- global most-severe-entities focus selection -- must stay exactly as
    it was before live_window existed. latest_transaction_id must be None (not
    populated) in this mode, since there is no one meaningful "latest" transaction
    across an entity's unwindowed, all-time history."""
    _seed_terminal_centric_stream(db_engine)
    resp = client.get("/stats/network")
    assert resp.status_code == 200
    body = resp.json()
    assert body["latest_transaction_id"] is None
