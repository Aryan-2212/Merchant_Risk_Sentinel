"""Tests for mrs.api.routers.recent -- the Simulated Recent Operational Stream API.

Mirrors tests/test_api_replay.py's structure/fixtures (isolated test database via
FastAPI dependency_overrides), but seeds a mix of benchmark (split="train") and
recent-stream (split="recent") rows to prove GET /recent/* only ever returns the
latter -- the two streams must stay cleanly separated through the API, not just in
the generator.
"""

from __future__ import annotations

import datetime as dt

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import insert
from sqlalchemy.orm import sessionmaker

from mrs.api.deps import get_db
from mrs.api.main import app
from mrs.db.models import Customer, Terminal, Transaction


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


def _seed_mixed_benchmark_and_recent(engine) -> None:
    """2 benchmark rows (2018, split="train") + 3 recent-stream rows (2026,
    split="recent"), sharing customer/terminal ids to also prove entity filters work."""
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
                for cid in (1, 2)
            ],
        )
        conn.execute(insert(Terminal.__table__), [{"terminal_id": 1, "x_terminal_id": 0.0, "y_terminal_id": 0.0}])
        conn.execute(
            insert(Transaction.__table__),
            [
                {
                    "transaction_id": 1,
                    "tx_datetime": dt.datetime(2018, 4, 1, 0, 0, 0),
                    "customer_id": 1,
                    "terminal_id": 1,
                    "tx_amount": 10.0,
                    "tx_time_seconds": 0,
                    "tx_time_days": 0,
                    "tx_fraud": 0,
                    "tx_fraud_scenario": 0,
                    "split": "train",
                },
                {
                    "transaction_id": 2,
                    "tx_datetime": dt.datetime(2018, 4, 2, 0, 0, 0),
                    "customer_id": 2,
                    "terminal_id": 1,
                    "tx_amount": 20.0,
                    "tx_time_seconds": 86_400,
                    "tx_time_days": 1,
                    "tx_fraud": 0,
                    "tx_fraud_scenario": 0,
                    "split": "train",
                },
                {
                    "transaction_id": 2_000_000_001,
                    "tx_datetime": dt.datetime(2026, 8, 14, 0, 0, 0),
                    "customer_id": 1,
                    "terminal_id": 1,
                    "tx_amount": 30.0,
                    "tx_time_seconds": 0,
                    "tx_time_days": 0,
                    "tx_fraud": 0,
                    "tx_fraud_scenario": 0,
                    "split": "recent",
                },
                {
                    "transaction_id": 2_000_000_002,
                    "tx_datetime": dt.datetime(2026, 8, 15, 0, 0, 0),
                    "customer_id": 2,
                    "terminal_id": 1,
                    "tx_amount": 40.0,
                    "tx_time_seconds": 86_400,
                    "tx_time_days": 1,
                    "tx_fraud": 0,
                    "tx_fraud_scenario": 0,
                    "split": "recent",
                },
                {
                    "transaction_id": 2_000_000_003,
                    "tx_datetime": dt.datetime(2026, 8, 16, 0, 0, 0),
                    "customer_id": 1,
                    "terminal_id": 1,
                    "tx_amount": 50.0,
                    "tx_time_seconds": 172_800,
                    "tx_time_days": 2,
                    "tx_fraud": 0,
                    "tx_fraud_scenario": 0,
                    "split": "recent",
                },
            ],
        )


# --------------------------------------------------------------- GET /recent/* isolation


def test_recent_bounds_only_covers_recent_split(db_engine, client):
    _seed_mixed_benchmark_and_recent(db_engine)
    resp = client.get("/recent/bounds")
    assert resp.status_code == 200
    body = resp.json()
    assert body["total_transactions"] == 3
    assert body["min_tx_datetime"] == "2026-08-14T00:00:00"
    assert body["max_tx_datetime"] == "2026-08-16T00:00:00"


def test_recent_transactions_never_returns_benchmark_rows(db_engine, client):
    _seed_mixed_benchmark_and_recent(db_engine)
    resp = client.get("/recent/transactions", params={"limit": 10})
    ids = [item["transaction"]["transaction_id"] for item in resp.json()["items"]]
    assert ids == [2_000_000_001, 2_000_000_002, 2_000_000_003]
    assert all(item["transaction"]["split"] == "recent" for item in resp.json()["items"])


def test_recent_bounds_empty_when_only_benchmark_rows_present(db_engine, client):
    with db_engine.begin() as conn:
        conn.execute(
            insert(Customer.__table__),
            [
                {
                    "customer_id": 1,
                    "x_customer_id": 0.0,
                    "y_customer_id": 0.0,
                    "mean_amount": 10.0,
                    "std_amount": 1.0,
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
                    "transaction_id": 1,
                    "tx_datetime": dt.datetime(2018, 4, 1, 0, 0, 0),
                    "customer_id": 1,
                    "terminal_id": 1,
                    "tx_amount": 10.0,
                    "tx_time_seconds": 0,
                    "tx_time_days": 0,
                    "tx_fraud": 0,
                    "tx_fraud_scenario": 0,
                    "split": "train",
                }
            ],
        )
    resp = client.get("/recent/bounds")
    assert resp.status_code == 404


def test_recent_customer_id_filter(db_engine, client):
    _seed_mixed_benchmark_and_recent(db_engine)
    resp = client.get("/recent/transactions", params={"customer_id": 2})
    ids = [item["transaction"]["transaction_id"] for item in resp.json()["items"]]
    assert ids == [2_000_000_002]


def test_recent_pagination_cursor_stays_within_recent_split(db_engine, client):
    _seed_mixed_benchmark_and_recent(db_engine)
    first = client.get("/recent/transactions", params={"limit": 1}).json()
    assert first["next_cursor"] is not None
    second = client.get("/recent/transactions", params={"after_cursor": first["next_cursor"]}).json()
    ids = [item["transaction"]["transaction_id"] for item in second["items"]]
    assert ids == [2_000_000_002, 2_000_000_003]


# ---------------------------------------------------- historical Replay stays historical


def test_historical_replay_bounds_excludes_recent_stream_rows(db_engine, client):
    _seed_mixed_benchmark_and_recent(db_engine)
    resp = client.get("/replay/bounds")
    assert resp.status_code == 200
    body = resp.json()
    assert body["total_transactions"] == 2
    assert body["max_tx_datetime"] == "2018-04-02T00:00:00"


def test_historical_replay_transactions_excludes_recent_stream_rows(db_engine, client):
    _seed_mixed_benchmark_and_recent(db_engine)
    resp = client.get("/replay/transactions", params={"limit": 10})
    ids = [item["transaction"]["transaction_id"] for item in resp.json()["items"]]
    assert ids == [1, 2]
    assert all(item["transaction"]["split"] == "train" for item in resp.json()["items"])
