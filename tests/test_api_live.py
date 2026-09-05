"""Tests for mrs.api.routers.live -- the Continuous Simulated Live Stream's
Start/Stop/Status control plane AND its GET /live/bounds, GET /live/transactions read
API (the fix for the Network investigation panel's "Recent Transactions" section
previously showing empty for a live-only customer/terminal). Isolated test database
only (tests/conftest.py's db_engine fixture via FastAPI dependency_overrides), same
pattern as tests/test_api_recent.py. Uses a real short interval (not a mock) so these
prove the actual background thread runs against the isolated test database, not the
real merchant_risk_sentinel one.
"""

from __future__ import annotations

import datetime as dt
import time

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import insert, select
from sqlalchemy.orm import sessionmaker

from mrs import config
from mrs.api.deps import get_db
from mrs.api.main import app
from mrs.db.models import Customer, RiskScore, Terminal, Transaction
from mrs.db.populate import customer_profile_rows, terminal_profile_rows
from mrs.live import manager as manager_module


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


@pytest.fixture()
def seeded_engine(db_engine):
    if not (config.REFERENCE_DIR / "customer_profiles.parquet").exists():
        pytest.skip("data/reference profiles not present; run scripts/03_reproduce_profiles.py")
    if not (config.MODELS_DIR / "xgboost_v1").exists():
        pytest.skip("models/xgboost_v1 not present; run scripts/07_train_xgboost.py")
    customer_profiles = pd.read_parquet(config.REFERENCE_DIR / "customer_profiles.parquet")
    terminal_profiles = pd.read_parquet(config.REFERENCE_DIR / "terminal_profiles.parquet")
    with db_engine.begin() as conn:
        conn.execute(insert(Customer.__table__), customer_profile_rows(customer_profiles))
        conn.execute(insert(Terminal.__table__), terminal_profile_rows(terminal_profiles))
    return db_engine


@pytest.fixture(autouse=True)
def _isolated_manager(seeded_engine, monkeypatch):
    """The live router uses mrs.live.manager's module-level singleton (correct for a
    real single-worker uvicorn process) -- for tests, give each test its own fresh
    instance pointed at the isolated test database, so tests never share producer
    state or touch the real merchant_risk_sentinel database."""
    from mrs.live.manager import LiveStreamManager

    fresh = LiveStreamManager()
    monkeypatch.setattr(manager_module, "manager", fresh)
    monkeypatch.setattr("mrs.api.routers.live.manager", fresh)
    monkeypatch.setattr("mrs.api.routers.live.get_engine", lambda: seeded_engine)
    yield fresh
    fresh.stop(timeout=5)


def test_status_before_starting(client):
    resp = client.get("/live/status")
    assert resp.status_code == 200
    body = resp.json()
    assert body["running"] is False
    assert body["n_generated"] == 0
    assert body["last_transaction_id"] is None


def test_start_then_status_shows_progress_then_stop_halts(client):
    start_resp = client.post("/live/start", params={"interval_seconds": 0.2})
    assert start_resp.status_code == 200
    assert start_resp.json()["running"] is True

    deadline = time.time() + 5
    n_generated = 0
    while time.time() < deadline:
        body = client.get("/live/status").json()
        n_generated = body["n_generated"]
        if n_generated >= 2:
            break
        time.sleep(0.2)
    assert n_generated >= 2

    stop_resp = client.post("/live/stop")
    assert stop_resp.status_code == 200
    assert stop_resp.json()["running"] is False

    after_stop = client.get("/live/status").json()["n_generated"]
    time.sleep(0.3)
    assert client.get("/live/status").json()["n_generated"] == after_stop


def test_starting_twice_is_a_clean_noop(client):
    first = client.post("/live/start", params={"interval_seconds": 0.2}).json()
    second = client.post("/live/start", params={"interval_seconds": 0.2}).json()
    assert first["running"] is True
    assert second["running"] is True
    # Started once, not restarted -- started_at is identical across both calls.
    assert first["started_at"] == second["started_at"]


def test_generated_transactions_appear_via_the_existing_network_endpoint(client):
    """The core cross-cutting requirement: a client watching the Live Entity Network
    (GET /stats/network?live_window=N, unmodified by this feature) must see newly
    generated transactions with no additional wiring."""
    client.post("/live/start", params={"interval_seconds": 0.2})
    deadline = time.time() + 5
    last_id = None
    while time.time() < deadline:
        status = client.get("/live/status").json()
        last_id = status["last_transaction_id"]
        if last_id is not None:
            break
        time.sleep(0.2)
    client.post("/live/stop")
    assert last_id is not None

    network = client.get("/stats/network", params={"live_window": 5}).json()
    assert network["latest_transaction_id"] == last_id


# ------------------------------------------ GET /live/bounds, GET /live/transactions


def _dt(minute: int) -> dt.datetime:
    return dt.datetime(2018, 4, 1, 0, minute, 0)


def _seed_mixed_benchmark_recent_and_live(engine, *, customer_id: int, terminal_id: int, seq: int = 1) -> None:
    """One benchmark row (split="train"), one recent-stream row (split="recent"), and
    one live-stream row (split="live"), all sharing the SAME customer/terminal ids --
    exactly the scenario that previously left the Network investigation panel's
    "Recent Transactions" section empty for a live-only entity, and the scenario a
    correct fix must keep the three streams cleanly separated under. `seq` offsets the
    ids so this can be called more than once in the same test without colliding."""
    with engine.begin() as conn:
        conn.execute(
            insert(Transaction.__table__),
            [
                {
                    "transaction_id": seq,
                    "tx_datetime": _dt(seq),
                    "customer_id": customer_id,
                    "terminal_id": terminal_id,
                    "tx_amount": 10.0,
                    "tx_time_seconds": 60,
                    "tx_time_days": 0,
                    "tx_fraud": 0,
                    "tx_fraud_scenario": 0,
                    "split": "train",
                },
                {
                    "transaction_id": config.RECENT_STREAM_TX_ID_OFFSET + seq,
                    "tx_datetime": dt.datetime(2026, 8, 15, 0, seq, 0),
                    "customer_id": customer_id,
                    "terminal_id": terminal_id,
                    "tx_amount": 20.0,
                    "tx_time_seconds": 60,
                    "tx_time_days": 0,
                    "tx_fraud": 0,
                    "tx_fraud_scenario": 0,
                    "split": "recent",
                },
                {
                    "transaction_id": config.LIVE_STREAM_TX_ID_OFFSET + seq,
                    "tx_datetime": dt.datetime(2026, 9, 5, 12, seq, 0),
                    "customer_id": customer_id,
                    "terminal_id": terminal_id,
                    "tx_amount": 30.0,
                    "tx_time_seconds": 60,
                    "tx_time_days": 0,
                    "tx_fraud": 0,
                    "tx_fraud_scenario": 0,
                    "split": "live",
                },
            ],
        )


def test_live_bounds_and_transactions_only_ever_return_live_split_rows(client, seeded_engine):
    customer_id = int(pd.read_parquet(config.REFERENCE_DIR / "customer_profiles.parquet")["CUSTOMER_ID"].iloc[0])
    terminal_id = int(pd.read_parquet(config.REFERENCE_DIR / "terminal_profiles.parquet")["TERMINAL_ID"].iloc[0])
    _seed_mixed_benchmark_recent_and_live(seeded_engine, customer_id=customer_id, terminal_id=terminal_id)

    bounds = client.get("/live/bounds")
    assert bounds.status_code == 200
    body = bounds.json()
    assert body["total_transactions"] == 1
    assert body["min_tx_datetime"] == "2026-09-05T12:01:00"

    txs = client.get("/live/transactions", params={"limit": 10}).json()
    ids = [item["transaction"]["transaction_id"] for item in txs["items"]]
    assert ids == [config.LIVE_STREAM_TX_ID_OFFSET + 1]
    assert all(item["transaction"]["split"] == "live" for item in txs["items"])


def test_replay_and_recent_endpoints_never_return_live_split_rows(client, seeded_engine):
    """Historical benchmark and recent-stream behavior must remain unchanged: neither
    existing endpoint's own split scoping is affected by GET /live/*'s addition."""
    customer_id = int(pd.read_parquet(config.REFERENCE_DIR / "customer_profiles.parquet")["CUSTOMER_ID"].iloc[0])
    terminal_id = int(pd.read_parquet(config.REFERENCE_DIR / "terminal_profiles.parquet")["TERMINAL_ID"].iloc[0])
    _seed_mixed_benchmark_recent_and_live(seeded_engine, customer_id=customer_id, terminal_id=terminal_id)

    replay = client.get("/replay/transactions", params={"limit": 10}).json()
    assert [item["transaction"]["transaction_id"] for item in replay["items"]] == [1]

    recent = client.get("/recent/transactions", params={"limit": 10}).json()
    assert [item["transaction"]["transaction_id"] for item in recent["items"]] == [config.RECENT_STREAM_TX_ID_OFFSET + 1]


def test_live_transactions_filtered_by_customer_and_terminal_id(client, seeded_engine):
    _seed_mixed_benchmark_recent_and_live(seeded_engine, customer_id=42, terminal_id=99, seq=1)
    _seed_mixed_benchmark_recent_and_live(seeded_engine, customer_id=7, terminal_id=8, seq=2)

    by_customer = client.get("/live/transactions", params={"customer_id": 42}).json()
    assert [i["transaction"]["customer_id"] for i in by_customer["items"]] == [42]

    by_terminal = client.get("/live/transactions", params={"terminal_id": 8}).json()
    assert [i["transaction"]["terminal_id"] for i in by_terminal["items"]] == [8]


def test_a_real_generated_live_transaction_appears_in_its_customers_and_terminals_transactions(client, seeded_engine):
    """The exact regression this fix targets: a transaction produced by the REAL
    continuous producer (not a hand-seeded fixture) must be retrievable via
    GET /live/transactions filtered by its own customer_id and by its own
    terminal_id -- precisely what the Network investigation panel's "Recent
    Transactions" section calls when a live-only entity is opened."""
    client.post("/live/start", params={"interval_seconds": 0.2})
    deadline = time.time() + 5
    last_id = None
    while time.time() < deadline:
        last_id = client.get("/live/status").json()["last_transaction_id"]
        if last_id is not None:
            break
        time.sleep(0.2)
    client.post("/live/stop")
    assert last_id is not None

    with seeded_engine.connect() as conn:
        tx = conn.execute(
            select(Transaction.customer_id, Transaction.terminal_id, Transaction.split).where(
                Transaction.transaction_id == last_id
            )
        ).one()
    assert tx.split == "live"

    by_customer = client.get("/live/transactions", params={"customer_id": tx.customer_id, "desc": True}).json()
    assert last_id in [item["transaction"]["transaction_id"] for item in by_customer["items"]]

    by_terminal = client.get("/live/transactions", params={"terminal_id": tx.terminal_id, "desc": True}).json()
    assert last_id in [item["transaction"]["transaction_id"] for item in by_terminal["items"]]

    # And it carries a real, computed risk score -- not a bare row -- exactly what the
    # panel's linked transaction detail page relies on.
    with seeded_engine.connect() as conn:
        risk = conn.execute(select(RiskScore).where(RiskScore.transaction_id == last_id)).one_or_none()
    assert risk is not None
    assert risk.model_version == "xgboost_v1"
