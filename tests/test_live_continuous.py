"""Tests for mrs.live.continuous and mrs.live.manager -- the Continuous Simulated
Live Stream (as opposed to mrs.live.simulate's playback of the fixed 21-day recent
stream, covered by tests/test_live_simulate.py). Live-Postgres integration tests only
(require_database fixture, the isolated merchant_risk_sentinel_test database via
tests/conftest.py's db_engine fixture -- never the real merchant_risk_sentinel
database).
"""

from __future__ import annotations

import datetime as dt
import time

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import insert, select

from mrs import config
from mrs.data.schema import RAW_COLUMNS
from mrs.db.models import Customer, RiskScore, Terminal, Transaction
from mrs.db.populate import customer_profile_rows, terminal_profile_rows
from mrs.live.continuous import (
    generate_next_live_transaction,
    load_live_stream_history,
    load_model_and_threshold,
    next_live_transaction_id,
    run_one_tick,
)
from mrs.live.manager import LiveStreamManager


@pytest.fixture(scope="module")
def customer_profiles() -> pd.DataFrame:
    if not (config.REFERENCE_DIR / "customer_profiles.parquet").exists():
        pytest.skip("data/reference profiles not present; run scripts/03_reproduce_profiles.py")
    return pd.read_parquet(config.REFERENCE_DIR / "customer_profiles.parquet")


@pytest.fixture()
def seeded_engine(db_engine, customer_profiles):
    """The isolated test database with EVERY real customer/terminal profile loaded
    (mirrors scripts/12's own population) -- required because the continuous
    producer draws a customer uniformly at random from the full real population, so
    any of the 5,000 could be the one a given tick picks."""
    if not (config.MODELS_DIR / "xgboost_v1").exists():
        pytest.skip("models/xgboost_v1 not present; run scripts/07_train_xgboost.py")
    terminal_profiles = pd.read_parquet(config.REFERENCE_DIR / "terminal_profiles.parquet")
    with db_engine.begin() as conn:
        conn.execute(insert(Customer.__table__), customer_profile_rows(customer_profiles))
        conn.execute(insert(Terminal.__table__), terminal_profile_rows(terminal_profiles))
    return db_engine


# ------------------------------------------------------------------- pure generation


def test_generate_next_live_transaction_uses_a_real_customer_and_their_own_terminal(customer_profiles):
    rng = np.random.default_rng(1)
    now = dt.datetime(2026, 9, 5, 12, 0, 0)
    row = generate_next_live_transaction(rng, config.LIVE_STREAM_TX_ID_OFFSET, customer_profiles, now, now)

    assert row["TRANSACTION_ID"].iloc[0] == config.LIVE_STREAM_TX_ID_OFFSET
    assert row["TX_DATETIME"].iloc[0] == now
    customer_id = int(row["CUSTOMER_ID"].iloc[0])
    terminal_id = int(row["TERMINAL_ID"].iloc[0])
    profile = customer_profiles[customer_profiles["CUSTOMER_ID"] == customer_id].iloc[0]
    assert terminal_id in list(profile["available_terminals"])
    assert row["TX_FRAUD"].iloc[0] == 0
    assert row["TX_FRAUD_SCENARIO"].iloc[0] == 0
    assert row["TX_AMOUNT"].iloc[0] >= 0.5


def test_generate_next_live_transaction_is_reproducible_given_the_same_rng_state(customer_profiles):
    """Not a determinism requirement for the producer overall (it is intentionally
    unseeded in production, see mrs.live.manager) -- but the pure generator function
    itself must be a deterministic function of its inputs, or its outputs would be
    untestable at all."""
    now = dt.datetime(2026, 9, 5, 12, 0, 0)
    a = generate_next_live_transaction(np.random.default_rng(7), 1, customer_profiles, now, now)
    b = generate_next_live_transaction(np.random.default_rng(7), 1, customer_profiles, now, now)
    pd.testing.assert_frame_equal(a, b)


# ----------------------------------------------------------------- id/history helpers


def test_next_live_transaction_id_starts_at_the_offset_on_an_empty_database(seeded_engine):
    assert next_live_transaction_id(seeded_engine) == config.LIVE_STREAM_TX_ID_OFFSET


def test_load_live_stream_history_is_empty_before_anything_is_generated(seeded_engine):
    history = load_live_stream_history(seeded_engine)
    assert len(history) == 0
    assert list(history.columns) == list(RAW_COLUMNS)


# --------------------------------------------------------- multiple ticks (core requirement)


def test_multiple_ticks_create_genuinely_new_transactions_with_advancing_timestamps(seeded_engine, customer_profiles):
    pipeline, threshold = load_model_and_threshold()
    rng = np.random.default_rng(42)
    history = load_live_stream_history(seeded_engine)
    next_id = next_live_transaction_id(seeded_engine)
    epoch = dt.datetime(2026, 9, 5, 0, 0, 0)

    generated_ids = []
    generated_times = []
    base_time = dt.datetime(2026, 9, 5, 12, 0, 0)
    for i in range(5):
        history, result, next_id = run_one_tick(
            seeded_engine,
            pipeline,
            threshold,
            rng,
            customer_profiles,
            history,
            next_id,
            epoch,
            now=base_time + dt.timedelta(seconds=i),
        )
        generated_ids.append(result["transaction_ids"][0])
        generated_times.append(history["TX_DATETIME"].iloc[-1])

    # Genuinely new: 5 distinct, monotonically increasing ids, none colliding with
    # the benchmark or the 21-day recent stream's id ranges.
    assert generated_ids == sorted(set(generated_ids))
    assert len(set(generated_ids)) == 5
    assert all(tid >= config.LIVE_STREAM_TX_ID_OFFSET for tid in generated_ids)

    # Timestamps genuinely advance -- never equal or decreasing.
    assert generated_times == sorted(generated_times)
    assert len(set(generated_times)) == 5

    with seeded_engine.connect() as conn:
        rows = conn.execute(
            select(Transaction.transaction_id, Transaction.split, Transaction.tx_datetime)
            .where(Transaction.transaction_id.in_(generated_ids))
            .order_by(Transaction.transaction_id)
        ).all()
    assert len(rows) == 5
    assert all(r.split == config.LIVE_STREAM_SPLIT_LABEL for r in rows)


def test_ticks_actually_run_risk_and_policy_processing(seeded_engine, customer_profiles):
    pipeline, threshold = load_model_and_threshold()
    rng = np.random.default_rng(99)
    history = load_live_stream_history(seeded_engine)
    next_id = next_live_transaction_id(seeded_engine)
    epoch = dt.datetime(2026, 9, 5, 0, 0, 0)

    _, result, _ = run_one_tick(seeded_engine, pipeline, threshold, rng, customer_profiles, history, next_id, epoch)
    tid = result["transaction_ids"][0]

    with seeded_engine.connect() as conn:
        risk = conn.execute(select(RiskScore).where(RiskScore.transaction_id == tid)).one()
    assert risk.model_version == "xgboost_v1"
    assert risk.transaction_risk_threshold == threshold
    assert risk.transaction_risk is not None
    assert risk.unified_risk_level in {"LOW", "MEDIUM", "HIGH", "CRITICAL", "INSUFFICIENT_EVIDENCE"}
    # A policy decision was actually recorded (mrs.policy.engine.decide_and_persist,
    # invoked inside mrs.live.simulate.ingest_batch) -- not merely scored.
    from mrs.db.models import AuditLog

    with seeded_engine.connect() as conn:
        audit = conn.execute(
            select(AuditLog).where(AuditLog.transaction_id == tid, AuditLog.event_type == "POLICY_DECISION")
        ).one_or_none()
    assert audit is not None


def test_a_later_tick_never_changes_an_earlier_ticks_persisted_risk(seeded_engine, customer_profiles):
    """No future leakage, continuous-stream version of the same guarantee
    tests/test_live_simulate.py already proves for the fixed recent stream."""
    pipeline, threshold = load_model_and_threshold()
    rng = np.random.default_rng(5)
    history = load_live_stream_history(seeded_engine)
    next_id = next_live_transaction_id(seeded_engine)
    epoch = dt.datetime(2026, 9, 5, 0, 0, 0)
    base_time = dt.datetime(2026, 9, 5, 12, 0, 0)

    history, first_result, next_id = run_one_tick(
        seeded_engine, pipeline, threshold, rng, customer_profiles, history, next_id, epoch, now=base_time
    )
    first_id = first_result["transaction_ids"][0]
    with seeded_engine.connect() as conn:
        before = conn.execute(select(RiskScore.transaction_risk).where(RiskScore.transaction_id == first_id)).scalar_one()

    for i in range(1, 5):
        history, _, next_id = run_one_tick(
            seeded_engine,
            pipeline,
            threshold,
            rng,
            customer_profiles,
            history,
            next_id,
            epoch,
            now=base_time + dt.timedelta(seconds=i),
        )

    with seeded_engine.connect() as conn:
        after = conn.execute(select(RiskScore.transaction_risk).where(RiskScore.transaction_id == first_id)).scalar_one()
    assert after == before


# --------------------------------------------------------------- background manager


def test_manager_start_generates_ticks_and_stop_halts_cleanly(seeded_engine):
    manager = LiveStreamManager()
    try:
        started = manager.start(seeded_engine, interval=0.05)
        assert started is True

        deadline = time.time() + 5
        while manager.status()["n_generated"] < 2 and time.time() < deadline:
            time.sleep(0.05)

        status = manager.status()
        assert status["running"] is True
        assert status["n_generated"] >= 2
        assert status["last_transaction_id"] is not None
        assert status["error"] is None

        stopped = manager.stop(timeout=5)
        assert stopped is True

        final = manager.status()
        assert final["running"] is False
        n_at_stop = final["n_generated"]

        # Genuinely stopped, not just "about to tick again": no further growth.
        time.sleep(0.3)
        assert manager.status()["n_generated"] == n_at_stop
    finally:
        manager.stop(timeout=5)


def test_manager_start_is_a_noop_while_already_running(seeded_engine):
    manager = LiveStreamManager()
    try:
        assert manager.start(seeded_engine, interval=0.05) is True
        assert manager.start(seeded_engine, interval=0.05) is False
    finally:
        manager.stop(timeout=5)


def test_manager_stop_is_a_noop_when_not_running():
    manager = LiveStreamManager()
    assert manager.stop() is False


# --------------------------------------------------- historical/recent-stream isolation


def test_continuous_stream_never_uses_the_recent_split_label(seeded_engine, customer_profiles):
    pipeline, threshold = load_model_and_threshold()
    rng = np.random.default_rng(3)
    history = load_live_stream_history(seeded_engine)
    next_id = next_live_transaction_id(seeded_engine)
    epoch = dt.datetime(2026, 9, 5, 0, 0, 0)

    _, result, _ = run_one_tick(seeded_engine, pipeline, threshold, rng, customer_profiles, history, next_id, epoch)
    tid = result["transaction_ids"][0]

    with seeded_engine.connect() as conn:
        split = conn.execute(select(Transaction.split).where(Transaction.transaction_id == tid)).scalar_one()
    assert split == "live"
    assert split != config.RECENT_STREAM_SPLIT_LABEL


def test_continuous_stream_ids_never_collide_with_recent_stream_or_benchmark_ranges():
    assert config.LIVE_STREAM_TX_ID_OFFSET > config.RECENT_STREAM_TX_ID_OFFSET
    # Generous headroom over the fixed 21-day stream's own known size (~41,610 rows).
    assert config.LIVE_STREAM_TX_ID_OFFSET - config.RECENT_STREAM_TX_ID_OFFSET > 100_000
