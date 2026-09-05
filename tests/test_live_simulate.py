"""Tests for mrs.live.simulate -- the Simulated Live Transaction Ingestion producer.

Live-Postgres integration tests only (require_database fixture, the isolated
merchant_risk_sentinel_test database via tests/conftest.py's db_engine fixture --
never the real merchant_risk_sentinel database). Uses a real slice of the actual
deterministic recent stream (mrs.data.recent_stream.generate_recent_stream), not a
synthetic stand-in, since the whole point of these tests is proving the live producer
reproduces the same pipeline results the batch script would have -- something a
hand-built fixture can't demonstrate.
"""

from __future__ import annotations

import pandas as pd
import pytest
from sqlalchemy import insert, select

from mrs import config
from mrs.data.recent_stream import generate_recent_stream
from mrs.db.models import Customer, RiskScore, Terminal, Transaction
from mrs.db.populate import customer_profile_rows, terminal_profile_rows
from mrs.live.simulate import (
    already_ingested_transaction_ids,
    ingest_batch,
    load_model_and_threshold,
    load_stream_and_pending,
)
from mrs.models.dataset import attach_labels, get_feature_matrix
from mrs.features.build import build_feature_frame


@pytest.fixture(scope="module")
def recent_slice() -> pd.DataFrame:
    """A real 40-row prefix of the actual deterministic recent stream -- small enough
    to run through the pipeline quickly in a test, large enough to span multiple
    customers/terminals and a couple of behavioral engine ticks."""
    if not (config.REFERENCE_DIR / "customer_profiles.parquet").exists():
        pytest.skip("data/reference profiles not present; run scripts/03_reproduce_profiles.py")
    if not (config.MODELS_DIR / "xgboost_v1").exists():
        pytest.skip("models/xgboost_v1 not present; run scripts/07_train_xgboost.py")
    return generate_recent_stream().head(40).reset_index(drop=True)


@pytest.fixture()
def seeded_engine(db_engine, recent_slice):
    """The isolated test database with exactly the customers/terminals recent_slice's
    40 rows reference (real profile rows, not synthetic placeholders) -- mirrors what
    scripts/12/14 already do, just scoped to the entities this slice actually needs."""
    customer_profiles = pd.read_parquet(config.REFERENCE_DIR / "customer_profiles.parquet")
    terminal_profiles = pd.read_parquet(config.REFERENCE_DIR / "terminal_profiles.parquet")
    needed_customers = customer_profiles[customer_profiles["CUSTOMER_ID"].isin(recent_slice["CUSTOMER_ID"])]
    needed_terminals = terminal_profiles[terminal_profiles["TERMINAL_ID"].isin(recent_slice["TERMINAL_ID"])]
    with db_engine.begin() as conn:
        conn.execute(insert(Customer.__table__), customer_profile_rows(needed_customers))
        conn.execute(insert(Terminal.__table__), terminal_profile_rows(needed_terminals))
    return db_engine


def test_load_stream_and_pending_starts_fully_pending_on_an_empty_database(seeded_engine, recent_slice):
    released, pending = load_stream_and_pending(seeded_engine)
    assert len(released) == 0
    # load_stream_and_pending regenerates the FULL deterministic stream (not just our
    # 40-row slice) -- pending is the whole thing until something is persisted.
    assert len(pending) == len(generate_recent_stream())
    assert pending["TRANSACTION_ID"].iloc[0] == recent_slice["TRANSACTION_ID"].iloc[0]


def test_already_ingested_ids_empty_on_a_fresh_database(seeded_engine):
    assert already_ingested_transaction_ids(seeded_engine) == set()


# --------------------------------------------------------------- chronological playback


def test_ingest_batch_persists_transactions_in_chronological_order(seeded_engine, recent_slice):
    pipeline, threshold = load_model_and_threshold()
    released = recent_slice.iloc[0:0]  # empty, correctly typed

    for start in range(0, 10):
        batch = recent_slice.iloc[start : start + 1]
        ingest_batch(seeded_engine, pipeline, threshold, released, batch)
        released = pd.concat([released, batch], ignore_index=True)

    with seeded_engine.connect() as conn:
        rows = conn.execute(select(Transaction.transaction_id, Transaction.tx_datetime).order_by(Transaction.transaction_id)).all()
    assert len(rows) == 10
    ids = [r.transaction_id for r in rows]
    assert ids == sorted(ids)
    tx_datetimes = [r.tx_datetime for r in rows]
    assert tx_datetimes == sorted(tx_datetimes)


def test_ingest_batch_is_deterministic_across_two_independent_runs(db_engine, recent_slice):
    """Same seed, same chronological order, same pipeline -> byte-identical
    transaction_risk and unified_risk_level, regardless of how many ticks it took to
    get there (one 10-row tick here vs. ten 1-row ticks in the previous test)."""
    customer_profiles = pd.read_parquet(config.REFERENCE_DIR / "customer_profiles.parquet")
    terminal_profiles = pd.read_parquet(config.REFERENCE_DIR / "terminal_profiles.parquet")
    needed_customers = customer_profiles[customer_profiles["CUSTOMER_ID"].isin(recent_slice["CUSTOMER_ID"])]
    needed_terminals = terminal_profiles[terminal_profiles["TERMINAL_ID"].isin(recent_slice["TERMINAL_ID"])]
    with db_engine.begin() as conn:
        conn.execute(insert(Customer.__table__), customer_profile_rows(needed_customers))
        conn.execute(insert(Terminal.__table__), terminal_profile_rows(needed_terminals))

    pipeline, threshold = load_model_and_threshold()
    released = recent_slice.iloc[0:0]
    batch = recent_slice.iloc[0:10]
    ingest_batch(db_engine, pipeline, threshold, released, batch)

    with db_engine.connect() as conn:
        one_tick = conn.execute(
            select(RiskScore.transaction_id, RiskScore.transaction_risk, RiskScore.unified_risk_level).order_by(RiskScore.transaction_id)
        ).all()

    # Full-batch reference computation over the same 10 rows, via the exact functions
    # scripts/14 itself uses -- not mrs.live.simulate at all, an independent path.
    features = build_feature_frame(batch, split_override=config.RECENT_STREAM_SPLIT_LABEL)
    full_df = attach_labels(features, batch)
    X = get_feature_matrix(full_df)
    reference_risk = pipeline.predict_proba(X)[:, 1]
    reference_by_id = dict(zip(full_df["TRANSACTION_ID"], reference_risk))

    for row in one_tick:
        assert row.transaction_risk == pytest.approx(reference_by_id[row.transaction_id], abs=1e-9)


# ------------------------------------------------------------------- idempotent ingestion


def test_rerunning_ingest_batch_on_the_same_rows_would_duplicate_without_the_producers_own_pending_check(seeded_engine, recent_slice):
    """mrs.live.simulate's idempotency guarantee lives in load_stream_and_pending
    (filters out already-ingested ids before ingest_batch is ever called), not inside
    ingest_batch itself -- this test proves that filtering actually works: after
    ingesting the first 5 rows, load_stream_and_pending's `pending` must exclude them."""
    pipeline, threshold = load_model_and_threshold()
    first_five = recent_slice.iloc[0:5]
    ingest_batch(seeded_engine, pipeline, threshold, recent_slice.iloc[0:0], first_five)

    ingested_ids = already_ingested_transaction_ids(seeded_engine)
    assert ingested_ids == set(first_five["TRANSACTION_ID"].tolist())

    released, pending = load_stream_and_pending(seeded_engine)
    assert set(released["TRANSACTION_ID"]) == ingested_ids
    assert not pending["TRANSACTION_ID"].isin(ingested_ids).any()

    with seeded_engine.connect() as conn:
        count = conn.execute(select(Transaction.transaction_id)).all()
    assert len(count) == 5  # not 10, not duplicated


# -------------------------------------------------------------- risk pipeline compatibility


def test_a_released_transactions_persisted_risk_never_changes_once_later_ones_arrive(seeded_engine, recent_slice):
    """No future leakage: transaction_risk (and every behavioral/aggregate value
    derived from it) for an already-released transaction must be bit-identical before
    and after later transactions are released -- if a later row could retroactively
    change an earlier one's persisted risk, that would mean the earlier row's features
    had somehow depended on the future."""
    pipeline, threshold = load_model_and_threshold()
    released = recent_slice.iloc[0:0]

    first_batch = recent_slice.iloc[0:3]
    ingest_batch(seeded_engine, pipeline, threshold, released, first_batch)
    released = pd.concat([released, first_batch], ignore_index=True)

    with seeded_engine.connect() as conn:
        before = {
            r.transaction_id: (r.transaction_risk, r.unified_risk_level, r.terminal_risk_state, r.customer_risk_state)
            for r in conn.execute(select(RiskScore)).all()
        }

    for start in range(3, 15):
        batch = recent_slice.iloc[start : start + 1]
        ingest_batch(seeded_engine, pipeline, threshold, released, batch)
        released = pd.concat([released, batch], ignore_index=True)

    with seeded_engine.connect() as conn:
        after = {
            r.transaction_id: (r.transaction_risk, r.unified_risk_level, r.terminal_risk_state, r.customer_risk_state)
            for r in conn.execute(select(RiskScore).where(RiskScore.transaction_id.in_(before.keys()))).all()
        }

    assert after == before


def test_ingest_batch_writes_all_pipeline_outputs(seeded_engine, recent_slice):
    pipeline, threshold = load_model_and_threshold()
    result = ingest_batch(seeded_engine, pipeline, threshold, recent_slice.iloc[0:0], recent_slice.iloc[0:5])

    assert len(result["transaction_ids"]) == 5
    assert set(result["unified_risk_levels"]) == set(result["transaction_ids"])
    assert set(result["unified_risk_levels"].values()) <= {"LOW", "MEDIUM", "HIGH", "CRITICAL", "INSUFFICIENT_EVIDENCE"}

    with seeded_engine.connect() as conn:
        risk_scores = conn.execute(select(RiskScore)).all()
    assert len(risk_scores) == 5
    for row in risk_scores:
        assert row.model_version == "xgboost_v1"
        assert row.transaction_risk_threshold == threshold
        assert row.transaction_risk is not None
