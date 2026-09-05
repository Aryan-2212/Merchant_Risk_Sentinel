"""Continuous Simulated Live Stream: a genuine, indefinitely-running transaction
producer -- distinct from mrs.live.simulate's playback of the fixed, pre-generated
21-day recent stream (mrs.data.recent_stream), which is exhausted once all ~41,610 of
its rows have been released. This module never runs out: each tick invents ONE new
transaction anchored to the real wall-clock time it is generated, so a continuously
running demo keeps producing fresh activity indefinitely.

Still not a second risk engine and not a second dataset in any deeper sense: every
generated transaction is scored and persisted via mrs.live.simulate.ingest_batch,
UNCHANGED -- the exact same mrs.features.build_feature_frame / frozen xgboost_v1
(inference only) / mrs.behavioral / mrs.risk.aggregate / mrs.db.populate /
mrs.policy.engine.decide_and_persist chain the recent stream's own playback already
uses. This module's only new responsibility is *inventing* one plausible transaction
at a time (reusing the real customer/terminal reference profiles, never fabricated
entities) and feeding it into that unchanged pipeline.

Temporal correctness: each new transaction's features/behavioral state are computed
over `released_so_far + new_row`, where `released_so_far` is the real, already-
persisted chronological history for the "recent" and "live" splits (see
load_live_stream_history) -- genuinely available before this transaction's own
tx_datetime, never anything from the future. The frozen benchmark (train/validation/
test) is never read here at all; it is a separate dataset by policy, not a source of
"prior history" for this stream.

Split: "live" (mrs.config.LIVE_STREAM_SPLIT_LABEL) -- distinct from "recent", so
GET /recent/* (scoped to split=="recent") never shows these, and the fixed 21-day
recent stream stays exactly what it was. GET /replay/* (scoped to train/validation/
test) already excludes both. GET /stats/network is unscoped by split (orders by
tx_datetime desc), so newly-generated "live" transactions -- timestamped at real
"now", later than every "recent"-split row once that stream's Sep 4, 2026 end date is
in the past -- appear there automatically, with no change needed to that endpoint.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
from sqlalchemy import select
from sqlalchemy.engine import Engine

from mrs import config
from mrs.data.schema import RAW_COLUMNS, normalize_dtypes
from mrs.db.models import Transaction
from mrs.live.simulate import ingest_batch, load_model_and_threshold  # noqa: F401  (re-exported for callers)

__all__ = [
    "load_model_and_threshold",
    "load_live_stream_history",
    "next_live_transaction_id",
    "generate_next_live_transaction",
    "run_one_tick",
]


def load_live_stream_history(engine: Engine) -> pd.DataFrame:
    """Real, already-persisted chronological history a new live transaction may
    legitimately depend on: every "recent" and "live" split row currently in the
    database, oldest first. The frozen benchmark is deliberately excluded -- it is a
    separate frozen dataset (Dev Plan Sec 2), never a source of "prior history" for
    this stream, and reading its ~1.75M rows here would also make every tick far
    slower for no temporal-correctness benefit (a live transaction's own customer/
    terminal history is what matters, and that only ever lives in "recent"/"live").
    """
    stmt = (
        select(
            Transaction.transaction_id,
            Transaction.tx_datetime,
            Transaction.customer_id,
            Transaction.terminal_id,
            Transaction.tx_amount,
            Transaction.tx_time_seconds,
            Transaction.tx_time_days,
            Transaction.tx_fraud,
            Transaction.tx_fraud_scenario,
        )
        .where(Transaction.split.in_((config.RECENT_STREAM_SPLIT_LABEL, config.LIVE_STREAM_SPLIT_LABEL)))
        .order_by(Transaction.tx_datetime, Transaction.transaction_id)
    )
    with engine.connect() as conn:
        rows = conn.execute(stmt).all()
    df = pd.DataFrame(
        rows,
        columns=[
            "TRANSACTION_ID",
            "TX_DATETIME",
            "CUSTOMER_ID",
            "TERMINAL_ID",
            "TX_AMOUNT",
            "TX_TIME_SECONDS",
            "TX_TIME_DAYS",
            "TX_FRAUD",
            "TX_FRAUD_SCENARIO",
        ],
    )
    if df.empty:
        # normalize_dtypes assumes real columns to cast; an empty frame with the right
        # dtypes still satisfies every downstream consumer (build_feature_frame etc.).
        df = df.astype(
            {
                "TRANSACTION_ID": "int64",
                "CUSTOMER_ID": "int32",
                "TERMINAL_ID": "int32",
                "TX_AMOUNT": "float64",
                "TX_TIME_SECONDS": "int64",
                "TX_TIME_DAYS": "int16",
                "TX_FRAUD": "int8",
                "TX_FRAUD_SCENARIO": "int8",
            }
        )
        df["TX_DATETIME"] = pd.to_datetime(df["TX_DATETIME"])
        return df[list(RAW_COLUMNS)]
    return normalize_dtypes(df[list(RAW_COLUMNS)])


def next_live_transaction_id(engine: Engine) -> int:
    """The next unused id in the live stream's own range -- resumable across
    stop/start cycles (and process restarts) without ever colliding with a
    previously-generated live transaction, the recent stream, or the benchmark."""
    with engine.connect() as conn:
        current_max = conn.execute(
            select(Transaction.transaction_id)
            .where(Transaction.transaction_id >= config.LIVE_STREAM_TX_ID_OFFSET)
            .order_by(Transaction.transaction_id.desc())
            .limit(1)
        ).scalar_one_or_none()
    return (current_max + 1) if current_max is not None else config.LIVE_STREAM_TX_ID_OFFSET


def generate_next_live_transaction(
    rng: np.random.Generator,
    transaction_id: int,
    customer_profiles: pd.DataFrame,
    tx_datetime: dt.datetime,
    stream_epoch: dt.datetime,
) -> pd.DataFrame:
    """One new, plausible transaction anchored to a real wall-clock timestamp --
    reusing an existing customer's real profile (mean_amount/std_amount, and one of
    their own real available_terminals) exactly like mrs.data.recent_stream's organic
    generation, just for a single row instead of a 21-day batch. No fraud injection
    here (TX_FRAUD/TX_FRAUD_SCENARIO are always 0/0): the continuous stream's purpose
    is proving new activity keeps arriving and flowing through the real pipeline, not
    re-manufacturing another engineered risk narrative -- the 21-day recent stream
    already does that (see mrs.data.recent_stream) and stays intact and unmodified.
    `stream_epoch` only anchors TX_TIME_SECONDS/TX_TIME_DAYS (lineage/audit fields the
    Transaction table stores; no feature or behavioral computation reads them).
    """
    row_idx = int(rng.integers(0, len(customer_profiles)))
    profile = customer_profiles.iloc[row_idx]
    customer_id = int(profile["CUSTOMER_ID"])
    available_terminals = profile["available_terminals"]
    terminal_id = int(available_terminals[int(rng.integers(0, len(available_terminals)))])
    amount = float(max(0.5, rng.normal(float(profile["mean_amount"]), float(profile["std_amount"]))))

    elapsed = tx_datetime - stream_epoch
    seconds_since_epoch = max(0, int(elapsed.total_seconds()))

    row = pd.DataFrame(
        [
            {
                "TRANSACTION_ID": transaction_id,
                "TX_DATETIME": tx_datetime,
                "CUSTOMER_ID": customer_id,
                "TERMINAL_ID": terminal_id,
                "TX_AMOUNT": amount,
                "TX_TIME_SECONDS": seconds_since_epoch,
                "TX_TIME_DAYS": seconds_since_epoch // 86_400,
                "TX_FRAUD": 0,
                "TX_FRAUD_SCENARIO": 0,
            }
        ]
    )
    return normalize_dtypes(row[list(RAW_COLUMNS)])


def run_one_tick(
    engine: Engine,
    pipeline,
    threshold: float,
    rng: np.random.Generator,
    customer_profiles: pd.DataFrame,
    released_so_far: pd.DataFrame,
    next_id: int,
    stream_epoch: dt.datetime,
    *,
    now: dt.datetime | None = None,
) -> tuple[pd.DataFrame, dict, int]:
    """Generate, score, and persist exactly one new live transaction. Pure orchestration
    over already-tested pieces (generate_next_live_transaction, mrs.live.simulate.
    ingest_batch) -- no policy/risk decision is made here. Returns
    (updated_released_so_far, ingest_result, next_id_after) so a caller (a test, or
    the background-thread loop in mrs.live.manager) can chain ticks without repeating
    this wiring. `now` is injectable for deterministic tests; defaults to the real
    wall clock, which is the whole point of this module in production use.
    """
    tx_datetime = now if now is not None else dt.datetime.now()
    if not released_so_far.empty:
        # Never let a fast tick land at/before the previous one -- ties are already
        # handled safely elsewhere (TRANSACTION_ID break ties, per mrs.features.
        # _temporal), but a strictly increasing wall clock is the honest, simplest
        # behavior for a "live" stream to have.
        last_dt = released_so_far["TX_DATETIME"].iloc[-1]
        if tx_datetime <= last_dt:
            tx_datetime = last_dt + dt.timedelta(microseconds=1)

    new_row = generate_next_live_transaction(rng, next_id, customer_profiles, tx_datetime, stream_epoch)
    result = ingest_batch(
        engine, pipeline, threshold, released_so_far, new_row, split_label=config.LIVE_STREAM_SPLIT_LABEL
    )
    updated = pd.concat([released_so_far, new_row], ignore_index=True)
    return updated, result, next_id + 1
