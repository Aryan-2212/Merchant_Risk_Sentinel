"""In-process background-thread controller for the Continuous Simulated Live Stream
(mrs.live.continuous). Lets the dashboard's "Start Simulation" / "Pause" controls
start and stop the producer directly via the API (mrs.api.routers.live), with no
separate manual script required and no external job queue/worker infrastructure --
a single daemon thread inside the same FastAPI process, matching this project's
repeated "no unnecessary infrastructure" constraint (no Kafka/Celery/Redis).

One process-wide singleton (`manager`, at module scope) -- correct for the single-
worker `uvicorn` process this project runs (see mrs.api.main's own docstring: "Run
locally with uvicorn ... --reload"); a real multi-worker deployment would need a
different mechanism, which this demo-scale project deliberately does not build.
"""

from __future__ import annotations

import datetime as dt
import threading

import numpy as np
import pandas as pd
from sqlalchemy.engine import Engine

from mrs import config
from mrs.live.continuous import (
    load_live_stream_history,
    load_model_and_threshold,
    next_live_transaction_id,
    run_one_tick,
)


class LiveStreamManager:
    """Owns at most one running producer thread at a time. All public methods are
    safe to call from FastAPI's request-handling threads concurrently."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._interval: float = config.LIVE_STREAM_DEFAULT_INTERVAL_SECONDS
        self._n_generated: int = 0
        self._last_transaction_id: int | None = None
        self._last_tx_datetime: dt.datetime | None = None
        self._started_at: dt.datetime | None = None
        self._error: str | None = None

    def start(self, engine: Engine, interval: float | None = None) -> bool:
        """Starts the producer if it is not already running. Returns False (a no-op,
        not an error) if it was already running -- clicking "Start" twice must not
        spawn a second thread or reset progress."""
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return False
            self._stop_event.clear()
            self._interval = interval if interval is not None else config.LIVE_STREAM_DEFAULT_INTERVAL_SECONDS
            self._n_generated = 0
            self._last_transaction_id = None
            self._last_tx_datetime = None
            self._started_at = dt.datetime.now()
            self._error = None
            self._thread = threading.Thread(target=self._run, args=(engine,), daemon=True, name="live-stream-producer")
            self._thread.start()
            return True

    def stop(self, *, timeout: float = 10.0) -> bool:
        """Signals the producer to stop and waits for the current tick (if any) to
        finish -- never kills it mid-write, so a stop can never leave a half-persisted
        transaction. Returns False if it was not running."""
        with self._lock:
            thread = self._thread
            if thread is None or not thread.is_alive():
                return False
            self._stop_event.set()
        thread.join(timeout=timeout)
        return True

    def status(self) -> dict:
        with self._lock:
            running = self._thread is not None and self._thread.is_alive()
            return {
                "running": running,
                "interval_seconds": self._interval,
                "n_generated": self._n_generated,
                "last_transaction_id": self._last_transaction_id,
                "last_tx_datetime": self._last_tx_datetime,
                "started_at": self._started_at,
                "error": self._error,
            }

    def _run(self, engine: Engine) -> None:
        try:
            pipeline, threshold = load_model_and_threshold()
            customer_profiles = pd.read_parquet(config.REFERENCE_DIR / "customer_profiles.parquet")
            # Not seeded: unlike mrs.data.recent_stream's deterministic 21-day batch,
            # the continuous stream is not meant to be byte-reproducible across runs
            # -- its purpose is to feel like fresh activity every time it starts.
            rng = np.random.default_rng()

            released_so_far = load_live_stream_history(engine)
            next_id = next_live_transaction_id(engine)
            stream_epoch = released_so_far["TX_DATETIME"].min() if not released_so_far.empty else dt.datetime.now()

            while not self._stop_event.is_set():
                released_so_far, result, next_id = run_one_tick(
                    engine, pipeline, threshold, rng, customer_profiles, released_so_far, next_id, stream_epoch
                )
                tid = result["transaction_ids"][0]
                with self._lock:
                    self._n_generated += 1
                    self._last_transaction_id = tid
                    self._last_tx_datetime = released_so_far["TX_DATETIME"].iloc[-1]

                self._stop_event.wait(self._interval)
        except Exception as exc:  # noqa: BLE001 -- a background thread's only way to
            # report failure back to API callers; never crashes the FastAPI process.
            with self._lock:
                self._error = f"{type(exc).__name__}: {exc}"


#: Process-wide singleton -- see class docstring for why one instance is correct here.
manager = LiveStreamManager()
