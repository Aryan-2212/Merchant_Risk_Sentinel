"""FastAPI application (Dev Plan §19; Phase 8 Step 4).

Read-only over already-persisted Phase 5/6/7/8 data (mrs.db.models) -- no route in
this application recomputes transaction ML risk, customer/terminal behavioral risk,
risk aggregation, or policy decisions; those remain exactly where the Dev Plan places
them (mrs.models / mrs.behavioral / mrs.risk / mrs.policy), untouched by this package.
The one exception is mrs.api.routers.live's POST /live/start and /live/stop, which
start/stop the Continuous Simulated Live Stream's background producer thread (see
that router's own docstring) -- they never decide risk themselves.

Run locally with: .venv/bin/uvicorn mrs.api.main:app --reload --env-file .env
"""

from __future__ import annotations

import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from mrs.api.routers import alerts, analyst, customers, health, live, recent, replay, stats, terminals, transactions

app = FastAPI(
    title="Merchant Risk Sentinel API",
    description=(
        "Read-only API over already-computed transaction ML risk, customer/terminal "
        "behavioral risk, unified risk aggregation, and deterministic policy "
        "decisions (Dev Plan Track 2). Simulated benchmark data only -- never real "
        "Razorpay production traffic (Dev Plan Sec 2)."
    ),
    version="0.1.0",
)

#: Phase 8 Step 7 (dashboard): the React/Vite dev server runs on a different origin
#: than this API, so a browser blocks the fetch without CORS. Origins are configurable
#: via MRS_FRONTEND_ORIGINS (comma-separated) for a non-default dev port; the Vite
#: defaults are included so a fresh checkout works with no extra setup. Unauthenticated
#: either way (Dev Plan Sec 26: no auth infra for this project) -- CORS here only
#: decides which origins a *browser* will let call it. POST is allowed only for
#: mrs.api.routers.live's Start/Stop controls (the sole non-GET routes in this app).
_default_origins = "http://localhost:5173,http://127.0.0.1:5173"
_origins = [o.strip() for o in os.environ.get("MRS_FRONTEND_ORIGINS", _default_origins).split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(transactions.router)
app.include_router(customers.router)
app.include_router(terminals.router)
app.include_router(alerts.router)
app.include_router(replay.router)
app.include_router(recent.router)
app.include_router(analyst.router)
app.include_router(stats.router)
app.include_router(live.router)
