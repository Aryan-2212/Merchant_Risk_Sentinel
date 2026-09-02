"""Liveness/readiness endpoint -- confirms the API can reach its database."""

from __future__ import annotations

import os

from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.orm import Session

from mrs.api.deps import get_db

router = APIRouter(tags=["health"])


@router.get("/health")
def health(db: Session = Depends(get_db)) -> dict:
    db.execute(text("SELECT 1"))
    #: Presence check only -- never calls the LLM (mrs.analyst.client.generate_explanation
    #: does that, on demand, per transaction). Lets the dashboard's System Health view show
    #: "Available" vs "Fallback" without triggering a real analyst call just to render a
    #: status page.
    return {
        "status": "ok",
        "ai_analyst_configured": bool(os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")),
    }
