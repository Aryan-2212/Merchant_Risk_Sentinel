"""Database engine/session factory (Dev Plan §33.10: credentials via environment
variables, never committed).

Reads the connection string from ``MRS_DATABASE_URL``. Falls back to a local-dev
default matching the ``merchant_risk_sentinel`` Postgres 15 database created for this
project (local peer/trust auth, no password) -- the same env-var-with-local-default
pattern :mod:`mrs.config` already uses for ``MRS_DATA_DIR``.
"""

from __future__ import annotations

import os

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

DEFAULT_DATABASE_URL = "postgresql+psycopg://localhost/merchant_risk_sentinel"


def get_database_url() -> str:
    return os.environ.get("MRS_DATABASE_URL", DEFAULT_DATABASE_URL)


def get_engine() -> Engine:
    return create_engine(get_database_url())


def get_session_factory(engine: Engine | None = None) -> sessionmaker[Session]:
    return sessionmaker(bind=engine or get_engine())
