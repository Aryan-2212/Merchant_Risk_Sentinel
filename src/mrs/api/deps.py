"""FastAPI dependency wiring (Dev Plan §19). Reuses mrs.db.engine's existing
session factory -- no new database configuration is introduced here.

Tests override get_db via FastAPI's app.dependency_overrides to point at the
isolated test database (tests/conftest.py's db_engine fixture), never this module's
default (which follows MRS_DATABASE_URL, the real database) -- see tests/test_api.py.
"""

from __future__ import annotations

from collections.abc import Iterator

from sqlalchemy.orm import Session

from mrs.db.engine import get_engine, get_session_factory

_session_factory = None


def _default_session_factory():
    global _session_factory
    if _session_factory is None:
        _session_factory = get_session_factory(get_engine())
    return _session_factory


def get_db() -> Iterator[Session]:
    session = _default_session_factory()()
    try:
        yield session
    finally:
        session.close()
