"""Shared pytest fixtures. Puts src/ and the repo root on sys.path for imports."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (_REPO_ROOT / "src", _REPO_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from mrs import config  # noqa: E402


def _raw_dataset_present() -> bool:
    return config.RAW_MANIFEST_PATH.exists() and any(config.RAW_DIR.glob("*.pkl"))


@pytest.fixture(scope="session")
def require_raw_dataset():
    """Skip a data-marked test when the downloaded dataset is not present locally."""
    if not _raw_dataset_present():
        pytest.skip("data/raw not populated; run scripts/01_download_raw.py")


def _processed_dataset_present() -> bool:
    return config.PROCESSED_TRANSACTIONS_DIR.exists() and any(
        config.PROCESSED_TRANSACTIONS_DIR.glob("*.parquet")
    )


@pytest.fixture(scope="session")
def require_processed_dataset():
    if not _processed_dataset_present():
        pytest.skip("data/processed not populated; run scripts/02_build_processed.py")


#: Deliberately a SEPARATE database from MRS_DATABASE_URL/mrs.db.engine's default.
#: db_engine below calls create_all/drop_all on every test -- running that against the
#: real `merchant_risk_sentinel` database (the one scripts/11 and scripts/12 populate)
#: would repeatedly drop the persisted Step 1/2 schema and data out from under a
#: session that had already loaded it. Override with MRS_TEST_DATABASE_URL if needed;
#: otherwise `createdb merchant_risk_sentinel_test` once, alongside the real database.
TEST_DATABASE_URL_ENV = "MRS_TEST_DATABASE_URL"
DEFAULT_TEST_DATABASE_URL = "postgresql+psycopg://localhost/merchant_risk_sentinel_test"


def _test_database_url() -> str:
    import os

    return os.environ.get(TEST_DATABASE_URL_ENV, DEFAULT_TEST_DATABASE_URL)


def _database_reachable() -> bool:
    try:
        from sqlalchemy import create_engine, text

        engine = create_engine(_test_database_url())
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception:
        return False


@pytest.fixture(scope="session")
def require_database():
    """Skip a DB-marked test when the test Postgres database is not reachable."""
    if not _database_reachable():
        pytest.skip(
            f"Postgres test database not reachable at {_test_database_url()!r}; "
            f"run `createdb merchant_risk_sentinel_test` or set {TEST_DATABASE_URL_ENV}. "
            "See .env.example."
        )


@pytest.fixture()
def db_engine(require_database):
    """A live engine, against the separate test database, with the Phase 8 schema
    applied; dropped again at teardown so tests never accumulate leftover rows."""
    from sqlalchemy import create_engine

    from mrs.db.base import create_all, drop_all

    engine = create_engine(_test_database_url())
    create_all(engine)
    yield engine
    drop_all(engine)
    engine.dispose()


@pytest.fixture()
def db_session(db_engine):
    from sqlalchemy.orm import sessionmaker

    Session = sessionmaker(bind=db_engine)
    session = Session()
    yield session
    session.close()
