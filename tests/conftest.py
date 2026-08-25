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
