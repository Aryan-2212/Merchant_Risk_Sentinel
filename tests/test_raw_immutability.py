"""Raw data must remain byte-identical after the processed layer is built (Dev Plan §33.5)."""

from __future__ import annotations

import stat

import pytest

from mrs import config
from mrs.data.download import verify_raw_integrity

pytestmark = pytest.mark.data


def test_raw_files_still_match_manifest_hashes(require_raw_dataset):
    problems = verify_raw_integrity()
    assert problems == [], f"raw data no longer matches its manifest: {problems}"


def test_raw_files_are_read_only_on_disk(require_raw_dataset):
    sample = next(config.RAW_DIR.glob("*.pkl"))
    mode = sample.stat().st_mode
    assert not (mode & stat.S_IWUSR), f"{sample.name} is writable by owner"
