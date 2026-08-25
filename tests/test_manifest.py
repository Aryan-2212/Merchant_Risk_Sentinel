"""Manifest completeness and date-contiguity tests."""

from __future__ import annotations

import datetime as dt

import pytest

from mrs import config
from mrs.data.download import load_manifest

pytestmark = pytest.mark.data


def test_manifest_has_expected_file_count(require_raw_dataset):
    manifest = load_manifest()
    assert manifest is not None
    assert manifest["file_count"] == config.EXPECTED_FILE_COUNT


def test_dates_are_contiguous_with_no_gaps_or_duplicates(require_raw_dataset):
    manifest = load_manifest()
    filenames = sorted(entry["filename"] for entry in manifest["files"])
    dates = [dt.datetime.strptime(name.removesuffix(".pkl"), "%Y-%m-%d").date() for name in filenames]

    assert len(dates) == len(set(dates)), "duplicate dates in manifest"

    expected = dates[0]
    for actual in dates:
        assert actual == expected, f"gap in date sequence at {actual} (expected {expected})"
        expected += dt.timedelta(days=1)

    assert dates[0] == dt.date.fromisoformat(config.EXPECTED_START_DATE)
    assert dates[-1] == dt.date.fromisoformat(config.EXPECTED_END_DATE)


def test_every_manifest_entry_has_provenance_fields(require_raw_dataset):
    manifest = load_manifest()
    required = {"filename", "sha256", "git_blob_sha", "size_bytes", "source_url", "retrieved_utc"}
    for entry in manifest["files"]:
        assert required <= entry.keys(), f"{entry['filename']} missing provenance fields"
