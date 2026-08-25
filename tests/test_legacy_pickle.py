"""Legacy pickle loader tests against a real raw file."""

from __future__ import annotations

import pytest

from mrs import config
from mrs.data.legacy_pickle import read_legacy_pickle
from mrs.data.schema import RAW_COLUMNS, validate_raw_frame

pytestmark = pytest.mark.data


def test_reads_first_day_with_expected_schema(require_raw_dataset):
    path = config.RAW_DIR / "2018-04-01.pkl"
    df = read_legacy_pickle(path)

    assert tuple(df.columns) == RAW_COLUMNS
    assert len(df) > 0
    validate_raw_frame(df, source=path.name)


def test_missing_file_raises(require_raw_dataset):
    from mrs.data.legacy_pickle import LegacyPickleError

    with pytest.raises((LegacyPickleError, FileNotFoundError)):
        read_legacy_pickle(config.RAW_DIR / "1999-01-01.pkl")
