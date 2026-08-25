"""Build the processed layer: raw daily .pkl files -> validated, normalized Parquet.

Each raw file is loaded, schema-validated in its original form, then cast to the
processed dtypes. Files are concatenated in filename (== chronological) order, since the
Handbook publishes one file per calendar day named by that date (Dev Plan §5: preserve
chronological ordering, no shuffling). The result is partitioned by year-month, which
keeps individual Parquet files a manageable size without requiring per-day files.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from mrs import config
from mrs.data.legacy_pickle import read_legacy_pickle
from mrs.data.schema import normalize_dtypes, validate_processed_frame, validate_raw_frame


def _raw_files_in_order() -> list[Path]:
    files = sorted(config.RAW_DIR.glob("*.pkl"))
    if not files:
        raise FileNotFoundError(
            f"No raw files found in {config.RAW_DIR}. Run scripts/01_download_raw.py first."
        )
    return files


def load_and_validate_raw(path: Path) -> pd.DataFrame:
    """Load one raw daily file and validate it against the documented schema."""
    df = read_legacy_pickle(path)
    validate_raw_frame(df, source=path.name)
    return df


def build_processed_dataset(*, verbose: bool = True) -> pd.DataFrame:
    """Load every raw file, validate, normalize, and concatenate chronologically.

    Returns the assembled processed DataFrame. Does not write anything.
    """
    frames: list[pd.DataFrame] = []
    per_day_counts: dict[str, int] = {}

    for path in _raw_files_in_order():
        df = load_and_validate_raw(path)
        per_day_counts[path.name] = len(df)
        frames.append(normalize_dtypes(df))
        if verbose:
            print(f"  {path.name}: {len(df):>6,} rows")

    processed = pd.concat(frames, ignore_index=True)
    processed = processed.sort_values("TX_DATETIME", kind="stable").reset_index(drop=True)

    expected_total = sum(per_day_counts.values())
    if len(processed) != expected_total:
        raise AssertionError(
            f"Row count mismatch after concatenation: {len(processed)} != {expected_total}"
        )

    validate_processed_frame(processed)
    return processed


def write_processed_dataset(df: pd.DataFrame) -> list[Path]:
    """Write the processed dataset as year-month partitioned Parquet files.

    Returns the list of files written. The processed directory is fully rebuilt from
    df — any stale partition files are removed first, since Phase 1 does not support
    incremental append semantics.
    """
    config.PROCESSED_TRANSACTIONS_DIR.mkdir(parents=True, exist_ok=True)
    for stale in config.PROCESSED_TRANSACTIONS_DIR.glob("*.parquet"):
        stale.unlink()

    year_month = df["TX_DATETIME"].dt.strftime("%Y-%m")
    written: list[Path] = []
    for period, group in df.groupby(year_month, sort=True):
        out_path = config.PROCESSED_TRANSACTIONS_DIR / f"transactions_{period}.parquet"
        group.to_parquet(out_path, index=False)
        written.append(out_path)
    return written
