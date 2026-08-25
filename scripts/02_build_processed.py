#!/usr/bin/env python
"""Build the processed (validated, normalized, Parquet) layer from raw data.

Run with: .venv/bin/python scripts/02_build_processed.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mrs.data.build_processed import build_processed_dataset, write_processed_dataset  # noqa: E402
from mrs.data.download import verify_raw_integrity  # noqa: E402


def main() -> None:
    problems = verify_raw_integrity()
    if problems:
        raise SystemExit("Raw integrity check failed:\n  " + "\n  ".join(problems))

    print("Loading and validating 183 raw files...")
    df = build_processed_dataset()

    print()
    print(f"Total rows:        {len(df):,}")
    print(f"Date range:        {df['TX_DATETIME'].min()} .. {df['TX_DATETIME'].max()}")
    print(f"Unique customers:  {df['CUSTOMER_ID'].nunique():,}")
    print(f"Unique terminals:  {df['TERMINAL_ID'].nunique():,}")
    print(f"Fraud count:       {int(df['TX_FRAUD'].sum()):,}")
    print(f"Fraud rate:        {df['TX_FRAUD'].mean():.4%}")

    written = write_processed_dataset(df)
    print()
    print(f"Wrote {len(written)} Parquet partitions to {written[0].parent}")


if __name__ == "__main__":
    main()
