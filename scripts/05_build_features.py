#!/usr/bin/env python
"""Build the Phase 3 feature layer from the processed transactions and write it to
data/features/*.parquet, partitioned by split (train/validation/test).

Run with: .venv/bin/python scripts/05_build_features.py

IMPORTANT for downstream consumers: unlike data/processed/'s month-partitioned files
(where filename order == chronological order), these three files are named by split
("features_test.parquet", "features_train.parquet", "features_validation.parquet"), and
alphabetical order is NOT chronological order (test < train < validation alphabetically,
but April < August < September chronologically). A naive `sorted(glob(...))` + concat
will silently produce rows out of chronological order. Each file's own row order IS
chronological; TX_DATETIME is kept in the output specifically so a consumer that needs
the full chronological stream can concat-then-sort rather than assume glob order is safe.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from mrs import config  # noqa: E402
from mrs.features.build import build_feature_frame  # noqa: E402


def load_processed() -> pd.DataFrame:
    parts = sorted(config.PROCESSED_TRANSACTIONS_DIR.glob("*.parquet"))
    if not parts:
        raise SystemExit("No processed data found. Run scripts/02_build_processed.py first.")
    frames = [pd.read_parquet(p) for p in parts]
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    print("Loading processed transactions...")
    transactions = load_processed()
    print(f"  {len(transactions):,} rows")

    print("Building feature layer (transaction + customer + terminal + relationship)...")
    start = time.time()
    features = build_feature_frame(transactions)
    elapsed = time.time() - start
    print(f"  built {len(features):,} feature rows in {elapsed:.1f}s")

    print()
    print("Split row counts:")
    print(features["split"].value_counts().to_string())

    features_dir = config.PROCESSED_DIR.parent / "features"
    features_dir.mkdir(parents=True, exist_ok=True)
    for stale in features_dir.glob("*.parquet"):
        stale.unlink()

    for split_name, part in features.groupby("split", sort=False):
        out_path = features_dir / f"features_{split_name}.parquet"
        part.to_parquet(out_path, index=False)
        print(f"  wrote {out_path} ({len(part):,} rows)")


if __name__ == "__main__":
    main()
