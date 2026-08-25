#!/usr/bin/env python
"""Reproduce customer/terminal profiles and validate them against transaction data.

Profiles are NOT published with the raw dataset (see
external/fraud_detection_handbook/NOTICE.md). This script regenerates them from the
official simulator's deterministic profile-generation code and writes them to
data/reference/ ONLY if they pass validation against the actual transaction data. If
validation fails, nothing is written and the script exits with an error — the absence
of profiles is never treated as permission to invent them.

Run with: .venv/bin/python scripts/03_reproduce_profiles.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT))

import pandas as pd  # noqa: E402

from mrs import config  # noqa: E402
from mrs.profiles.reproduce import reproduce_profiles  # noqa: E402
from mrs.profiles.validate import validate_profiles_against_transactions  # noqa: E402


def _load_processed_transactions() -> pd.DataFrame:
    parts = sorted(config.PROCESSED_TRANSACTIONS_DIR.glob("*.parquet"))
    if not parts:
        raise SystemExit(
            f"No processed data in {config.PROCESSED_TRANSACTIONS_DIR}. "
            "Run scripts/02_build_processed.py first."
        )
    return pd.concat(
        (pd.read_parquet(p, columns=["CUSTOMER_ID", "TERMINAL_ID"]) for p in parts),
        ignore_index=True,
    )


def main() -> None:
    print("Reproducing customer/terminal profiles from official simulator source...")
    customer_profiles, terminal_profiles = reproduce_profiles()
    print(f"  customers: {len(customer_profiles):,}  terminals: {len(terminal_profiles):,}")
    print(f"  customers with zero available terminals: {(customer_profiles['nb_terminals'] == 0).sum()}")

    print("Loading processed transactions for validation...")
    transactions = _load_processed_transactions()

    print("Validating reproduced profiles against observed (CUSTOMER_ID, TERMINAL_ID) usage...")
    result = validate_profiles_against_transactions(customer_profiles, transactions)

    if not result.passed:
        print("VALIDATION FAILED. No profile table will be written.")
        for problem in result.problems:
            print(f"  - {problem}")
        raise SystemExit(1)

    print("Validation PASSED: profile reproduction is consistent with observed transactions.")

    config.REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
    customer_out = customer_profiles.copy()
    customer_out["available_terminals"] = customer_out["available_terminals"].apply(list)
    customer_out.to_parquet(config.REFERENCE_DIR / "customer_profiles.parquet", index=False)
    terminal_profiles.to_parquet(config.REFERENCE_DIR / "terminal_profiles.parquet", index=False)

    print(f"Wrote customer_profiles.parquet and terminal_profiles.parquet to {config.REFERENCE_DIR}")


if __name__ == "__main__":
    main()
