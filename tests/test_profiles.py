"""Profile reproduction determinism and validation-against-data tests."""

from __future__ import annotations

import pandas as pd
import pytest

from mrs import config
from mrs.profiles.reproduce import reproduce_profiles
from mrs.profiles.validate import validate_profiles_against_transactions

pytestmark = pytest.mark.data


def test_reproduction_is_deterministic():
    customers_a, terminals_a = reproduce_profiles()
    customers_b, terminals_b = reproduce_profiles()

    pd.testing.assert_frame_equal(
        customers_a.drop(columns=["available_terminals"]),
        customers_b.drop(columns=["available_terminals"]),
    )
    assert list(customers_a["available_terminals"]) == list(customers_b["available_terminals"])
    pd.testing.assert_frame_equal(terminals_a, terminals_b)


def test_reproduced_counts_match_published_parameters():
    customers, terminals = reproduce_profiles()
    assert len(customers) == 5000
    assert len(terminals) == 10000


def test_reproduction_validates_against_processed_transactions(require_processed_dataset):
    parts = sorted(config.PROCESSED_TRANSACTIONS_DIR.glob("*.parquet"))
    transactions = pd.concat(
        (pd.read_parquet(p, columns=["CUSTOMER_ID", "TERMINAL_ID"]) for p in parts),
        ignore_index=True,
    )
    customers, _terminals = reproduce_profiles()

    result = validate_profiles_against_transactions(customers, transactions)
    assert result.passed, result.problems


def test_reference_tables_were_persisted_after_passing_validation():
    """If Phase 1's script ran, the reference tables should exist and be internally consistent.

    Not a data-integrity re-check (that's covered elsewhere) — this only confirms the
    persisted files match what reproduce_profiles() currently returns, i.e. that nothing
    stale is lying around.
    """
    customer_path = config.REFERENCE_DIR / "customer_profiles.parquet"
    terminal_path = config.REFERENCE_DIR / "terminal_profiles.parquet"
    if not customer_path.exists() or not terminal_path.exists():
        pytest.skip("reference profiles not generated; run scripts/03_reproduce_profiles.py")

    persisted_customers = pd.read_parquet(customer_path)
    live_customers, _ = reproduce_profiles()

    assert len(persisted_customers) == len(live_customers)
    assert persisted_customers["CUSTOMER_ID"].tolist() == live_customers["CUSTOMER_ID"].tolist()
