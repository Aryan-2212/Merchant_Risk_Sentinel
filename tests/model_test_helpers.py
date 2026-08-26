"""Synthetic data helpers for Phase 4 model tests.

Test-support only (not part of mrs.models). Produces small, fast frames shaped like the
real Phase 3 feature layer / processed transactions layer so unit tests never need to
touch the real 1.75M-row dataset. Not collected by pytest (filename doesn't match the
test_*.py pattern).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mrs.features.registry import FEATURE_NAMES

FEATURE_COLUMNS_FOR_TESTS: tuple[str, ...] = tuple(sorted(FEATURE_NAMES))

_FLAG_COLUMNS = {"customer_new_terminal_flag", "pair_is_new_relationship"}


def _synthetic_feature_column(rng: np.random.Generator, name: str, n: int) -> np.ndarray:
    if name in _FLAG_COLUMNS:
        return rng.integers(0, 2, size=n).astype(float)
    if "count" in name:
        return rng.integers(0, 20, size=n).astype(float)
    return rng.normal(loc=50.0, scale=15.0, size=n)


def make_synthetic_features(
    n_rows: int,
    *,
    split_name: str,
    date_start: str,
    nan_rate: float = 0.1,
    seed: int = 0,
    start_transaction_id: int = 0,
) -> pd.DataFrame:
    """A frame shaped like one split of the real persisted feature layer -- every
    registered feature column present, TX_DATETIME/CUSTOMER_ID/TERMINAL_ID/split present,
    and (matching the real files) no TX_FRAUD/TX_FRAUD_SCENARIO.
    """
    rng = np.random.default_rng(seed)
    transaction_ids = np.arange(start_transaction_id, start_transaction_id + n_rows)
    timestamps = pd.to_datetime(date_start) + pd.to_timedelta(
        np.sort(rng.integers(0, 3600 * 24 * 20, size=n_rows)), unit="s"
    )

    data: dict[str, object] = {
        "TRANSACTION_ID": transaction_ids,
        "TX_DATETIME": timestamps,
        "CUSTOMER_ID": rng.integers(0, 50, size=n_rows),
        "TERMINAL_ID": rng.integers(0, 30, size=n_rows),
    }
    for col in FEATURE_COLUMNS_FOR_TESTS:
        values = _synthetic_feature_column(rng, col, n_rows)
        nan_mask = rng.random(n_rows) < nan_rate
        values[nan_mask] = np.nan
        data[col] = values
    data["split"] = split_name

    return pd.DataFrame(data)


def make_synthetic_labels(transaction_ids, *, fraud_rate: float = 0.08, seed: int = 0) -> pd.DataFrame:
    """A frame shaped like the label columns of the real processed transactions layer."""
    rng = np.random.default_rng(seed)
    n = len(transaction_ids)
    fraud = (rng.random(n) < fraud_rate).astype(int)
    scenario = np.where(fraud == 1, rng.integers(1, 4, size=n), 0)
    return pd.DataFrame(
        {
            "TRANSACTION_ID": np.asarray(transaction_ids),
            "TX_FRAUD": fraud,
            "TX_FRAUD_SCENARIO": scenario,
        }
    )


def make_synthetic_labeled_frame(
    n_rows: int,
    *,
    split_name: str,
    date_start: str,
    fraud_rate: float = 0.08,
    nan_rate: float = 0.1,
    seed: int = 0,
    start_transaction_id: int = 0,
) -> pd.DataFrame:
    """Features + labels already joined -- matches what mrs.models.dataset.load_split()
    returns. The join itself is tested independently in test_model_dataset.py; this is
    for tests (e.g. train_baseline) that just need a realistic labeled frame.
    """
    features = make_synthetic_features(
        n_rows,
        split_name=split_name,
        date_start=date_start,
        nan_rate=nan_rate,
        seed=seed,
        start_transaction_id=start_transaction_id,
    )
    labels = make_synthetic_labels(
        features["TRANSACTION_ID"],
        fraud_rate=fraud_rate,
        seed=seed + 1,
    )
    return features.merge(labels, on="TRANSACTION_ID", how="inner")
