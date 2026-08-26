"""Joins the Phase 3 feature layer to TX_FRAUD labels.

The feature layer (mrs.features.build.build_feature_frame) never contains TX_FRAUD or
TX_FRAUD_SCENARIO by construction (Dev Plan Sec 34.1) -- they live only in the processed
transactions layer. Phase 4 needs TX_FRAUD for training/evaluation and TX_FRAUD_SCENARIO
for scenario-specific error analysis (Dev Plan Sec 11, allowed per Sec 34.3), so this
module performs that join explicitly, once, at the model layer -- never inside
mrs.features, which must stay leakage-safe by construction.
"""

from __future__ import annotations

import pandas as pd

from mrs import config
from mrs.data.schema import LABEL_COLUMNS
from mrs.features.registry import FEATURE_NAMES

#: Deterministic column order for the model's design matrix. Sorted so it never depends
#: on registry declaration order or dict/set iteration order (reproducibility, Sec 33.4).
FEATURE_COLUMNS: tuple[str, ...] = tuple(sorted(FEATURE_NAMES))

#: Columns carried alongside the label join for identification/analysis, never fed to a
#: model as inputs.
_LABEL_JOIN_COLUMNS = ("TRANSACTION_ID", "TX_FRAUD", "TX_FRAUD_SCENARIO")


def attach_labels(features: pd.DataFrame, labels_source: pd.DataFrame) -> pd.DataFrame:
    """Join TX_FRAUD/TX_FRAUD_SCENARIO onto a feature frame by TRANSACTION_ID.

    `features` is any frame produced by build_feature_frame (or a split slice of one).
    `labels_source` is the processed transactions frame (must contain TRANSACTION_ID,
    TX_FRAUD, TX_FRAUD_SCENARIO). Row count must be preserved exactly -- a silent partial
    join would corrupt training data, so any mismatch raises instead of dropping rows.
    """
    missing = [c for c in _LABEL_JOIN_COLUMNS if c not in labels_source.columns]
    if missing:
        raise ValueError(f"attach_labels: labels_source missing columns: {missing}")

    joined = features.merge(
        labels_source[list(_LABEL_JOIN_COLUMNS)],
        on="TRANSACTION_ID",
        how="inner",
        validate="one_to_one",
    )
    if len(joined) != len(features):
        raise ValueError(
            f"attach_labels: row count changed during label join "
            f"({len(features)} -> {len(joined)}); features and labels_source disagree "
            "on which TRANSACTION_IDs exist"
        )
    return joined


def get_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Return only the registered model input columns, in deterministic order.

    Raises if a label column is registered as a feature -- a defensive check so a caller
    can never accidentally hand TX_FRAUD/TX_FRAUD_SCENARIO to a model as an input.
    """
    leaked = LABEL_COLUMNS & set(FEATURE_COLUMNS)
    if leaked:
        raise AssertionError(f"get_feature_matrix: label columns registered as features: {leaked}")
    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"get_feature_matrix: missing feature columns: {missing}")
    return df[list(FEATURE_COLUMNS)]


def load_processed_transactions() -> pd.DataFrame:
    """Load the full processed transactions layer (for the label join)."""
    parts = sorted(config.PROCESSED_TRANSACTIONS_DIR.glob("*.parquet"))
    if not parts:
        raise FileNotFoundError(
            f"No processed data found under {config.PROCESSED_TRANSACTIONS_DIR}. "
            "Run scripts/02_build_processed.py first."
        )
    return pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)


def load_split(split: str, labels_source: pd.DataFrame | None = None) -> pd.DataFrame:
    """Load one split's persisted feature file and attach TX_FRAUD/TX_FRAUD_SCENARIO.

    `labels_source` may be pre-loaded and passed in to avoid re-reading the full
    processed layer once per split; defaults to loading it fresh.
    """
    path = config.FEATURES_DIR / f"features_{split}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"No feature file for split={split!r} at {path}. "
            "Run scripts/05_build_features.py first."
        )
    features = pd.read_parquet(path)
    if labels_source is None:
        labels_source = load_processed_transactions()
    return attach_labels(features, labels_source)
