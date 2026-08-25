"""Orchestrates the full feature build: transaction + customer + terminal + relationship
features, joined into one frame, with the chronological split label attached.

Per Dev Plan §13/§39 and the Phase 3 handoff §11: features are built once over the
COMPLETE chronologically-ordered dataset, not separately per split -- customer/terminal
history legitimately continues across the train/validation/test boundaries. The split
column is attached only at the very end, purely as a downstream evaluation label; it never
participates in any aggregation above.
"""

from __future__ import annotations

import pandas as pd

from mrs.data.schema import LABEL_COLUMNS
from mrs.data.splits import assign_split
from mrs.features import _temporal as T
from mrs.features.customer import build_customer_features
from mrs.features.registry import FEATURE_NAMES, NON_FEATURE_COLUMNS
from mrs.features.relationship import build_relationship_features
from mrs.features.terminal import build_terminal_features
from mrs.features.transaction import build_transaction_features

#: Raw columns build_feature_frame requires as input.
REQUIRED_INPUT_COLUMNS = (
    "TRANSACTION_ID",
    "TX_DATETIME",
    "CUSTOMER_ID",
    "TERMINAL_ID",
    "TX_AMOUNT",
    "TX_FRAUD",
)


def build_feature_frame(transactions: pd.DataFrame) -> pd.DataFrame:
    """Build the full feature layer from a processed transactions frame.

    `transactions` need not be pre-sorted. The output is in canonical chronological order
    (TX_DATETIME, TRANSACTION_ID) -- not necessarily the input's row order -- since that
    canonical order is what every downstream consumer (models, evaluation, replay) should
    rely on (Dev Plan §5).
    """
    missing = [c for c in REQUIRED_INPUT_COLUMNS if c not in transactions.columns]
    if missing:
        raise ValueError(f"build_feature_frame: missing required input columns: {missing}")

    ordered = T.sort_canonical(transactions)

    tx_feats = build_transaction_features(ordered)
    cust_feats = build_customer_features(ordered)
    term_feats = build_terminal_features(ordered)
    rel_feats = build_relationship_features(ordered)

    result = tx_feats.merge(
        cust_feats, on="TRANSACTION_ID", how="inner", validate="one_to_one"
    ).merge(
        term_feats, on="TRANSACTION_ID", how="inner", validate="one_to_one"
    ).merge(
        rel_feats, on="TRANSACTION_ID", how="inner", validate="one_to_one"
    )

    if len(result) != len(ordered):
        raise AssertionError(
            f"build_feature_frame: row count changed during join "
            f"({len(ordered)} -> {len(result)})"
        )

    # Join on TRANSACTION_ID explicitly rather than trusting that four chained merges
    # preserved row order/positional alignment with `ordered` -- pandas' inner-merge does
    # preserve left-frame order (verified), but this makes correctness independent of that
    # implementation detail instead of relying on it silently.
    result = result.merge(
        ordered[["TRANSACTION_ID", "TX_DATETIME"]], on="TRANSACTION_ID", how="inner",
        validate="one_to_one",
    )
    result["split"] = assign_split(result["TX_DATETIME"])
    # TX_DATETIME is kept (not dropped) so a consumer can always recover chronological
    # order and re-sort after loading, without rejoining to the processed layer -- see
    # scripts/05_build_features.py's docstring for why this matters when the output is
    # partitioned by split.

    # Enforce the canonical chronological row order explicitly (by TRANSACTION_ID lookup,
    # not by hoping prior operations preserved it) -- this is the order every downstream
    # consumer should be able to rely on.
    result = (
        result.set_index("TRANSACTION_ID")
        .loc[ordered["TRANSACTION_ID"]]
        .reset_index()
    )

    generated_feature_columns = set(result.columns) - NON_FEATURE_COLUMNS
    if generated_feature_columns != set(FEATURE_NAMES):
        missing_from_registry = generated_feature_columns - set(FEATURE_NAMES)
        missing_from_output = set(FEATURE_NAMES) - generated_feature_columns
        raise AssertionError(
            "build_feature_frame: generated columns and the feature registry disagree.\n"
            f"  generated but not registered: {sorted(missing_from_registry)}\n"
            f"  registered but not generated: {sorted(missing_from_output)}"
        )

    leaked_labels = generated_feature_columns & LABEL_COLUMNS
    if leaked_labels:
        raise AssertionError(f"build_feature_frame: label columns leaked as features: {leaked_labels}")

    return result
