"""Customer-terminal relationship features (Dev Plan §7.2's "new terminal" bullet;
handoff §9's relationship-feature family).

"Whether the terminal is new for the customer" and "number of terminals historically
used" are already computed in mrs.features.customer (they only need the CUSTOMER_ID grouping).
This module adds the one piece of information customer.py does not compute: how many times
THIS customer has used THIS SPECIFIC terminal before -- a property of the (customer,
terminal) pair, not of either entity alone.

Identifiers (CUSTOMER_ID, TERMINAL_ID) are preserved in the output specifically so a later
phase can use them for relationship/graph visualization (handoff §21, §9) -- no graph
construction happens here, only the underlying measurable signal and the identifiers
needed to join it back to entities later.
"""

from __future__ import annotations

import pandas as pd

from mrs.features import _temporal as T

_PAIR_KEY_COLUMN = "_customer_terminal_pair"


def build_relationship_features(df: pd.DataFrame) -> pd.DataFrame:
    """Customer-terminal pair features. Input order is preserved; output aligned 1:1."""
    keyed = df.assign(
        **{
            _PAIR_KEY_COLUMN: (
                df["CUSTOMER_ID"].astype(str) + "_" + df["TERMINAL_ID"].astype(str)
            )
        }
    )

    prior_interaction_count = T.expanding_prior(
        keyed, _PAIR_KEY_COLUMN, "TX_AMOUNT", "count"
    )
    is_new_relationship = (prior_interaction_count == 0).astype(int)

    return pd.DataFrame(
        {
            "TRANSACTION_ID": df["TRANSACTION_ID"].to_numpy(),
            "CUSTOMER_ID": df["CUSTOMER_ID"].to_numpy(),
            "TERMINAL_ID": df["TERMINAL_ID"].to_numpy(),
            "pair_prior_interaction_count": prior_interaction_count,
            "pair_is_new_relationship": is_new_relationship,
        }
    )
