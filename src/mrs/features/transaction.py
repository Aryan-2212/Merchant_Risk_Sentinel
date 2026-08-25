"""Transaction-level features (Dev Plan §7.1).

Pure row-wise derivations of the current transaction's own fields -- no history, no
cross-row aggregation, so nothing here can leak future information by construction.

The Dev Plan lists "time since previous transaction" and "transaction frequency in
recent windows" under this section, but a "previous transaction" or "recent frequency"
is only meaningful relative to some entity. Those are implemented once, as customer-scoped
features, in mrs.features.customer (see docs/FEATURE_SPEC.md for the explicit mapping);
this module does not duplicate them.
"""

from __future__ import annotations

import pandas as pd

#: This project's own choice (not ported from the Handbook's feature-engineering chapter,
#: which is outside the GPL-isolated scope of external/fraud_detection_handbook/).
NIGHT_START_HOUR = 22
NIGHT_END_HOUR = 6


def build_transaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Row-wise transaction context. Input order is preserved; output is aligned 1:1."""
    hour = df["TX_DATETIME"].dt.hour
    day_of_week = df["TX_DATETIME"].dt.dayofweek  # 0=Mon..6=Sun

    return pd.DataFrame(
        {
            "TRANSACTION_ID": df["TRANSACTION_ID"].to_numpy(),
            "tx_amount": df["TX_AMOUNT"].to_numpy(),
            "tx_hour": hour.to_numpy(),
            "tx_day_of_week": day_of_week.to_numpy(),
            "tx_is_weekend": (day_of_week >= 5).astype(int).to_numpy(),
            "tx_is_night": ((hour >= NIGHT_START_HOUR) | (hour < NIGHT_END_HOUR))
            .astype(int)
            .to_numpy(),
        }
    )
