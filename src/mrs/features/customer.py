"""Customer behavioral features (Dev Plan §7.2).

Every historical feature here uses only transactions strictly before the current row
(mrs.features._temporal enforces this by construction -- see that module's docstring for
the proven properties). None of these read TX_FRAUD or TX_FRAUD_SCENARIO.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mrs.features import _temporal as T

#: "Recent window" granularities for customer velocity (Dev Plan §7.2 minimum features;
#: also satisfies §7.1's "transaction frequency in recent windows" -- see transaction.py).
_VELOCITY_WINDOWS = {"10min": "10min", "1h": "1h", "24h": "24h"}


def _circular_hour_deviation(df: pd.DataFrame) -> np.ndarray:
    """Hours between the current transaction's hour and the customer's historical
    (strictly-prior) circular-mean hour-of-day. Circular, not linear, mean: a linear mean
    of hours 23 and 1 gives 12 (noon) which is wrong -- the circular mean correctly gives
    0 (midnight). NaN when the customer has no prior history.
    """
    hour = df["TX_DATETIME"].dt.hour.to_numpy()
    angle = 2 * np.pi * hour / 24.0
    sin_df = df.assign(_sin=np.sin(angle))
    cos_df = df.assign(_cos=np.cos(angle))

    hist_sin = T.expanding_prior(sin_df, "CUSTOMER_ID", "_sin", "mean")
    hist_cos = T.expanding_prior(cos_df, "CUSTOMER_ID", "_cos", "mean")

    hist_angle = np.arctan2(hist_sin, hist_cos)
    hist_hour = (hist_angle / (2 * np.pi) * 24.0) % 24.0

    raw_diff = np.abs(hour - hist_hour)
    circular_diff = np.minimum(raw_diff, 24.0 - raw_diff)
    # NaN (no history) propagates naturally through the arithmetic above.
    return circular_diff


def build_customer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Customer behavioral features. Input order is preserved; output aligned 1:1."""
    out: dict[str, np.ndarray] = {"TRANSACTION_ID": df["TRANSACTION_ID"].to_numpy()}

    for suffix, window in _VELOCITY_WINDOWS.items():
        out[f"customer_tx_count_{suffix}"] = T.rolling_count(df, "CUSTOMER_ID", window)

    hist_mean = T.expanding_prior(df, "CUSTOMER_ID", "TX_AMOUNT", "mean")
    hist_std = T.expanding_prior(df, "CUSTOMER_ID", "TX_AMOUNT", "std")
    hist_count = T.expanding_prior(df, "CUSTOMER_ID", "TX_AMOUNT", "count")

    out["customer_hist_amount_mean"] = hist_mean
    out["customer_hist_amount_std"] = hist_std
    out["customer_prior_tx_count"] = hist_count

    amount_deviation = df["TX_AMOUNT"].to_numpy() - hist_mean
    out["customer_amount_deviation"] = amount_deviation
    with np.errstate(invalid="ignore", divide="ignore"):
        zscore = np.where(
            (hist_std > 0) & ~np.isnan(hist_std), amount_deviation / hist_std, np.nan
        )
    out["customer_amount_zscore"] = zscore

    out["customer_time_since_prev_tx_seconds"] = T.time_since_previous(df, "CUSTOMER_ID")

    is_new_terminal, unique_terminals = T.first_occurrence_and_prior_unique_count(
        df, "CUSTOMER_ID", "TERMINAL_ID"
    )
    out["customer_new_terminal_flag"] = is_new_terminal
    out["customer_unique_terminals_count"] = unique_terminals

    out["customer_hour_deviation"] = _circular_hour_deviation(df)

    return pd.DataFrame(out)
