"""Terminal behavioral features (Dev Plan §7.3).

Every historical feature here uses only transactions strictly before the current row
(mrs.features._temporal enforces this by construction). The fraud-count/fraud-rate
features are the most sensitive in the whole feature layer (Dev Plan §10, §34.2): they use
TX_FRAUD, but ONLY as an aggregated label over strictly-prior transactions of this
terminal -- never the current row's own label, and never a future transaction's label.
This is the one place in mrs.features that reads TX_FRAUD at all.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mrs.features import _temporal as T

_VELOCITY_WINDOWS = {"10min": "10min", "1h": "1h", "24h": "24h"}

#: Window used for "recent" fraud rate (Dev Plan §7.3: "terminal recent fraud count",
#: "terminal recent fraud rate"). 24h chosen for consistency with the longest customer/
#: terminal velocity window already defined above -- documented choice, not implied by
#: the Dev Plan, which does not specify an exact window here (Dev Plan §33.11).
_RECENT_FRAUD_WINDOW = "24h"

#: Minimum prior history (in hours) required before a volume-deviation baseline is
#: considered stable enough to report; below this, "prior transactions per hour of history"
#: is too noisy to be a meaningful comparison (Dev Plan §8.7: handle missing history
#: explicitly rather than reporting a misleading statistic).
_MIN_HOURS_FOR_RATE_BASELINE = 1.0


def build_terminal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Terminal behavioral features. Input order is preserved; output aligned 1:1."""
    out: dict[str, np.ndarray] = {"TRANSACTION_ID": df["TRANSACTION_ID"].to_numpy()}

    for suffix, window in _VELOCITY_WINDOWS.items():
        out[f"terminal_tx_count_{suffix}"] = T.rolling_count(df, "TERMINAL_ID", window)

    hist_mean = T.expanding_prior(df, "TERMINAL_ID", "TX_AMOUNT", "mean")
    hist_std = T.expanding_prior(df, "TERMINAL_ID", "TX_AMOUNT", "std")
    hist_count = T.expanding_prior(df, "TERMINAL_ID", "TX_AMOUNT", "count")
    out["terminal_hist_amount_mean"] = hist_mean
    out["terminal_hist_amount_std"] = hist_std
    out["terminal_prior_tx_count"] = hist_count

    out["terminal_time_since_prev_tx_seconds"] = T.time_since_previous(df, "TERMINAL_ID")

    _, unique_customers = T.first_occurrence_and_prior_unique_count(
        df, "TERMINAL_ID", "CUSTOMER_ID"
    )
    out["terminal_unique_customers_count"] = unique_customers

    # --- fraud-history features: strictly-prior TX_FRAUD labels only ---
    recent_fraud_count = T.rolling_sum(df, "TERMINAL_ID", "TX_FRAUD", _RECENT_FRAUD_WINDOW)
    recent_tx_count = T.rolling_count(df, "TERMINAL_ID", _RECENT_FRAUD_WINDOW)
    with np.errstate(invalid="ignore", divide="ignore"):
        recent_fraud_rate = np.where(
            recent_tx_count > 0, recent_fraud_count / recent_tx_count, np.nan
        )
    out["terminal_recent_fraud_count_24h"] = recent_fraud_count
    out["terminal_recent_fraud_rate_24h"] = recent_fraud_rate

    hist_fraud_count = T.expanding_prior(df, "TERMINAL_ID", "TX_FRAUD", "sum")
    with np.errstate(invalid="ignore", divide="ignore"):
        hist_fraud_rate = np.where(
            hist_count > 0, hist_fraud_count / hist_count, np.nan
        )
    out["terminal_hist_fraud_count"] = hist_fraud_count
    out["terminal_hist_fraud_rate"] = hist_fraud_rate

    out["terminal_fraud_rate_deviation"] = recent_fraud_rate - hist_fraud_rate

    # --- volume deviation: current recent pace vs. this terminal's own long-run pace ---
    first_seen = T.first_seen_timestamp(df, "TERMINAL_ID")
    hours_since_first_seen = (
        df["TX_DATETIME"].to_numpy() - first_seen
    ) / np.timedelta64(1, "h")
    stable_baseline = (hist_count > 0) & (hours_since_first_seen >= _MIN_HOURS_FOR_RATE_BASELINE)
    with np.errstate(invalid="ignore", divide="ignore"):
        hist_hourly_rate = np.where(
            stable_baseline, hist_count / hours_since_first_seen, np.nan
        )
    current_1h_count = out["terminal_tx_count_1h"]
    out["terminal_volume_deviation"] = current_1h_count - hist_hourly_rate

    return pd.DataFrame(out)
