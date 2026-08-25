"""Feature registry: the single source of truth for feature metadata (Dev Plan §12).

Every column produced by mrs.features.build.build_feature_frame (excluding join-key/
identifier columns -- TRANSACTION_ID, CUSTOMER_ID, TERMINAL_ID, split) must have exactly
one FeatureSpec entry here, and every entry here must correspond to an actual generated
column. tests/test_feature_registry.py enforces this round-trip so no undocumented column
can silently appear. docs/FEATURE_SPEC.md is generated from this list.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    level: str  # "transaction" | "customer" | "terminal" | "relationship"
    definition: str
    historical_window: str
    excludes_current_row: str
    cold_start_behavior: str
    zero_variance_behavior: str
    missing_history_behavior: str
    uses_labels: bool
    assumptions: str = ""


#: Columns produced by the feature build that are identifiers/join keys, not features.
#: Excluded from the registry contract by design (Dev Plan §34.4: entity IDs are not
#: ordinary numeric features).
NON_FEATURE_COLUMNS: frozenset[str] = frozenset(
    {"TRANSACTION_ID", "TX_DATETIME", "CUSTOMER_ID", "TERMINAL_ID", "split"}
)

_NOT_APPLICABLE = "n/a (not an aggregate)"

FEATURE_SPECS: tuple[FeatureSpec, ...] = (
    # --- transaction.py ---
    FeatureSpec(
        "tx_amount", "transaction", "The current transaction's own TX_AMOUNT.",
        _NOT_APPLICABLE, "n/a", _NOT_APPLICABLE, _NOT_APPLICABLE, _NOT_APPLICABLE, False,
    ),
    FeatureSpec(
        "tx_hour", "transaction", "Hour of day (0-23) of TX_DATETIME.",
        _NOT_APPLICABLE, "n/a", _NOT_APPLICABLE, _NOT_APPLICABLE, _NOT_APPLICABLE, False,
    ),
    FeatureSpec(
        "tx_day_of_week", "transaction", "Day of week of TX_DATETIME (0=Mon..6=Sun).",
        _NOT_APPLICABLE, "n/a", _NOT_APPLICABLE, _NOT_APPLICABLE, _NOT_APPLICABLE, False,
    ),
    FeatureSpec(
        "tx_is_weekend", "transaction", "1 if tx_day_of_week in {5,6} (Sat/Sun), else 0.",
        _NOT_APPLICABLE, "n/a", _NOT_APPLICABLE, _NOT_APPLICABLE, _NOT_APPLICABLE, False,
    ),
    FeatureSpec(
        "tx_is_night", "transaction",
        "1 if tx_hour >= 22 or tx_hour < 6, else 0. Threshold is this project's own choice "
        "(not sourced from the Handbook's later chapters, which are outside the GPL-"
        "isolated scope of external/).",
        _NOT_APPLICABLE, "n/a", _NOT_APPLICABLE, _NOT_APPLICABLE, _NOT_APPLICABLE, False,
        assumptions="Night = 22:00-06:00, a documented project choice, not a frozen Dev Plan value.",
    ),
    # --- customer.py ---
    FeatureSpec(
        "customer_tx_count_10min", "customer",
        "Count of this customer's transactions in the last 10 minutes.",
        "[t-10min, t)", "yes", "0 (no prior transactions in window)", _NOT_APPLICABLE,
        "0 if no prior transactions exist at all", False,
        assumptions="Also satisfies Dev Plan §7.1's 'transaction frequency in recent windows' "
        "(implemented once, customer-scoped, per docs/FEATURE_SPEC.md).",
    ),
    FeatureSpec(
        "customer_tx_count_1h", "customer",
        "Count of this customer's transactions in the last 1 hour.",
        "[t-1h, t)", "yes", "0", _NOT_APPLICABLE, "0", False,
    ),
    FeatureSpec(
        "customer_tx_count_24h", "customer",
        "Count of this customer's transactions in the last 24 hours.",
        "[t-24h, t)", "yes", "0", _NOT_APPLICABLE, "0", False,
    ),
    FeatureSpec(
        "customer_hist_amount_mean", "customer",
        "Mean TX_AMOUNT over all this customer's strictly-prior transactions.",
        "all prior history", "yes", "NaN (no prior transactions)", _NOT_APPLICABLE,
        "NaN", False,
    ),
    FeatureSpec(
        "customer_hist_amount_std", "customer",
        "Std of TX_AMOUNT over all this customer's strictly-prior transactions.",
        "all prior history", "yes", "NaN (0 or 1 prior transactions)",
        "reported as a true 0.0 when >=2 identical prior amounts, distinguishable from "
        "cold-start NaN via customer_prior_tx_count", "NaN", False,
    ),
    FeatureSpec(
        "customer_prior_tx_count", "customer",
        "Count of this customer's strictly-prior transactions (all history).",
        "all prior history", "yes", "0", _NOT_APPLICABLE, "0", False,
    ),
    FeatureSpec(
        "customer_amount_deviation", "customer",
        "tx_amount - customer_hist_amount_mean.",
        "all prior history", "yes", "NaN (mean undefined)", _NOT_APPLICABLE, "NaN", False,
    ),
    FeatureSpec(
        "customer_amount_zscore", "customer",
        "customer_amount_deviation / customer_hist_amount_std, only when std is defined "
        "and > 0.",
        "all prior history", "yes", "NaN", "NaN when std==0 (never divides by zero)",
        "NaN", False,
    ),
    FeatureSpec(
        "customer_time_since_prev_tx_seconds", "customer",
        "Seconds between this transaction and this customer's immediately preceding one.",
        "single most recent prior transaction", "yes", "NaN (no prior transaction)",
        _NOT_APPLICABLE, "NaN", False,
        assumptions="Also satisfies Dev Plan §7.1's 'time since previous transaction' "
        "(customer-scoped; see docs/FEATURE_SPEC.md). For two transactions tied at the "
        "exact same timestamp, this is exactly 0.0, a real (not erroneous) value.",
    ),
    FeatureSpec(
        "customer_new_terminal_flag", "customer",
        "1 if this is the first transaction ever between this customer and this terminal.",
        "all prior history", "yes", "1 (first transaction is always a new pairing)",
        _NOT_APPLICABLE, _NOT_APPLICABLE, False,
    ),
    FeatureSpec(
        "customer_unique_terminals_count", "customer",
        "Count of distinct terminals this customer has used, strictly before this "
        "transaction.",
        "all prior history", "yes", "0", _NOT_APPLICABLE, "0", False,
    ),
    FeatureSpec(
        "customer_hour_deviation", "customer",
        "Circular distance (hours, 0-12) between tx_hour and this customer's historical "
        "circular-mean hour-of-day.",
        "all prior history", "yes", "NaN (no prior transactions)", _NOT_APPLICABLE,
        "NaN", False,
        assumptions="Uses a circular (sin/cos) mean, not a linear mean, so hours 23 and 1 "
        "correctly average to 0 rather than 12.",
    ),
    # --- terminal.py ---
    FeatureSpec(
        "terminal_tx_count_10min", "terminal",
        "Count of this terminal's transactions in the last 10 minutes.",
        "[t-10min, t)", "yes", "0", _NOT_APPLICABLE, "0", False,
    ),
    FeatureSpec(
        "terminal_tx_count_1h", "terminal",
        "Count of this terminal's transactions in the last 1 hour.",
        "[t-1h, t)", "yes", "0", _NOT_APPLICABLE, "0", False,
    ),
    FeatureSpec(
        "terminal_tx_count_24h", "terminal",
        "Count of this terminal's transactions in the last 24 hours.",
        "[t-24h, t)", "yes", "0", _NOT_APPLICABLE, "0", False,
    ),
    FeatureSpec(
        "terminal_hist_amount_mean", "terminal",
        "Mean TX_AMOUNT over all this terminal's strictly-prior transactions.",
        "all prior history", "yes", "NaN", _NOT_APPLICABLE, "NaN", False,
    ),
    FeatureSpec(
        "terminal_hist_amount_std", "terminal",
        "Std of TX_AMOUNT over all this terminal's strictly-prior transactions.",
        "all prior history", "yes", "NaN (0 or 1 prior transactions)",
        "true 0.0 when >=2 identical prior amounts", "NaN", False,
    ),
    FeatureSpec(
        "terminal_prior_tx_count", "terminal",
        "Count of this terminal's strictly-prior transactions (all history).",
        "all prior history", "yes", "0", _NOT_APPLICABLE, "0", False,
    ),
    FeatureSpec(
        "terminal_time_since_prev_tx_seconds", "terminal",
        "Seconds between this transaction and this terminal's immediately preceding one.",
        "single most recent prior transaction", "yes", "NaN", _NOT_APPLICABLE, "NaN",
        False,
    ),
    FeatureSpec(
        "terminal_unique_customers_count", "terminal",
        "Count of distinct customers this terminal has served, strictly before this "
        "transaction.",
        "all prior history", "yes", "0", _NOT_APPLICABLE, "0", False,
    ),
    FeatureSpec(
        "terminal_recent_fraud_count_24h", "terminal",
        "Count of this terminal's fraudulent (TX_FRAUD=1) transactions in the last 24h.",
        "[t-24h, t)", "yes", "0", _NOT_APPLICABLE, "0", True,
        assumptions="Uses only strictly-prior TX_FRAUD labels of this terminal -- never "
        "the current row's own label, never a future transaction's label (Dev Plan §34.2).",
    ),
    FeatureSpec(
        "terminal_recent_fraud_rate_24h", "terminal",
        "terminal_recent_fraud_count_24h / (count of this terminal's transactions in the "
        "same 24h window).",
        "[t-24h, t)", "yes", "NaN (no transactions in window, not 0% fraud)",
        _NOT_APPLICABLE, "NaN", True,
    ),
    FeatureSpec(
        "terminal_hist_fraud_count", "terminal",
        "Count of this terminal's fraudulent transactions over all strictly-prior "
        "history.",
        "all prior history", "yes", "0", _NOT_APPLICABLE, "0", True,
    ),
    FeatureSpec(
        "terminal_hist_fraud_rate", "terminal",
        "terminal_hist_fraud_count / terminal_prior_tx_count.",
        "all prior history", "yes", "NaN (no prior transactions, not 0% fraud)",
        _NOT_APPLICABLE, "NaN", True,
    ),
    FeatureSpec(
        "terminal_fraud_rate_deviation", "terminal",
        "terminal_recent_fraud_rate_24h - terminal_hist_fraud_rate.",
        "[t-24h,t) vs. all prior history", "yes", "NaN if either side is undefined",
        _NOT_APPLICABLE, "NaN", True,
    ),
    FeatureSpec(
        "terminal_volume_deviation", "terminal",
        "terminal_tx_count_1h minus this terminal's own long-run implied hourly rate "
        "(terminal_prior_tx_count / hours since this terminal's first-ever transaction).",
        "1h current vs. all prior history", "yes",
        "NaN until at least 1 hour of prior history has elapsed for this terminal",
        _NOT_APPLICABLE, "NaN", False,
        assumptions="Baseline requires >=1 hour of elapsed history (project choice, Dev "
        "Plan does not specify a threshold) so the implied rate is not dominated by a "
        "near-zero time denominator.",
    ),
    # --- relationship.py ---
    FeatureSpec(
        "pair_prior_interaction_count", "relationship",
        "Count of strictly-prior transactions between this exact (CUSTOMER_ID, "
        "TERMINAL_ID) pair.",
        "all prior history", "yes", "0", _NOT_APPLICABLE, "0", False,
    ),
    FeatureSpec(
        "pair_is_new_relationship", "relationship",
        "1 if pair_prior_interaction_count == 0 (this transaction is the first-ever "
        "between this customer and this terminal).",
        "all prior history", "yes", "1", _NOT_APPLICABLE, _NOT_APPLICABLE, False,
    ),
)

FEATURE_NAMES: frozenset[str] = frozenset(spec.name for spec in FEATURE_SPECS)
