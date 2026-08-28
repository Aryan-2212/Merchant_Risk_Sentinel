"""Customer behavioral-risk engine (Dev Plan Sec 7.2/12/18; Phase 7 handoff).

A separate, non-ML, interpretable statistical state machine that tracks each customer's
behavioral state over time -- NORMAL / RISK_RISING / HIGH_RISK / RECOVERY -- built
entirely from Phase 3's already-computed customer features. This module does NOT
duplicate feature computation, does NOT fit a model, and does NOT invent a new feature:
it consumes `customer_prior_tx_count`, `customer_hist_amount_mean`,
`customer_amount_zscore`, `customer_amount_deviation`, and `customer_new_terminal_flag`
exactly as `mrs.features.customer.build_customer_features` already produces them.

Same state-machine philosophy as the validated mrs.behavioral.terminal engine (Dev Plan
Sec 18 explicitly wants the same NORMAL -> RISING -> HIGH -> RECOVERY -> NORMAL framework
applied to both compromised-terminal and compromised-customer scenarios), but NOT a
mechanical port: the driving signal here is `customer_amount_zscore` (Scenario 3 --
compromised customer -- is an abnormal-spending-amount pattern, Dev Plan Sec 2), not a
fraud-rate deviation, because Phase 3 never computed a customer-side fraud-rate feature
(every customer feature has uses_labels=False in the registry) and none is invented here.

This module does NOT expose a normalized/bounded "customer_risk_score". The raw
`customer_amount_zscore` (the driving signal), `customer_amount_deviation` (raw signed
currency-unit context), and `customer_new_terminal_flag` (Dev Plan Sec 7.2's named
compromised-customer signal) are exposed as-is, alongside the categorical
`customer_risk_state`. Any normalization needed for Risk Aggregation is a decision for
that later phase, made against its actual specification -- not invented here.

Temporal safety: every input this module reads is already a strictly-prior, leakage-safe
aggregate by construction (mrs.features._temporal / mrs.features.customer). This module
adds no new aggregation over TX_AMOUNT and never reads TX_FRAUD or TX_FRAUD_SCENARIO; it
only assigns a state to each transaction via a single forward pass over each customer's
own transactions in chronological order, so a transaction's assigned state can never
depend on any later transaction of that customer (or of any other customer -- customers
are fully isolated from each other).

State transition table -- identical topology to the approved terminal engine, applied to
`level = _target_level(zscore)` in {0=calm, 1=rising, 2=high}:

    From \\ To          INSUFFICIENT   NORMAL          RISK_RISING     HIGH_RISK        RECOVERY
    INSUFFICIENT_HISTORY  stays*        level0 (first)  level1 (first)  level2 (first)   never
    NORMAL                 never         stays (lvl0)    -> lvl1         -> lvl2 (**)      never
    RISK_RISING             never         -> lvl0          stays (lvl1)   -> lvl2          never
    HIGH_RISK                 never         never directly   never directly  stays (lvl2)   -> whenever level<2 (only exit)
    RECOVERY                   never         -> after 3 confirmations  relapse if lvl1  relapse if lvl2   stays (counting or streak-reset)

    (*) INSUFFICIENT_HISTORY is entered once at the start of a customer's life and left
        permanently -- customer_prior_tx_count only increases for a real customer, so it
        can never be re-entered.
    (**) Intentional direct jump: NORMAL/INSUFFICIENT_HISTORY -> HIGH_RISK is allowed
         when a single transaction's z-score exceeds HIGH_RISK_THRESHOLD directly. A
         sudden severe spend spike is not artificially delayed through an intermediate
         RISK_RISING label first.

    Positive-only escalation: only a positive z-score (spending MORE than the customer's
    own historical baseline) drives escalation. A customer suddenly spending LESS than
    usual is not the Scenario-3 pattern and floors to level 0 (calm), exactly as only
    positive terminal_fraud_rate_deviation drives the terminal engine.

    Asymmetry (intentional, matching the terminal engine): RISK_RISING -> NORMAL resolves
    immediately on the next calm transaction (no confirmation streak). HIGH_RISK can only
    be exited through a *confirmed* RECOVERY (RECOVERY_CONFIRM_COUNT consecutive
    transactions at or below RECOVERY_CONFIRM_THRESHOLD).

    From any state: whenever customer_amount_zscore is NaN -- which, once a customer has
    sufficient history, only happens when customer_hist_amount_std == 0 (a customer whose
    prior spending has been perfectly uniform so far; the registry's "never divides by
    zero" case) -- the current state and streak are held unchanged. This is a genuinely
    different underlying cause than the terminal engine's "empty recent window" (there is
    no recent-window concept here; customer_amount_zscore is always a current-vs-all-time
    comparison), but the same "do not manufacture a score from unavailable evidence" rule.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mrs.features._temporal import sort_canonical

#: A customer needs at least this many strictly-prior transactions before their
#: historical amount baseline (customer_hist_amount_mean) is considered stable enough to
#: assess behavioral state against -- a project choice (the Dev Plan does not specify
#: one), reusing the same literal value as MIN_TERMINAL_HISTORY in
#: mrs.behavioral.terminal for consistency, not independently tuned.
MIN_CUSTOMER_HISTORY = 10

#: customer_amount_zscore thresholds, in standard-deviation units -- conventional
#: statistical significance bands (>2 sigma "notably unusual", >4 sigma "extreme"), not
#: tuned against this dataset's outcomes.
RISING_THRESHOLD = 2.0
HIGH_RISK_THRESHOLD = 4.0

#: A customer must show a z-score at or below their own historical mean (zscore <= 0)
#: for RECOVERY_CONFIRM_COUNT consecutive evaluable transactions before being declared
#: fully NORMAL again -- reuses the same values as the terminal engine.
RECOVERY_CONFIRM_THRESHOLD = 0.0
RECOVERY_CONFIRM_COUNT = 3

INSUFFICIENT_HISTORY = "INSUFFICIENT_HISTORY"
NORMAL = "NORMAL"
RISK_RISING = "RISK_RISING"
HIGH_RISK = "HIGH_RISK"
RECOVERY = "RECOVERY"

#: Columns this module requires -- all already produced by
#: mrs.features.build.build_feature_frame; nothing here is recomputed.
REQUIRED_COLUMNS = (
    "TRANSACTION_ID",
    "CUSTOMER_ID",
    "TX_DATETIME",
    "customer_prior_tx_count",
    "customer_hist_amount_mean",
    "customer_amount_zscore",
    "customer_amount_deviation",
    "customer_new_terminal_flag",
)


def _target_level(zscore: float) -> int:
    """0=calm, 1=rising, 2=high -- a pure threshold lookup on one already-computed value."""
    if zscore > HIGH_RISK_THRESHOLD:
        return 2
    if zscore > RISING_THRESHOLD:
        return 1
    return 0


def _level_to_state(level: int) -> str:
    return {0: NORMAL, 1: RISK_RISING, 2: HIGH_RISK}[level]


def _step(state: str, streak: int, has_history: bool, zscore: float) -> tuple[str, int]:
    """One customer's state transition for one transaction.

    A pure function: given the same (state, streak, has_history, zscore) it always
    returns the same next (state, streak) -- identical structure to
    mrs.behavioral.terminal._step, applied to the z-score signal. See the module
    docstring for the full transition table this implements.
    """
    if not has_history:
        return INSUFFICIENT_HISTORY, 0

    if np.isnan(zscore):
        return state, streak

    level = _target_level(zscore)

    if state == INSUFFICIENT_HISTORY:
        return _level_to_state(level), 0

    if state == RECOVERY:
        if level >= 1:
            return _level_to_state(level), 0  # relapse: renewed elevation ends recovery
        if zscore <= RECOVERY_CONFIRM_THRESHOLD:
            streak += 1
            if streak >= RECOVERY_CONFIRM_COUNT:
                return NORMAL, 0
            return RECOVERY, streak
        return RECOVERY, 0  # calm-ish (level 0) but not below the stricter confirm bar

    # state in {NORMAL, RISK_RISING, HIGH_RISK}
    if state == HIGH_RISK and level < 2:
        # The only way out of HIGH_RISK is a confirmed RECOVERY, never a direct drop back
        # to NORMAL/RISK_RISING.
        initial_streak = 1 if zscore <= RECOVERY_CONFIRM_THRESHOLD else 0
        return RECOVERY, initial_streak

    return _level_to_state(level), 0


def compute_customer_behavioral_states(df: pd.DataFrame) -> pd.DataFrame:
    """Assign a behavioral state to every transaction via one forward pass per customer,
    in chronological order.

    Returns a frame with:
      - TRANSACTION_ID
      - customer_risk_state: one of INSUFFICIENT_HISTORY/NORMAL/RISK_RISING/HIGH_RISK/RECOVERY
      - customer_amount_zscore: the raw, signed Phase 3 driving signal, passed through
        unchanged, for explainability (Dev Plan Sec 17)
      - customer_amount_deviation: the raw, signed Phase 3 currency-unit evidence field,
        passed through unchanged
      - customer_new_terminal_flag: the raw Phase 3 contextual evidence field, passed
        through unchanged

    No normalized/bounded risk score is computed here -- see module docstring.

    Output row order is the canonical chronological order (Dev Plan Sec 5), not
    necessarily the caller's input order; join back on TRANSACTION_ID if a different
    order is needed.
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"compute_customer_behavioral_states: missing columns: {missing}")

    ordered = sort_canonical(df)
    customer_ids = ordered["CUSTOMER_ID"].to_numpy()
    prior_counts = ordered["customer_prior_tx_count"].to_numpy()
    hist_means = ordered["customer_hist_amount_mean"].to_numpy()
    zscores = ordered["customer_amount_zscore"].to_numpy()

    states = np.empty(len(ordered), dtype=object)
    customer_memory: dict[object, tuple[str, int]] = {}

    for i in range(len(ordered)):
        customer_id = customer_ids[i]
        state, streak = customer_memory.get(customer_id, (INSUFFICIENT_HISTORY, 0))
        has_history = bool(prior_counts[i] >= MIN_CUSTOMER_HISTORY and not np.isnan(hist_means[i]))
        next_state, next_streak = _step(state, streak, has_history, zscores[i])
        states[i] = next_state
        customer_memory[customer_id] = (next_state, next_streak)

    return pd.DataFrame(
        {
            "TRANSACTION_ID": ordered["TRANSACTION_ID"].to_numpy(),
            "customer_risk_state": states,
            "customer_amount_zscore": zscores,
            "customer_amount_deviation": ordered["customer_amount_deviation"].to_numpy(),
            "customer_new_terminal_flag": ordered["customer_new_terminal_flag"].to_numpy(),
        }
    )
