"""Terminal behavioral-risk engine (Dev Plan Sec 12/18; Phase 6 handoff Sec 6/10/11).

A separate, non-ML, interpretable statistical state machine that tracks each terminal's
behavioral state over time -- NORMAL / RISK_RISING / HIGH_RISK / RECOVERY -- built
entirely from Phase 3's already-computed terminal features. This module does NOT
duplicate feature computation and does NOT fit a model: it consumes
`terminal_prior_tx_count`, `terminal_hist_fraud_rate`, and `terminal_fraud_rate_deviation`
exactly as `mrs.features.terminal.build_terminal_features` already produces them, and
only adds a stateful, causal interpretation layer on top.

Architecturally this sits parallel to, not inside, the Phase 5 XGBoost transaction model
(Phase 6 handoff Sec 6):

    Transaction ML Risk (mrs.models.train_xgboost)
                    +
    Terminal Behavioral Risk (this module)
                    |
                    v
            Risk Aggregation (a later phase)

Temporal safety: every input this module reads is already a strictly-prior, leakage-safe
aggregate by construction (mrs.features._temporal / mrs.features.terminal -- see
docs/FEATURE_SPEC.md Sec 2 and Sec 7). This module adds no new aggregation over TX_FRAUD
or TX_AMOUNT; it only assigns a state to each transaction via a single forward pass over
each terminal's own transactions in chronological order, so a transaction's assigned
state can never depend on any later transaction of that terminal (or of any other
terminal -- terminals are fully isolated from each other).

State transition table (INSUFFICIENT_HISTORY / NORMAL / RISK_RISING / HIGH_RISK /
RECOVERY), where "level" = _target_level(deviation) in {0=calm, 1=rising, 2=high}:

    From \\ To          INSUFFICIENT   NORMAL          RISK_RISING     HIGH_RISK        RECOVERY
    INSUFFICIENT_HISTORY  stays*        level0 (first)  level1 (first)  level2 (first)   never
    NORMAL                 never         stays (lvl0)    -> lvl1         -> lvl2 (**)      never
    RISK_RISING             never         -> lvl0          stays (lvl1)   -> lvl2          never
    HIGH_RISK                 never         never directly   never directly  stays (lvl2)   -> whenever level<2 (only exit)
    RECOVERY                   never         -> after 3 confirmations  relapse if lvl1  relapse if lvl2   stays (counting or streak-reset)

    (*) INSUFFICIENT_HISTORY is entered once at the start of a terminal's life and left
        permanently -- terminal_prior_tx_count only increases for a real terminal, so it
        can never be re-entered.
    (**) Intentional direct jump: NORMAL/INSUFFICIENT_HISTORY -> HIGH_RISK is allowed
         when a single transaction's deviation exceeds HIGH_RISK_THRESHOLD directly. A
         sudden severe spike is not artificially delayed through an intermediate
         RISK_RISING label first -- RISK_RISING is the state for a moderate, watch-worthy
         elevation, not a mandatory waypoint for a severe one.

    Asymmetry (intentional): RISK_RISING -> NORMAL resolves immediately on the very next
    calm transaction (no confirmation streak). HIGH_RISK can only be exited through a
    *confirmed* RECOVERY (RECOVERY_CONFIRM_COUNT consecutive transactions at or below
    RECOVERY_CONFIRM_THRESHOLD). RISK_RISING is inherently tentative and cheap to
    reverse; a confirmed severe episode warrants a confirmed recovery before being
    cleared (Dev Plan Sec 18: concept drift and temporary fraud must be considered).

    From any state: whenever terminal_fraud_rate_deviation is NaN (no transactions in the
    recent 24h window to assess), the current state and streak are held unchanged --
    Dev Plan Sec 28: do not manufacture an anomaly score/transition from missing data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mrs.features._temporal import sort_canonical

#: A terminal needs at least this many strictly-prior transactions before its historical
#: fraud-rate baseline (terminal_hist_fraud_rate) is considered stable enough to assess
#: behavioral state against -- a project choice (the Dev Plan does not specify one),
#: mirroring the same kind of "how much history is enough" judgment already made for
#: terminal_volume_deviation's 1-hour stability threshold in mrs.features.terminal.
MIN_TERMINAL_HISTORY = 10

#: terminal_fraud_rate_deviation (recent 24h fraud rate minus historical fraud rate)
#: thresholds, in fraud-rate percentage points. Chosen against the measured ~0.8-0.9%
#: dataset-wide fraud rate (docs/DATASET_REPORT.md): a genuine Scenario-2 compromise
#: drives a terminal's recent rate far above these thresholds (compromised terminals are
#: overwhelmingly fraudulent during their active window -- 357 of 10,000 terminals were
#: ever compromised, docs/DATASET_REPORT.md Sec 5), while an isolated single fraud event
#: on an otherwise-normal terminal should not, by itself, cross RISING_THRESHOLD.
RISING_THRESHOLD = 0.05
HIGH_RISK_THRESHOLD = 0.15

#: A terminal must show recent-rate at or below its historical baseline (deviation <= 0)
#: for RECOVERY_CONFIRM_COUNT consecutive evaluable transactions before being declared
#: fully NORMAL again -- a small confirmation window so one quiet transaction cannot
#: erase a HIGH_RISK episode's memory on its own.
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
    "TERMINAL_ID",
    "TX_DATETIME",
    "terminal_prior_tx_count",
    "terminal_hist_fraud_rate",
    "terminal_fraud_rate_deviation",
)


def _target_level(deviation: float) -> int:
    """0=calm, 1=rising, 2=high -- a pure threshold lookup on one already-computed value."""
    if deviation > HIGH_RISK_THRESHOLD:
        return 2
    if deviation > RISING_THRESHOLD:
        return 1
    return 0


def _level_to_state(level: int) -> str:
    return {0: NORMAL, 1: RISK_RISING, 2: HIGH_RISK}[level]


def _step(state: str, streak: int, has_history: bool, deviation: float) -> tuple[str, int]:
    """One terminal's state transition for one transaction.

    A pure function: given the same (state, streak, has_history, deviation) it always
    returns the same next (state, streak) -- the transition logic is unit-testable in
    complete isolation from any DataFrame or looping machinery. See the module docstring
    for the full transition table this implements.
    """
    if not has_history:
        return INSUFFICIENT_HISTORY, 0

    if np.isnan(deviation):
        return state, streak

    level = _target_level(deviation)

    if state == INSUFFICIENT_HISTORY:
        return _level_to_state(level), 0

    if state == RECOVERY:
        if level >= 1:
            return _level_to_state(level), 0  # relapse: renewed elevation ends recovery
        if deviation <= RECOVERY_CONFIRM_THRESHOLD:
            streak += 1
            if streak >= RECOVERY_CONFIRM_COUNT:
                return NORMAL, 0
            return RECOVERY, streak
        return RECOVERY, 0  # calm-ish (level 0) but not below the stricter confirm bar

    # state in {NORMAL, RISK_RISING, HIGH_RISK}
    if state == HIGH_RISK and level < 2:
        # The only way out of HIGH_RISK is a confirmed RECOVERY, never a direct drop back
        # to NORMAL/RISK_RISING.
        initial_streak = 1 if deviation <= RECOVERY_CONFIRM_THRESHOLD else 0
        return RECOVERY, initial_streak

    return _level_to_state(level), 0


def _risk_score(state: str, deviation: float) -> float:
    """Bounded, non-negative, directly-interpretable risk score for Risk Aggregation.

    NaN when the state is INSUFFICIENT_HISTORY (no baseline yet) or when
    terminal_fraud_rate_deviation itself is NaN (empty recent window) -- a genuine
    "not yet assessable" (Dev Plan Sec 28), distinct from a confident 0.0. Otherwise
    relu(deviation) clipped to its own natural theoretical ceiling of 1.0 (deviation is a
    difference of two rates each in [0, 1], so it can never exceed 1.0 in the first
    place -- the clip is a defensive bound, not a rescaling). 0.0 means the terminal's
    recent behavior is at or better than its own historical baseline; a positive value is
    "recent 24h fraud rate exceeds this terminal's own historical baseline by that many
    percentage points."
    """
    if state == INSUFFICIENT_HISTORY or np.isnan(deviation):
        return np.nan
    return float(min(1.0, max(0.0, deviation)))


def compute_terminal_behavioral_states(df: pd.DataFrame) -> pd.DataFrame:
    """Assign a behavioral state and risk score to every transaction via one forward pass
    per terminal, in chronological order.

    Returns a frame with:
      - TRANSACTION_ID
      - terminal_risk_state: one of INSUFFICIENT_HISTORY/NORMAL/RISK_RISING/HIGH_RISK/RECOVERY
      - terminal_risk_score: bounded [0.0, 1.0] or NaN -- see _risk_score()
      - terminal_fraud_rate_deviation: the raw, signed Phase 3 evidence field, passed
        through unchanged, for explainability (Dev Plan Sec 17)

    Output row order is the canonical chronological order (Dev Plan Sec 5), not
    necessarily the caller's input order; join back on TRANSACTION_ID if a different
    order is needed.
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"compute_terminal_behavioral_states: missing columns: {missing}")

    ordered = sort_canonical(df)
    terminal_ids = ordered["TERMINAL_ID"].to_numpy()
    prior_counts = ordered["terminal_prior_tx_count"].to_numpy()
    hist_rates = ordered["terminal_hist_fraud_rate"].to_numpy()
    deviations = ordered["terminal_fraud_rate_deviation"].to_numpy()

    states = np.empty(len(ordered), dtype=object)
    scores = np.empty(len(ordered), dtype=np.float64)
    terminal_memory: dict[object, tuple[str, int]] = {}

    for i in range(len(ordered)):
        terminal_id = terminal_ids[i]
        state, streak = terminal_memory.get(terminal_id, (INSUFFICIENT_HISTORY, 0))
        has_history = bool(prior_counts[i] >= MIN_TERMINAL_HISTORY and not np.isnan(hist_rates[i]))
        next_state, next_streak = _step(state, streak, has_history, deviations[i])
        states[i] = next_state
        scores[i] = _risk_score(next_state, deviations[i])
        terminal_memory[terminal_id] = (next_state, next_streak)

    return pd.DataFrame(
        {
            "TRANSACTION_ID": ordered["TRANSACTION_ID"].to_numpy(),
            "terminal_risk_state": states,
            "terminal_risk_score": scores,
            "terminal_fraud_rate_deviation": deviations,
        }
    )
