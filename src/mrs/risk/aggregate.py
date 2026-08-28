"""Risk Aggregation (Dev Plan Sec 13/14/38; Phase 7 handoff, approved Option A design).

Combines three already-computed component signals into a unified risk level plus
structured evidence:

    Transaction ML Risk (mrs.models.train_xgboost, via predict_proba)
                    +
    Customer Behavioral Risk (mrs.behavioral.customer.compute_customer_behavioral_states)
                    +
    Terminal Behavioral Risk (mrs.behavioral.terminal.compute_terminal_behavioral_states)
                    v
        Rule/State-Based Risk Aggregation (this module)
                    v
        Unified Risk Level + Structured Evidence

This is rule/state-based aggregation, NOT a second ML model and NOT a weighted numeric
blend. No permanent weights are fitted or invented anywhere in this module. It performs
no model inference, no feature engineering, no feature recomputation, and no file I/O --
it is a pure function over three already-computed pandas DataFrames, and never reads
TX_FRAUD or TX_FRAUD_SCENARIO. This component does not determine fraud independently and
does not execute any action; it produces evidence for a later policy/analyst phase.

Component representation:

- Terminal and customer behavioral risk are represented by their NATIVE state
  (NORMAL/RISK_RISING/HIGH_RISK/RECOVERY/INSUFFICIENT_HISTORY), mapped to a shared 0/1/2
  severity ordinal below. Customer risk is deliberately NOT represented by a normalized
  score -- mrs.behavioral.customer intentionally does not expose one, and that decision
  is unchanged here.
- Transaction ML risk uses the caller-supplied, already-validated Phase 5 operating
  threshold (models/xgboost_v1/metadata.json["threshold"], confirmed to be the actual
  max-F1-on-validation threshold used to compute both Phase 5's validation and test
  metrics) -- never hardcoded or duplicated inside this module.
- No independent temporal/spike risk component is currently implemented. It is
  therefore excluded from this aggregation version rather than being manufactured or
  implicitly represented as another signal -- the two behavioral engines' own use of
  temporal history does not make them a separate temporal_risk component.

Severity mapping:

    NORMAL               -> 0
    RISK_RISING          -> 1
    RECOVERY             -> 1
    HIGH_RISK            -> 2
    INSUFFICIENT_HISTORY -> unavailable (never treated as NORMAL/0)
    NaN / missing        -> unavailable (never treated as NORMAL/0 or as calm evidence)

    transaction_risk >= threshold -> 2
    transaction_risk <  threshold -> 0
    NaN transaction_risk          -> unavailable

Unified risk level decision table (over the AVAILABLE component severities only):

    all components unavailable          -> INSUFFICIENT_EVIDENCE
    >= 2 components at severity 2        -> CRITICAL
    max available severity == 2          -> HIGH
    max available severity == 1          -> MEDIUM
    max available severity == 0          -> LOW

This is a transparent max-of-severities-plus-corroboration rule, not a weighted sum and
not a fitted model (Dev Plan Sec 13/38).

contributing_signals semantics (generated only from actual structured component
fields -- nothing invented):

    LOW / INSUFFICIENT_EVIDENCE -> []
    MEDIUM                      -> every component at severity 1
    HIGH / CRITICAL              -> every component at severity 2

A component present in the output's raw state/value columns but NOT at the level's
target severity is never listed as a contributing signal, even though its raw
state/value remains visible for explainability (Dev Plan Sec 38: keep component scores
separately, show why risk increased -- without implying a component "caused" an outcome
it did not actually drive).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

NORMAL = "NORMAL"
RISK_RISING = "RISK_RISING"
HIGH_RISK = "HIGH_RISK"
RECOVERY = "RECOVERY"
INSUFFICIENT_HISTORY = "INSUFFICIENT_HISTORY"

_KNOWN_STATES = frozenset({NORMAL, RISK_RISING, RECOVERY, HIGH_RISK, INSUFFICIENT_HISTORY})

_STATE_SEVERITY: dict[str, int] = {
    NORMAL: 0,
    RISK_RISING: 1,
    RECOVERY: 1,
    HIGH_RISK: 2,
    # INSUFFICIENT_HISTORY intentionally absent: has no severity, handled explicitly.
}

LOW = "LOW"
MEDIUM = "MEDIUM"
HIGH = "HIGH"
CRITICAL = "CRITICAL"
INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"

_TRANSACTION_SIGNAL = "transaction_ml_risk"
_TERMINAL_SIGNAL = "terminal_behavioral_risk"
_CUSTOMER_SIGNAL = "customer_behavioral_risk"
_CANONICAL_SIGNAL_ORDER = (_TRANSACTION_SIGNAL, _TERMINAL_SIGNAL, _CUSTOMER_SIGNAL)

REQUIRED_TRANSACTION_COLUMNS = ("TRANSACTION_ID", "transaction_risk")
REQUIRED_TERMINAL_COLUMNS = ("TRANSACTION_ID", "terminal_risk_state")
REQUIRED_CUSTOMER_COLUMNS = ("TRANSACTION_ID", "customer_risk_state")

OUTPUT_COLUMNS = (
    "TRANSACTION_ID",
    "unified_risk_level",
    "transaction_risk",
    "transaction_risk_severity",
    "terminal_risk_state",
    "terminal_risk_severity",
    "customer_risk_state",
    "customer_risk_severity",
    "contributing_signals",
)


def _is_missing(value) -> bool:
    return value is None or (isinstance(value, float) and np.isnan(value))


def _behavioral_severity(state) -> int | None:
    """Map a behavioral engine's native state to a shared 0/1/2 severity, or None
    ("unavailable") for INSUFFICIENT_HISTORY or a missing/NaN state. Absence of evidence
    is never treated as evidence of calm (Dev Plan Sec 28/33.7).
    """
    if _is_missing(state):
        return None
    if state not in _KNOWN_STATES:
        raise ValueError(f"aggregate_risk: unrecognized behavioral state: {state!r}")
    if state == INSUFFICIENT_HISTORY:
        return None
    return _STATE_SEVERITY[state]


def _transaction_severity(transaction_risk, threshold: float) -> int | None:
    """2 if transaction_risk >= threshold, else 0; None ("unavailable") if missing/NaN.

    No invented middle tier -- this is the single already-validated Phase 5 cut-point,
    supplied by the caller, never hardcoded here.
    """
    if _is_missing(transaction_risk):
        return None
    return 2 if transaction_risk >= threshold else 0


def _unified_level(available_severities: dict[str, int]) -> str:
    if not available_severities:
        return INSUFFICIENT_EVIDENCE
    n_severe = sum(1 for v in available_severities.values() if v == 2)
    if n_severe >= 2:
        return CRITICAL
    max_severity = max(available_severities.values())
    if max_severity == 2:
        return HIGH
    if max_severity == 1:
        return MEDIUM
    return LOW


def _signal_text(name: str, transaction_risk, terminal_state, customer_state, threshold: float) -> str:
    if name == _TRANSACTION_SIGNAL:
        return f"{_TRANSACTION_SIGNAL} >= {threshold}"
    if name == _TERMINAL_SIGNAL:
        return f"{_TERMINAL_SIGNAL}: {terminal_state}"
    return f"{_CUSTOMER_SIGNAL}: {customer_state}"


def _contributing_signals(
    level: str,
    available_severities: dict[str, int],
    transaction_risk,
    terminal_state,
    customer_state,
    threshold: float,
) -> list[str]:
    if level in (LOW, INSUFFICIENT_EVIDENCE):
        return []
    target_severity = 1 if level == MEDIUM else 2  # HIGH or CRITICAL
    return [
        _signal_text(name, transaction_risk, terminal_state, customer_state, threshold)
        for name in _CANONICAL_SIGNAL_ORDER
        if available_severities.get(name) == target_severity
    ]


def _validate_columns(df: pd.DataFrame, required: tuple[str, ...], name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"aggregate_risk: {name} missing columns: {missing}")


def _validate_unique_transaction_id(df: pd.DataFrame, name: str) -> None:
    if df["TRANSACTION_ID"].duplicated().any():
        dupes = sorted(df.loc[df["TRANSACTION_ID"].duplicated(keep=False), "TRANSACTION_ID"].unique().tolist())
        raise ValueError(
            f"aggregate_risk: {name} has duplicate TRANSACTION_ID values: {dupes[:10]}"
            f"{' ...' if len(dupes) > 10 else ''} -- each input frame must have exactly "
            "one row per TRANSACTION_ID."
        )


def aggregate_risk(
    transaction_df: pd.DataFrame,
    terminal_df: pd.DataFrame,
    customer_df: pd.DataFrame,
    transaction_risk_threshold: float,
) -> pd.DataFrame:
    """Combine three already-computed component signals into a unified risk assessment.

    transaction_df: TRANSACTION_ID, transaction_risk (e.g. Phase 5's predict_proba output).
    terminal_df: output of mrs.behavioral.terminal.compute_terminal_behavioral_states
        (only TRANSACTION_ID/terminal_risk_state are used).
    customer_df: output of mrs.behavioral.customer.compute_customer_behavioral_states
        (only TRANSACTION_ID/customer_risk_state are used).
    transaction_risk_threshold: the caller-supplied, already-validated Phase 5 operating
        threshold (e.g. read from models/xgboost_v1/metadata.json["threshold"]).

    Joins the three inputs on TRANSACTION_ID via an outer merge: a TRANSACTION_ID absent
    from any one input is explicitly "unavailable" for that component, never silently
    treated as NORMAL/calm. Each input must have a unique TRANSACTION_ID per row (a
    duplicate is treated as a caller contract violation and raises ValueError).

    Returns one row per TRANSACTION_ID present in the union of all three inputs, with
    columns exactly matching OUTPUT_COLUMNS, sorted by TRANSACTION_ID for a deterministic,
    reproducible order (this function has no cross-row dependency at all, so output order
    carries no chronological meaning -- it is not a temporal computation).
    """
    _validate_columns(transaction_df, REQUIRED_TRANSACTION_COLUMNS, "transaction_df")
    _validate_columns(terminal_df, REQUIRED_TERMINAL_COLUMNS, "terminal_df")
    _validate_columns(customer_df, REQUIRED_CUSTOMER_COLUMNS, "customer_df")
    _validate_unique_transaction_id(transaction_df, "transaction_df")
    _validate_unique_transaction_id(terminal_df, "terminal_df")
    _validate_unique_transaction_id(customer_df, "customer_df")

    tx = transaction_df[["TRANSACTION_ID", "transaction_risk"]]
    term = terminal_df[["TRANSACTION_ID", "terminal_risk_state"]]
    cust = customer_df[["TRANSACTION_ID", "customer_risk_state"]]

    merged = tx.merge(term, on="TRANSACTION_ID", how="outer").merge(cust, on="TRANSACTION_ID", how="outer")
    merged = merged.sort_values("TRANSACTION_ID").reset_index(drop=True)

    rows = []
    for row in merged.itertuples(index=False):
        transaction_risk = row.transaction_risk
        terminal_state = row.terminal_risk_state
        customer_state = row.customer_risk_state

        tx_severity = _transaction_severity(transaction_risk, transaction_risk_threshold)
        term_severity = _behavioral_severity(terminal_state)
        cust_severity = _behavioral_severity(customer_state)

        available = {
            k: v
            for k, v in (
                (_TRANSACTION_SIGNAL, tx_severity),
                (_TERMINAL_SIGNAL, term_severity),
                (_CUSTOMER_SIGNAL, cust_severity),
            )
            if v is not None
        }

        level = _unified_level(available)
        signals = _contributing_signals(
            level, available, transaction_risk, terminal_state, customer_state, transaction_risk_threshold
        )

        rows.append(
            {
                "TRANSACTION_ID": row.TRANSACTION_ID,
                "unified_risk_level": level,
                "transaction_risk": transaction_risk,
                "transaction_risk_severity": tx_severity,
                "terminal_risk_state": terminal_state,
                "terminal_risk_severity": term_severity,
                "customer_risk_state": customer_state,
                "customer_risk_severity": cust_severity,
                "contributing_signals": signals,
            }
        )

    return pd.DataFrame(rows, columns=list(OUTPUT_COLUMNS))
