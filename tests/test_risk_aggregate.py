"""Tests for mrs.risk.aggregate -- Phase 7 Risk Aggregation (rule/state-based, Option A).

Covers severity mapping, the unified-level decision table, contributing_signals
semantics, missing/NaN handling, duplicate-TRANSACTION_ID contract, outer-merge
coverage, determinism, row-order independence, the exact output contract, and a
representative sweep of all-three-component combinations. Small synthetic fixtures
only -- this module has no cross-row dependency, so no real dataset is needed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mrs.risk.aggregate import (
    CRITICAL,
    HIGH,
    HIGH_RISK,
    INSUFFICIENT_EVIDENCE,
    INSUFFICIENT_HISTORY,
    LOW,
    MEDIUM,
    NORMAL,
    OUTPUT_COLUMNS,
    RECOVERY,
    RISK_RISING,
    _behavioral_severity,
    _transaction_severity,
    aggregate_risk,
)

THRESHOLD = 0.97


def _tx_df(pairs):
    ids, risks = zip(*pairs)
    return pd.DataFrame({"TRANSACTION_ID": list(ids), "transaction_risk": list(risks)})


def _term_df(pairs):
    ids, states = zip(*pairs)
    return pd.DataFrame({"TRANSACTION_ID": list(ids), "terminal_risk_state": list(states)})


def _cust_df(pairs):
    ids, states = zip(*pairs)
    return pd.DataFrame({"TRANSACTION_ID": list(ids), "customer_risk_state": list(states)})


def _single_case(transaction_risk, terminal_state, customer_state, threshold=THRESHOLD, tid=1):
    result = aggregate_risk(
        _tx_df([(tid, transaction_risk)]),
        _term_df([(tid, terminal_state)]),
        _cust_df([(tid, customer_state)]),
        threshold,
    )
    return result.iloc[0]


# =====================================================================================
# _behavioral_severity: pure mapping function
# =====================================================================================


def test_behavioral_severity_normal_is_zero():
    assert _behavioral_severity(NORMAL) == 0


def test_behavioral_severity_risk_rising_is_one():
    assert _behavioral_severity(RISK_RISING) == 1


def test_behavioral_severity_recovery_is_one():
    assert _behavioral_severity(RECOVERY) == 1


def test_behavioral_severity_high_risk_is_two():
    assert _behavioral_severity(HIGH_RISK) == 2


def test_behavioral_severity_insufficient_history_is_unavailable_not_zero():
    assert _behavioral_severity(INSUFFICIENT_HISTORY) is None


def test_behavioral_severity_nan_is_unavailable():
    assert _behavioral_severity(np.nan) is None


def test_behavioral_severity_none_is_unavailable():
    assert _behavioral_severity(None) is None


def test_behavioral_severity_raises_on_unrecognized_state():
    with pytest.raises(ValueError, match="unrecognized behavioral state"):
        _behavioral_severity("SOMETHING_ELSE")


# =====================================================================================
# _transaction_severity: pure mapping function
# =====================================================================================


def test_transaction_severity_below_threshold_is_zero():
    assert _transaction_severity(0.5, 0.97) == 0


def test_transaction_severity_at_threshold_is_two():
    # >= is inclusive.
    assert _transaction_severity(0.97, 0.97) == 2


def test_transaction_severity_above_threshold_is_two():
    assert _transaction_severity(0.99, 0.97) == 2


def test_transaction_severity_nan_is_unavailable():
    assert _transaction_severity(np.nan, 0.97) is None


def test_transaction_severity_uses_supplied_threshold_not_a_fixed_value():
    # The same raw score is above one threshold and below another -- proves the
    # parameter is actually used, not a hardcoded 0.97 baked into the function.
    assert _transaction_severity(0.6, threshold=0.5) == 2
    assert _transaction_severity(0.6, threshold=0.97) == 0


# =====================================================================================
# aggregate_risk: input validation
# =====================================================================================


def test_raises_on_missing_transaction_df_columns():
    bad = pd.DataFrame({"TRANSACTION_ID": [1]})
    with pytest.raises(ValueError, match="transaction_df missing columns"):
        aggregate_risk(bad, _term_df([(1, NORMAL)]), _cust_df([(1, NORMAL)]), THRESHOLD)


def test_raises_on_missing_terminal_df_columns():
    bad = pd.DataFrame({"TRANSACTION_ID": [1]})
    with pytest.raises(ValueError, match="terminal_df missing columns"):
        aggregate_risk(_tx_df([(1, 0.1)]), bad, _cust_df([(1, NORMAL)]), THRESHOLD)


def test_raises_on_missing_customer_df_columns():
    bad = pd.DataFrame({"TRANSACTION_ID": [1]})
    with pytest.raises(ValueError, match="customer_df missing columns"):
        aggregate_risk(_tx_df([(1, 0.1)]), _term_df([(1, NORMAL)]), bad, THRESHOLD)


def test_raises_on_duplicate_transaction_id_in_transaction_df():
    dup = pd.DataFrame({"TRANSACTION_ID": [1, 1], "transaction_risk": [0.1, 0.9]})
    with pytest.raises(ValueError, match="transaction_df has duplicate TRANSACTION_ID"):
        aggregate_risk(dup, _term_df([(1, NORMAL)]), _cust_df([(1, NORMAL)]), THRESHOLD)


def test_raises_on_duplicate_transaction_id_in_terminal_df():
    dup = pd.DataFrame({"TRANSACTION_ID": [1, 1], "terminal_risk_state": [NORMAL, HIGH_RISK]})
    with pytest.raises(ValueError, match="terminal_df has duplicate TRANSACTION_ID"):
        aggregate_risk(_tx_df([(1, 0.1)]), dup, _cust_df([(1, NORMAL)]), THRESHOLD)


def test_raises_on_duplicate_transaction_id_in_customer_df():
    dup = pd.DataFrame({"TRANSACTION_ID": [1, 1], "customer_risk_state": [NORMAL, HIGH_RISK]})
    with pytest.raises(ValueError, match="customer_df has duplicate TRANSACTION_ID"):
        aggregate_risk(_tx_df([(1, 0.1)]), _term_df([(1, NORMAL)]), dup, THRESHOLD)


# =====================================================================================
# Unified risk level decision table
# =====================================================================================


def test_all_normal_and_below_threshold_yields_low():
    row = _single_case(0.1, NORMAL, NORMAL)
    assert row["unified_risk_level"] == LOW
    assert row["contributing_signals"] == []


def test_one_risk_rising_component_yields_medium():
    row = _single_case(0.1, RISK_RISING, NORMAL)
    assert row["unified_risk_level"] == MEDIUM


def test_one_recovery_component_yields_medium():
    row = _single_case(0.1, RECOVERY, NORMAL)
    assert row["unified_risk_level"] == MEDIUM


def test_one_high_risk_component_yields_high():
    row = _single_case(0.1, HIGH_RISK, NORMAL)
    assert row["unified_risk_level"] == HIGH


def test_transaction_alone_at_high_severity_yields_high():
    row = _single_case(0.99, NORMAL, NORMAL)
    assert row["unified_risk_level"] == HIGH


def test_two_high_risk_components_yields_critical():
    row = _single_case(0.1, HIGH_RISK, HIGH_RISK)
    assert row["unified_risk_level"] == CRITICAL


def test_three_high_severity_components_yields_critical():
    row = _single_case(0.99, HIGH_RISK, HIGH_RISK)
    assert row["unified_risk_level"] == CRITICAL


def test_transaction_and_one_behavioral_high_yields_critical():
    row = _single_case(0.99, HIGH_RISK, NORMAL)
    assert row["unified_risk_level"] == CRITICAL


def test_all_unavailable_yields_insufficient_evidence():
    row = _single_case(np.nan, INSUFFICIENT_HISTORY, INSUFFICIENT_HISTORY)
    assert row["unified_risk_level"] == INSUFFICIENT_EVIDENCE
    assert row["contributing_signals"] == []


def test_insufficient_history_not_treated_as_normal_does_not_dilute_to_low():
    # If INSUFFICIENT_HISTORY were wrongly treated as severity 0, this would still
    # correctly be HIGH (transaction is severe) -- but the point is severity 2 count
    # must be based on truly AVAILABLE components, not padded with a fake 0.
    row = _single_case(0.99, INSUFFICIENT_HISTORY, INSUFFICIENT_HISTORY)
    assert row["unified_risk_level"] == HIGH
    assert row["terminal_risk_severity"] is None
    assert row["customer_risk_severity"] is None


def test_high_risk_plus_unavailable_component_yields_high_not_critical():
    # Only ONE component is actually available at severity 2 -- INSUFFICIENT_HISTORY
    # must not be silently counted as a second corroborating severe signal.
    row = _single_case(np.nan, HIGH_RISK, INSUFFICIENT_HISTORY)
    assert row["unified_risk_level"] == HIGH
    assert row["transaction_risk_severity"] is None
    assert row["customer_risk_severity"] is None


# =====================================================================================
# contributing_signals semantics
# =====================================================================================


def test_contributing_signals_empty_for_low():
    row = _single_case(0.1, NORMAL, NORMAL)
    assert row["contributing_signals"] == []


def test_contributing_signals_empty_for_insufficient_evidence():
    row = _single_case(np.nan, INSUFFICIENT_HISTORY, INSUFFICIENT_HISTORY)
    assert row["contributing_signals"] == []


def test_contributing_signals_for_medium_lists_all_severity_one_components():
    row = _single_case(0.1, RISK_RISING, RECOVERY)
    assert row["unified_risk_level"] == MEDIUM
    assert row["contributing_signals"] == [
        "terminal_behavioral_risk: RISK_RISING",
        "customer_behavioral_risk: RECOVERY",
    ]


def test_contributing_signals_for_high_lists_only_the_severe_component():
    # The exact worked example from the approved spec.
    row = _single_case(0.42, HIGH_RISK, RISK_RISING, threshold=0.97)
    assert row["unified_risk_level"] == HIGH
    assert row["contributing_signals"] == ["terminal_behavioral_risk: HIGH_RISK"]


def test_contributing_signals_for_critical_lists_all_severe_components():
    row = _single_case(0.99, HIGH_RISK, NORMAL, threshold=0.97)
    assert row["unified_risk_level"] == CRITICAL
    assert row["contributing_signals"] == [
        "transaction_ml_risk >= 0.97",
        "terminal_behavioral_risk: HIGH_RISK",
    ]


def test_contributing_signals_does_not_mention_a_present_but_non_driving_component():
    row = _single_case(0.1, HIGH_RISK, RISK_RISING)
    assert row["unified_risk_level"] == HIGH
    assert "customer_behavioral_risk" not in " ".join(row["contributing_signals"])
    # But the raw state remains visible for explainability.
    assert row["customer_risk_state"] == RISK_RISING


def test_contributing_signals_transaction_text_uses_actual_supplied_threshold():
    row = _single_case(0.9, NORMAL, NORMAL, threshold=0.83)
    assert row["unified_risk_level"] == HIGH
    assert row["contributing_signals"] == ["transaction_ml_risk >= 0.83"]


# =====================================================================================
# Outer-merge coverage / missing component rows
# =====================================================================================


def test_transaction_id_present_only_in_terminal_df_is_still_output():
    result = aggregate_risk(
        _tx_df([(2, 0.5)]),
        _term_df([(1, HIGH_RISK), (2, NORMAL)]),
        _cust_df([(2, NORMAL)]),
        THRESHOLD,
    )
    row1 = result.set_index("TRANSACTION_ID").loc[1]
    assert pd.isna(row1["transaction_risk"])
    # A column mixing None (this row, unavailable) and an int (another row's severity)
    # is coerced by pandas to float64, so an unavailable severity surfaces as NaN here,
    # not literally None -- check with pd.isna(), not `is None` (mirrors the same
    # pandas behavior already documented in tests/test_model_compare.py).
    assert pd.isna(row1["transaction_risk_severity"])
    assert pd.isna(row1["customer_risk_state"])
    assert pd.isna(row1["customer_risk_severity"])
    assert row1["terminal_risk_state"] == HIGH_RISK
    assert row1["unified_risk_level"] == HIGH


def test_output_row_count_equals_union_of_all_input_transaction_ids():
    tx = _tx_df([(1, 0.1), (2, 0.2)])
    term = _term_df([(2, NORMAL), (3, NORMAL)])
    cust = _cust_df([(3, NORMAL), (4, NORMAL)])
    result = aggregate_risk(tx, term, cust, THRESHOLD)
    assert set(result["TRANSACTION_ID"]) == {1, 2, 3, 4}
    assert len(result) == 4
    assert not result["TRANSACTION_ID"].duplicated().any()


# =====================================================================================
# Determinism / row-order independence / no cross-row leakage
# =====================================================================================


def test_deterministic_repeated_calls():
    tx = _tx_df([(1, 0.1), (2, 0.99)])
    term = _term_df([(1, NORMAL), (2, HIGH_RISK)])
    cust = _cust_df([(1, NORMAL), (2, RISK_RISING)])
    result_a = aggregate_risk(tx, term, cust, THRESHOLD)
    result_b = aggregate_risk(tx, term, cust, THRESHOLD)
    pd.testing.assert_frame_equal(result_a, result_b)


def test_input_row_order_independence():
    tx = _tx_df([(1, 0.1), (2, 0.99), (3, 0.5)])
    term = _term_df([(1, NORMAL), (2, HIGH_RISK), (3, RISK_RISING)])
    cust = _cust_df([(1, NORMAL), (2, NORMAL), (3, NORMAL)])

    result_in_order = aggregate_risk(tx, term, cust, THRESHOLD)
    shuffled_result = aggregate_risk(
        tx.sample(frac=1, random_state=1).reset_index(drop=True),
        term.sample(frac=1, random_state=2).reset_index(drop=True),
        cust.sample(frac=1, random_state=3).reset_index(drop=True),
        THRESHOLD,
    )
    pd.testing.assert_frame_equal(result_in_order, shuffled_result)


def test_changing_one_row_does_not_affect_another_rows_result():
    tx = _tx_df([(1, 0.1), (2, 0.1)])
    term = _term_df([(1, NORMAL), (2, NORMAL)])
    cust = _cust_df([(1, NORMAL), (2, NORMAL)])
    before = aggregate_risk(tx, term, cust, THRESHOLD).set_index("TRANSACTION_ID").loc[1]

    mutated_term = _term_df([(1, NORMAL), (2, HIGH_RISK)])  # only row 2 changes
    after = aggregate_risk(tx, mutated_term, cust, THRESHOLD).set_index("TRANSACTION_ID").loc[1]

    assert before["unified_risk_level"] == after["unified_risk_level"] == LOW
    assert before["contributing_signals"] == after["contributing_signals"]


# =====================================================================================
# Output contract
# =====================================================================================


def test_output_columns_exactly_match_documented_contract():
    result = aggregate_risk(_tx_df([(1, 0.1)]), _term_df([(1, NORMAL)]), _cust_df([(1, NORMAL)]), THRESHOLD)
    assert list(result.columns) == list(OUTPUT_COLUMNS)


def test_no_customer_risk_score_column_is_introduced():
    result = aggregate_risk(_tx_df([(1, 0.1)]), _term_df([(1, NORMAL)]), _cust_df([(1, NORMAL)]), THRESHOLD)
    assert "customer_risk_score" not in result.columns


def test_raw_states_and_transaction_risk_value_are_preserved_regardless_of_level():
    row = _single_case(0.31, RECOVERY, RISK_RISING)
    assert row["transaction_risk"] == pytest.approx(0.31)
    assert row["terminal_risk_state"] == RECOVERY
    assert row["customer_risk_state"] == RISK_RISING


def test_no_tx_fraud_or_tx_fraud_scenario_columns_required():
    # These input frames never contain label columns at all -- confirms the function
    # neither requires nor reads them.
    tx = _tx_df([(1, 0.1)])
    term = _term_df([(1, NORMAL)])
    cust = _cust_df([(1, NORMAL)])
    assert "TX_FRAUD" not in tx.columns and "TX_FRAUD" not in term.columns and "TX_FRAUD" not in cust.columns
    assert "TX_FRAUD_SCENARIO" not in tx.columns
    result = aggregate_risk(tx, term, cust, THRESHOLD)
    assert "TX_FRAUD" not in result.columns
    assert "TX_FRAUD_SCENARIO" not in result.columns


def test_extra_unrelated_columns_in_inputs_are_ignored_not_leaked_into_output():
    tx = _tx_df([(1, 0.1)])
    tx["some_other_column"] = "decoy"
    term = _term_df([(1, NORMAL)])
    term["terminal_risk_score"] = 0.0  # a real Phase 6 column, still shouldn't leak through
    cust = _cust_df([(1, NORMAL)])
    cust["customer_amount_zscore"] = 1.23
    result = aggregate_risk(tx, term, cust, THRESHOLD)
    assert "some_other_column" not in result.columns
    assert "terminal_risk_score" not in result.columns
    assert "customer_amount_zscore" not in result.columns


# =====================================================================================
# Representative combined-scenario sweep (all three component types together)
# =====================================================================================


@pytest.mark.parametrize(
    "transaction_risk,terminal_state,customer_state,expected_level",
    [
        (0.1, NORMAL, NORMAL, LOW),
        (0.1, RISK_RISING, NORMAL, MEDIUM),
        (0.1, NORMAL, RECOVERY, MEDIUM),
        (0.1, RISK_RISING, RISK_RISING, MEDIUM),
        (0.1, HIGH_RISK, NORMAL, HIGH),
        (0.1, NORMAL, HIGH_RISK, HIGH),
        (0.99, NORMAL, NORMAL, HIGH),
        (0.99, HIGH_RISK, NORMAL, CRITICAL),
        (0.99, NORMAL, HIGH_RISK, CRITICAL),
        (0.1, HIGH_RISK, HIGH_RISK, CRITICAL),
        (0.99, HIGH_RISK, HIGH_RISK, CRITICAL),
        (np.nan, NORMAL, NORMAL, LOW),
        (np.nan, RISK_RISING, INSUFFICIENT_HISTORY, MEDIUM),
        (np.nan, INSUFFICIENT_HISTORY, INSUFFICIENT_HISTORY, INSUFFICIENT_EVIDENCE),
        (0.99, INSUFFICIENT_HISTORY, INSUFFICIENT_HISTORY, HIGH),
    ],
)
def test_representative_combinations(transaction_risk, terminal_state, customer_state, expected_level):
    row = _single_case(transaction_risk, terminal_state, customer_state, threshold=THRESHOLD)
    assert row["unified_risk_level"] == expected_level
