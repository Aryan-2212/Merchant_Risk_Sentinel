"""Tests for mrs.behavioral.customer -- the Phase 7 customer behavioral state machine.

Mirrors tests/test_behavioral_terminal.py's structure and depth: the exact approved
transition table (see the module docstring in src/mrs/behavioral/customer.py), tested
both as a pure per-step function in isolation and via the batch
compute_customer_behavioral_states() integration behavior (chronological ordering,
cold-start, customer isolation, missing/zero-variance data, temporal leakage,
determinism, and the evidence-field pass-through contract -- no risk-score field exists
in this module by design). Small synthetic fixtures only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mrs.behavioral.customer import (
    HIGH_RISK,
    HIGH_RISK_THRESHOLD,
    INSUFFICIENT_HISTORY,
    MIN_CUSTOMER_HISTORY,
    NORMAL,
    RECOVERY,
    RECOVERY_CONFIRM_COUNT,
    RECOVERY_CONFIRM_THRESHOLD,
    RISING_THRESHOLD,
    RISK_RISING,
    _step,
    _target_level,
    compute_customer_behavioral_states,
)

CALM = 0.0
MODERATE = (RISING_THRESHOLD + HIGH_RISK_THRESHOLD) / 2  # strictly between the two thresholds
SEVERE = HIGH_RISK_THRESHOLD + 2.0
JUST_BELOW_RISING = RISING_THRESHOLD - 0.01


# =====================================================================================
# _step: pure transition function, tested in complete isolation
# =====================================================================================


def test_step_no_history_returns_insufficient_history_and_resets_streak():
    state, streak = _step(NORMAL, 5, has_history=False, zscore=SEVERE)
    assert state == INSUFFICIENT_HISTORY
    assert streak == 0


# --- INSUFFICIENT_HISTORY -> {NORMAL, RISK_RISING, HIGH_RISK} on first eligible row ---


def test_step_insufficient_history_to_normal_on_first_eligible_calm():
    assert _step(INSUFFICIENT_HISTORY, 0, True, CALM) == (NORMAL, 0)


def test_step_insufficient_history_to_risk_rising_on_first_eligible_moderate():
    assert _step(INSUFFICIENT_HISTORY, 0, True, MODERATE) == (RISK_RISING, 0)


def test_step_insufficient_history_to_high_risk_direct_jump_on_first_eligible_severe():
    assert _step(INSUFFICIENT_HISTORY, 0, True, SEVERE) == (HIGH_RISK, 0)


# --- NORMAL -> {NORMAL, RISK_RISING, HIGH_RISK} ---


def test_step_normal_stays_normal_when_calm():
    assert _step(NORMAL, 0, True, CALM) == (NORMAL, 0)


def test_step_normal_stays_normal_when_moderately_negative():
    # Positive-only escalation: a customer spending LESS than usual is not the Scenario-3
    # pattern and must not trigger any elevation.
    assert _step(NORMAL, 0, True, -10.0) == (NORMAL, 0)


def test_step_normal_to_risk_rising():
    assert _step(NORMAL, 0, True, MODERATE) == (RISK_RISING, 0)


def test_step_normal_to_high_risk_direct_jump():
    assert _step(NORMAL, 0, True, SEVERE) == (HIGH_RISK, 0)


# --- RISK_RISING -> {NORMAL, RISK_RISING, HIGH_RISK} ---


def test_step_risk_rising_to_normal_resolves_immediately_no_confirmation_needed():
    assert _step(RISK_RISING, 0, True, CALM) == (NORMAL, 0)


def test_step_risk_rising_stays_risk_rising_when_still_moderate():
    assert _step(RISK_RISING, 0, True, MODERATE) == (RISK_RISING, 0)


def test_step_risk_rising_to_high_risk():
    assert _step(RISK_RISING, 0, True, SEVERE) == (HIGH_RISK, 0)


# --- HIGH_RISK -> {HIGH_RISK, RECOVERY} -- never directly to NORMAL/RISK_RISING ---


def test_step_high_risk_stays_high_risk_when_still_severe():
    assert _step(HIGH_RISK, 0, True, SEVERE) == (HIGH_RISK, 0)


def test_step_high_risk_to_recovery_when_level_drops_to_calm():
    state, streak = _step(HIGH_RISK, 0, True, CALM)
    assert state == RECOVERY
    assert streak == 1  # CALM (0.0) <= RECOVERY_CONFIRM_THRESHOLD (0.0), counts immediately


def test_step_high_risk_to_recovery_when_level_drops_to_moderate():
    state, streak = _step(HIGH_RISK, 0, True, MODERATE)
    assert state == RECOVERY
    assert streak == 0


@pytest.mark.parametrize("zscore", [CALM, MODERATE, JUST_BELOW_RISING, -5.0])
def test_step_high_risk_never_drops_directly_to_normal_or_risk_rising(zscore):
    state, _ = _step(HIGH_RISK, 0, True, zscore)
    assert state == RECOVERY
    assert state not in (NORMAL, RISK_RISING)


# --- RECOVERY -> {RECOVERY, NORMAL, RISK_RISING, HIGH_RISK} ---


def test_step_recovery_streak_increments_when_calm_below_confirm_threshold():
    assert _step(RECOVERY, 1, True, -0.5) == (RECOVERY, 2)


def test_step_recovery_confirms_to_normal_at_exactly_the_required_count():
    assert _step(RECOVERY, 0, True, CALM) == (RECOVERY, 1)
    assert _step(RECOVERY, 1, True, CALM) == (RECOVERY, 2)
    assert RECOVERY_CONFIRM_COUNT == 3
    assert _step(RECOVERY, 2, True, CALM) == (NORMAL, 0)


def test_step_recovery_streak_resets_without_relapse_when_calm_but_above_confirm_threshold():
    zscore = RECOVERY_CONFIRM_THRESHOLD + 0.5
    assert zscore <= RISING_THRESHOLD  # still level 0
    state, streak = _step(RECOVERY, 2, True, zscore)
    assert (state, streak) == (RECOVERY, 0)


def test_step_recovery_relapses_to_risk_rising_on_moderate_elevation():
    assert _step(RECOVERY, 2, True, MODERATE) == (RISK_RISING, 0)


def test_step_recovery_relapses_to_high_risk_on_severe_elevation():
    assert _step(RECOVERY, 2, True, SEVERE) == (HIGH_RISK, 0)


# --- NaN zscore (zero-variance case): hold current state/streak, from every state ---


@pytest.mark.parametrize("state,streak", [
    (INSUFFICIENT_HISTORY, 0), (NORMAL, 0), (RISK_RISING, 0), (HIGH_RISK, 0), (RECOVERY, 2),
])
def test_step_holds_state_and_streak_on_nan_zscore_from_any_state(state, streak):
    assert _step(state, streak, True, float("nan")) == (state, streak)


def test_step_is_pure_and_deterministic():
    args = (RISK_RISING, 0, True, MODERATE)
    assert _step(*args) == _step(*args)


def test_target_level_boundaries():
    assert _target_level(RISING_THRESHOLD) == 0  # inclusive boundary: exactly at threshold is calm
    assert _target_level(RISING_THRESHOLD + 0.0001) == 1
    assert _target_level(HIGH_RISK_THRESHOLD) == 1  # inclusive boundary: exactly at threshold is rising
    assert _target_level(HIGH_RISK_THRESHOLD + 0.0001) == 2


# =====================================================================================
# compute_customer_behavioral_states: batch/integration behavior
# =====================================================================================


def _sequence_df(entries: list[tuple[int, float, float, float, int]], customer_id: int = 1, start: str = "2018-04-01") -> pd.DataFrame:
    """entries: (prior_count, hist_amount_mean, zscore, deviation, new_terminal_flag)
    tuples, one row per transaction, already in the intended chronological order for one
    customer."""
    n = len(entries)
    return pd.DataFrame(
        {
            "TRANSACTION_ID": np.arange(n) + customer_id * 100_000,
            "CUSTOMER_ID": customer_id,
            "TX_DATETIME": pd.to_datetime(start) + pd.to_timedelta(np.arange(n), unit="h"),
            "customer_prior_tx_count": [e[0] for e in entries],
            "customer_hist_amount_mean": [e[1] for e in entries],
            "customer_amount_zscore": [e[2] for e in entries],
            "customer_amount_deviation": [e[3] for e in entries],
            "customer_new_terminal_flag": [e[4] for e in entries],
        }
    )


def test_raises_on_missing_required_columns():
    df = _sequence_df([(10, 100.0, CALM, 0.0, 0)])
    incomplete = df.drop(columns=["customer_amount_zscore"])
    with pytest.raises(ValueError, match="missing columns"):
        compute_customer_behavioral_states(incomplete)


def test_full_lifecycle_single_customer_normal_rising_high_recovery_normal():
    entries = [
        (10, 100.0, CALM, 0.0, 0),          # -> NORMAL (first eligible)
        (11, 100.0, MODERATE, 300.0, 0),    # -> RISK_RISING
        (12, 100.0, SEVERE, 800.0, 1),      # -> HIGH_RISK
        (13, 100.0, SEVERE, 800.0, 0),      # -> HIGH_RISK (stays)
        (14, 100.0, CALM, 0.0, 0),          # -> RECOVERY, streak=1
        (15, 100.0, CALM, 0.0, 0),          # -> RECOVERY, streak=2
        (16, 100.0, CALM, 0.0, 0),          # -> NORMAL (confirmed)
    ]
    df = _sequence_df(entries)
    result = compute_customer_behavioral_states(df)

    assert list(result["customer_risk_state"]) == [
        NORMAL, RISK_RISING, HIGH_RISK, HIGH_RISK, RECOVERY, RECOVERY, NORMAL,
    ]


def test_cold_start_customer_stays_insufficient_history_until_min_transactions():
    entries = [(i, np.nan, np.nan, np.nan, 1) for i in range(MIN_CUSTOMER_HISTORY)]
    entries.append((MIN_CUSTOMER_HISTORY, 100.0, CALM, 0.0, 0))
    df = _sequence_df(entries)
    result = compute_customer_behavioral_states(df)

    states = list(result["customer_risk_state"])
    assert states[:-1] == [INSUFFICIENT_HISTORY] * MIN_CUSTOMER_HISTORY
    assert states[-1] == NORMAL


def test_insufficient_history_when_hist_amount_mean_is_nan_even_with_enough_prior_count():
    df = _sequence_df([(MIN_CUSTOMER_HISTORY, np.nan, CALM, 0.0, 0)])
    result = compute_customer_behavioral_states(df)
    assert result["customer_risk_state"].iloc[0] == INSUFFICIENT_HISTORY


def test_zero_variance_nan_zscore_holds_state_despite_sufficient_history():
    # A customer with perfectly uniform prior spending (hist_std == 0): zscore is
    # genuinely NaN despite ample history and a defined mean -- must hold, not reset to
    # cold-start and not manufacture a score.
    entries = [
        (10, 100.0, MODERATE, 300.0, 0),  # -> RISK_RISING
        (11, 100.0, np.nan, 0.0, 0),      # zero-variance gap -> hold RISK_RISING
        (12, 100.0, np.nan, 0.0, 0),      # still hold
        (13, 100.0, CALM, 0.0, 0),        # resolves -> NORMAL
    ]
    df = _sequence_df(entries)
    result = compute_customer_behavioral_states(df)
    assert list(result["customer_risk_state"]) == [RISK_RISING, RISK_RISING, RISK_RISING, NORMAL]


def test_customers_are_isolated_from_each_other():
    hot = _sequence_df(
        [(10, 100.0, CALM, 0.0, 0), (11, 100.0, SEVERE, 900.0, 0), (12, 100.0, SEVERE, 900.0, 0)],
        customer_id=1, start="2018-04-01 00:00",
    )
    calm = _sequence_df(
        [(10, 100.0, CALM, 0.0, 0), (11, 100.0, CALM, 0.0, 0), (12, 100.0, CALM, 0.0, 0)],
        customer_id=2, start="2018-04-01 00:30",
    )
    combined = pd.concat([hot, calm], ignore_index=True)

    result = compute_customer_behavioral_states(combined)
    by_id = result.set_index("TRANSACTION_ID")

    assert list(by_id.loc[hot["TRANSACTION_ID"], "customer_risk_state"]) == [NORMAL, HIGH_RISK, HIGH_RISK]
    assert list(by_id.loc[calm["TRANSACTION_ID"], "customer_risk_state"]) == [NORMAL, NORMAL, NORMAL]


def test_output_order_is_canonical_chronological_regardless_of_input_row_order():
    df = _sequence_df([(10, 100.0, CALM, 0.0, 0), (11, 100.0, MODERATE, 300.0, 0), (12, 100.0, SEVERE, 900.0, 1)])
    shuffled_input = df.sample(frac=1, random_state=0).reset_index(drop=True)

    result_from_shuffled = compute_customer_behavioral_states(shuffled_input)

    assert list(result_from_shuffled["TRANSACTION_ID"]) == list(df["TRANSACTION_ID"])


def test_row_shuffle_does_not_change_computed_states_for_any_transaction():
    hot = _sequence_df(
        [(10, 100.0, CALM, 0.0, 0), (11, 100.0, MODERATE, 300.0, 0), (12, 100.0, SEVERE, 900.0, 0), (13, 100.0, CALM, 0.0, 0)],
        customer_id=1, start="2018-04-01 00:00",
    )
    calm = _sequence_df(
        [(10, 100.0, CALM, 0.0, 0), (11, 100.0, CALM, 0.0, 0), (12, 100.0, CALM, 0.0, 0)],
        customer_id=2, start="2018-04-01 00:15",
    )
    combined = pd.concat([hot, calm], ignore_index=True)
    shuffled = combined.sample(frac=1, random_state=42).reset_index(drop=True)

    result_unshuffled = compute_customer_behavioral_states(combined).set_index("TRANSACTION_ID")
    result_shuffled = compute_customer_behavioral_states(shuffled).set_index("TRANSACTION_ID")

    pd.testing.assert_series_equal(
        result_unshuffled["customer_risk_state"].sort_index(),
        result_shuffled["customer_risk_state"].sort_index(),
    )


def test_temporal_leakage_future_row_mutation_does_not_affect_earlier_state():
    entries = [
        (10, 100.0, CALM, 0.0, 0),  # transaction of interest: established as NORMAL
        (11, 100.0, CALM, 0.0, 0),
        (12, 100.0, CALM, 0.0, 0),
    ]
    df = _sequence_df(entries)
    before = compute_customer_behavioral_states(df)
    established_id = df["TRANSACTION_ID"].iloc[0]
    established_before = before.set_index("TRANSACTION_ID").loc[established_id]

    mutated = df.copy()
    mutated.loc[mutated.index[-1], "customer_amount_zscore"] = SEVERE
    mutated.loc[mutated.index[-1], "customer_amount_deviation"] = 5000.0
    after = compute_customer_behavioral_states(mutated)
    established_after = after.set_index("TRANSACTION_ID").loc[established_id]

    assert established_before["customer_risk_state"] == established_after["customer_risk_state"]
    assert established_before["customer_amount_zscore"] == pytest.approx(established_after["customer_amount_zscore"])


def test_output_is_deterministic_across_repeated_calls():
    df = _sequence_df([(10, 100.0, CALM, 0.0, 0), (11, 100.0, SEVERE, 900.0, 1), (12, 100.0, CALM, 0.0, 0)])
    result_a = compute_customer_behavioral_states(df)
    result_b = compute_customer_behavioral_states(df)
    pd.testing.assert_frame_equal(result_a, result_b)


def test_evidence_fields_pass_through_unchanged_including_sign_and_nan():
    entries = [
        (10, 100.0, -1.5, -150.0, 0),
        (11, 100.0, 3.0, 300.0, 1),
        (12, 100.0, np.nan, 0.0, 0),
    ]
    df = _sequence_df(entries)
    result = compute_customer_behavioral_states(df)

    expected_z = [e[2] for e in entries]
    expected_dev = [e[3] for e in entries]
    expected_flag = [e[4] for e in entries]

    for exp, act in zip(expected_z, result["customer_amount_zscore"].tolist()):
        if np.isnan(exp):
            assert np.isnan(act)
        else:
            assert act == pytest.approx(exp)
    for exp, act in zip(expected_dev, result["customer_amount_deviation"].tolist()):
        assert act == pytest.approx(exp)
    assert result["customer_new_terminal_flag"].tolist() == expected_flag


def test_no_risk_score_field_is_present_by_design():
    # Explicit guard against silently reintroducing a normalized score field: the
    # approved design deliberately excludes one until Risk Aggregation defines its own
    # normalization requirement.
    df = _sequence_df([(10, 100.0, CALM, 0.0, 0)])
    result = compute_customer_behavioral_states(df)
    assert "customer_risk_score" not in result.columns


def test_output_columns_are_exactly_the_documented_contract():
    df = _sequence_df([(10, 100.0, CALM, 0.0, 0)])
    result = compute_customer_behavioral_states(df)
    assert list(result.columns) == [
        "TRANSACTION_ID", "customer_risk_state", "customer_amount_zscore",
        "customer_amount_deviation", "customer_new_terminal_flag",
    ]


def test_output_row_count_and_transaction_id_coverage_matches_input():
    df = _sequence_df([(10, 100.0, CALM, 0.0, 0), (11, 100.0, MODERATE, 300.0, 0), (12, 100.0, SEVERE, 900.0, 1)])
    result = compute_customer_behavioral_states(df)

    assert len(result) == len(df)
    assert set(result["TRANSACTION_ID"]) == set(df["TRANSACTION_ID"])
    assert not result["TRANSACTION_ID"].duplicated().any()
