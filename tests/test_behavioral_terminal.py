"""Tests for mrs.behavioral.terminal -- the Phase 6 terminal behavioral state machine.

Tests the exact approved transition table (see the module docstring in
src/mrs/behavioral/terminal.py): both the pure per-step transition function in complete
isolation, and the batch compute_terminal_behavioral_states() integration behavior
(chronological ordering, cold-start, terminal isolation, missing data, temporal leakage,
determinism, and the score/evidence-field contract). Small synthetic fixtures only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mrs.behavioral.terminal import (
    HIGH_RISK,
    HIGH_RISK_THRESHOLD,
    INSUFFICIENT_HISTORY,
    MIN_TERMINAL_HISTORY,
    NORMAL,
    RECOVERY,
    RECOVERY_CONFIRM_COUNT,
    RECOVERY_CONFIRM_THRESHOLD,
    RISING_THRESHOLD,
    RISK_RISING,
    _risk_score,
    _step,
    compute_terminal_behavioral_states,
)

CALM = 0.0
MODERATE = (RISING_THRESHOLD + HIGH_RISK_THRESHOLD) / 2  # strictly between the two thresholds
SEVERE = HIGH_RISK_THRESHOLD + 0.10
JUST_ABOVE_RISING = RISING_THRESHOLD + 0.001
JUST_BELOW_RISING = RISING_THRESHOLD - 0.001


# =====================================================================================
# _step: pure transition function, tested in complete isolation
# =====================================================================================


def test_step_no_history_returns_insufficient_history_and_resets_streak():
    state, streak = _step(NORMAL, 5, has_history=False, deviation=SEVERE)
    assert state == INSUFFICIENT_HISTORY
    assert streak == 0


# --- INSUFFICIENT_HISTORY -> {NORMAL, RISK_RISING, HIGH_RISK} on first eligible row ---


def test_step_insufficient_history_to_normal_on_first_eligible_calm():
    state, streak = _step(INSUFFICIENT_HISTORY, 0, has_history=True, deviation=CALM)
    assert (state, streak) == (NORMAL, 0)


def test_step_insufficient_history_to_risk_rising_on_first_eligible_moderate():
    state, streak = _step(INSUFFICIENT_HISTORY, 0, has_history=True, deviation=MODERATE)
    assert (state, streak) == (RISK_RISING, 0)


def test_step_insufficient_history_to_high_risk_direct_jump_on_first_eligible_severe():
    # Documented intentional transition: cold start can jump straight to HIGH_RISK.
    state, streak = _step(INSUFFICIENT_HISTORY, 0, has_history=True, deviation=SEVERE)
    assert (state, streak) == (HIGH_RISK, 0)


# --- NORMAL -> {NORMAL, RISK_RISING, HIGH_RISK} ---


def test_step_normal_stays_normal_when_calm():
    assert _step(NORMAL, 0, True, CALM) == (NORMAL, 0)


def test_step_normal_to_risk_rising():
    assert _step(NORMAL, 0, True, MODERATE) == (RISK_RISING, 0)


def test_step_normal_to_high_risk_direct_jump():
    # Documented intentional transition: a severe spike is not delayed through RISK_RISING.
    assert _step(NORMAL, 0, True, SEVERE) == (HIGH_RISK, 0)


# --- RISK_RISING -> {NORMAL, RISK_RISING, HIGH_RISK} ---


def test_step_risk_rising_to_normal_resolves_immediately_no_confirmation_needed():
    # Asymmetry vs. HIGH_RISK: RISK_RISING clears on the very next calm transaction.
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
    assert streak == 0  # MODERATE > RECOVERY_CONFIRM_THRESHOLD, doesn't count toward exit yet


@pytest.mark.parametrize("deviation", [CALM, MODERATE, JUST_BELOW_RISING])
def test_step_high_risk_never_drops_directly_to_normal_or_risk_rising(deviation):
    state, _ = _step(HIGH_RISK, 0, True, deviation)
    assert state == RECOVERY
    assert state not in (NORMAL, RISK_RISING)


# --- RECOVERY -> {RECOVERY, NORMAL, RISK_RISING, HIGH_RISK} ---


def test_step_recovery_streak_increments_when_calm_below_confirm_threshold():
    assert _step(RECOVERY, 1, True, -0.01) == (RECOVERY, 2)


def test_step_recovery_confirms_to_normal_at_exactly_the_required_count():
    # Not before: streaks 1 and 2 (of 3) must stay in RECOVERY.
    assert _step(RECOVERY, 0, True, CALM) == (RECOVERY, 1)
    assert _step(RECOVERY, 1, True, CALM) == (RECOVERY, 2)
    assert RECOVERY_CONFIRM_COUNT == 3
    assert _step(RECOVERY, 2, True, CALM) == (NORMAL, 0)


def test_step_recovery_streak_resets_without_relapse_when_calm_but_above_confirm_threshold():
    # level 0 (<= RISING_THRESHOLD) but > RECOVERY_CONFIRM_THRESHOLD: still "calm enough"
    # to stay in RECOVERY (not a relapse), but doesn't count toward the exit streak.
    deviation = RECOVERY_CONFIRM_THRESHOLD + 0.01
    assert deviation <= RISING_THRESHOLD  # still level 0
    state, streak = _step(RECOVERY, 2, True, deviation)
    assert (state, streak) == (RECOVERY, 0)


def test_step_recovery_relapses_to_risk_rising_on_moderate_elevation():
    assert _step(RECOVERY, 2, True, MODERATE) == (RISK_RISING, 0)


def test_step_recovery_relapses_to_high_risk_on_severe_elevation():
    assert _step(RECOVERY, 2, True, SEVERE) == (HIGH_RISK, 0)


# --- NaN deviation: hold current state/streak, from every state ---


@pytest.mark.parametrize("state,streak", [
    (INSUFFICIENT_HISTORY, 0), (NORMAL, 0), (RISK_RISING, 0), (HIGH_RISK, 0), (RECOVERY, 2),
])
def test_step_holds_state_and_streak_on_nan_deviation_from_any_state(state, streak):
    assert _step(state, streak, True, float("nan")) == (state, streak)


def test_step_is_pure_and_deterministic():
    args = (RISK_RISING, 0, True, MODERATE)
    assert _step(*args) == _step(*args)


# =====================================================================================
# _risk_score
# =====================================================================================


def test_risk_score_nan_when_insufficient_history():
    assert np.isnan(_risk_score(INSUFFICIENT_HISTORY, SEVERE))


def test_risk_score_nan_when_deviation_is_nan():
    assert np.isnan(_risk_score(NORMAL, float("nan")))


def test_risk_score_zero_when_deviation_negative():
    assert _risk_score(NORMAL, -0.5) == 0.0


def test_risk_score_zero_when_deviation_exactly_zero():
    assert _risk_score(NORMAL, 0.0) == 0.0


def test_risk_score_equals_deviation_when_in_bounds():
    assert _risk_score(RISK_RISING, MODERATE) == pytest.approx(MODERATE)
    assert _risk_score(HIGH_RISK, SEVERE) == pytest.approx(SEVERE)


def test_risk_score_capped_at_one_for_an_out_of_range_deviation():
    # Deviation cannot exceed 1.0 in practice (difference of two rates in [0,1]), but the
    # clip is a defensive bound -- verify it actually clips rather than just trusting it.
    assert _risk_score(HIGH_RISK, 1.5) == 1.0


def test_risk_score_is_never_negative_across_a_range_of_inputs():
    for deviation in (-1.0, -0.3, -0.001, 0.0, 0.001, 0.3, 1.0):
        score = _risk_score(NORMAL, deviation)
        assert score >= 0.0


# =====================================================================================
# compute_terminal_behavioral_states: batch/integration behavior
# =====================================================================================


def _sequence_df(entries: list[tuple[int, float, float]], terminal_id: int = 1, start: str = "2018-04-01") -> pd.DataFrame:
    """entries: (prior_count, hist_fraud_rate, deviation) tuples, one row per
    transaction, already in the intended chronological order for one terminal."""
    n = len(entries)
    return pd.DataFrame(
        {
            "TRANSACTION_ID": np.arange(n) + terminal_id * 100_000,
            "TERMINAL_ID": terminal_id,
            "TX_DATETIME": pd.to_datetime(start) + pd.to_timedelta(np.arange(n), unit="h"),
            "terminal_prior_tx_count": [e[0] for e in entries],
            "terminal_hist_fraud_rate": [e[1] for e in entries],
            "terminal_fraud_rate_deviation": [e[2] for e in entries],
        }
    )


def test_raises_on_missing_required_columns():
    df = _sequence_df([(10, 0.01, CALM)])
    incomplete = df.drop(columns=["terminal_fraud_rate_deviation"])
    with pytest.raises(ValueError, match="missing columns"):
        compute_terminal_behavioral_states(incomplete)


def test_full_lifecycle_single_terminal_normal_rising_high_recovery_normal():
    entries = [
        (10, 0.01, CALM),      # -> NORMAL (first eligible)
        (11, 0.01, MODERATE),  # -> RISK_RISING
        (12, 0.01, SEVERE),    # -> HIGH_RISK
        (13, 0.01, SEVERE),    # -> HIGH_RISK (stays)
        (14, 0.01, CALM),      # -> RECOVERY, streak=1
        (15, 0.01, CALM),      # -> RECOVERY, streak=2
        (16, 0.01, CALM),      # -> NORMAL (confirmed)
    ]
    df = _sequence_df(entries)
    result = compute_terminal_behavioral_states(df)

    assert list(result["terminal_risk_state"]) == [
        NORMAL, RISK_RISING, HIGH_RISK, HIGH_RISK, RECOVERY, RECOVERY, NORMAL,
    ]


def test_cold_start_terminal_stays_insufficient_history_until_min_transactions():
    # prior_count below MIN_TERMINAL_HISTORY for the first few rows.
    entries = [(i, np.nan, np.nan) for i in range(MIN_TERMINAL_HISTORY)]  # 0..MIN-1 prior
    entries.append((MIN_TERMINAL_HISTORY, 0.01, CALM))  # now eligible
    df = _sequence_df(entries)
    result = compute_terminal_behavioral_states(df)

    states = list(result["terminal_risk_state"])
    assert states[:-1] == [INSUFFICIENT_HISTORY] * MIN_TERMINAL_HISTORY
    assert states[-1] == NORMAL


def test_insufficient_history_when_hist_fraud_rate_is_nan_even_with_enough_prior_count():
    # Guards the AND in has_history: prior_count alone is not sufficient.
    df = _sequence_df([(MIN_TERMINAL_HISTORY, np.nan, CALM)])
    result = compute_terminal_behavioral_states(df)
    assert result["terminal_risk_state"].iloc[0] == INSUFFICIENT_HISTORY


def test_terminals_are_isolated_from_each_other():
    # Terminal 1 escalates to HIGH_RISK; terminal 2 stays calm throughout, interleaved
    # chronologically with terminal 1's transactions.
    hot = _sequence_df(
        [(10, 0.01, CALM), (11, 0.01, SEVERE), (12, 0.01, SEVERE)], terminal_id=1, start="2018-04-01 00:00"
    )
    calm = _sequence_df(
        [(10, 0.01, CALM), (11, 0.01, CALM), (12, 0.01, CALM)], terminal_id=2, start="2018-04-01 00:30"
    )
    combined = pd.concat([hot, calm], ignore_index=True)

    result = compute_terminal_behavioral_states(combined)
    by_id = result.set_index("TRANSACTION_ID")

    hot_ids = hot["TRANSACTION_ID"]
    calm_ids = calm["TRANSACTION_ID"]
    assert list(by_id.loc[hot_ids, "terminal_risk_state"]) == [NORMAL, HIGH_RISK, HIGH_RISK]
    assert list(by_id.loc[calm_ids, "terminal_risk_state"]) == [NORMAL, NORMAL, NORMAL]


def test_output_order_is_canonical_chronological_regardless_of_input_row_order():
    df = _sequence_df([(10, 0.01, CALM), (11, 0.01, MODERATE), (12, 0.01, SEVERE)])
    shuffled_input = df.sample(frac=1, random_state=0).reset_index(drop=True)

    result_from_shuffled = compute_terminal_behavioral_states(shuffled_input)

    assert list(result_from_shuffled["TRANSACTION_ID"]) == list(df["TRANSACTION_ID"])


def test_row_shuffle_does_not_change_computed_states_for_any_transaction():
    hot = _sequence_df(
        [(10, 0.01, CALM), (11, 0.01, MODERATE), (12, 0.01, SEVERE), (13, 0.01, CALM)],
        terminal_id=1, start="2018-04-01 00:00",
    )
    calm = _sequence_df(
        [(10, 0.01, CALM), (11, 0.01, CALM), (12, 0.01, CALM)], terminal_id=2, start="2018-04-01 00:15"
    )
    combined = pd.concat([hot, calm], ignore_index=True)
    shuffled = combined.sample(frac=1, random_state=42).reset_index(drop=True)

    result_unshuffled = compute_terminal_behavioral_states(combined).set_index("TRANSACTION_ID")
    result_shuffled = compute_terminal_behavioral_states(shuffled).set_index("TRANSACTION_ID")

    pd.testing.assert_series_equal(
        result_unshuffled["terminal_risk_state"].sort_index(),
        result_shuffled["terminal_risk_state"].sort_index(),
    )


def test_missing_recent_window_holds_previous_state_across_a_gap():
    entries = [
        (10, 0.01, MODERATE),  # -> RISK_RISING
        (11, 0.01, np.nan),    # no recent activity -> hold RISK_RISING
        (12, 0.01, np.nan),    # still hold
        (13, 0.01, CALM),      # now resolves -> NORMAL
    ]
    df = _sequence_df(entries)
    result = compute_terminal_behavioral_states(df)

    assert list(result["terminal_risk_state"]) == [RISK_RISING, RISK_RISING, RISK_RISING, NORMAL]
    # Score must also reflect "not assessable" during the held gap, not a stale number.
    assert np.isnan(result["terminal_risk_score"].iloc[1])
    assert np.isnan(result["terminal_risk_score"].iloc[2])


def test_temporal_leakage_future_row_mutation_does_not_affect_earlier_state():
    entries = [
        (10, 0.01, CALM),      # transaction of interest: established as NORMAL
        (11, 0.01, CALM),
        (12, 0.01, CALM),
    ]
    df = _sequence_df(entries)
    before = compute_terminal_behavioral_states(df)
    established_id = df["TRANSACTION_ID"].iloc[0]
    established_before = before.set_index("TRANSACTION_ID").loc[established_id]

    mutated = df.copy()
    # Mutate a LATER transaction's deviation to an extreme value.
    mutated.loc[mutated.index[-1], "terminal_fraud_rate_deviation"] = SEVERE
    after = compute_terminal_behavioral_states(mutated)
    established_after = after.set_index("TRANSACTION_ID").loc[established_id]

    assert established_before["terminal_risk_state"] == established_after["terminal_risk_state"]
    assert established_before["terminal_risk_score"] == pytest.approx(established_after["terminal_risk_score"])


def test_output_is_deterministic_across_repeated_calls():
    df = _sequence_df([(10, 0.01, CALM), (11, 0.01, SEVERE), (12, 0.01, CALM)])
    result_a = compute_terminal_behavioral_states(df)
    result_b = compute_terminal_behavioral_states(df)
    pd.testing.assert_frame_equal(result_a, result_b)


def test_evidence_field_passes_through_unchanged_including_sign_and_nan():
    entries = [(10, 0.01, -0.2), (11, 0.01, 0.3), (12, 0.01, np.nan)]
    df = _sequence_df(entries)
    result = compute_terminal_behavioral_states(df)

    expected = [e[2] for e in entries]
    actual = result["terminal_fraud_rate_deviation"].tolist()
    for exp, act in zip(expected, actual):
        if np.isnan(exp):
            assert np.isnan(act)
        else:
            assert act == pytest.approx(exp)


def test_output_row_count_and_transaction_id_coverage_matches_input():
    df = _sequence_df([(10, 0.01, CALM), (11, 0.01, MODERATE), (12, 0.01, SEVERE)])
    result = compute_terminal_behavioral_states(df)

    assert len(result) == len(df)
    assert set(result["TRANSACTION_ID"]) == set(df["TRANSACTION_ID"])
    assert not result["TRANSACTION_ID"].duplicated().any()


def test_output_columns_are_exactly_the_documented_contract():
    df = _sequence_df([(10, 0.01, CALM)])
    result = compute_terminal_behavioral_states(df)
    assert list(result.columns) == [
        "TRANSACTION_ID", "terminal_risk_state", "terminal_risk_score", "terminal_fraud_rate_deviation",
    ]
