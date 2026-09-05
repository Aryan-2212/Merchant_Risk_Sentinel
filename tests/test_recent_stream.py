"""Tests for mrs.data.recent_stream -- the Simulated Recent Operational Stream.

Covers exactly the properties the addendum spec calls out: determinism, the 21-day
window, schema conformance, temporal integrity, label isolation, behavioral evolution,
and compatibility with the existing feature/model/behavioral/aggregation pipeline.
None of these touch a database -- everything here is in-memory pandas, like the rest
of the Phase 1-7 test suite.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mrs import config
from mrs.behavioral.customer import compute_customer_behavioral_states
from mrs.behavioral.terminal import compute_terminal_behavioral_states
from mrs.data.recent_stream import generate_recent_stream
from mrs.data.schema import LABEL_COLUMNS
from mrs.features.build import build_feature_frame
from mrs.features.registry import FEATURE_NAMES
from mrs.models.dataset import attach_labels, get_feature_matrix
from mrs.risk.aggregate import aggregate_risk


@pytest.fixture(scope="module")
def recent_df() -> pd.DataFrame:
    """Generated once per test module -- generation is pure/deterministic and every
    test in this module reads it, never mutates it (pandas frames are not copied
    defensively here, so a test that needs to mutate must .copy() first)."""
    if not (config.REFERENCE_DIR / "customer_profiles.parquet").exists():
        pytest.skip("data/reference profiles not present; run scripts/03_reproduce_profiles.py")
    return generate_recent_stream()


@pytest.fixture(scope="module")
def full_df(recent_df: pd.DataFrame) -> pd.DataFrame:
    features = build_feature_frame(recent_df, split_override=config.RECENT_STREAM_SPLIT_LABEL)
    joined = attach_labels(features, recent_df)
    return joined.merge(
        recent_df[["TRANSACTION_ID", "TX_TIME_SECONDS", "TX_TIME_DAYS"]],
        on="TRANSACTION_ID",
        how="left",
        validate="one_to_one",
    )


# ------------------------------------------------------------------------ determinism


def test_same_seed_produces_identical_stream():
    a = generate_recent_stream(seed=config.RECENT_STREAM_SEED)
    b = generate_recent_stream(seed=config.RECENT_STREAM_SEED)
    pd.testing.assert_frame_equal(a, b)


def test_different_seed_produces_a_different_stream():
    a = generate_recent_stream(seed=config.RECENT_STREAM_SEED)
    b = generate_recent_stream(seed=config.RECENT_STREAM_SEED + 1)
    assert not a["TX_AMOUNT"].equals(b["TX_AMOUNT"])


def test_same_seed_and_start_date_produces_identical_stream_regardless_of_when_called():
    """start_date is a configuration value (mrs.config.RECENT_STREAM_START_DATE by
    default), never derived from datetime.now() -- passing the same one explicitly,
    twice, must be exactly as deterministic as relying on the default."""
    a = generate_recent_stream(seed=config.RECENT_STREAM_SEED, start_date="2027-01-01")
    b = generate_recent_stream(seed=config.RECENT_STREAM_SEED, start_date="2027-01-01")
    pd.testing.assert_frame_equal(a, b)


def test_start_date_is_configurable_and_shifts_the_window(recent_df: pd.DataFrame):
    shifted = generate_recent_stream(seed=config.RECENT_STREAM_SEED, start_date="2027-01-01")
    assert shifted["TX_DATETIME"].min().normalize() == pd.Timestamp("2027-01-01")
    assert shifted["TX_TIME_DAYS"].nunique() == config.RECENT_STREAM_DAYS
    # Same seed, same relative cohort/phase structure -- only calendar time moved.
    assert len(shifted) == len(recent_df)


# ----------------------------------------------------------------------------- window


def test_exactly_21_days_represented(recent_df: pd.DataFrame):
    assert recent_df["TX_TIME_DAYS"].nunique() == config.RECENT_STREAM_DAYS == 21


def test_date_range_matches_configured_window(recent_df: pd.DataFrame):
    assert recent_df["TX_DATETIME"].min().normalize() == pd.Timestamp(config.RECENT_STREAM_START_DATE)
    assert recent_df["TX_DATETIME"].max().normalize() == pd.Timestamp(config.RECENT_STREAM_END_DATE)


def test_transaction_count_is_approximately_the_target(recent_df: pd.DataFrame):
    target = config.RECENT_STREAM_DAYS * config.RECENT_STREAM_TX_PER_DAY
    # Injected compromised-terminal traffic adds on top of the organic per-day budget,
    # so the total is >= the organic target, not exactly equal to it -- "approximately"
    # per the spec, bounded to a sane multiple rather than left unchecked.
    assert target <= len(recent_df) <= target * 1.5


# ----------------------------------------------------------------------------- schema


#: The frozen benchmark's real, verified max TRANSACTION_ID (1,754,155 rows, 0-indexed
#: -- confirmed against the populated database during end-to-end validation). Asserted
#: directly here, not just against the offset constant, so this test would fail if
#: RECENT_STREAM_TX_ID_OFFSET were ever misconfigured below the benchmark's actual range.
_BENCHMARK_MAX_TRANSACTION_ID = 1_754_154


def test_generated_frame_has_no_transaction_id_collision_with_benchmark(recent_df: pd.DataFrame):
    assert config.RECENT_STREAM_TX_ID_OFFSET > _BENCHMARK_MAX_TRANSACTION_ID
    assert (recent_df["TRANSACTION_ID"] > _BENCHMARK_MAX_TRANSACTION_ID).all()
    assert (recent_df["TRANSACTION_ID"] >= config.RECENT_STREAM_TX_ID_OFFSET).all()
    assert not recent_df["TRANSACTION_ID"].duplicated().any()


def test_generated_frame_reuses_existing_entity_ids(recent_df: pd.DataFrame):
    customer_profiles = pd.read_parquet(config.REFERENCE_DIR / "customer_profiles.parquet")
    terminal_profiles = pd.read_parquet(config.REFERENCE_DIR / "terminal_profiles.parquet")
    assert set(recent_df["CUSTOMER_ID"]).issubset(set(customer_profiles["CUSTOMER_ID"]))
    assert set(recent_df["TERMINAL_ID"]).issubset(set(terminal_profiles["TERMINAL_ID"]))


def test_schema_matches_processed_layer_conventions(recent_df: pd.DataFrame):
    # generate_recent_stream() already calls validate_processed_frame internally;
    # this test asserts that contract explicitly rather than only implicitly.
    from mrs.data.schema import RAW_COLUMNS, validate_processed_frame

    assert tuple(recent_df.columns) == RAW_COLUMNS
    validate_processed_frame(recent_df, source="test_recent_stream")


# ----------------------------------------------------------- temporal integrity / leakage


def test_split_is_recent_never_the_frozen_benchmark_labels(full_df: pd.DataFrame):
    assert set(full_df["split"].unique()) == {config.RECENT_STREAM_SPLIT_LABEL}


def test_customer_history_features_never_see_a_later_transaction(full_df: pd.DataFrame):
    """For every customer, customer_prior_tx_count at row i must equal exactly the
    number of that customer's OWN transactions strictly before row i's timestamp --
    i.e. it cannot have counted anything from later in the stream."""
    merged = full_df.sort_values(["TX_DATETIME", "TRANSACTION_ID"]).reset_index(drop=True)
    for customer_id, group in merged.groupby("CUSTOMER_ID"):
        expected = np.arange(len(group))  # 0 prior, 1 prior, 2 prior, ... in chronological order
        assert (group["customer_prior_tx_count"].to_numpy() == expected).all(), customer_id


def test_terminal_history_features_never_see_a_later_transaction(full_df: pd.DataFrame):
    merged = full_df.sort_values(["TX_DATETIME", "TRANSACTION_ID"]).reset_index(drop=True)
    for terminal_id, group in merged.groupby("TERMINAL_ID"):
        expected = np.arange(len(group))
        assert (group["terminal_prior_tx_count"].to_numpy() == expected).all(), terminal_id


# --------------------------------------------------------------------- label isolation


def test_tx_fraud_and_scenario_are_never_registered_as_features():
    assert LABEL_COLUMNS.isdisjoint(FEATURE_NAMES)


def test_tx_fraud_and_scenario_absent_from_the_model_feature_matrix(full_df: pd.DataFrame):
    X = get_feature_matrix(full_df)
    assert "TX_FRAUD" not in X.columns
    assert "TX_FRAUD_SCENARIO" not in X.columns


def test_genuine_rows_have_scenario_zero_and_fraud_rows_have_nonzero_scenario(recent_df: pd.DataFrame):
    genuine = recent_df["TX_FRAUD"] == 0
    assert (recent_df.loc[genuine, "TX_FRAUD_SCENARIO"] == 0).all()
    assert (recent_df.loc[~genuine, "TX_FRAUD_SCENARIO"] != 0).all()


# ------------------------------------------------------------------ behavioral evolution


def test_customer_engine_shows_the_full_state_progression(full_df: pd.DataFrame):
    states = compute_customer_behavioral_states(full_df)
    observed = set(states["customer_risk_state"].unique())
    assert {"NORMAL", "RISK_RISING", "HIGH_RISK", "RECOVERY"}.issubset(observed)


def test_terminal_engine_shows_the_full_state_progression(full_df: pd.DataFrame):
    states = compute_terminal_behavioral_states(full_df)
    observed = set(states["terminal_risk_state"].unique())
    assert {"NORMAL", "RISK_RISING", "HIGH_RISK", "RECOVERY"}.issubset(observed)


def test_at_least_one_terminal_completes_the_full_normal_to_recovery_arc(recent_df: pd.DataFrame, full_df: pd.DataFrame):
    states = compute_terminal_behavioral_states(full_df)
    joined = states.merge(recent_df[["TRANSACTION_ID", "TERMINAL_ID"]], on="TRANSACTION_ID")
    arcs = joined.groupby("TERMINAL_ID")["terminal_risk_state"].apply(set)
    full_arc = {"NORMAL", "RISK_RISING", "HIGH_RISK", "RECOVERY"}
    assert (arcs.apply(lambda s: full_arc.issubset(s))).any()


def test_week_one_is_baseline_no_elevated_terminal_states_yet(recent_df: pd.DataFrame, full_df: pd.DataFrame):
    """Day numbers 1-7 (TX_TIME_DAYS 0-6) are the deliberately un-elevated baseline
    window (start_day is sampled from [8, 10]) -- no terminal should be RISK_RISING/
    HIGH_RISK/RECOVERY that early."""
    states = compute_terminal_behavioral_states(full_df)
    joined = states.merge(recent_df[["TRANSACTION_ID", "TX_TIME_DAYS"]], on="TRANSACTION_ID")
    week1 = joined[joined["TX_TIME_DAYS"] < 7]
    assert not week1["terminal_risk_state"].isin(["RISK_RISING", "HIGH_RISK", "RECOVERY"]).any()


def test_recovery_entities_genuinely_return_toward_baseline_in_chronological_order(
    recent_df: pd.DataFrame, full_df: pd.DataFrame
):
    """Not just 'RECOVERY appears somewhere in the set of states' (already covered by
    test_at_least_one_terminal_completes_the_full_normal_to_recovery_arc) but the
    actual chronological ORDER: for at least one terminal, a HIGH_RISK transaction is
    genuinely followed later (not earlier) by a RECOVERY transaction, which is in turn
    followed later by a return to NORMAL -- i.e. a real NORMAL -> ... -> HIGH_RISK ->
    RECOVERY -> NORMAL arc in time, not merely all four labels appearing anywhere in an
    unordered set for that entity."""
    states = compute_terminal_behavioral_states(full_df)
    joined = states.merge(
        recent_df[["TRANSACTION_ID", "TERMINAL_ID", "TX_DATETIME"]], on="TRANSACTION_ID"
    ).sort_values(["TERMINAL_ID", "TX_DATETIME"])

    found = False
    for _terminal_id, group in joined.groupby("TERMINAL_ID"):
        seq = group["terminal_risk_state"].tolist()
        high_idx = seq.index("HIGH_RISK") if "HIGH_RISK" in seq else None
        if high_idx is None:
            continue
        after_high = seq[high_idx + 1 :]
        if "RECOVERY" not in after_high:
            continue
        recovery_idx = after_high.index("RECOVERY")
        after_recovery = after_high[recovery_idx + 1 :]
        if "NORMAL" in after_recovery:
            found = True
            break
    assert found, "no terminal showed a genuine chronological HIGH_RISK -> RECOVERY -> NORMAL sequence"


def test_behavioral_risk_is_time_sensitive_not_a_permanent_blacklist(recent_df: pd.DataFrame, full_df: pd.DataFrame):
    """Behavioral risk stays time-sensitive, not a permanent-blacklist-style engine:
    among terminals that ever reach HIGH_RISK, at least one is later reassessed as
    calmer (its last observed state is NOT HIGH_RISK/RISK_RISING) -- this is the
    "elevated_recover" cohort's whole point (mrs.data.recent_stream module docstring).
    A second subset ("elevated_sustain") deliberately stays elevated through day 21,
    by design, to also demonstrate a persisting episode -- so this asserts "at least
    one recovers", not "every elevated terminal recovers"."""
    states = compute_terminal_behavioral_states(full_df)
    joined = states.merge(
        recent_df[["TRANSACTION_ID", "TERMINAL_ID", "TX_DATETIME"]], on="TRANSACTION_ID"
    ).sort_values(["TERMINAL_ID", "TX_DATETIME"])

    ever_high = joined.groupby("TERMINAL_ID").filter(lambda g: (g["terminal_risk_state"] == "HIGH_RISK").any())
    assert not ever_high.empty

    last_states = ever_high.groupby("TERMINAL_ID")["terminal_risk_state"].last()
    assert (~last_states.isin({"HIGH_RISK", "RISK_RISING"})).any()
    # And, matching the module docstring, not every elevated terminal is required to
    # have recovered by day 21 either -- both outcomes should be observable.
    assert (last_states.isin({"HIGH_RISK", "RISK_RISING"})).any()


# --------------------------------------------------------------- existing-pipeline compatibility


def test_full_pipeline_runs_without_error_using_the_frozen_model_and_engines(full_df: pd.DataFrame):
    from mrs.models.persistence import load_model

    version_dir = config.MODELS_DIR / "xgboost_v1"
    if not version_dir.exists():
        pytest.skip("models/xgboost_v1 not present; run scripts/07_train_xgboost.py")

    pipeline, metadata = load_model(version_dir)
    X = get_feature_matrix(full_df)
    risk = pipeline.predict_proba(X)[:, 1]
    assert len(risk) == len(full_df)
    assert ((risk >= 0) & (risk <= 1)).all()

    transaction_df = pd.DataFrame({"TRANSACTION_ID": full_df["TRANSACTION_ID"].to_numpy(), "transaction_risk": risk})
    terminal_df = compute_terminal_behavioral_states(full_df)[["TRANSACTION_ID", "terminal_risk_state"]]
    customer_df = compute_customer_behavioral_states(full_df)[["TRANSACTION_ID", "customer_risk_state"]]

    result = aggregate_risk(transaction_df, terminal_df, customer_df, metadata["threshold"])
    assert len(result) == len(full_df)
    assert set(result["unified_risk_level"].unique()) <= {"LOW", "MEDIUM", "HIGH", "CRITICAL", "INSUFFICIENT_EVIDENCE"}


# ------------------------------------------------------------------- historical isolation


def test_build_feature_frame_default_behavior_is_unchanged_for_benchmark_style_dates():
    """mrs.features.build_feature_frame's split_override parameter (added to support
    the recent stream) must be a pure opt-in: every existing caller passes nothing and
    gets EXACTLY the old behavior -- mrs.data.splits.assign_split decides the split
    from the frozen train/validation/test boundaries, and a 2018 (in-boundary)
    timestamp is labeled accordingly. This is a synthetic-but-realistic frame (not the
    real 1.75M-row dataset -- that is covered by the full existing suite, e.g.
    tests/test_feature_build_integration.py, which is unaffected by this change and
    still passes unmodified)."""
    from mrs.features.build import build_feature_frame

    synthetic = pd.DataFrame(
        {
            "TRANSACTION_ID": [1, 2, 3],
            "TX_DATETIME": pd.to_datetime(["2018-04-01", "2018-08-15", "2018-09-15"]),
            "CUSTOMER_ID": [1, 1, 1],
            "TERMINAL_ID": [1, 1, 1],
            "TX_AMOUNT": [10.0, 20.0, 30.0],
            "TX_FRAUD": [0, 0, 0],
        }
    )
    features = build_feature_frame(synthetic)  # no split_override -- default path
    by_id = features.set_index("TRANSACTION_ID")["split"]
    assert by_id[1] == "train"
    assert by_id[2] == "validation"
    assert by_id[3] == "test"


def test_recent_stream_generation_does_not_mutate_the_frozen_split_boundaries():
    """Generating and feature-building the recent stream must not have touched
    mrs.data.splits.SPLIT_BOUNDARIES as an importable module-level value -- confirms
    the split_override escape hatch is read-only with respect to the frozen
    boundaries, not a code path that could ever rewrite them."""
    from mrs.data.splits import SPLIT_BOUNDARIES

    before = dict(SPLIT_BOUNDARIES)
    generate_recent_stream()
    assert dict(SPLIT_BOUNDARIES) == before == {
        "train": ("2018-04-01", "2018-07-31"),
        "validation": ("2018-08-01", "2018-08-31"),
        "test": ("2018-09-01", "2018-09-30"),
    }
