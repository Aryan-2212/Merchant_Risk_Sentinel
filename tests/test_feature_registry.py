"""Registry round-trip tests (Dev Plan Sec 12): registered FeatureSpec entries -> the
feature builder -> output columns must match exactly, with no undocumented extras and no
missing features. mrs.features.registry is a declarative FeatureSpec list plus
build_feature_frame(); this file tests that contract, not an invented registry API."""

from __future__ import annotations

import pandas as pd
import pytest

from mrs.features.build import build_feature_frame
from mrs.features.registry import FEATURE_NAMES, FEATURE_SPECS, NON_FEATURE_COLUMNS

_VALID_LEVELS = {"transaction", "customer", "terminal", "relationship"}


def _synthetic_transactions():
    # Same (customer, terminal) pair repeated 4 times so prior-history counts, amount
    # stats, and fraud stats all have real, hand-checkable non-cold-start values by the
    # last row. TX_FRAUD and TX_FRAUD_SCENARIO are deliberately present in the input so
    # the test can confirm they never become output columns despite being available to
    # the one permitted internal consumer (terminal.py's fraud-history features).
    return pd.DataFrame({
        "TRANSACTION_ID": [1, 2, 3, 4],
        "CUSTOMER_ID": [1, 1, 1, 1],
        "TERMINAL_ID": [100, 100, 100, 100],
        "TX_DATETIME": pd.to_datetime([
            "2018-04-01 00:00:00", "2018-04-01 01:00:00", "2018-04-01 02:00:00", "2018-04-01 03:00:00",
        ]),
        "TX_AMOUNT": [10.0, 20.0, 30.0, 40.0],
        "TX_FRAUD": [0, 0, 1, 0],
        "TX_FRAUD_SCENARIO": [0, 0, 2, 0],
    })


def test_no_duplicate_names_in_registry():
    # registry.py's own docstring promises "exactly one FeatureSpec entry" per column.
    names = [spec.name for spec in FEATURE_SPECS]
    assert len(names) == len(set(names))


def test_non_feature_columns_and_feature_names_do_not_overlap():
    assert NON_FEATURE_COLUMNS.isdisjoint(FEATURE_NAMES)


def test_every_spec_has_required_metadata_populated():
    # Dev Plan Sec 12 requires each feature to document its level, definition, window,
    # current-row-exclusion, and cold-start behavior.
    for spec in FEATURE_SPECS:
        assert spec.name
        assert spec.level in _VALID_LEVELS, f"{spec.name}: invalid level {spec.level!r}"
        assert spec.definition
        assert spec.historical_window
        assert spec.excludes_current_row
        assert spec.cold_start_behavior
        assert isinstance(spec.uses_labels, bool)


def test_feature_count_per_level_matches_verified_registry_composition():
    # Counts confirmed by direct inspection of the current FEATURE_SPECS tuple, not
    # copied from the Dev Plan or an earlier draft.
    counts: dict[str, int] = {}
    for spec in FEATURE_SPECS:
        counts[spec.level] = counts.get(spec.level, 0) + 1
    assert counts == {"transaction": 5, "customer": 12, "terminal": 14, "relationship": 2}


def test_uses_labels_flags_exactly_the_verified_fraud_derived_features():
    # Verified directly against terminal.py: these five columns are the only ones that
    # read TX_FRAUD. terminal_recent_tx_count is an internal-only intermediate (never a
    # registered/output column) and terminal_volume_deviation does not touch TX_FRAUD.
    expected = {
        "terminal_recent_fraud_count_24h",
        "terminal_recent_fraud_rate_24h",
        "terminal_hist_fraud_count",
        "terminal_hist_fraud_rate",
        "terminal_fraud_rate_deviation",
    }
    actual = {spec.name for spec in FEATURE_SPECS if spec.uses_labels}
    assert actual == expected


def test_build_feature_frame_output_matches_registry_exactly():
    result = build_feature_frame(_synthetic_transactions())
    generated = set(result.columns) - NON_FEATURE_COLUMNS
    assert generated == set(FEATURE_NAMES)  # no undocumented extras, nothing missing


def test_labels_present_in_input_never_become_output_columns():
    result = build_feature_frame(_synthetic_transactions())
    assert "TX_FRAUD" not in result.columns
    assert "TX_FRAUD_SCENARIO" not in result.columns


def test_build_feature_frame_produces_correct_values_not_just_correct_column_names():
    # A registry can declare every name correctly while the builder is wired wrong
    # underneath. Check actual computed values on row 4 (real, non-cold-start history).
    result = build_feature_frame(_synthetic_transactions())
    row4 = result[result["TRANSACTION_ID"] == 4].iloc[0]

    assert row4["customer_prior_tx_count"] == 3
    assert row4["customer_hist_amount_mean"] == pytest.approx(20.0)  # mean([10,20,30])
    assert row4["terminal_prior_tx_count"] == 3
    assert row4["terminal_hist_fraud_count"] == 1  # from row3 (TX_FRAUD=1)
    assert row4["terminal_hist_fraud_rate"] == pytest.approx(1.0 / 3.0)
    assert row4["pair_prior_interaction_count"] == 3
    assert row4["split"] == "train"  # 2018-04-01 falls in the frozen train range
