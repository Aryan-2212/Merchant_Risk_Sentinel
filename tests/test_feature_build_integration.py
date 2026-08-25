"""Real-data integration test for the Phase 3 feature layer (final feature-layer
validation test). Uses the actual Fraud Detection Handbook processed dataset at
data/processed/transactions/*.parquet (Phase 1's validated, chronologically-ordered
processed layer, 1,754,155 real rows) -- never a synthetic or substitute dataset. Every
check reuses the actual production entry point, mrs.features.build.build_feature_frame,
and the live registry (mrs.features.registry.FEATURE_NAMES / NON_FEATURE_COLUMNS) --
nothing is reimplemented here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mrs import config
from mrs.data.schema import LABEL_COLUMNS, RAW_COLUMNS
from mrs.features.build import REQUIRED_INPUT_COLUMNS, build_feature_frame
from mrs.features.registry import FEATURE_NAMES, NON_FEATURE_COLUMNS

pytestmark = pytest.mark.data

# Deterministic real transactions, selected by direct one-time inspection of the actual
# dataset (see the Phase 3 completion report for exactly how each was found). None are
# fabricated; all values below were read from the real processed Parquet files.
_ESTABLISHED_TXN_ID = 16740          # customer=2464, terminal=5271, 2018-04-02 15:40:09
_ESTABLISHED_CUSTOMER_ID = 2464
_ESTABLISHED_TERMINAL_ID = 5271
_FUTURE_SAME_PAIR_TXN_ID = 1726847   # same customer+terminal, 2018-09-28 06:17:42
_GLOBAL_FIRST_TXN_ID = 0             # 2018-04-01 00:00:31, customer=596, terminal=3156
_TERMINAL_FIRST_FRAUD_TXN_ID = 561077        # terminal 4's first-ever fraud
_TERMINAL_NEXT_AFTER_FRAUD_TXN_ID = 562833   # terminal 4's next transaction afterward


def _load_processed() -> pd.DataFrame:
    parts = sorted(config.PROCESSED_TRANSACTIONS_DIR.glob("*.parquet"))
    return pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)


@pytest.fixture(scope="module")
def raw_transactions(require_processed_dataset):
    return _load_processed()


@pytest.fixture(scope="module")
def full_features(raw_transactions):
    return build_feature_frame(raw_transactions)


# --- A. real data loading ---

def test_raw_dataset_has_expected_columns_and_is_nonempty(raw_transactions):
    assert not raw_transactions.empty
    assert set(RAW_COLUMNS) <= set(raw_transactions.columns)
    for col in REQUIRED_INPUT_COLUMNS:
        assert col in raw_transactions.columns


def test_tx_datetime_is_usable_as_datetime(raw_transactions):
    assert pd.api.types.is_datetime64_any_dtype(raw_transactions["TX_DATETIME"])
    assert raw_transactions["TX_DATETIME"].dt.hour.notna().all()


def test_raw_dataset_size_matches_known_dataset(raw_transactions):
    # Regression pin against the Phase 1/2 measured total (docs/PHASE1_REPORT.md,
    # docs/DATASET_REPORT.md) -- confirms this really is the full Handbook dataset.
    assert len(raw_transactions) == 1_754_155


# --- G. raw immutability ---

def test_raw_dataframe_unchanged_after_feature_build(raw_transactions, full_features):
    # Compare against an independently reloaded copy rather than assuming object
    # identity or no-copy semantics.
    independent_reload = _load_processed()
    pd.testing.assert_frame_equal(
        raw_transactions.sort_values("TRANSACTION_ID").reset_index(drop=True),
        independent_reload.sort_values("TRANSACTION_ID").reset_index(drop=True),
    )


# --- B. feature building ---

def test_feature_build_completes_and_is_nonempty(full_features):
    assert not full_features.empty


def test_transaction_id_uniquely_identifies_output_rows(raw_transactions, full_features):
    assert len(full_features) == len(raw_transactions)
    assert full_features["TRANSACTION_ID"].is_unique
    assert set(full_features["TRANSACTION_ID"]) == set(raw_transactions["TRANSACTION_ID"])


# --- C. registry contract ---

def test_output_columns_match_live_registry_exactly(full_features):
    generated = set(full_features.columns) - NON_FEATURE_COLUMNS
    assert generated == set(FEATURE_NAMES)  # no undocumented extras, nothing missing


# --- D. label leakage ---

def test_label_columns_never_appear_in_output(full_features):
    assert not (LABEL_COLUMNS & set(full_features.columns))


def test_no_feature_column_is_a_renamed_copy_of_a_label(raw_transactions, full_features):
    aligned = full_features.merge(
        raw_transactions[["TRANSACTION_ID", "TX_FRAUD", "TX_FRAUD_SCENARIO"]],
        on="TRANSACTION_ID",
    )
    feature_columns = set(full_features.columns) - NON_FEATURE_COLUMNS
    fraud_values = aligned["TX_FRAUD"].to_numpy(dtype=float)
    scenario_values = aligned["TX_FRAUD_SCENARIO"].to_numpy(dtype=float)
    for col in feature_columns:
        values = aligned[col].to_numpy(dtype=float)
        assert not np.array_equal(values, fraud_values), f"{col} is a copy of TX_FRAUD"
        assert not np.array_equal(values, scenario_values), f"{col} is a copy of TX_FRAUD_SCENARIO"


# --- I. label-derived features: historical and current-row-excluded, not leakage ---

def test_terminal_fraud_history_excludes_the_transactions_own_label(full_features):
    # This IS a legitimate, documented use of TX_FRAUD (Dev Plan Sec 10/34.2): a
    # historical, strictly-prior aggregate -- not the current row's own label.
    fraud_row = full_features[full_features.TRANSACTION_ID == _TERMINAL_FIRST_FRAUD_TXN_ID].iloc[0]
    next_row = full_features[full_features.TRANSACTION_ID == _TERMINAL_NEXT_AFTER_FRAUD_TXN_ID].iloc[0]
    assert fraud_row["terminal_hist_fraud_count"] == 0  # its own fraud=1 label excluded
    assert next_row["terminal_hist_fraud_count"] == 1   # now sees that prior fraud


# --- E. temporal integrity: independent recomputation (not just "output is sorted") ---

def test_established_transaction_matches_independently_recomputed_history(raw_transactions, full_features):
    established = full_features[full_features.TRANSACTION_ID == _ESTABLISHED_TXN_ID].iloc[0]
    established_ts = raw_transactions.loc[
        raw_transactions.TRANSACTION_ID == _ESTABLISHED_TXN_ID, "TX_DATETIME"
    ].iloc[0]

    # Independent recomputation via plain pandas filtering on the raw processed data --
    # not calling any mrs.features code -- to cross-check the pipeline's own output.
    prior_customer_rows = raw_transactions[
        (raw_transactions.CUSTOMER_ID == _ESTABLISHED_CUSTOMER_ID)
        & (raw_transactions.TX_DATETIME < established_ts)
    ]
    prior_terminal_rows = raw_transactions[
        (raw_transactions.TERMINAL_ID == _ESTABLISHED_TERMINAL_ID)
        & (raw_transactions.TX_DATETIME < established_ts)
    ]

    assert established["customer_prior_tx_count"] == len(prior_customer_rows)
    assert established["customer_hist_amount_mean"] == pytest.approx(prior_customer_rows["TX_AMOUNT"].mean())
    assert established["customer_hist_amount_std"] == pytest.approx(prior_customer_rows["TX_AMOUNT"].std())
    assert established["terminal_prior_tx_count"] == len(prior_terminal_rows)
    assert established["terminal_hist_amount_mean"] == pytest.approx(prior_terminal_rows["TX_AMOUNT"].mean())
    assert established["terminal_hist_fraud_count"] == prior_terminal_rows["TX_FRAUD"].sum()


# --- E. temporal integrity: real-data future-row mutation check ---

@pytest.fixture(scope="module")
def entity_complete_real_subset(raw_transactions):
    # Real rows only -- a genuine, unmodified slice of the actual dataset, restricted to
    # the union of the two entities involved in the established transaction.
    # mrs.features.customer/terminal group strictly by CUSTOMER_ID/TERMINAL_ID (proven
    # entity-isolated in tests/test_features_temporal.py and the customer/terminal test
    # files), so this subset reproduces IDENTICAL real feature values for TXN 16740's own
    # customer- and terminal-scoped features as the full 1,754,155-row build -- verified
    # below, not assumed -- while being far cheaper to rebuild twice for the mutation
    # check than the full dataset.
    mask = (raw_transactions.CUSTOMER_ID == _ESTABLISHED_CUSTOMER_ID) | (
        raw_transactions.TERMINAL_ID == _ESTABLISHED_TERMINAL_ID
    )
    return raw_transactions[mask].copy()


def test_subset_reproduces_identical_established_feature_values_as_full_dataset(
    entity_complete_real_subset, full_features
):
    subset_features = build_feature_frame(entity_complete_real_subset)
    subset_row = subset_features[subset_features.TRANSACTION_ID == _ESTABLISHED_TXN_ID].iloc[0]
    full_row = full_features[full_features.TRANSACTION_ID == _ESTABLISHED_TXN_ID].iloc[0]

    for col in (
        "customer_prior_tx_count", "customer_hist_amount_mean", "customer_hist_amount_std",
        "terminal_prior_tx_count", "terminal_hist_amount_mean", "terminal_hist_fraud_count",
        "pair_prior_interaction_count",
    ):
        left, right = subset_row[col], full_row[col]
        if pd.isna(left) or pd.isna(right):
            assert pd.isna(left) and pd.isna(right)
        else:
            assert left == pytest.approx(right)


def test_future_row_mutation_does_not_alter_earlier_established_real_transaction(
    entity_complete_real_subset,
):
    before_features = build_feature_frame(entity_complete_real_subset)
    before_row = before_features[before_features.TRANSACTION_ID == _ESTABLISHED_TXN_ID].iloc[0]

    mutated = entity_complete_real_subset.copy()
    future_mask = mutated.TRANSACTION_ID == _FUTURE_SAME_PAIR_TXN_ID
    assert future_mask.sum() == 1
    mutated.loc[future_mask, "TX_AMOUNT"] = 99999.0
    mutated.loc[future_mask, "TX_FRAUD"] = 1

    after_features = build_feature_frame(mutated)
    after_row = after_features[after_features.TRANSACTION_ID == _ESTABLISHED_TXN_ID].iloc[0]

    for col in (
        "customer_prior_tx_count", "customer_hist_amount_mean", "customer_hist_amount_std",
        "customer_tx_count_10min", "customer_tx_count_1h", "customer_tx_count_24h",
        "terminal_prior_tx_count", "terminal_hist_amount_mean", "terminal_hist_fraud_count",
        "terminal_hist_fraud_rate", "pair_prior_interaction_count",
    ):
        left, right = before_row[col], after_row[col]
        if pd.isna(left) or pd.isna(right):
            assert pd.isna(left) and pd.isna(right)
        else:
            assert left == right


# --- F. cold start / missingness ---

def test_global_first_transaction_is_a_genuine_cold_start(full_features):
    row = full_features[full_features.TRANSACTION_ID == _GLOBAL_FIRST_TXN_ID].iloc[0]
    assert row["customer_prior_tx_count"] == 0
    assert row["terminal_prior_tx_count"] == 0
    assert np.isnan(row["customer_hist_amount_mean"])
    assert np.isnan(row["terminal_hist_amount_mean"])
    assert np.isnan(row["customer_time_since_prev_tx_seconds"])
    assert np.isnan(row["terminal_time_since_prev_tx_seconds"])
    assert row["customer_new_terminal_flag"] == 1
    assert row["pair_is_new_relationship"] == 1
    assert np.isnan(row["terminal_hist_fraud_rate"])  # not a false 0%


def test_cold_start_rows_match_known_active_customer_count(full_features):
    # Each active customer has exactly one cold-start row (their first-ever transaction).
    # 4,990 active customers is the exact figure independently measured in Phase 1/2
    # (docs/PHASE1_REPORT.md, docs/DATASET_REPORT.md).
    cold_start = full_features[full_features["customer_prior_tx_count"] == 0]
    assert len(cold_start) == 4_990


# --- H. shape / sanity ---

def test_no_unexpected_duplicate_transaction_ids(full_features):
    assert not full_features["TRANSACTION_ID"].duplicated().any()


def test_feature_dtypes_are_numeric_or_split_label(full_features):
    feature_columns = set(full_features.columns) - NON_FEATURE_COLUMNS
    for col in feature_columns:
        assert pd.api.types.is_numeric_dtype(full_features[col]), (
            f"{col} has non-numeric dtype {full_features[col].dtype}"
        )
    assert full_features["split"].dtype == object


def test_split_column_covers_all_three_configured_splits(full_features):
    assert set(full_features["split"].unique()) == {"train", "validation", "test"}
