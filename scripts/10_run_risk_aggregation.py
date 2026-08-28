#!/usr/bin/env python
"""Phase 7 real-data integration run: Risk Aggregation over the full frozen dataset.

Pure orchestrator, mirroring scripts/06-09's pattern: loads the persisted Phase 3
feature splits (read-only) via the established mrs.models.dataset loaders, scores them
with the already-trained, unmodified Phase 5 XGBoost pipeline (inference only, no
retraining), runs the unmodified Phase 6/7 behavioral engines, and feeds the three
already-computed component outputs into mrs.risk.aggregate.aggregate_risk() unchanged.

Does not modify mrs.risk.aggregate, mrs.behavioral.terminal, mrs.behavioral.customer,
or any Phase 5 model artifact. Does not read TX_FRAUD/TX_FRAUD_SCENARIO as an
aggregation input (both remain present in the loaded frame only for the sanity-check
section at the very end, exactly as scripts/09 used TX_FRAUD_SCENARIO post-hoc).

Run with: .venv/bin/python scripts/10_run_risk_aggregation.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from mrs import config  # noqa: E402
from mrs.behavioral.customer import compute_customer_behavioral_states  # noqa: E402
from mrs.behavioral.terminal import compute_terminal_behavioral_states  # noqa: E402
from mrs.models.dataset import get_feature_matrix, load_processed_transactions, load_split  # noqa: E402
from mrs.models.persistence import load_model  # noqa: E402
from mrs.risk.aggregate import (  # noqa: E402
    CRITICAL,
    HIGH,
    INSUFFICIENT_EVIDENCE,
    LOW,
    MEDIUM,
    aggregate_risk,
)

XGBOOST_VERSION_DIR = config.MODELS_DIR / "xgboost_v1"
OUTPUT_PATH = config.MODELS_DIR.parent / "docs" / "risk_aggregation_report_data.json"


def main() -> None:
    print("=" * 70)
    print("STEP 1: Load the frozen dataset via the established pipeline")
    print("=" * 70)
    labels_source = load_processed_transactions()
    train_df = load_split("train", labels_source)
    validation_df = load_split("validation", labels_source)
    test_df = load_split("test", labels_source)
    full_df = pd.concat([train_df, validation_df, test_df], ignore_index=True)
    print("Input files (via mrs.models.dataset, unmodified):")
    print("  data/processed/transactions/*.parquet (labels_source)")
    print("  data/features/features_train.parquet")
    print("  data/features/features_validation.parquet")
    print("  data/features/features_test.parquet")
    print(f"Row counts: train={len(train_df):,} validation={len(validation_df):,} test={len(test_df):,}")
    print(f"Total: {len(full_df):,}")
    assert len(full_df) == 1_754_155, "frozen dataset row count mismatch"
    assert not full_df["TRANSACTION_ID"].duplicated().any(), "duplicate TRANSACTION_ID in loaded dataset"

    print()
    print("=" * 70)
    print("STEP 2: Load the frozen Phase 5 XGBoost model + validated threshold")
    print("=" * 70)
    pipeline, metadata = load_model(XGBOOST_VERSION_DIR)
    threshold = metadata["threshold"]
    print(f"Model: {XGBOOST_VERSION_DIR / 'model.joblib'}")
    print(f"Threshold source: {XGBOOST_VERSION_DIR / 'metadata.json'}['threshold'] = {threshold}")
    print(f"threshold_selection: {metadata['threshold_selection']}")

    print()
    print("=" * 70)
    print("STEP 3: Score transaction ML risk (inference only, no retraining)")
    print("=" * 70)
    X = get_feature_matrix(full_df)
    t0 = time.time()
    transaction_risk = pipeline.predict_proba(X)[:, 1]
    score_time = time.time() - t0
    print(f"Scored {len(X):,} rows in {score_time:.2f}s")
    print(f"transaction_risk NaN count: {int(np.isnan(transaction_risk).sum())}")
    transaction_df = pd.DataFrame(
        {"TRANSACTION_ID": full_df["TRANSACTION_ID"].to_numpy(), "transaction_risk": transaction_risk}
    )

    print()
    print("=" * 70)
    print("STEP 4: Run the unmodified Phase 6/7 behavioral engines")
    print("=" * 70)
    t0 = time.time()
    terminal_full = compute_terminal_behavioral_states(full_df)
    terminal_time = time.time() - t0
    print(f"mrs.behavioral.terminal.compute_terminal_behavioral_states: {len(terminal_full):,} rows in {terminal_time:.2f}s")
    terminal_df = terminal_full[["TRANSACTION_ID", "terminal_risk_state"]]

    t0 = time.time()
    customer_full = compute_customer_behavioral_states(full_df)
    customer_time = time.time() - t0
    print(f"mrs.behavioral.customer.compute_customer_behavioral_states: {len(customer_full):,} rows in {customer_time:.2f}s")
    customer_df = customer_full[["TRANSACTION_ID", "customer_risk_state"]]

    print()
    print("=" * 70)
    print("STEP 5: Verify TRANSACTION_ID uniqueness/contracts before aggregation")
    print("=" * 70)
    for name, df in (("transaction_df", transaction_df), ("terminal_df", terminal_df), ("customer_df", customer_df)):
        dup_count = int(df["TRANSACTION_ID"].duplicated().sum())
        print(f"  {name}: rows={len(df):,} duplicate_TRANSACTION_ID={dup_count}")
        assert dup_count == 0, f"{name} has duplicate TRANSACTION_ID -- aggregation contract violated"
    tx_ids = set(transaction_df["TRANSACTION_ID"])
    term_ids = set(terminal_df["TRANSACTION_ID"])
    cust_ids = set(customer_df["TRANSACTION_ID"])
    union_ids = tx_ids | term_ids | cust_ids
    print(f"  transaction_df/terminal_df/customer_df TRANSACTION_ID sets identical: {tx_ids == term_ids == cust_ids}")
    print(f"  union size: {len(union_ids):,}")

    print()
    print("=" * 70)
    print("STEP 6: Run aggregate_risk() -- unmodified, no tuning")
    print("=" * 70)
    t0 = time.time()
    result = aggregate_risk(transaction_df, terminal_df, customer_df, threshold)
    agg_time = time.time() - t0
    print(f"Aggregated {len(result):,} rows in {agg_time:.2f}s")

    print()
    print("=" * 70)
    print("STEP 7: Row-count / TRANSACTION_ID coverage check")
    print("=" * 70)
    result_ids = set(result["TRANSACTION_ID"])
    print(f"  output rows: {len(result):,}  union of inputs: {len(union_ids):,}  match: {len(result) == len(union_ids)}")
    print(f"  output TRANSACTION_ID set == union of inputs: {result_ids == union_ids}")
    print(f"  duplicate TRANSACTION_ID in output: {int(result['TRANSACTION_ID'].duplicated().sum())}")

    print()
    print("=" * 70)
    print("STEP 8: unified_risk_level distribution")
    print("=" * 70)
    level_counts = result["unified_risk_level"].value_counts()
    level_pcts = (level_counts / len(result) * 100).round(4)
    for level in [LOW, MEDIUM, HIGH, CRITICAL, INSUFFICIENT_EVIDENCE]:
        c = int(level_counts.get(level, 0))
        p = float(level_pcts.get(level, 0.0))
        print(f"  {level:<22} {c:>10,}  ({p:.4f}%)")

    print()
    print("=" * 70)
    print("STEP 9: Component severity distributions")
    print("=" * 70)
    for col in ("transaction_risk_severity", "terminal_risk_severity", "customer_risk_severity"):
        print(f"  {col}:")
        counts = result[col].value_counts(dropna=False)
        for value, count in counts.items():
            label = "unavailable (NaN)" if pd.isna(value) else f"severity {int(value)}"
            print(f"    {label}: {count:,} ({count / len(result) * 100:.4f}%)")

    print()
    print("=" * 70)
    print("STEP 10: Missing/unavailable component counts")
    print("=" * 70)
    tx_unavailable = int(result["transaction_risk_severity"].isna().sum())
    term_unavailable = int(result["terminal_risk_severity"].isna().sum())
    cust_unavailable = int(result["customer_risk_severity"].isna().sum())
    all_unavailable = int((result["unified_risk_level"] == INSUFFICIENT_EVIDENCE).sum())
    print(f"  transaction_risk unavailable: {tx_unavailable:,}")
    print(f"  terminal_risk unavailable: {term_unavailable:,}")
    print(f"  customer_risk unavailable: {cust_unavailable:,}")
    print(f"  all three unavailable (INSUFFICIENT_EVIDENCE): {all_unavailable:,}")

    print()
    print("=" * 70)
    print("STEP 11: Top contributing_signals patterns")
    print("=" * 70)
    signal_patterns = result["contributing_signals"].apply(lambda s: tuple(s)).value_counts().head(15)
    for pattern, count in signal_patterns.items():
        label = pattern if pattern else "() [no contributing signals]"
        print(f"  {count:>10,}  {label}")

    print()
    print("=" * 70)
    print("STEP 12: Sanity checks -- unexpected nulls, impossible states, invariants")
    print("=" * 70)
    # Every row must have a level.
    print(f"  rows with null unified_risk_level: {int(result['unified_risk_level'].isna().sum())}")
    # CRITICAL rows must have >=2 severity-2 components among transaction/terminal/customer.
    critical_rows = result[result["unified_risk_level"] == CRITICAL]
    severity_cols = ["transaction_risk_severity", "terminal_risk_severity", "customer_risk_severity"]
    n_severe = (critical_rows[severity_cols] == 2).sum(axis=1)
    bad_critical = int((n_severe < 2).sum())
    print(f"  CRITICAL rows with < 2 severity-2 components (should be 0): {bad_critical}")
    # HIGH rows must have exactly 1 severity-2 component.
    high_rows = result[result["unified_risk_level"] == HIGH]
    n_severe_high = (high_rows[severity_cols] == 2).sum(axis=1)
    bad_high = int((n_severe_high != 1).sum())
    print(f"  HIGH rows without exactly 1 severity-2 component (should be 0): {bad_high}")
    # LOW rows must have max available severity 0 (no severity 1 or 2 anywhere).
    low_rows = result[result["unified_risk_level"] == LOW]
    bad_low = int(((low_rows[severity_cols] == 1) | (low_rows[severity_cols] == 2)).any(axis=1).sum())
    print(f"  LOW rows with a severity 1 or 2 component present (should be 0): {bad_low}")
    # INSUFFICIENT_EVIDENCE rows must have all three severities NaN.
    insuff_rows = result[result["unified_risk_level"] == INSUFFICIENT_EVIDENCE]
    bad_insuff = int((~insuff_rows[severity_cols].isna()).any(axis=1).sum())
    print(f"  INSUFFICIENT_EVIDENCE rows with any available severity (should be 0): {bad_insuff}")
    # contributing_signals must be empty for LOW/INSUFFICIENT_EVIDENCE.
    bad_empty_signals = int(
        (result["unified_risk_level"].isin([LOW, INSUFFICIENT_EVIDENCE]) & (result["contributing_signals"].apply(len) > 0)).sum()
    )
    print(f"  LOW/INSUFFICIENT_EVIDENCE rows with non-empty contributing_signals (should be 0): {bad_empty_signals}")

    print()
    print("=" * 70)
    print("STEP 13: Cross-check against TX_FRAUD_SCENARIO (post-hoc reporting only,")
    print("         never an aggregation input)")
    print("=" * 70)
    scenario_lookup = full_df.set_index("TRANSACTION_ID")["TX_FRAUD_SCENARIO"]
    result_with_scenario = result.set_index("TRANSACTION_ID").join(scenario_lookup)
    for level in [LOW, MEDIUM, HIGH, CRITICAL, INSUFFICIENT_EVIDENCE]:
        subset = result_with_scenario[result_with_scenario["unified_risk_level"] == level]
        if len(subset) == 0:
            continue
        fraud_rate = (subset["TX_FRAUD_SCENARIO"] != 0).mean() if "TX_FRAUD_SCENARIO" in subset else float("nan")
        print(f"  {level}: n={len(subset):,} genuine-fraud-rate-within-level={fraud_rate:.4f}")

    report = {
        "input_files": [
            "data/processed/transactions/*.parquet",
            "data/features/features_train.parquet",
            "data/features/features_validation.parquet",
            "data/features/features_test.parquet",
        ],
        "threshold": threshold,
        "total_rows": len(full_df),
        "output_rows": len(result),
        "level_distribution": {k: int(v) for k, v in level_counts.to_dict().items()},
        "transaction_unavailable": tx_unavailable,
        "terminal_unavailable": term_unavailable,
        "customer_unavailable": cust_unavailable,
        "insufficient_evidence_count": all_unavailable,
        "sanity_check_failures": {
            "bad_critical": bad_critical,
            "bad_high": bad_high,
            "bad_low": bad_low,
            "bad_insufficient_evidence": bad_insuff,
            "bad_empty_signals": bad_empty_signals,
        },
        "runtime_seconds": {
            "transaction_scoring": score_time,
            "terminal_engine": terminal_time,
            "customer_engine": customer_time,
            "aggregation": agg_time,
        },
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(report, f, indent=2, sort_keys=True, default=str)
    print()
    print(f"Report data written to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
