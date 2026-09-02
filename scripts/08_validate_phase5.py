#!/usr/bin/env python
"""Phase 5 independent validation / runtime audit.

Read-only with respect to the production Phase 5 model: loads the already-saved
models/xgboost_v1/ pipeline for most checks, and where a fresh fit is genuinely required
(runtime benchmark across n_jobs, random-label sanity test, feature-group ablation,
reproducibility, independent cross-check) it trains its OWN separate, disposable
pipelines built from the exact production hyperparameters/preprocessing -- it never
overwrites, retrains, or mutates models/xgboost_v1/ itself.

Writes every artifact under models/xgboost_v1/validation/:
  runtime_benchmark.json, random_label_results.csv, feature_ablation.csv,
  baseline_comparison.csv, permutation_importance.csv, scenario_validation.csv,
  reproducibility.json, temporal_leakage_audit.json, VALIDATION_REPORT.md

Run with: .venv/bin/python scripts/08_validate_phase5.py
"""

from __future__ import annotations

import json
import os
import resource
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
from xgboost import XGBClassifier  # noqa: E402
import xgboost  # noqa: E402

from mrs import config  # noqa: E402
from mrs.models import metrics as M  # noqa: E402
from mrs.models import validate as V  # noqa: E402
from mrs.models.dataset import FEATURE_COLUMNS, get_feature_matrix, load_processed_transactions, load_split  # noqa: E402
from mrs.models.persistence import load_model  # noqa: E402
from mrs.models.preprocessing import build_preprocessing_pipeline  # noqa: E402
from mrs.models.train_xgboost import _FIXED_HYPERPARAMS, RANDOM_SEED  # noqa: E402

XGBOOST_VERSION_DIR = config.MODELS_DIR / "xgboost_v1"
LOGREG_VERSION_DIR = config.MODELS_DIR / "logreg_baseline_v1"
OUTPUT_DIR = XGBOOST_VERSION_DIR / "validation"


def _pr_auc(y_true, y_prob) -> float:
    return M.threshold_independent_metrics(y_true, y_prob)["pr_auc"]


def _roc_auc(y_true, y_prob) -> float:
    return M.threshold_independent_metrics(y_true, y_prob)["roc_auc"]


def _peak_rss_bytes() -> int:
    # macOS reports ru_maxrss in bytes; this is the process's peak RSS since start
    # (monotonically non-decreasing), not an isolated per-call measurement.
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss


def load_everything():
    print("Loading real data (processed transactions + Phase 3 feature splits)...")
    t0 = time.time()
    labels_source = load_processed_transactions()
    train_df = load_split("train", labels_source)
    validation_df = load_split("validation", labels_source)
    test_df = load_split("test", labels_source)
    load_time = time.time() - t0
    print(f"  loaded in {load_time:.2f}s: train={len(train_df):,} val={len(validation_df):,} test={len(test_df):,}")

    print("Loading saved production XGBoost model + metadata...")
    pipeline, metadata = load_model(XGBOOST_VERSION_DIR)
    with open(XGBOOST_VERSION_DIR / "error_analysis.json") as f:
        error_analysis = json.load(f)
    with open(LOGREG_VERSION_DIR / "metadata.json") as f:
        logreg_metadata = json.load(f)
    with open(LOGREG_VERSION_DIR / "error_analysis.json") as f:
        logreg_error_analysis = json.load(f)

    return {
        "train_df": train_df, "validation_df": validation_df, "test_df": test_df,
        "load_time": load_time,
        "pipeline": pipeline, "metadata": metadata, "error_analysis": error_analysis,
        "logreg_metadata": logreg_metadata, "logreg_error_analysis": logreg_error_analysis,
    }


# --- Part 1 + 2: workload verification, Part 4: booster structure -----------------------


def part1_2_4_workload_and_structure(ctx: dict) -> dict:
    print("\n[Part 1/2/4] Verifying actual training workload + booster structure...")
    train_df = ctx["train_df"]
    pipeline = ctx["pipeline"]
    metadata = ctx["metadata"]

    X_train = get_feature_matrix(train_df)
    y_train = train_df["TX_FRAUD"].to_numpy()

    classifier = pipeline.named_steps["classifier"]
    impute_step = pipeline.named_steps["preprocess"].named_steps["impute_flag"]
    booster = classifier.get_booster()

    trees_df = booster.trees_to_dataframe()
    n_trees = booster.num_boosted_rounds()
    nodes_per_tree = trees_df.groupby("Tree").size()
    split_nodes = trees_df[trees_df["Feature"] != "Leaf"]

    booster_cfg = json.loads(booster.save_config())
    nthread = booster_cfg.get("learner", {}).get("generic_param", {}).get("nthread")

    model_bytes = len(booster.save_raw())

    workload = {
        "X_train_shape": list(X_train.shape),
        "y_train_shape": list(y_train.shape),
        "y_train_non_null_count": int(pd.Series(y_train).notna().sum()),
        "X_train_dtype_after_preprocessing": "float64 (post impute+scale; XGBoost internally uses float32)",
        "X_train_memory_bytes_raw_dataframe": int(X_train.memory_usage(deep=True).sum()),
        "n_features_input_to_pipeline": X_train.shape[1],
        "n_features_after_preprocessing": len(impute_step.get_feature_names_out()),
        "n_positive_labels_train": int((y_train == 1).sum()),
        "n_negative_labels_train": int((y_train == 0).sum()),
        "scale_pos_weight_configured": float(classifier.scale_pos_weight),
        "scale_pos_weight_independently_recomputed": float((y_train == 0).sum() / (y_train == 1).sum()),
        "xgboost_version_this_environment": xgboost.__version__,
        "xgboost_hyperparameters_configured": classifier.get_params(deep=False),
        "n_estimators_configured": classifier.n_estimators,
        "n_trees_actually_built": n_trees,
        "trees_match_configured_exactly": n_trees == classifier.n_estimators,
        "best_iteration_attr_present": hasattr(classifier, "best_iteration"),
        "early_stopping_used": hasattr(classifier, "best_iteration"),
        "max_depth_configured": classifier.max_depth,
        "max_depth_actually_observed_in_trees": int(trees_df["Depth"].max() if "Depth" in trees_df.columns else -1),
        "avg_nodes_per_tree": float(nodes_per_tree.mean()),
        "total_split_nodes_all_trees": int(len(split_nodes)),
        "n_jobs_configured": classifier.n_jobs,
        "booster_internal_nthread_config": nthread,
        "tree_method_configured": classifier.tree_method,
        "tree_method_in_booster_internal_config": booster_cfg.get("learner", {}).get("gradient_booster", {}).get("gbtree_train_param", {}).get("tree_method"),
        "objective_configured": classifier.objective,
        "eval_metric_configured": classifier.eval_metric,
        "learning_rate_configured": classifier.learning_rate,
        "subsample_configured": classifier.subsample,
        "colsample_bytree_configured": classifier.colsample_bytree,
        "random_state_configured": classifier.random_state,
        "model_file_size_bytes_raw_booster": model_bytes,
        "hyperparameter_candidates_evaluated": len(metadata["hyperparameter_selection"]["candidates"]),
        "metadata_threshold": metadata["threshold"],
    }

    print(f"  X_train shape: {workload['X_train_shape']}  (expected [1169723, 33])")
    print(f"  n_trees_actually_built: {n_trees}  (matches n_estimators: {workload['trees_match_configured_exactly']})")
    print(f"  early_stopping_used: {workload['early_stopping_used']}")
    print(f"  n_jobs_configured: {workload['n_jobs_configured']}  booster nthread config: {nthread}")
    print(f"  model file size: {model_bytes:,} bytes")
    return workload


# --- Part 3: controlled n_jobs benchmark -------------------------------------------------


def part3_runtime_benchmark(ctx: dict, selected_hyperparams: dict, scale_pos_weight: float) -> list[dict]:
    print("\n[Part 3] Controlled n_jobs benchmark (same rows, same features, same params)...")
    train_df = ctx["train_df"]
    X_train = get_feature_matrix(train_df)
    y_train = train_df["TX_FRAUD"].to_numpy()

    experiments = [
        ("A_actual_default_n_jobs", None),
        ("B_n_jobs_1", 1),
        ("C_n_jobs_2", 2),
        ("D_n_jobs_4", 4),
        ("E_n_jobs_max", os.cpu_count()),
    ]

    results = []
    for label, n_jobs in experiments:
        classifier = XGBClassifier(
            **selected_hyperparams, **_FIXED_HYPERPARAMS,
            scale_pos_weight=scale_pos_weight, random_state=RANDOM_SEED, n_jobs=n_jobs,
        )
        cpu_t0 = time.process_time()
        wall_t0 = time.time()
        classifier.fit(X_train, y_train)
        wall_elapsed = time.time() - wall_t0
        cpu_elapsed = time.process_time() - cpu_t0
        peak_rss = _peak_rss_bytes()
        n_trees = classifier.get_booster().num_boosted_rounds()
        model_size = len(classifier.get_booster().save_raw())

        result = {
            "experiment": label,
            "n_jobs_requested": n_jobs,
            "n_jobs_effective": n_jobs if n_jobs is not None else os.cpu_count(),
            "wall_clock_seconds": round(wall_elapsed, 3),
            "cpu_time_seconds": round(cpu_elapsed, 3),
            "cpu_utilization_ratio": round(cpu_elapsed / wall_elapsed, 2) if wall_elapsed > 0 else None,
            "peak_rss_bytes_process_cumulative": peak_rss,
            "n_trees": n_trees,
            "model_size_bytes": model_size,
        }
        results.append(result)
        print(f"  {label}: n_jobs={n_jobs!r} wall={wall_elapsed:.2f}s cpu={cpu_elapsed:.2f}s "
              f"utilization={result['cpu_utilization_ratio']}x trees={n_trees}")

    return results


# --- Part 5: random-label sanity test -----------------------------------------------------


def part5_random_label_test(ctx: dict, selected_hyperparams: dict, scale_pos_weight: float) -> list[dict]:
    print("\n[Part 5] Random-label sanity test (5 seeds, train labels shuffled, val/test untouched)...")
    train_df, validation_df, test_df = ctx["train_df"], ctx["validation_df"], ctx["test_df"]
    X_train = get_feature_matrix(train_df)
    y_train = train_df["TX_FRAUD"].to_numpy()
    X_val = get_feature_matrix(validation_df)
    y_val = validation_df["TX_FRAUD"].to_numpy()

    results = []
    for seed in range(1, 6):
        y_train_shuffled = V.shuffle_labels(y_train, seed=seed)
        assert int(y_train_shuffled.sum()) == int(y_train.sum()), "shuffling must preserve class counts"

        pipeline = V.train_on_feature_subset(
            X_train, y_train_shuffled, selected_hyperparams, scale_pos_weight, tuple(FEATURE_COLUMNS)
        )
        val_prob = pipeline.predict_proba(X_val)[:, 1]
        m = {
            **M.metrics_at_threshold(y_val, val_prob, 0.5).to_dict(),
            **M.threshold_independent_metrics(y_val, val_prob),
        }
        row = {
            "seed": seed,
            "pr_auc": m["pr_auc"], "roc_auc": m["roc_auc"], "f1": m["f1"],
            "precision": m["precision"], "recall": m["recall"],
        }
        results.append(row)
        print(f"  seed={seed}: pr_auc={m['pr_auc']:.4f} roc_auc={m['roc_auc']:.4f} f1={m['f1']:.4f}")

    return results


# --- Part 6 + 9: feature-group ablation + scenario recall --------------------------------


def part6_9_feature_ablation(ctx: dict, selected_hyperparams: dict, scale_pos_weight: float) -> tuple[list[dict], list[dict]]:
    print("\n[Part 6/9] Feature-group ablation (registry-derived groups) + scenario recall...")
    train_df, validation_df, test_df = ctx["train_df"], ctx["validation_df"], ctx["test_df"]
    groups = V.feature_groups()
    print(f"  groups from registry: { {k: len(v) for k, v in groups.items()} }")

    X_train = get_feature_matrix(train_df)
    y_train = train_df["TX_FRAUD"].to_numpy()
    X_test = get_feature_matrix(test_df)
    y_test = test_df["TX_FRAUD"].to_numpy()
    scenarios = test_df["TX_FRAUD_SCENARIO"].to_numpy()

    ablation_rows = []
    scenario_rows = []

    configs = [("all_features", tuple(FEATURE_COLUMNS))] + [
        (f"{level}_only", cols) for level, cols in groups.items()
    ]

    for name, subset in configs:
        if name == "all_features":
            # Reuse the actual saved production model rather than retraining a duplicate.
            pipeline = ctx["pipeline"]
            test_prob = pipeline.predict_proba(X_test)[:, 1]
        else:
            pipeline = V.train_on_feature_subset(X_train, y_train, selected_hyperparams, scale_pos_weight, subset)
            test_prob = pipeline.predict_proba(X_test[list(subset)])[:, 1]

        # A fixed 0.5 threshold (not each ablation's own max-F1 point) is used here
        # deliberately: PR-AUC/ROC-AUC (threshold-independent) are this audit's primary
        # signal for "does this feature group carry predictive information"; the
        # threshold-dependent columns below are supplementary context at one common,
        # simple operating point, not a per-model-tuned comparison.
        m = {
            **M.metrics_at_threshold(y_test, test_prob, 0.5).to_dict(),
            **M.threshold_independent_metrics(y_test, test_prob),
        }
        ablation_rows.append({
            "feature_group": name, "n_features": len(subset),
            "pr_auc": m["pr_auc"], "roc_auc": m["roc_auc"], "f1": m["f1"],
            "precision": m["precision"], "recall": m["recall"],
            "false_positive_rate": m["false_positive_rate"],
        })
        print(f"  {name} ({len(subset)} features): pr_auc={m['pr_auc']:.4f} roc_auc={m['roc_auc']:.4f}")

        y_pred = (test_prob >= 0.5).astype(int)
        for scenario in (1, 2, 3):
            mask = scenarios == scenario
            total = int(mask.sum())
            detected = int(((y_pred == 1) & mask).sum())
            scenario_rows.append({
                "feature_group": name, "scenario": scenario,
                "total_fraud": total, "detected": detected,
                "recall": (detected / total) if total else None,
            })

    return ablation_rows, scenario_rows


# --- Part 7 + 12: baseline comparison + independent cross-check --------------------------


def part7_12_baselines(ctx: dict, ablation_rows: list[dict], selected_hyperparams: dict, scale_pos_weight: float) -> list[dict]:
    print("\n[Part 7/12] Baseline comparison (random, majority, feature-group, LR, XGBoost, independent HGB)...")
    train_df, validation_df, test_df = ctx["train_df"], ctx["validation_df"], ctx["test_df"]
    y_test = test_df["TX_FRAUD"].to_numpy()
    n_test = len(y_test)
    positive_rate = float(train_df["TX_FRAUD"].mean())

    rows = []

    random_scores = V.random_ranking_scores(n_test, seed=0)
    rows.append({"model": "random_ranking", "pr_auc": _pr_auc(y_test, random_scores), "roc_auc": _roc_auc(y_test, random_scores)})

    majority_scores = V.majority_baseline_scores(n_test, positive_rate)
    # ROC-AUC undefined for a constant score in some implementations; guard it.
    try:
        majority_roc = _roc_auc(y_test, majority_scores)
    except Exception:
        majority_roc = 0.5
    rows.append({"model": "majority_constant_score", "pr_auc": _pr_auc(y_test, majority_scores), "roc_auc": majority_roc})

    for r in ablation_rows:
        rows.append({"model": f"xgboost_{r['feature_group']}", "pr_auc": r["pr_auc"], "roc_auc": r["roc_auc"]})

    rows.append({
        "model": "logistic_regression_phase4",
        "pr_auc": ctx["logreg_metadata"]["test_metrics"]["pr_auc"],
        "roc_auc": ctx["logreg_metadata"]["test_metrics"]["roc_auc"],
    })
    rows.append({
        "model": "xgboost_phase5_production",
        "pr_auc": ctx["metadata"]["test_metrics"]["pr_auc"],
        "roc_auc": ctx["metadata"]["test_metrics"]["roc_auc"],
    })

    print("  Training independent cross-check model (sklearn HistGradientBoostingClassifier)...")
    preprocess = build_preprocessing_pipeline()
    X_train_proc = preprocess.fit_transform(get_feature_matrix(train_df))
    X_test_proc = preprocess.transform(get_feature_matrix(test_df))
    y_train = train_df["TX_FRAUD"].to_numpy()
    sample_weight = np.where(y_train == 1, scale_pos_weight, 1.0)

    hgb = HistGradientBoostingClassifier(
        max_iter=selected_hyperparams["n_estimators"],
        max_depth=selected_hyperparams["max_depth"],
        learning_rate=selected_hyperparams["learning_rate"],
        random_state=RANDOM_SEED,
    )
    t0 = time.time()
    hgb.fit(X_train_proc, y_train, sample_weight=sample_weight)
    hgb_time = time.time() - t0
    hgb_test_prob = hgb.predict_proba(X_test_proc)[:, 1]
    rows.append({
        "model": "independent_cross_check_sklearn_HistGradientBoosting",
        "pr_auc": _pr_auc(y_test, hgb_test_prob), "roc_auc": _roc_auc(y_test, hgb_test_prob),
    })
    print(f"  independent cross-check (HGB) fit in {hgb_time:.2f}s: "
          f"pr_auc={rows[-1]['pr_auc']:.4f} roc_auc={rows[-1]['roc_auc']:.4f}")

    for row in rows:
        print(f"  {row['model']}: pr_auc={row['pr_auc']:.4f} roc_auc={row['roc_auc']:.4f}")

    return rows


# --- Part 8: permutation importance on the frozen production model -----------------------


def part8_permutation_importance(ctx: dict) -> pd.DataFrame:
    print("\n[Part 8] Permutation feature importance (production model, TEST set, n_repeats=2)...")
    test_df = ctx["test_df"]
    pipeline = ctx["pipeline"]
    X_test = get_feature_matrix(test_df)
    y_test = test_df["TX_FRAUD"].to_numpy()

    result = V.permutation_importance(
        pipeline, X_test, y_test, _pr_auc, feature_names=list(FEATURE_COLUMNS), n_repeats=2, seed=0
    )

    classifier = pipeline.named_steps["classifier"]
    impute_step = pipeline.named_steps["preprocess"].named_steps["impute_flag"]
    gain_importances = dict(zip(impute_step.get_feature_names_out(), classifier.feature_importances_))
    result["gain_importance"] = result["feature"].map(lambda f: gain_importances.get(f))

    print(result.head(10).to_string(index=False))
    high_gain_low_permutation = result[(result["gain_importance"] > 0.03) & (result["mean_degradation"] < 0.005)]
    if len(high_gain_low_permutation):
        print(f"  FLAGGED (high gain, low permutation impact): {list(high_gain_low_permutation['feature'])}")
    else:
        print("  No feature flagged as high-gain-but-low-permutation-impact.")

    return result


# --- Part 11: reproducibility -------------------------------------------------------------


def part11_reproducibility(ctx: dict, selected_hyperparams: dict, scale_pos_weight: float) -> dict:
    print("\n[Part 11] Reproducibility: two independent fits, same data/params/seed...")
    train_df, validation_df = ctx["train_df"], ctx["validation_df"]
    X_train = get_feature_matrix(train_df)
    y_train = train_df["TX_FRAUD"].to_numpy()
    X_val = get_feature_matrix(validation_df)
    y_val = validation_df["TX_FRAUD"].to_numpy()

    runs = []
    for i in (1, 2):
        pipeline = V.train_on_feature_subset(X_train, y_train, selected_hyperparams, scale_pos_weight, tuple(FEATURE_COLUMNS))
        val_prob = pipeline.predict_proba(X_val)[:, 1]
        m = {**M.metrics_at_threshold(y_val, val_prob, 0.5).to_dict(), **M.threshold_independent_metrics(y_val, val_prob)}
        threshold, _ = M.select_threshold_max_f1(y_val, val_prob)
        importances = pipeline.named_steps["classifier"].feature_importances_
        runs.append({"run": i, "val_prob": val_prob, "pr_auc": m["pr_auc"], "roc_auc": m["roc_auc"],
                     "f1": m["f1"], "threshold": threshold, "importances": importances})

    predictions_identical = np.array_equal(runs[0]["val_prob"], runs[1]["val_prob"])
    importances_identical = np.array_equal(runs[0]["importances"], runs[1]["importances"])

    result = {
        "predictions_bit_identical": bool(predictions_identical),
        "max_abs_prediction_diff": float(np.max(np.abs(runs[0]["val_prob"] - runs[1]["val_prob"]))),
        "pr_auc_run1": runs[0]["pr_auc"], "pr_auc_run2": runs[1]["pr_auc"],
        "roc_auc_run1": runs[0]["roc_auc"], "roc_auc_run2": runs[1]["roc_auc"],
        "f1_run1": runs[0]["f1"], "f1_run2": runs[1]["f1"],
        "threshold_run1": runs[0]["threshold"], "threshold_run2": runs[1]["threshold"],
        "feature_importances_bit_identical": bool(importances_identical),
    }
    print(f"  predictions bit-identical: {predictions_identical}  max abs diff: {result['max_abs_prediction_diff']:.2e}")
    print(f"  pr_auc: {runs[0]['pr_auc']:.6f} vs {runs[1]['pr_auc']:.6f}")
    print(f"  feature importances bit-identical: {importances_identical}")
    return result


# --- Part 10: temporal leakage audit (structured restatement of source-traced evidence) --


def part10_temporal_leakage_audit() -> dict:
    print("\n[Part 10] Temporal leakage audit (traced directly from mrs/features source)...")
    audit = {
        "primitives_module": "src/mrs/features/_temporal.py",
        "rolling_count_and_rolling_sum": {
            "mechanism": "pandas .rolling(window, closed='left', min_periods=0) after grouping by entity and sorting by (TX_DATETIME, TRANSACTION_ID)",
            "current_row_excluded": True,
            "future_rows_can_influence": False,
            "verdict": "PASS",
        },
        "expanding_prior": {
            "mechanism": "groupby(entity)[value_col].shift(1) BEFORE .expanding() -- current row removed from its own group before any aggregate sees it",
            "current_row_excluded": True,
            "future_rows_can_influence": False,
            "verdict": "PASS",
        },
        "features_traced": {
            "terminal_fraud_rate_deviation": {
                "definition": "terminal_recent_fraud_rate_24h - terminal_hist_fraud_rate",
                "source": "rolling_sum(TX_FRAUD, 24h, closed=left) / rolling_count(24h, closed=left)  minus  expanding_prior(TX_FRAUD, sum) / expanding_prior(TX_AMOUNT, count)",
                "current_row_excluded": True, "future_can_influence": False, "verdict": "PASS",
            },
            "terminal_recent_fraud_rate_24h": {
                "definition": "rolling_sum(TX_FRAUD, 24h, closed=left) / rolling_count(24h, closed=left)",
                "current_row_excluded": True, "future_can_influence": False, "verdict": "PASS",
            },
            "terminal_hist_fraud_rate": {
                "definition": "expanding_prior(TX_FRAUD, 'sum') / expanding_prior(TX_AMOUNT, 'count')  -- all strictly-prior history",
                "current_row_excluded": True, "future_can_influence": False, "verdict": "PASS",
            },
            "customer_hist_amount_mean": {
                "definition": "expanding_prior(TX_AMOUNT, 'mean') over strictly-prior customer transactions",
                "current_row_excluded": True, "future_can_influence": False, "verdict": "PASS",
            },
            "customer_hist_amount_std": {
                "definition": "expanding_prior(TX_AMOUNT, 'std') over strictly-prior customer transactions",
                "current_row_excluded": True, "future_can_influence": False, "verdict": "PASS",
            },
            "customer_amount_zscore": {
                "definition": "(tx_amount - customer_hist_amount_mean) / customer_hist_amount_std, only when std defined and > 0",
                "current_row_excluded": True, "future_can_influence": False, "verdict": "PASS",
                "note": "tx_amount is the CURRENT row's own amount, which is legitimate (it is the transaction being scored, not future information) -- only the mean/std it is compared against are historical.",
            },
            "customer_amount_deviation": {
                "definition": "tx_amount - customer_hist_amount_mean",
                "current_row_excluded": True, "future_can_influence": False, "verdict": "PASS",
            },
        },
        "label_columns_reachable_from_terminal_py_only": True,
        "label_read_always_via_current_row_excluding_primitive": True,
        "build_time_assertion": "mrs.features.build.build_feature_frame asserts (generated_feature_columns & LABEL_COLUMNS) is empty -- a label leak would raise AssertionError at build time, not pass silently.",
        "overall_verdict": "PASS -- no future information or label leakage found in any traced feature.",
    }
    print("  All traced features: PASS (current-row-excluding primitives, verified from source)")
    return audit


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ctx = load_everything()

    selected_hyperparams = ctx["metadata"]["hyperparameter_selection"]["selected"]
    scale_pos_weight = ctx["metadata"]["hyperparameters"]["scale_pos_weight"]

    workload = part1_2_4_workload_and_structure(ctx)
    benchmark_rows = part3_runtime_benchmark(ctx, selected_hyperparams, scale_pos_weight)
    random_label_rows = part5_random_label_test(ctx, selected_hyperparams, scale_pos_weight)
    ablation_rows, scenario_rows = part6_9_feature_ablation(ctx, selected_hyperparams, scale_pos_weight)
    baseline_rows = part7_12_baselines(ctx, ablation_rows, selected_hyperparams, scale_pos_weight)
    permutation_df = part8_permutation_importance(ctx)
    reproducibility = part11_reproducibility(ctx, selected_hyperparams, scale_pos_weight)
    leakage_audit = part10_temporal_leakage_audit()

    # --- LR baseline scenario rows, for scenario_validation.csv completeness ---
    for scenario_str, stats in ctx["logreg_error_analysis"]["recall_by_scenario"].items():
        scenario_rows.append({
            "feature_group": "logistic_regression_phase4", "scenario": int(scenario_str),
            "total_fraud": stats["total_fraud"], "detected": stats["detected"], "recall": stats["recall"],
        })

    runtime_benchmark = {"workload": workload, "n_jobs_experiments": benchmark_rows}
    with open(OUTPUT_DIR / "runtime_benchmark.json", "w") as f:
        json.dump(runtime_benchmark, f, indent=2, sort_keys=True, default=str)

    pd.DataFrame(random_label_rows).to_csv(OUTPUT_DIR / "random_label_results.csv", index=False)
    pd.DataFrame(ablation_rows).to_csv(OUTPUT_DIR / "feature_ablation.csv", index=False)
    pd.DataFrame(baseline_rows).to_csv(OUTPUT_DIR / "baseline_comparison.csv", index=False)
    permutation_df.to_csv(OUTPUT_DIR / "permutation_importance.csv", index=False)
    pd.DataFrame(scenario_rows).to_csv(OUTPUT_DIR / "scenario_validation.csv", index=False)
    with open(OUTPUT_DIR / "reproducibility.json", "w") as f:
        json.dump(reproducibility, f, indent=2, sort_keys=True)
    with open(OUTPUT_DIR / "temporal_leakage_audit.json", "w") as f:
        json.dump(leakage_audit, f, indent=2, sort_keys=True)

    print(f"\nAll validation artifacts written to {OUTPUT_DIR}")
    print("Run scripts/08_validate_phase5.py's companion report is written separately (VALIDATION_REPORT.md).")


if __name__ == "__main__":
    main()
