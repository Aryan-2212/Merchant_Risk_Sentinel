# Feature Specification — Phase 3 (Feature Engineering)

Per Dev Plan §7/§8/§12/§27 (Phase 3) and CLAUDE.md §13. This document is generated from
`src/mrs/features/registry.py` (`FEATURE_SPECS`), the single source of truth for feature
metadata — every field below is read directly from that registry, not retyped by hand, so
this document cannot silently drift from the code. `tests/test_feature_registry.py`
enforces the reverse direction: every column the feature builder actually produces has
exactly one registry entry, and vice versa.

## 1. Architecture

```
data/processed/transactions/*.parquet (Phase 1, chronologically ordered)
        ↓
mrs.features._temporal          leakage-safe primitives (rolling/expanding/first-seen)
        ↓
mrs.features.transaction        5 features  — pure row-wise, no history
mrs.features.customer           12 features — customer behavioral history
mrs.features.terminal           14 features — terminal behavioral + fraud history
mrs.features.relationship       2 features  — customer-terminal pair history
        ↓
mrs.features.build.build_feature_frame   joins all four, attaches `split` label
        ↓
data/features/features_{train,validation,test}.parquet
```

33 features total. `src/mrs/data/schema.LABEL_COLUMNS` remains the one authoritative
exclusion mechanism for `TX_FRAUD`/`TX_FRAUD_SCENARIO`; nothing in `mrs.features`
redefines it.

## 2. Temporal-safety design (Dev Plan §33.6 — non-negotiable)

Every historical feature is built on two proven primitives in `mrs.features._temporal`
(see that module's docstring and `tests/test_features_temporal.py` for the properties
proven before any feature module was built on top of them):

- **Rolling time-windows** (`customer_tx_count_10min`, `..._1h`, `..._24h`, and the
  terminal/fraud equivalents) use `closed="left"`: the window is `[t-window, t)` — the
  left boundary (exactly `window` ago) is included, the current row's own timestamp is
  never included, even when another row shares that exact timestamp.
- **Expanding ("all prior history") statistics** (`*_hist_amount_mean/std`,
  `*_prior_tx_count`, `*_hist_fraud_count/rate`, `pair_prior_interaction_count`, …) use
  `shift(1)` before `expanding()`, which removes the current row from its own aggregate
  before the expanding window ever sees it.
- **Duplicate timestamps** (231,973 real rows in this dataset share a timestamp with at
  least one other row; 58 at customer granularity, 24 at terminal granularity — this is
  not a hypothetical edge case) are broken deterministically by `TRANSACTION_ID`
  (verified globally unique in Phase 1). Proven in both directions: the later-ID row sees
  the earlier-ID row's history, the earlier-ID row never sees the later one.
- **Features are built once over the complete chronologically-ordered dataset**, not
  separately per split (Dev Plan §13/§39): customer/terminal history legitimately
  continues across the train/validation/test boundaries. The `split` column is attached
  only at the end, purely as a downstream evaluation label — it never participates in any
  aggregation.

Real-data proof, not just synthetic proof: `tests/test_feature_build_integration.py`
independently recomputes several of a real established transaction's historical features
via plain pandas filtering (not calling `mrs.features` at all) and confirms an exact match;
it also mutates a genuine future real transaction and confirms an earlier, established
real transaction's features are byte-identical before and after.

## 3. Two Dev Plan bullets deliberately implemented once, not duplicated

- §7.1's *"time since previous transaction"* and *"transaction frequency in recent
  windows"* are the same underlying computation as §7.2's customer-scoped
  `customer_time_since_prev_tx_seconds` and `customer_tx_count_{10min,1h,24h}` — a
  "previous transaction" or "recent frequency" is only meaningful relative to an entity.
  Implemented once, customer-scoped; not duplicated as an entity-less transaction-level
  feature.
- §7.2's *"whether the terminal is new for the customer"* and the relationship family's
  *"whether this customer-terminal relationship is new"* are the same fact. Exposed once,
  as `customer_new_terminal_flag` (customer.py) — `relationship.py` computes the distinct
  piece customer.py does not: `pair_prior_interaction_count`, how many times this
  *specific* pair has interacted, not just whether this terminal is new to the customer.

## 4. Cold-start and missing-history convention (Dev Plan §8, handoff §4)

No fabricated defaults, applied uniformly:

- **Statistical descriptors** (mean, std, z-score, deviation, hour-deviation, time-since-
  previous) → `NaN` when undefined. A `NaN` here means "not yet knowable," never "zero."
- **Counts** (`*_prior_tx_count`, `*_tx_count_{10min,1h,24h}`, `*_unique_*_count`,
  `pair_prior_interaction_count`, fraud counts) → `0` when no prior history exists — a
  real, known answer, not a missing one.
- **Zero variance** (≥2 identical prior amounts) → a true `0.0`, distinguishable from a
  cold-start `NaN` via the paired `*_prior_tx_count` column. Z-scores never divide by a
  zero standard deviation — they return `NaN` instead.
- **Rates** (fraud rates, deviations) → `NaN` when the denominator (a transaction count)
  is `0`, never a false `0%`.

## 5. Transaction-level features (5)

Pure row-wise derivations of the current transaction's own fields — no history, no
cross-row aggregation, so nothing here can leak future information by construction.

| Feature | Definition |
|---|---|
| `tx_amount` | The current transaction's own `TX_AMOUNT`. |
| `tx_hour` | Hour of day (0–23) of `TX_DATETIME`. |
| `tx_day_of_week` | Day of week (0=Mon..6=Sun). |
| `tx_is_weekend` | 1 if Saturday/Sunday, else 0. |
| `tx_is_night` | 1 if hour ≥ 22 or hour < 6, else 0. **This threshold (22:00–06:00) is this project's own choice** — not sourced from the Handbook's later chapters, which are outside the GPL-isolated scope of `external/`. |

## 6. Customer behavioral features (12)

All historical; all use only transactions strictly before the current row.

| Feature | Definition | Window | Cold-start | Notes |
|---|---|---|---|---|
| `customer_tx_count_10min` | Count of prior transactions | `[t-10min, t)` | 0 | Also covers §7.1's "recent window frequency" |
| `customer_tx_count_1h` | Count of prior transactions | `[t-1h, t)` | 0 | |
| `customer_tx_count_24h` | Count of prior transactions | `[t-24h, t)` | 0 | |
| `customer_hist_amount_mean` | Mean `TX_AMOUNT`, all prior history | all prior | NaN | |
| `customer_hist_amount_std` | Std `TX_AMOUNT`, all prior history | all prior | NaN (0–1 prior) | True 0.0 if ≥2 identical prior amounts |
| `customer_prior_tx_count` | Count of all prior transactions | all prior | 0 | |
| `customer_amount_deviation` | `tx_amount − customer_hist_amount_mean` | all prior | NaN | |
| `customer_amount_zscore` | deviation / std, only if std defined and > 0 | all prior | NaN | NaN when std==0, never divides by zero |
| `customer_time_since_prev_tx_seconds` | Seconds since this customer's previous transaction | single most recent | NaN | Also covers §7.1's "time since previous transaction"; exactly 0.0 for tied-timestamp rows |
| `customer_new_terminal_flag` | 1 if this is the first-ever transaction at this terminal for this customer | all prior | 1 | |
| `customer_unique_terminals_count` | Distinct terminals used, strictly before this row | all prior | 0 | |
| `customer_hour_deviation` | Circular distance (0–12h) between this hour and the customer's historical circular-mean hour | all prior | NaN | Circular (sin/cos) mean, not linear — hours 23 and 1 correctly average to 0, not 12 |

## 7. Terminal behavioral features (14)

All historical; five (marked †) legitimately read `TX_FRAUD` — the only place in
`mrs.features` that does — strictly as a historical, prior-only aggregate (Dev Plan §10,
§34.2). None ever read the current row's own label.

| Feature | Definition | Window | Cold-start | Uses labels |
|---|---|---|---|---|
| `terminal_tx_count_10min` | Count of prior transactions | `[t-10min, t)` | 0 | |
| `terminal_tx_count_1h` | Count of prior transactions | `[t-1h, t)` | 0 | |
| `terminal_tx_count_24h` | Count of prior transactions | `[t-24h, t)` | 0 | |
| `terminal_hist_amount_mean` | Mean `TX_AMOUNT`, all prior history | all prior | NaN | |
| `terminal_hist_amount_std` | Std `TX_AMOUNT`, all prior history | all prior | NaN (0–1 prior) | |
| `terminal_prior_tx_count` | Count of all prior transactions | all prior | 0 | |
| `terminal_time_since_prev_tx_seconds` | Seconds since this terminal's previous transaction | single most recent | NaN | |
| `terminal_unique_customers_count` | Distinct customers served, strictly before this row | all prior | 0 | |
| `terminal_recent_fraud_count_24h` † | Count of fraudulent prior transactions | `[t-24h, t)` | 0 | ✓ |
| `terminal_recent_fraud_rate_24h` † | recent fraud count / recent transaction count | `[t-24h, t)` | NaN (not 0%) | ✓ |
| `terminal_hist_fraud_count` † | Count of fraudulent transactions, all prior history | all prior | 0 | ✓ |
| `terminal_hist_fraud_rate` † | hist fraud count / prior tx count | all prior | NaN (not 0%) | ✓ |
| `terminal_fraud_rate_deviation` † | recent rate − historical rate | both | NaN if either undefined | ✓ |
| `terminal_volume_deviation` | current 1h count − implied long-run hourly rate (`prior_tx_count / hours since first-ever transaction`) | 1h vs. all prior | NaN until ≥1h of history elapsed | |

`terminal_volume_deviation`'s 1-hour stability threshold is a project choice (the Dev
Plan does not specify one) — below it, an implied rate would be dominated by a near-zero
time denominator.

## 8. Customer-terminal relationship features (2)

`CUSTOMER_ID`/`TERMINAL_ID` identifiers are preserved in this module's output
specifically so a later phase can use them for relationship/graph visualization (handoff
§9, §21) — no graph construction happens in Phase 3.

| Feature | Definition | Cold-start |
|---|---|---|
| `pair_prior_interaction_count` | Count of strictly-prior transactions between this exact (customer, terminal) pair | 0 |
| `pair_is_new_relationship` | 1 if `pair_prior_interaction_count == 0` | 1 |

## 9. Split interaction (Dev Plan §13)

`build_feature_frame` attaches `mrs.data.splits.assign_split()`'s label as a `split`
column after all feature computation — it is a downstream evaluation label, never a
feature-generation boundary. A validation/test-period transaction legitimately uses
customer/terminal history from train-period transactions, because that history would
genuinely have existed in the real system at scoring time. Test labels never influence
feature construction.

## 10. Test coverage

91 tests across 7 files: `test_features_temporal.py` (19, the leakage-critical
primitives), `test_features_transaction.py` (8), `test_features_customer.py` (14),
`test_features_terminal.py` (15), `test_features_relationship.py` (9),
`test_feature_registry.py` (8, the registry↔builder round-trip contract), and
`test_feature_build_integration.py` (18, real-data: 1,754,155 rows, independent
recomputation, and a real future-row mutation check). See the Phase 3 completion report
for full results.

## 11. Known limitations

- `terminal_volume_deviation`'s "implied hourly rate" is a simplification (total prior
  count / elapsed hours), not a smoothed or seasonally-adjusted rate — stated explicitly
  rather than presented as more sophisticated than it is.
- Compromise-episode structure (from Phase 2's `DATASET_REPORT.md`) is not yet
  incorporated into any feature here — Phase 3 builds the general-purpose behavioral
  layer; scenario-specific risk modeling is a later-phase concern.
- No feature selection, scaling, or encoding has been applied — this is the raw feature
  layer, not a model-ready matrix. That belongs to Phase 4/5.
