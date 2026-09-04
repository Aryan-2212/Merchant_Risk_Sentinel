# Simulated Recent Operational Stream

## What this is, and what it is not

The Simulated Recent Operational Stream is a deterministic, 21-day, ~41.6k-transaction
demo dataset layered **on top of, never replacing,** the frozen Fraud Detection
Handbook benchmark (`mrs.config.EXPECTED_START_DATE`/`EXPECTED_END_DATE`,
`mrs.data.splits`). It exists to demonstrate how customer and terminal behavioral risk
evolve over a recent operating period -- something the frozen 2018 benchmark cannot
show live, because its evaluation is already fixed and its dates never move.

**This is simulated operational/demo data. It is NOT real Razorpay production traffic,
NOT "live" data, and NOT an official benchmark evaluation.** Every synthetic
`TX_FRAUD`/`TX_FRAUD_SCENARIO` label in it is a simulation annotation only, generated
by this project, exactly like the benchmark's own labels -- never presented as ground
truth about real payment activity, and never fed to any model as an input feature.

## Why it is separate from the benchmark

- **Different purpose.** The benchmark is the frozen, chronologically-split dataset
  Phase 4/5's model was trained and evaluated against once (Dev Plan Sec 6/37) -- its
  numbers must never move again. The recent stream is a rolling demo surface whose only
  job is to show the *existing* frozen model and behavioral engines running against a
  fresh window of activity.
- **Different date range.** 2026-08-15 through 2026-09-04 by default (`mrs.config.
  RECENT_STREAM_START_DATE`/`RECENT_STREAM_END_DATE`) vs. the benchmark's
  2018-04-01 through 2018-09-30. The start date is configurable --
  `generate_recent_stream(start_date=...)` -- rather than hard-coded at every call
  site; `mrs.config.RECENT_STREAM_START_DATE` is only the default used when no
  `start_date` is passed. Whichever start date is used, it is always a fixed constant
  for that run (never derived from "today"), so a given (seed, start_date) pair stays
  reproducible regardless of when it is generated.
- **Different `split` value.** `split="recent"` (`mrs.config.RECENT_STREAM_SPLIT_LABEL`),
  never `"train"`/`"validation"`/`"test"` -- `mrs.data.splits.assign_split` (the frozen
  benchmark split boundaries) is never called on it. See "How it flows through the
  existing architecture" below for how that separation is enforced end to end.
- **Different transaction-id range.** IDs start at `mrs.config.RECENT_STREAM_TX_ID_OFFSET`
  (2,000,000,000), far above the benchmark's max id (1,754,154), so the two can never
  collide in Postgres.

## The 21-day window and deterministic generation

`mrs.data.recent_stream.generate_recent_stream(seed=mrs.config.RECENT_STREAM_SEED,
start_date=mrs.config.RECENT_STREAM_START_DATE)` generates exactly `mrs.config.
RECENT_STREAM_DAYS` (21) calendar days, targeting `mrs.config.RECENT_STREAM_TX_PER_DAY`
(1,800) organic transactions per day (injected attack traffic, below, adds on top of
that budget, so the actual daily/total count is "approximately" the target, not
exactly it).

Every random draw goes through one `numpy.random.default_rng(seed)` created at the top
of the function -- same (seed, start_date), byte-identical output (`tests/
test_recent_stream.py::test_same_seed_produces_identical_stream` asserts this with
`pandas.testing.assert_frame_equal`). Both are fixed values for a given call, never
computed from "today", so the stream stays reproducible regardless of when it is
generated.

## Simulated cohorts and the behavioral-evolution narrative

The generator samples `mrs.config.RECENT_STREAM_N_CUSTOMERS` (250) existing customer
ids and `RECENT_STREAM_N_TERMINALS` (100) existing terminal ids from `data/reference/
customer_profiles.parquet` / `terminal_profiles.parquet` -- no new entities are
invented. 80% of each sampled population stays a `"normal"` cohort for the entire
21 days. The remaining 20% is split into `"elevated_recover"` (12%) and
`"elevated_sustain"` (8%), each assigned a randomized (seeded) start day in week 2
(day 8-10):

- **Days 1-7 (baseline / week 1):** every entity is in its `"normal"` cohort phase --
  normal transaction amounts, normal customer/terminal velocity, zero injected fraud.
  `tests/test_recent_stream.py::test_week_one_is_baseline_no_elevated_terminal_states_yet`
  asserts no terminal reaches an elevated state this early.
- **Days ~8-10 through ~+3 (rising / week 2):** an elevated customer's transaction
  amounts are inflated 2.0x-2.6x its own baseline (customer_hist_amount_std is ~0.5x
  customer_hist_amount_mean for this dataset, so this lands `customer_amount_zscore`
  around 2.0-3.2, just past `mrs.behavioral.customer.RISING_THRESHOLD=2.0`); an
  elevated terminal receives 10-14 injected transactions/day at a 50% fraud rate,
  pushing `terminal_fraud_rate_deviation` past `mrs.behavioral.terminal.
  RISING_THRESHOLD=0.05`.
- **The following days (high-risk / week 3):** the same entities' signal strengthens --
  customer amount multiplier 3.2x-4.2x (z ~4.4-6.4, past `HIGH_RISK_THRESHOLD=4.0`);
  terminal injected volume rises to 18-26/day at a 90% fraud rate (past
  `HIGH_RISK_THRESHOLD=0.15`).
- **Final days (recovery, `"elevated_recover"` cohort only):** on a randomized
  (seeded) day in 17-19, that subset of entities reverts to baseline behavior --
  amounts/fraud-injection stop entirely -- and the existing, unmodified behavioral
  engines' own confirmation-streak logic (`RECOVERY_CONFIRM_COUNT=3`) eventually
  declares them `NORMAL` again. The `"elevated_sustain"` subset never reverts, staying
  elevated through day 21, so the demo can show both a completed recovery arc and a
  persisting episode -- no entity is permanently labeled malicious by construction, and
  not every elevated entity magically recovers either.

`tests/test_recent_stream.py` verifies the resulting stream actually contains all four
states (`NORMAL`/`RISK_RISING`/`HIGH_RISK`/`RECOVERY`) for both engines, and that at
least one terminal completes the full arc.

## How it flows through the existing architecture

The recent stream is not a second, competing pipeline. `scripts/14_ingest_recent_stream.py`
calls the exact same functions `scripts/12_populate_db.py` already used for the
benchmark, unmodified, just on the recent-stream frame instead:

```
generate_recent_stream()                              (new)
        v
validate_processed_frame()                             (reused, unmodified)
        v
build_feature_frame(..., split_override="recent")      (reused; one new optional param)
        v
load_model("xgboost_v1") + pipeline.predict_proba()     (reused, unmodified -- NOT retrained)
        v
compute_customer_behavioral_states()                    (reused, unmodified)
compute_terminal_behavioral_states()                     (reused, unmodified)
        v
aggregate_risk()                                         (reused, unmodified)
        v
populate_transactions/_features/_risk_scores()           (reused, unmodified)
        v
apply_policy()                                           (reused; one bug fix, see below)
```

Two small, deliberate, additive changes were needed to make this possible without
touching frozen benchmark behavior:

1. **`mrs.features.build.build_feature_frame` gained one optional keyword,
   `split_override`.** Every existing caller passes nothing and gets byte-identical
   behavior (`assign_split` against the frozen 2018-2018 boundaries, exactly as
   before). Only `scripts/14_ingest_recent_stream.py` passes `split_override="recent"`,
   which skips `assign_split` entirely -- the recent stream's 2026 timestamps never
   touch the frozen benchmark split boundaries, and `mrs.data.splits.SPLIT_BOUNDARIES`
   itself was not edited.
2. **`mrs.policy.engine.apply_policy`'s alert/audit inserts are now sub-batched** (at
   most 5,000 rows per `INSERT`, independent of the existing 50,000-row read-chunk
   size). This is a genuine bug fix, not a design change: the recent stream's alert
   rate (~30%, by design, for a demonstrable dataset) is far higher than the frozen
   benchmark's, and a single 41,610-row chunk's alert `INSERT` exceeded Postgres's
   65,535-bound-parameter limit the first time this ran. The fix sub-batches the
   `INSERT` itself; it changes no policy decision, no evidence, and no persisted value
   -- confirmed by re-running the real ingestion end to end and cross-checking the
   frozen benchmark's `unified_risk_level` distribution afterward against
   `docs/risk_aggregation_report_data.json` (see Validation below): unchanged, to the
   row.

### Historical Replay stays historical

`GET /replay/bounds` and `GET /replay/transactions` (`mrs.api.routers.replay`) now
filter `Transaction.split.in_(mrs.data.splits.SPLIT_ORDER)` -- the smallest change that
keeps "historical Replay" meaning exactly what it always has. Without this, inserting
2026-dated rows into the same `transactions` table would have silently widened
`/replay/bounds`' `max_tx_datetime` and let the recent stream leak into the historical
replay feed. `tests/test_api_recent.py::test_historical_replay_bounds_excludes_recent_stream_rows`
and `test_historical_replay_transactions_excludes_recent_stream_rows` seed a mixed
benchmark+recent database and assert the historical endpoints only ever see the
benchmark rows.

## How it is exposed

`mrs.api.routers.recent` is a thin, permanently-scoped sibling of `mrs.api.routers.
replay` -- same response schemas (`ReplayBounds`/`ReplayPage`/`ReplayItemOut`), same
keyset-cursor pagination convention, but every query is filtered to
`split == "recent"` by construction (not by caller-supplied parameter, so it can never
be accidentally left off):

- `GET /recent/bounds`
- `GET /recent/transactions` (supports `after_cursor`, `start`, `end`, `customer_id`,
  `terminal_id`, `limit` -- the same filters `/replay/transactions` supports, minus
  `desc`, which nothing currently needs for the recent stream)

Every other existing read endpoint (`/customers/{id}/risk`, `/terminals/{id}/risk`,
`/stats/*`, `/alerts`, `/transactions/{id}`, `/transactions/{id}/analyst`) needed no
changes at all: since the recent stream reuses real, existing customer/terminal ids and
is persisted into the same `risk_scores`/`alerts`/`transaction_features` tables, those
endpoints already show a customer's or terminal's 2026 activity alongside its 2018
activity, and the AI Risk Analyst already explains a recent-stream transaction from its
already-computed evidence exactly as it does a benchmark one (verified against a real
recent-stream CRITICAL/ESCALATE transaction during validation, below).

The frontend's Replay page (`frontend/src/pages/Replay.tsx`) gained a two-button
"Benchmark Dataset" / "Recent Simulated Stream" toggle that switches which pair of
endpoints it calls (`api.replayBounds`/`api.replayTransactions` vs.
`api.recentBounds`/`api.recentTransactions`) and updates its subtitle to name the
active source and date range explicitly -- reusing 100% of the existing feed-rendering,
speed-control, and row-click-to-investigate UI. No new page, no new components.

## How to run ingestion

```
.venv/bin/python scripts/14_ingest_recent_stream.py
```

Idempotent in the same sense as `scripts/12`/`scripts/13`: refuses to run again if any
`transaction_id >= mrs.config.RECENT_STREAM_TX_ID_OFFSET` already exists (mirrors
`mrs.db.populate.assert_transactions_table_empty`'s guard, scoped to this id range so
the frozen benchmark's own rows are never touched by this check). To regenerate,
delete the recent stream's rows first:

```sql
DELETE FROM alerts WHERE transaction_id >= 2000000000;
DELETE FROM audit_logs WHERE transaction_id >= 2000000000;
DELETE FROM risk_scores WHERE transaction_id >= 2000000000;
DELETE FROM transaction_features WHERE transaction_id >= 2000000000;
DELETE FROM transactions WHERE transaction_id >= 2000000000;
```

Does **not** call `mrs.db.populate.populate_customers_and_terminals` -- the recent
stream reuses customer/terminal ids already present in those tables from the benchmark
load; re-inserting them would violate their primary keys.

**Discovered while reloading the stream during development:** `DELETE FROM alerts`
was effectively hanging on a database already holding the 1.75M-row benchmark, because
`audit_logs.alert_id` (a foreign key to `alerts.alert_id`) had no index -- Postgres
must sequentially scan the whole referencing table once per deleted row to check that
FK, and `audit_logs` is the largest table in the schema. `mrs.db.models.AuditLog` now
declares `Index("ix_audit_logs_alert_id", "alert_id")`; a fresh `scripts/
11_init_db_schema.py` run picks it up automatically, and an existing database needs a
one-time `CREATE INDEX IF NOT EXISTS ix_audit_logs_alert_id ON audit_logs (alert_id);`
(no migration tool is in use in this project; this project has no other schema
migrations either, so a direct `CREATE INDEX` is consistent with how `scripts/
11_init_db_schema.py` itself applies schema changes). This is a genuine latent
performance fix, unrelated to the recent stream's own correctness -- it changes no
persisted value, only how fast a deletion is.

## How it differs from official benchmark evaluation

The recent stream is never used for, and must never be cited as, model evaluation:

- It is not part of any train/validation/test split, and Phase 5's frozen model
  metrics (`docs/MODEL_REPORT.md`) are computed only against the benchmark test split,
  unaffected by anything in this document.
- The frozen Phase 5 XGBoost model is used for **inference only** here -- it is never
  retrained, refit, or re-thresholded on recent-stream data.
- Its customer/terminal behavioral histories start fresh (cold start) at the beginning
  of the 21-day window, independent of that same customer/terminal's 2018 benchmark
  history -- an intentional, documented modeling choice (an ~8-year real-world gap
  between the two datasets makes carrying 2018 statistics into a 2026 baseline
  meaningless), not an oversight.
- Its synthetic `TX_FRAUD`/`TX_FRAUD_SCENARIO` labels exist only so the terminal
  behavioral engine's already-existing fraud-rate-deviation signal has something to
  react to (exactly mirroring how the benchmark's own labels work) -- they are
  simulation annotations, never surfaced as "real fraud" anywhere in the API or UI, and
  never fed to any model as an input feature (`mrs.models.dataset.get_feature_matrix`
  structurally cannot include them; `tests/test_recent_stream.py` asserts this
  explicitly for the recent-stream frame too, not just the benchmark).
