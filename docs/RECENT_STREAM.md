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

## Simulated Live Transaction Ingestion (fixed-stream playback)

An optional way to get the SAME deterministic recent-stream data into the database:
instead of one bulk insert (`scripts/14_ingest_recent_stream.py`), release it
progressively -- a few transactions at a time, paced by a configurable interval -- so a
dashboard watching the database can show it "arriving" for a demo. See
`mrs.live.simulate` and `scripts/15_run_live_simulation.py`.

This mode is finite: once all ~41,610 rows of the fixed 21-day stream have been
released, there is nothing left to release (`scripts/16_reset_recent_stream.py` clears
it to demo the playback again from the start). For a producer that never runs out, see
"Continuous Simulated Live Stream" below -- that is what the dashboard's Network page
Start/Stop control actually drives; this section's CLI script remains available for
scripted/headless demos of the fixed stream specifically.

**This is not real payment traffic and not a live production feed** -- it is playback
of the same 21-day simulated stream, one (or a few) transaction(s) at a time. The
Entity Network page's "Live" mode is always labeled `SIMULATED LIVE STREAM`, never
"live production" or any variant implying real traffic.

- **Same pipeline, same source of truth.** `mrs.live.simulate.ingest_batch` calls the
  identical `mrs.features.build_feature_frame` / frozen `xgboost_v1` / behavioral
  engines / `mrs.risk.aggregate` / `mrs.db.populate` chain as the batch script -- it is
  not a second risk engine. A batch-computed reference run and a two-tick live run over
  the same 15 rows produce bit-identical `transaction_risk` values (verified in
  `tests/test_live_simulate.py`); the full 41,610-row stream, run entirely through the
  live path during validation, produced the exact same total transaction (41,610) and
  alert (12,752) counts as the batch path did.
- **Temporal correctness.** Each tick recomputes features/behavioral state over
  `released_so_far + new_batch` (a real, growing, strictly-chronological prefix of the
  same deterministic stream) -- never anything not yet released. Only `new_batch`'s own
  rows are ever written; an already-released transaction's persisted risk never changes
  as later ones arrive (verified directly in `tests/test_live_simulate.py`).
- **Idempotent / resumable.** `mrs.live.simulate.load_stream_and_pending` regenerates
  the deterministic stream and filters out whatever `transaction_id`s already exist
  (from a prior live run, or the batch script) before releasing anything -- a stopped
  and restarted run picks up exactly where it left off, never duplicating a
  transaction, risk score, audit entry, or alert.
- **Policy writes without the batch idempotency scan.** `mrs.policy.engine.apply_policy`
  (the batch/bulk entry point) scans the entire `audit_logs` table on every call to
  find already-decided transactions -- appropriate for an occasional bulk run, far too
  slow to call once per live tick. `mrs.policy.engine.decide_and_persist` (extracted
  from the same module, sharing the same `_persist_decisions` write path so no decision
  logic is duplicated) evaluates and writes only the small, already-known-fresh row
  list a live tick just inserted.
- **Reset.** `scripts/16_reset_recent_stream.py` deletes only `transaction_id >=
  mrs.config.RECENT_STREAM_TX_ID_OFFSET` rows (across alerts/audit_logs/risk_scores/
  transaction_features/transactions) so a live demo can be replayed from the start; it
  verifies the benchmark's own transaction count is unchanged before exiting. No reset
  control is exposed in the UI -- this is a script, run deliberately, not a button a
  demo viewer could click by accident.

Run with, e.g.:

```
.venv/bin/python scripts/16_reset_recent_stream.py       # start from a clean slate
.venv/bin/python scripts/15_run_live_simulation.py --interval 2
```

## Continuous Simulated Live Stream

A genuinely continuous producer -- unlike the fixed-stream playback above, this one
never runs out. While running, it generates ONE new transaction roughly every
`--interval`/`interval_seconds` (default ~2s), timestamped at the real wall clock
"now" (not August/September 2026), using a real, randomly-chosen existing customer
profile and one of that customer's own real `available_terminals` -- never a
fabricated entity. See `mrs.live.continuous` (generation + one-tick scoring) and
`mrs.live.manager` (the background-thread controller the API/UI drive).

**This is simulated demo data, not real payment traffic and not a live production
feed.** Every generated transaction carries `split="live"` (`mrs.config.
LIVE_STREAM_SPLIT_LABEL`) -- a THIRD split, distinct from both the frozen benchmark
(`train`/`validation`/`test`) and the fixed 21-day `"recent"` stream, so:

- `GET /replay/*` (scoped to `train`/`validation`/`test`) and `GET /recent/*` (scoped
  to `"recent"`) never show these rows -- both existing streams stay exactly what they
  were, with no code change needed in either router.
- `GET /stats/network` is unscoped by split (already orders by `tx_datetime desc`), so
  newly-generated `"live"` rows appear there automatically once the fixed recent
  stream's own end date (Sep 4, 2026) is in the past -- no change was needed there
  either, beyond the pre-existing `live_window` parameter (see below).
- Transaction ids start at `mrs.config.LIVE_STREAM_TX_ID_OFFSET` (3,000,000,000), well
  clear of both the benchmark's and the 21-day stream's own ranges.

**Same pipeline, still no second risk engine.** Each tick calls `mrs.live.simulate.
ingest_batch` (unmodified, now accepting an optional `split_label` so it can stamp
`"live"` instead of its original `"recent"` default) -- the identical
`build_feature_frame` / frozen `xgboost_v1` (inference only) / behavioral engines /
`aggregate_risk` / `mrs.db.populate` / `decide_and_persist` chain every other
ingestion path in this project uses. Temporal history for a new transaction comes from
`mrs.live.continuous.load_live_stream_history` -- every already-persisted `"recent"`
and `"live"` row, chronologically -- never the frozen benchmark (a separate dataset by
policy) and never anything not yet generated.

**Backend-controlled start/stop -- no manual script required.** `mrs.live.manager.
LiveStreamManager` runs the producer as a daemon thread inside the same FastAPI
process (no Celery/Redis/queue -- a single process-wide singleton, correct for this
project's single-worker `uvicorn` deployment). `GET /live/status`, `POST /live/start`
(optional `interval_seconds`, default from `mrs.config.
LIVE_STREAM_DEFAULT_INTERVAL_SECONDS`), and `POST /live/stop` are the only non-GET
routes in this API (`mrs.api.routers.live`) -- CORS was extended to allow `POST`
specifically for these. The Network page's "Live" mode Start Simulation/Pause button
(`frontend/src/pages/Network.tsx`) calls these directly; `livePlaying` is derived from
the real polled `/live/status` response, not a client-side-only toggle, so a page
reload or a second browser tab always agrees with the actual producer state. Clicking
Start while already running, or Stop while already stopped, is a safe no-op.

**Not seeded.** Unlike the deterministic 21-day recent stream, the continuous
producer's random generator is intentionally NOT fixed to a constant seed -- its whole
purpose is to feel like fresh, different activity every time it runs, not to be
byte-reproducible across runs (`tests/test_live_continuous.py` still proves the
underlying *generation function* is a pure, deterministic function of whatever
`rng`/inputs it is given, which is what makes it testable at all).

### Live Entity Network

`GET /stats/network?live_window=N` (N transactions, most recent first, same "last N"
convention `GET /stats/recent-activity` already uses) restricts the graph to real
customer/terminal relationships from only that rolling window -- not the entity's full
history -- and names the single newest transaction in `latest_transaction_id` so a
client can highlight it. It is a sibling branch inside the existing
`GET /stats/network` handler (`mrs.api.routers.stats._live_window_network`), not a
duplicate endpoint; omitting `live_window` leaves the original (default/focus) behavior
byte-for-byte unchanged. This endpoint needed no changes at all to support the
Continuous Simulated Live Stream: it already read the most-recent transactions
regardless of split, so newly-generated `"live"` rows simply appear as soon as they
exist. The frontend's Network page gained a minimal "Investigate" / "Live" mode toggle
plus the Start/Pause control described above (`frontend/src/pages/Network.tsx`), which
polls this endpoint every 1.5s while the producer is running, reusing the existing
`EntityNetworkGraph` component, its existing risk-state color language, and its
existing `is_focus` (square/glow) styling to highlight the newest arrival's customer
and terminal -- no new visual language was introduced.

**Known limitation:** the Network page's "Recent Transactions" side panel (shown when
drilling into one node) only queries `GET /replay/transactions` or `GET /recent/
transactions` -- there is no dedicated `GET /live/transactions` listing endpoint yet,
so that one panel shows empty for a customer/terminal whose only activity is in the
`"live"` split. The graph itself, `GET /transactions/{id}`, alerts, and the AI analyst
all work correctly for `"live"` transactions regardless.
