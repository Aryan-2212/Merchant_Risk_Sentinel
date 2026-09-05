# Merchant Risk Sentinel

**An explainable, AI-assisted merchant risk-intelligence system** — not a fraud
classifier. Merchant Risk Sentinel combines transaction-level machine learning,
customer behavioral risk, terminal/merchant behavioral risk, temporal context, entity
network relationships, deterministic policy, and evidence-grounded AI explanation into
a single unified risk assessment with a bounded, non-executing recommended action.

Built for the Track 2 ("AI Risk Manager") track. See
`Merchant_Risk_Sentinel_Development_Plan.md` for the full architecture and roadmap, and
`CLAUDE.md` for the standing engineering rules this project follows.

> **Data note:** every dataset in this project — the historical benchmark, the recent
> operational stream, and the live stream — is **simulated**. None of it is real
> Razorpay production traffic, and it is never presented as such anywhere in the code,
> API, or UI.

---

## Table of contents

- [What this is](#what-this-is)
- [Architecture](#architecture)
- [Data](#data)
- [Temporal integrity](#temporal-integrity)
- [Transaction ML risk](#transaction-ml-risk)
- [Behavioral risk](#behavioral-risk)
- [Risk aggregation and policy](#risk-aggregation-and-policy)
- [AI Risk Analyst](#ai-risk-analyst)
- [Dashboard](#dashboard)
- [Entity network and live processing](#entity-network-and-live-processing)
- [Replay](#replay)
- [Auditability](#auditability)
- [Repository structure](#repository-structure)
- [Setup and running](#setup-and-running)
- [Testing](#testing)
- [Limitations](#limitations)
- [Demo flow](#demo-flow)

---

## What this is

Most fraud-detection projects stop at "train a classifier, report AUC." Merchant Risk
Sentinel is built around the observation that a single transaction-level model misses
two important fraud patterns: a customer suddenly spending far outside their own
history, and a terminal being actively compromised over a window of time. Neither is
best captured by a per-row ML score — both are *behavioral drift* signals that need a
notion of state over time.

The system therefore runs three complementary risk components side by side, combines
them with a transparent (not fitted) aggregation rule, hands the result to a
deterministic policy engine for a bounded action, and only then asks an LLM to explain
— in plain language, grounded strictly in the evidence already computed — why the risk
level is what it is. The LLM never scores, never decides, and never invents evidence.

## Architecture

```
Historical Benchmark Data  +  Recent Simulated Stream  +  Live Simulated Stream
                              (all simulated; §Data)
                    ↓
          Feature Engineering (leakage-safe, chronological)
                    ↓
 ┌──────────────────────────────────────────────────────────┐
 │  Transaction ML Risk      Customer Behavioral Risk        │
 │  (XGBoost, threshold-     (NORMAL → RISK_RISING →         │
 │   gated probability)       HIGH_RISK → RECOVERY)          │
 │                                                            │
 │                     Terminal Behavioral Risk                │
 │                     (same 4-state machine, fraud-rate       │
 │                      deviation driven)                      │
 └──────────────────────────────────────────────────────────┘
                    ↓
     Risk Aggregation (rule/state-based, max-severity +
                        corroboration — not a fitted model)
                    ↓
     Deterministic Policy Engine → bounded action
     (ALLOW / MONITOR / STEP_UP_VERIFICATION /
      TEMPORARY_REVIEW / ESCALATE)
                    ↓
     AI Risk Analyst (Gemini) — explains evidence,
     recommends (advisory only); deterministic fallback
     if the LLM is unavailable
                    ↓
     Dashboard · Alerts · Entity Network · Replay ·
     Audit Trail
```

Each stage is a separate, inspectable module (`src/mrs/models`, `src/mrs/behavioral`,
`src/mrs/risk`, `src/mrs/policy`, `src/mrs/analyst`) — none of them are collapsed into
one opaque scoring function, and the ML/statistical layers determine risk; the LLM only
explains it.

## Data

Merchant Risk Sentinel uses **three** simulated datasets, kept structurally separate
end to end (a distinct `split` value per row, distinct transaction-id ranges, distinct
API routers), so none of them can leak into or be confused with another.

| | Historical benchmark | Recent operational stream | Live simulated stream |
|---|---|---|---|
| Source | Fraud Detection Handbook public simulator | This project's own generator, seeded | This project's own generator, unseeded |
| Rows | 1,754,155 | ~41,610 (21 days × ~1,800/day) | Unbounded, grows while running |
| Date range | 2018-04-01 → 2018-09-30 | 2026-08-15 → 2026-09-04 | Real wall-clock time, going forward |
| `split` value | `train` / `validation` / `test` | `recent` | `live` |
| Reproducibility | Fixed upstream dataset | Deterministic — same seed, byte-identical output | Not seeded — fresh each run by design |
| Used for | Model training/evaluation | Behavioral demo, historical replay | Continuous simulated ingestion, live network |
| Model use | Trained and evaluated here | Frozen model, inference only | Frozen model, inference only |

**Historical benchmark** — the Fraud Detection Handbook's public simulated dataset
(`external/fraud_detection_handbook/`, ported under GPL-3.0 and isolated from the rest
of `src/mrs`, per `external/fraud_detection_handbook/NOTICE.md`). This is the frozen
benchmark the transaction ML model is trained and evaluated against; its numbers never
move again once computed. See `docs/DATASET_REPORT.md`.

**Recent simulated operational stream** — a deterministic, seeded 21-day, ~41.6k
transaction dataset layered on top of the frozen benchmark to demonstrate how customer
and terminal behavioral risk evolve over a *recent* operating window (something a
frozen 2018 benchmark cannot show). It reuses real, existing customer/terminal ids
sampled from the benchmark's own profile tables — no invented entities — and is
generated to walk a subset of entities through the full behavioral arc
(`NORMAL → RISK_RISING → HIGH_RISK → RECOVERY`). Its own `TX_FRAUD`/`TX_FRAUD_SCENARIO`
labels are simulation annotations only, never fed to any model as a feature. See
`docs/RECENT_STREAM.md`.

**Live simulated stream** — a continuous, in-process producer that generates one new
transaction roughly every ~2 seconds, timestamped at the real current time, using real
existing customer/terminal profiles. It is explicitly labeled `SIMULATED LIVE STREAM`
everywhere it appears in the UI — never "live production" or any variant implying real
payment traffic. See [Entity network and live processing](#entity-network-and-live-processing).

None of this data is, or is ever described as, real Razorpay production traffic.

## Temporal integrity

Chronological correctness is enforced throughout the pipeline, not just claimed:

- Every historical feature (`src/mrs/features/`) is built from strictly-prior,
  leakage-safe rolling/expanding aggregates — a transaction's features can never depend
  on a later transaction of the same or any other entity (`tests/test_features_temporal.py`).
- `TX_FRAUD` and `TX_FRAUD_SCENARIO` are structurally excluded from the feature matrix
  (`mrs.data.schema.LABEL_COLUMNS`); `tests/test_feature_registry.py` and
  `tests/test_model_dataset.py` assert this rather than trusting it by convention.
- The final benchmark evaluation uses **chronological** train (Apr–Jul 2018) /
  validation (Aug 2018) / test (Sep 2018) splits, never a random split
  (`src/mrs/data/splits.py`, `docs/DATASET_REPORT.md` §7).
- The recent and live streams are each processed as their own strictly-increasing
  chronological sequence, with cold-start (fresh, empty) behavioral history at the
  start of their own window — never backfilled from the unrelated 2018 benchmark.

## Transaction ML risk

Two models were trained and compared on the frozen benchmark's chronological splits;
XGBoost is the model in production use. Both are documented in full in
`docs/MODEL_REPORT.md`.

| Metric | Logistic Regression | XGBoost |
|---|---:|---:|
| Precision | 0.348 | **0.772** |
| Recall | **0.733** | 0.663 |
| F1 | 0.472 | **0.713** |
| PR-AUC | 0.412 | **0.763** |
| ROC-AUC | 0.962 | **0.981** |
| Selected threshold | 0.930 | **0.970** |

Both thresholds are selected by max-F1 on the **validation** split only, then frozen
for a single test-set evaluation — the test set is never used to pick a threshold.

**Scenario-level recall (test set):**

| Scenario | Logistic Regression | XGBoost |
|---|---:|---:|
| 1 — high-value fraud | 32.6% | **84.7%** |
| 2 — compromised terminal | **75.6%** | 59.3% |
| 3 — compromised customer | 76.1% | **77.7%** |

XGBoost is the clear net improvement (large precision/F1/PR-AUC/ROC-AUC gains, an 86%
cut in false positives, and the targeted Scenario 1 fix the baseline was weak on), with
one honestly-documented trade-off: Scenario 2 (compromised terminal) recall regresses
by 16.3 points relative to the baseline at these independently-selected thresholds.
This is exactly the gap the terminal behavioral engine below is designed to cover — the
project's answer to a transaction-model weakness is a complementary detector, not a
model rewrite.

Both models' output is documented as an `uncalibrated_probability_estimate` — a
relative risk ranking, not a literal fraud probability (see `docs/MODEL_REPORT.md` §2/§8).

## Behavioral risk

Customer and terminal risk are **not** ML models — they are separate, interpretable,
non-ML statistical state machines (`src/mrs/behavioral/customer.py`,
`src/mrs/behavioral/terminal.py`) that track each entity's own behavior over time:

```
NORMAL → RISK_RISING → HIGH_RISK → RECOVERY → NORMAL
```

- **Terminal risk** is driven by `terminal_fraud_rate_deviation` — how far a terminal's
  recent (24h) fraud rate has drifted above its own historical baseline.
- **Customer risk** is driven by `customer_amount_zscore` — how far a transaction's
  amount deviates from that customer's own historical spending baseline.
- A single severe reading can jump an entity straight from NORMAL to HIGH_RISK; exiting
  HIGH_RISK requires a *confirmed* recovery (three consecutive calm transactions), so a
  brief dip is not mistaken for resolution.
- No entity is ever permanently labeled malicious from one historical event — the state
  machine is explicitly designed to track drift and recovery, not to blacklist.

**Real-data validation** (`docs/TERMINAL_BEHAVIORAL_REPORT.md`, run once over the full
1,754,155-row benchmark):

| | Value |
|---|---:|
| Scenario-2 fraud transactions flagged elevated at that moment | 91.4% |
| ...flagged specifically HIGH_RISK at that moment | **90.4%** |
| Terminal-level recall (compromised terminals ever detected) | **98.3%** (351/357) |
| Terminal-level precision | **11.75%** (351/2,986) |

The 90.4% transaction-level detection substantially exceeds XGBoost's own 59.3%
Scenario-2 recall — confirming the terminal behavioral engine is a genuinely
complementary signal, not a redundant one.

**On the 11.75% terminal-level precision figure:** this is a **behavioral terminal-state**
detection rate, not a transaction-level fraud classification rate. It means most
terminals that ever reach `HIGH_RISK` were not, in fact, part of a labeled Scenario-2
compromise — usually because a single unrelated high-value or compromised-customer
fraud transaction passed through an otherwise-normal terminal and briefly spiked its
24h fraud rate. `HIGH_RISK` asserts "this terminal's behavior is currently anomalous,"
not "this transaction is fraud" — combining that behavioral context with the
transaction-level ML score is exactly what Risk Aggregation, below, is for. This
trade-off (high recall, low precision) is a deliberate, documented property of a
sensitivity-first single-signal detector, not a defect.

## Risk aggregation and policy

`src/mrs/risk/aggregate.py` combines the three already-computed component signals —
transaction ML risk (threshold-gated), customer behavioral state, terminal behavioral
state — into one `unified_risk_level`. This is a **transparent max-of-severities-plus-
corroboration rule**, not a second ML model and not a fitted weighted blend:

- Each component maps to a 0/1/2 severity (calm / elevated / severe); a component with
  insufficient history is `unavailable`, never silently treated as calm.
- All components unavailable → `INSUFFICIENT_EVIDENCE`.
- Two or more components at severity 2 → `CRITICAL`.
- Otherwise the level is set by the single highest available severity
  (`HIGH` / `MEDIUM` / `LOW`).

`src/mrs/policy/rules.py` then maps that level to one bounded, non-financial defensive
action — deterministically, with no randomness and no LLM involvement:

| Unified risk level | Action |
|---|---|
| LOW | ALLOW |
| MEDIUM | MONITOR |
| HIGH | STEP_UP_VERIFICATION |
| CRITICAL | ESCALATE |
| INSUFFICIENT_EVIDENCE | TEMPORARY_REVIEW |

The policy engine is the sole authority on the action taken; the AI Risk Analyst below
may recommend a different action, but that recommendation is advisory only and never
overrides this table.

## AI Risk Analyst

`src/mrs/analyst/client.py` makes a single, non-agentic, structured call to **Google
Gemini** (`gemini-3.5-flash-lite` by default, configurable via `GEMINI_MODEL`) per
requested explanation — no tool use, no multi-step loop, no autonomous agent behavior.

```
Already-computed evidence (mrs.analyst.evidence)
        ↓
One structured Gemini call (response_schema-validated)
        ↓
Automated evidence-grounding check (blocks fraud-certainty language)
        ↓
AnalystResult — explanation, or a deterministic fallback
```

The analyst is given **only** the already-computed evidence for one transaction:
unified risk level, transaction ML score/severity, customer/terminal behavioral
states, the specific contributing signals that drove the level, and the already-decided
policy action. It:

- **may** summarize the situation, explain which supplied signals drove the
  assessment, and recommend one of the five bounded actions;
- **may not** invent a fact, number, date, or signal not present in the evidence;
  assert a transaction definitely is or is not fraud; change a risk score; or override
  the deterministic policy action.

An automated grounding check scans every response for fraud-certainty language ("is
fraud", "confirmed fraud", etc.) as a hard backstop on top of the system prompt's
instruction. If the LLM call fails, times out, is blocked, or fails the grounding
check, the endpoint falls back to a **deterministic explanation built entirely from the
same computed evidence fields** — the core risk system keeps functioning and always
returns a usable response, with the failure category (never raw provider error text)
surfaced to the caller. This is exercised via `GET /transactions/{id}/analyst`, which
returns HTTP 200 with a fallback explanation even with no API key configured.

## Dashboard

The frontend (`frontend/`, React + TypeScript + Vite + TanStack Query) implements the
following pages, reachable from the sidebar:

| Page | Route | Purpose |
|---|---|---|
| **Overview** | `/` | KPI strip, risk-signal breakdown, recent high-risk activity, and risk-trend/behavioral-shift panels — the command-center landing page. |
| **Terminals** | `/terminals` | Search/browse terminals. |
| **Terminal Detail** | `/terminals/:id` | One terminal's risk history over time, its behavioral-state timeline, and deviation from its own baseline. |
| **Customers** | `/customers` | Search/browse customers. |
| **Customer Detail** | `/customers/:id` | One customer's risk history, behavioral-state timeline, and spending-deviation view. |
| **Alerts** | `/alerts` | The alert queue — every non-ALLOW policy decision, filterable and paginated. |
| **Alert Detail** | `/alerts/:id` | One alert's full evidence, policy decision, and the AI Risk Analyst's explanation and recommendation. |
| **Network** | `/network` | Entity network graph (see below), with an **Investigate** mode (explore existing customer/terminal relationships) and a **Live** mode (watch the simulated live stream arrive). |
| **Replay** | `/replay` | Chronological playback of either the historical benchmark or the recent simulated stream. |
| **Transactions** | `/transactions` | Transaction explorer/search. |
| **Transaction Detail** | `/transactions/:id` | One transaction's full record, ML score, behavioral context, audit trail, and AI explanation. |
| **System health** | `/system` | Live status of the database, API, and AI analyst — every row backed by an actual query result, never an assumed "operational" claim. |

## Entity network and live processing

The **Entity Network** (`GET /stats/network`, `frontend/src/components/network/EntityNetworkGraph.tsx`)
connects customers and the payment terminals they have transacted at, so an analyst can
see at a glance whether elevated risk is:

- isolated to a single customer,
- concentrated around one terminal, or
- distributed across a connected cluster of entities.

The Network page's **Live** mode drives this same graph off the Continuous Simulated
Live Stream — `src/mrs/live/continuous.py` (generation + one-tick scoring) and
`src/mrs/live/manager.py` (a background-thread producer, started/stopped from the UI
via `POST /live/start` / `POST /live/stop`, the only non-GET routes in the API). Each
tick runs through the **exact same** feature → ML → behavioral → aggregation → policy
pipeline as every other ingestion path — there is no second risk engine. The producer
runs as a single daemon thread inside the existing FastAPI process; it deliberately
does **not** require Kafka, Redis, Celery, Kubernetes, or any other message-queue/
orchestration infrastructure, consistent with this project's "don't build unnecessary
infrastructure" principle. Because start/stop state lives in the backend (`GET
/live/status`), a page reload or a second browser tab always agrees on whether the
producer is running.

This is simulated demonstration data, generated in-process — it is never described as
real payment processing or a production feed.

## Replay

`GET /replay/*` chronologically replays the frozen historical benchmark; `GET
/recent/*` does the same for the 21-day recent simulated stream — each processes
transactions strictly in order so behavioral states evolve the way they originally did,
with no future information visible ahead of its timestamp. The two are kept structurally
separate (the historical replay endpoints are explicitly scoped to exclude recent/live
rows), so "Replay" always means exactly the dataset the UI says it does. The Replay page
lets a reviewer switch between the two datasets and control playback speed.

## Auditability

Every transaction that reaches a policy decision is recorded in `audit_logs`, whether
or not it is alert-worthy; every non-ALLOW decision additionally produces an `alerts`
row. Each stored risk score carries its `model_version`, `feature_version`, and the
`transaction_risk_threshold` that was actually used, so any past decision can be traced
back to the exact model/feature/threshold combination that produced it — never just a
bare number. Risk explanations shown in the UI are always built from these same
persisted evidence fields, and the AI analyst's fallback path guarantees an evidence-
backed explanation exists even when the LLM is unavailable.

## Repository structure

```
src/mrs/                    Core library
  data/                       Acquisition, schema validation, chronological splits,
                               recent-stream generator
  features/                   Leakage-safe feature engineering (transaction, customer,
                               terminal, relationship — 33 features total)
  models/                     Preprocessing, training, persistence for the LR baseline
                               and XGBoost model
  behavioral/                 Customer and terminal behavioral state machines
  risk/                       Risk aggregation (rule/state-based, not a model)
  policy/                     Deterministic bounded-action policy engine
  analyst/                    AI Risk Analyst (Gemini call, evidence assembly,
                               grounding check, deterministic fallback)
  live/                       Simulated live-ingestion pipeline (fixed-stream replay,
                               continuous producer, background manager)
  db/                         SQLAlchemy models, engine, population
  api/                        FastAPI app and routers (transactions, customers,
                               terminals, alerts, replay, recent, analyst, stats, live)

external/fraud_detection_handbook/   GPL-3.0 code ported from the official Handbook
                                      simulator, isolated and never imported by src/mrs
                                      outside profile reproduction — see its NOTICE.md

frontend/                   React + TypeScript + Vite dashboard (see Dashboard above)

scripts/                    Numbered, runnable pipeline steps — see Setup below
tests/                      pytest suite (646 tests at time of writing)
docs/                       Phase/model/dataset/feature/behavioral reports
data/                       Not committed to git — raw/processed/features/reference
models/                     Trained model artifacts (logreg_baseline_v1, xgboost_v1)
Merchant_Risk_Sentinel_Development_Plan.md   Full architecture and roadmap
CLAUDE.md                   Standing engineering rules this project follows
```

## Setup and running

### 1. Backend environment

```bash
/opt/homebrew/bin/python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

`requirements.lock.txt` records the exact resolved versions this project was built and
verified against. On macOS, `xgboost`'s native library additionally requires the OpenMP
runtime: `brew install libomp`.

### 2. Environment variables

Copy `.env.example` to `.env` and fill in what you need — everything is optional
except when you want the corresponding feature:

| Variable | Purpose |
|---|---|
| `MRS_DATA_DIR` | Override the data root if kept outside the repo. |
| `MRS_DATABASE_URL` | SQLAlchemy/psycopg connection string for the app database. Defaults to `postgresql+psycopg://localhost/merchant_risk_sentinel`. |
| `MRS_TEST_DATABASE_URL` | Separate database used by the pytest DB fixtures. Defaults to `postgresql+psycopg://localhost/merchant_risk_sentinel_test`. |
| `GEMINI_API_KEY` (or `GOOGLE_API_KEY`) | Enables live AI Risk Analyst explanations. Without it, the analyst endpoint still returns 200 with a deterministic fallback. Get a key at https://aistudio.google.com/apikey. |
| `GEMINI_MODEL` | Overrides the Gemini model, default `gemini-3.5-flash-lite`. |
| `MRS_FRONTEND_ORIGINS` | Comma-separated CORS origins for the frontend dev server (defaults cover the standard Vite port). |

Never commit `.env` — only `.env.example` is tracked.

### 3. Data acquisition and pipeline

```bash
.venv/bin/python scripts/01_download_raw.py          # fetch ~183 daily raw files (~107 MB)
.venv/bin/python scripts/02_build_processed.py        # validate, normalize, write Parquet
.venv/bin/python scripts/03_reproduce_profiles.py     # reproduce + validate customer/terminal profiles
.venv/bin/python scripts/05_build_features.py         # build the 33-feature layer
```

### 4. Models

```bash
.venv/bin/python scripts/06_train_baseline.py         # Logistic Regression baseline
.venv/bin/python scripts/07_train_xgboost.py           # XGBoost + comparison vs. baseline
```

### 5. Database

Requires a local PostgreSQL instance and a database created for `MRS_DATABASE_URL`
(e.g. `createdb merchant_risk_sentinel`; `createdb merchant_risk_sentinel_test` for
tests).

```bash
.venv/bin/python scripts/11_init_db_schema.py          # DDL only
.venv/bin/python scripts/12_populate_db.py             # populate with benchmark pipeline output
.venv/bin/python scripts/13_run_policy_engine.py        # run the deterministic policy engine
.venv/bin/python scripts/14_ingest_recent_stream.py      # add the 21-day recent simulated stream
```

### 6. Backend API

```bash
.venv/bin/uvicorn mrs.api.main:app --reload --env-file .env
```

### 7. Frontend

```bash
cd frontend
npm install
cp .env.example .env    # only needed if the backend isn't on http://localhost:8000
npm run dev
```

### 8. Simulated live stream (optional)

The continuous producer behind the Network page's **Live** mode is started/stopped
from the UI itself (`POST /live/start` / `POST /live/stop`) — no separate process is
required. For a scripted/headless demo of the fixed 21-day stream instead:

```bash
.venv/bin/python scripts/16_reset_recent_stream.py     # clear any prior run
.venv/bin/python scripts/15_run_live_simulation.py --interval 2
```

## Testing

```bash
.venv/bin/pytest -q
```

At time of writing: **646/646 backend tests passing** (data-dependent tests are marked
`data` and skip automatically if `data/raw` is absent). Frontend TypeScript check and
production build are both clean:

```bash
cd frontend
npm run build     # tsc -b && vite build
```

## Limitations

- **All data is simulated.** The historical benchmark, the recent operational stream,
  and the live stream are all synthetic — none of it is connected to real payment
  infrastructure or reflects real Razorpay transactions.
- **The AI Risk Analyst is advisory, not authoritative.** It explains and can suggest a
  bounded action, but the deterministic policy engine's decision is always the one that
  stands; the analyst cannot change a risk score or execute anything.
- **Behavioral risk is statistical, not proof.** A `HIGH_RISK` behavioral state
  describes anomalous recent activity relative to an entity's own baseline — it is not
  a fraud determination, and terminal-level precision is deliberately low (11.75%) at
  the current thresholds in exchange for high recall (98.3%); see [Behavioral risk](#behavioral-risk).
- **Both ML models are uncalibrated.** `predict_proba()` output is a relative risk
  ranking, not a literal fraud probability (`docs/MODEL_REPORT.md` §8).
- **XGBoost trades away some Scenario-2 recall** relative to the Logistic Regression
  baseline at matched, independently-selected thresholds (59.3% vs. 75.6%) — the
  terminal behavioral engine is this project's documented answer to that gap, not a
  retrained model.
- **No authentication.** There is no auth layer on the API (a deliberate scope
  decision for this project, not an oversight) — do not expose it beyond a local/demo
  environment.
- **Single-process live producer.** The continuous live stream runs as one background
  thread inside the API process; it is not designed for multi-worker/horizontally
  scaled deployment.
- **No dedicated live-transactions list endpoint yet.** The Network page's side panel
  for "recent transactions" queries the historical/recent replay endpoints; it does not
  yet have a dedicated listing endpoint for `split="live"` rows specifically, so that
  one panel can show empty for an entity whose only activity is in the live stream. The
  graph itself, transaction detail, alerts, and the AI analyst all work correctly for
  live-stream transactions regardless.

## Demo flow

A suggested path for a reviewer or judge, roughly in order:

1. **Overview** — the command-center landing page: KPI strip, risk-signal breakdown,
   recent high-risk activity.
2. **Customer / Transaction investigation** — drill from a customer or transaction into
   its full risk history and evidence.
3. **Terminal investigation** — a terminal's behavioral-state timeline, showing the
   NORMAL → RISK_RISING → HIGH_RISK → RECOVERY arc.
4. **Alerts → Alert Detail** — the alert queue, then one alert's full evidence, policy
   decision, and AI Risk Analyst explanation.
5. **Network (Investigate mode)** — the entity graph, showing whether risk is isolated
   or clustered.
6. **Network (Live mode)** — start the continuous simulated live stream and watch new
   transactions, risk scores, and network updates arrive in real time.
7. **Replay** — chronological playback of the historical benchmark or recent stream.
8. **System health** — live status of the database, API, and AI analyst.
