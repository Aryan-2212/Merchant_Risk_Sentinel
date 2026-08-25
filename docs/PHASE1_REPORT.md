# Phase 1 Completion Report — Repository + Data

Per Dev Plan §45 (Phase Handoff Protocol) and CLAUDE.md §13.

## Files created

```
.gitignore, .env.example, pyproject.toml, requirements.txt, requirements.lock.txt, README.md

src/mrs/__init__.py
src/mrs/config.py
src/mrs/data/__init__.py
src/mrs/data/schema.py
src/mrs/data/legacy_pickle.py
src/mrs/data/download.py
src/mrs/data/build_processed.py
src/mrs/profiles/__init__.py
src/mrs/profiles/reproduce.py
src/mrs/profiles/validate.py

external/fraud_detection_handbook/__init__.py
external/fraud_detection_handbook/simulator_profiles.py
external/fraud_detection_handbook/NOTICE.md

scripts/01_download_raw.py
scripts/02_build_processed.py
scripts/03_reproduce_profiles.py

tests/conftest.py, tests/test_schema.py, tests/test_legacy_pickle.py,
tests/test_manifest.py, tests/test_processed.py, tests/test_raw_immutability.py,
tests/test_profiles.py

docs/PHASE1_REPORT.md (this file)
```

Data produced (not committed to git): `data/raw/` (183 `.pkl`, 103 MB, read-only, plus
the tracked `MANIFEST.json`), `data/processed/transactions/` (6 Parquet partitions,
38 MB), `data/reference/` (`customer_profiles.parquet`, `terminal_profiles.parquet`,
1.2 MB).

## Functionality implemented

**Environment.** Project-local `.venv` on Python 3.12.13, chosen because the raw
dataset's `.pkl` files were written by pandas 1.x and reference
`pandas.core.indexes.numeric.Int64Index` (removed in pandas 2.0), while the system
Python was a bare 3.14 with no packages installed. Resolved dependencies are frozen in
`requirements.lock.txt`: pandas 2.3.3, numpy 2.5.2, pyarrow 25.0.1, requests 2.34.2,
pytest 9.1.1.

**Acquisition (`src/mrs/data/download.py`, `scripts/01_download_raw.py`).** Idempotent
download of all 183 daily files from `Fraud-Detection-Handbook/simulated-data-raw`.
Each file's content is verified against its upstream git blob SHA-1 at download time,
then set read-only (`0o444`). `data/raw/MANIFEST.json` records, per file: SHA-256, git
blob SHA, size, source URL, and retrieval timestamp, plus the upstream commit
(`6e67dbd0a3bfe0d7ec33abc4bce5f37cd4ff0d6a`) the whole raw layer is pinned to. Re-running
the script skips files whose hash already matches the manifest.

**Legacy-pickle compatibility (`src/mrs/data/legacy_pickle.py`).** Read-only loader.
`pd.read_pickle` on the pinned pandas 2.3.3 succeeded unaided on every file — pandas
still ships the `pickle_compat` shim that remaps the removed `Int64Index` class. A
manual `Unpickler` with class-relocation fallback is retained as a defensive path for a
future dependency bump, but was not needed in practice. The raw file is only ever opened
`"rb"`; nothing is rewritten.

**Schema validation (`src/mrs/data/schema.py`).** `validate_raw_frame` enforces: exact
column set and order, no nulls, non-negative amounts, `TX_FRAUD ∈ {0,1}`,
`TX_FRAUD_SCENARIO ∈ {0,1,2,3}`, and label-consistency (`TX_FRAUD=0 ⟺
TX_FRAUD_SCENARIO=0`), plus that each file holds exactly one calendar day sorted by
`TX_TIME_SECONDS`. `LABEL_COLUMNS = {TX_FRAUD, TX_FRAUD_SCENARIO}` is exported so any
future feature-building code has one canonical way to exclude labels
(`feature_candidate_columns`) rather than re-deriving the exclusion each time.

**Processed layer (`src/mrs/data/build_processed.py`,
`scripts/02_build_processed.py`).** Loads and validates all 183 raw files, normalizes
four columns that the simulator emits as `object`-dtype Python ints
(`CUSTOMER_ID`, `TERMINAL_ID`, `TX_TIME_SECONDS`, `TX_TIME_DAYS`) to real integer
dtypes, concatenates in filename (= chronological) order, and writes six year-month
Parquet partitions. Verifies raw integrity before building and validates chronological
ordering plus global `TRANSACTION_ID` uniqueness after.

**Profile reproduction (`src/mrs/profiles/`, `external/fraud_detection_handbook/`,
`scripts/03_reproduce_profiles.py`).** See "Profile availability finding" below.

## Tests executed and results

```
.venv/bin/pytest -q
33 passed in 2.43s
```

`.venv/bin/pytest -q -m data -v` (the 15 tests that require the downloaded dataset):
```
tests/test_legacy_pickle.py ..
tests/test_manifest.py ...
tests/test_processed.py ....
tests/test_profiles.py ....
tests/test_raw_immutability.py ..
15 passed, 18 deselected in 2.25s
```

18 schema-validator unit tests run against synthetic frames without requiring the
dataset — the suite passes cleanly (with the 15 data tests auto-skipped) on a checkout
that hasn't downloaded anything.

Coverage: schema validator edge cases (missing/extra columns, nulls, negative amounts,
out-of-domain and mutually-inconsistent labels, unsorted timestamps, mixed calendar
days, non-castable IDs, dtype normalization); a real raw file loads and matches the
documented schema; manifest completeness and date contiguity (no gaps, no duplicates,
`2018-04-01`–`2018-09-30`); processed-layer row count, dtypes, chronological order, ID
uniqueness; raw SHA-256 unchanged and files still read-only after the processed build
ran; profile reproduction is deterministic and validates against the actual processed
transactions.

## Measured dataset statistics

| Metric | Value |
|---|---|
| Total transactions | 1,754,155 |
| Date range | 2018-04-01 00:00:31 – 2018-09-30 23:59:57 |
| Fraud count | 14,681 |
| Fraud rate | 0.8369% |
| Unique terminals | 10,000 |
| Unique customers active in transactions | 4,990 (of 5,000 profiled) |

These match the Dev Plan §2 headline figures (~1,754,155 / 14,681 / ~0.84%). Reconciling
them formally and analyzing distributions is Phase 2's `DATASET_REPORT.md` — reported
here only as measured, not asserted as passing a spec check.

## Profile availability finding

Neither `simulated-data-raw` nor `simulated-data-transformed` (the only two Handbook
data repositories) publishes customer or terminal profile tables — both hold exactly the
183 daily transaction files plus a README. The generators exist only in
`Chapter_3_GettingStarted/SimulatedDataset.ipynb` in the `fraud-detection-handbook` repo.

Investigation established the generation is **deterministic and independent of the
transaction stream**: `generate_customer_profiles_table` seeds `np.random.seed(0)`,
`generate_terminal_profiles_table` seeds `np.random.seed(1)`, and `available_terminals`
is pure geometry with no randomness. The published dataset's exact parameters are
recorded in the same notebook: `n_customers=5000, n_terminals=10000, nb_days=183,
start_date="2018-04-01", r=5`.

These functions were ported verbatim into `external/fraud_detection_handbook/`
(isolated because the source is GPL-3.0-licensed; see its `NOTICE.md`) and re-run via
`scripts/03_reproduce_profiles.py`. Reproduction was empirically validated rather than
trusted on inspection alone: every `(CUSTOMER_ID, TERMINAL_ID)` pair observed in the
1,754,155 processed transactions was checked against that customer's reproduced
`available_terminals` list — **validation passed with zero containment failures**. The
script writes `data/reference/*.parquet` only when validation passes; had it failed,
nothing would have been persisted and this would be reported as a blocker instead.

One downstream artifact of this: 10 of the 5,000 profiled customers never appear in the
transaction data. Checked individually — each has a `mean_nb_tx_per_day` very close to
zero (drawn from `Uniform(0,4)`) and a non-trivial Poisson probability of zero
transactions across 183 days (e.g. customer 513: `mean_nb_tx_per_day=0.00029`,
`P(zero) ≈ 0.95`). This is expected simulator randomness, not a data quality problem —
and it is independent corroborating evidence that the profile reproduction is correct,
since it explains an otherwise-unexplained gap in the transaction data using only the
reproduced profiles.

## Decisions made

- Python 3.12 venv, pandas pinned `<3`, for legacy-pickle compatibility.
- Raw `.pkl` files kept byte-identical and read-only; all normalization happens only
  when building the processed layer, never in place.
- Processed layer stored as year-month partitioned Parquet.
- Customer/terminal profiles reproduced from the official simulator source rather than
  fabricated, and persisted only after empirical validation against real transaction
  data passed.
- Ported GPL-3.0 simulator code isolated under `external/`, run only as an offline
  script; product code under `src/mrs/` never imports it.
- **No train/validation/test split boundaries exist anywhere in Phase 1**, including as
  placeholder or "proposed" constants — per explicit instruction, these are determined
  from the Phase 2 dataset analysis, not assumed in advance.

## Known limitations

- `data/` is not committed to git (only `MANIFEST.json` is tracked) — a fresh checkout
  must run the three pipeline scripts before the data-marked tests will run.
- Reading the simulator source revealed two facts that affect how later-phase results
  should be interpreted, carried forward as findings rather than analyzed here:
  - **Scenario 1 is the deterministic rule `TX_AMOUNT > 220`** — any model that learns
    to threshold on amount will trivially "detect" this scenario, so scenario-1
    performance says little about the model beyond that rule.
  - **`TX_FRAUD_SCENARIO` is assigned by sequential overwrite (1 → 2 → 3)** in the
    simulator's `add_frauds`, not by exclusive membership — a transaction that matches
    both scenario 2 and scenario 3 conditions ends up labeled 3 only. Scenario-specific
    evaluation in Phase 2/5 needs to account for this rather than treating the scenario
    label as a clean partition.
- No performance profiling was done on the processed-layer build; at 1.75M rows the
  current implementation ran well within acceptable time, so no optimization was
  pursued (Dev Plan §14: don't add complexity without a demonstrated need).

## Recommended next phase

Phase 2 — Data Understanding: complete dataset statistics, verify distributions,
scenario-specific analysis (accounting for the sequential-overwrite behavior above), and
produce `DATASET_REPORT.md`. Train/validation/test split boundaries should be decided as
part of this phase, based on the actual date-level transaction and fraud distributions.

## Blockers

None. Phase 1 is complete and the repository is in a clean, tested, reproducible state.
