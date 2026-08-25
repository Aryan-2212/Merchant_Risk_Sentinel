# Merchant Risk Sentinel

Track 2 AI Risk Manager: a transaction-, customer-, and terminal-behavioral risk
intelligence system built on the Fraud Detection Handbook's public simulated benchmark
dataset. See `Merchant_Risk_Sentinel_Development_Plan.md` for the full architecture and
`CLAUDE.md` for the standing engineering rules this project follows.

This is public **simulated** data — it is not real Razorpay production traffic and must
never be described as such.

## Current status: Phase 1 — Repository + Data

See `docs/PHASE1_REPORT.md` for the full phase completion report.

## Setup

```bash
/opt/homebrew/bin/python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

`requirements.lock.txt` records the exact resolved versions used to produce the results
in `docs/PHASE1_REPORT.md`.

## Running the Phase 1 pipeline

```bash
.venv/bin/python scripts/01_download_raw.py     # fetch the 183 daily raw files (~107 MB)
.venv/bin/python scripts/02_build_processed.py  # validate, normalize, write Parquet
.venv/bin/python scripts/03_reproduce_profiles.py  # reproduce + validate customer/terminal profiles
.venv/bin/pytest -q                             # full test suite (skips data tests if data/ absent)
```

## Layout

```
src/mrs/                   Core library
  config.py                 Paths and data-source constants (no split boundaries — see below)
  data/                      Acquisition, legacy-pickle compatibility, schema validation, processed build
  profiles/                  Reproduction and validation of customer/terminal profile tables

external/fraud_detection_handbook/   GPL-3.0 code ported from the official Handbook simulator,
                                      isolated and never imported by src/mrs/. See its NOTICE.md.

scripts/                    Runnable pipeline steps (thin wrappers over src/mrs)
tests/                      pytest suite; data-dependent tests are marked `data`
docs/                       Phase reports

data/                       Not committed to git (see .gitignore)
  raw/                       Original .pkl files, read-only, plus MANIFEST.json (tracked exception)
  processed/                 Validated, normalized, chronologically ordered Parquet
  reference/                 Reproduced + validated customer/terminal profile tables
```

## Data provenance

Raw data is fetched from `Fraud-Detection-Handbook/simulated-data-raw` on GitHub. Every
file's SHA-256, the upstream git blob SHA, source URL, and retrieval timestamp are
recorded in `data/raw/MANIFEST.json`, which is the one file inside `data/` that stays
tracked in git.

## Note on configuration

`src/mrs/config.py` intentionally contains no train/validation/test split boundaries.
Splits are determined from the Phase 2 dataset analysis, not assumed in advance.
