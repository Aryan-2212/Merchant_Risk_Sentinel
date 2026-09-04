# Simulated Recent Operational Stream

Merchant Risk Sentinel now supports a separate 21-day simulated operational window in addition to the frozen Fraud Detection Handbook benchmark.

## Purpose

The Handbook dataset remains the official benchmark for model development and chronological evaluation. The recent stream demonstrates how the same risk-intelligence architecture behaves when transactions arrive in a recent operating window.

It is **not live payment traffic** and must not be described as real Razorpay production data.

## Window

- 21 days
- 15 Aug 2026 through 4 Sep 2026
- deterministic generator seed: `20260904`
- transaction IDs begin at `2,000,000` to keep the stream distinct from the benchmark

## Controlled patterns

The generator establishes a baseline during the first week, increases selected customer spending and selected terminal activity during weeks two and three, and reduces those changes during the final days to exercise the existing behavioral state machine.

The intended observable progression is:

`NORMAL -> RISK_RISING -> HIGH_RISK -> RECOVERY -> NORMAL`

Synthetic `TX_FRAUD` / `TX_FRAUD_SCENARIO` values are scenario annotations for exercising the existing terminal fraud-history features. They are never model inputs and are excluded from official benchmark metrics.

## Processing

```text
Simulated recent transaction
        -> schema validation
        -> existing Phase-3 feature builders
        -> frozen XGBoost inference
        -> existing customer behavioral engine
        -> existing terminal behavioral engine
        -> existing risk aggregation
        -> deterministic policy engine
        -> persisted alert + audit trail
```

No retraining occurs. No benchmark split is changed.

## Run

After the Phase 8 database has been initialized and populated:

```bash
.venv/bin/python scripts/14_ingest_recent_stream.py
```

The generated artifacts are written under `data/recent/` locally and are intentionally not committed as a large generated dataset. The generator is the reproducible source of that data.

## API

After ingestion:

- `GET /recent/bounds`
- `GET /recent/transactions`

These endpoints are read-only and expose persisted recent results to the dashboard.
