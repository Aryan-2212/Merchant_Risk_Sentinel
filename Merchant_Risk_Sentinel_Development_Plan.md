# Merchant Risk Sentinel --- Claude Development Plan

## 0. Purpose

This document is the **implementation handoff/specification for**
**Claude**. The goal is to move directly from planning into development
without repeatedly revisiting product scope, architecture, dataset
choice, or technology decisions. **Do not redesign the project unless a
concrete technical blocker is** **discovered.** Build the project in the
order specified below, keep the implementation modular, and preserve the
separation between ML risk detection, behavioral intelligence, LLM
explanation, and defensive policy.

------------------------------------------------------------------------

# 1. Project Definition

## Project Name

**Merchant Risk Sentinel**

## Track

**Track 2 --- AI Risk Manager**

## Core Problem

Payment fraud should not be treated only as an isolated
transaction-classification problem. The system must detect:

1.  suspicious individual transactions,
2.  abnormal customer behavior,
3.  abnormal merchant/payment-terminal behavior,
4.  emerging fraud patterns over time,

and then explain the evidence and recommend a bounded defensive
response.

## One-line Product Definition

> Merchant Risk Sentinel combines transaction-level ML with customer and
> merchant-terminal behavioral monitoring to detect emerging fraud
> patterns, explain why risk is increasing, and recommend bounded
> defensive actions.

------------------------------------------------------------------------

# 2. Frozen Dataset Decision

## Primary Dataset

Use the **Fraud Detection Handbook simulated dataset** as the primary
and initial complete data source. Do NOT use the user's previous
PCA-transformed credit-card dataset. The Handbook dataset provides:

-   5,000 customers
-   10,000 terminals
-   183 days
-   April 1--September 30, 2018
-   1,754,155 transactions
-   14,681 fraudulent transactions
-   approximately 0.84% fraud
-   customer profiles
-   terminal profiles
-   customer-terminal relationships
-   timestamps
-   transaction amounts
-   fraud labels
-   fraud scenarios
-   time-dependent fraud behavior

The simulator contains three fraud scenarios:

### Scenario 1 --- High-value fraud

Transactions above the specified high-value threshold are fraudulent.

### Scenario 2 --- Compromised terminal

A small number of terminals are compromised for a temporary period,
producing fraudulent activity associated with those terminals.

### Scenario 3 --- Compromised customer

A small number of customers are compromised for a temporary period, with
abnormal transaction amounts and fraudulent activity. Scenarios 2 and 3
intentionally introduce temporal behavior/concept drift.

## Data-source principle

Treat the Handbook dataset as a **public simulated benchmark**, not as
real Razorpay production data. Never claim that it represents actual
Razorpay transaction traffic.

------------------------------------------------------------------------

# 3. Data Acquisition

The raw transaction files are daily `.pkl` files available from the
official Fraud Detection Handbook simulated-data repository. The
development environment can be Google Colab. Initial data-loading
workflow:

1.  Load daily raw files from the official repository.
2.  Inspect all available dates/files.
3.  Load the full dataset only after schema validation.
4.  Preserve chronological ordering.
5.  Normalize IDs and timestamps.
6.  Keep raw data immutable.
7.  Create processed/feature datasets separately.

Do not manually copy/paste transaction data.

------------------------------------------------------------------------

# 4. Data Model

The system should conceptually work with three source entities.

## 4.1 Customer Profile

Expected conceptual information:

-   customer ID
-   customer location
-   typical transaction amount
-   transaction amount variability
-   typical transaction frequency
-   available terminals

## 4.2 Terminal Profile

Expected conceptual information:

-   terminal ID
-   terminal location

Treat the terminal as the merchant/payment acceptance entity for this
simulated environment. Do not call it a full real-world merchant
account.

## 4.3 Transaction

Fields available in the raw transaction table include:

-   `TRANSACTION_ID`
-   `TX_DATETIME`
-   `CUSTOMER_ID`
-   `TERMINAL_ID`
-   `TX_AMOUNT`
-   `TX_TIME_SECONDS`
-   `TX_TIME_DAYS`
-   `TX_FRAUD`
-   `TX_FRAUD_SCENARIO`

------------------------------------------------------------------------

# 5. Data Processing Rules

## Critical rule: prevent temporal leakage

Never calculate historical features using transactions that occur after
the transaction being scored. For a transaction at time T:

> Every behavioral feature must use only information available before T.

Do not use a normal random train/test split for the final temporal
evaluation.

------------------------------------------------------------------------

# 6. Train / Validation / Test Strategy

Use a chronological split. Initial proposal:

-   **Train:** April--July
-   **Validation:** August
-   **Test:** September

Before finalizing exact boundaries, inspect transaction and fraud
distribution by date. If the distribution is materially uneven, adjust
boundaries while preserving strict chronological order. The final test
set must remain untouched until model/threshold decisions are frozen.

------------------------------------------------------------------------

# 7. Feature Engineering Layer

Build a dedicated feature-engineering module. Do not mix feature
generation into API endpoints or model code. The feature engine should
generate three groups of features.

------------------------------------------------------------------------

## 7.1 Transaction-Level Features

Direct/derived transaction context:

-   transaction amount
-   hour
-   day
-   day of week
-   weekend indicator
-   night indicator
-   time since previous transaction
-   transaction frequency in recent windows

------------------------------------------------------------------------

## 7.2 Customer Behavioral Features

Build features from historical customer behavior. Minimum target
features:

-   customer transaction count in last 10 minutes
-   customer transaction count in last hour
-   customer transaction count in last 24 hours
-   historical customer average amount
-   historical customer amount standard deviation
-   current amount deviation from customer baseline
-   customer amount z-score where statistically valid
-   time since customer's previous transaction
-   number of terminals historically used
-   whether the terminal is new for the customer
-   customer's normal transaction hour / temporal deviation

Purpose:

> Detect behavior that is unusual for the individual customer.

This is primarily aimed at compromised-customer behavior.

------------------------------------------------------------------------

## 7.3 Terminal / Merchant Behavioral Features

Minimum target features:

-   terminal transactions in last 10 minutes
-   terminal transactions in last hour
-   terminal transactions in last 24 hours
-   terminal average transaction amount
-   terminal unique-customer count
-   terminal recent fraud count
-   terminal historical fraud rate
-   terminal recent fraud rate
-   time since terminal's previous transaction
-   deviation from terminal's historical transaction volume
-   deviation from terminal's historical fraud baseline

Purpose:

> Detect unusual behavior at a payment terminal/merchant entity.

This is primarily aimed at compromised-terminal behavior.

------------------------------------------------------------------------

# 8. Feature Engineering Quality Rules

For every feature:

1.  Define exactly what it means.
2.  Define the historical lookback window.
3.  Ensure no future information is used.
4.  Document whether it is transaction-, customer-, or terminal-level.
5.  Handle cold-start cases.
6.  Handle zero standard deviation safely.
7.  Handle missing history explicitly.

Do not create dozens of features without understanding them. Start with
a small interpretable feature set and expand only when justified by
validation results.

------------------------------------------------------------------------

# 9. ML Risk Model

## Objective

Predict: `P(transaction is fraudulent)`

## Model progression

Implement in this order:

### Baseline

Logistic Regression.

### Primary model

XGBoost.

### Optional comparison

Random Forest or another strong tabular baseline only if useful. Do not
spend time on deep learning unless the initial models demonstrate a
clear limitation that justifies it.

------------------------------------------------------------------------

# 10. Model Evaluation

Because fraud is highly imbalanced, accuracy is NOT a primary metric.
Report:

-   Precision
-   Recall
-   F1
-   PR-AUC
-   ROC-AUC
-   False Positive Rate
-   confusion matrix
-   false-positive count
-   false-negative count

PR-AUC is particularly important. Also analyze the business tradeoff
between:

-   missed fraud
-   legitimate transactions incorrectly flagged

------------------------------------------------------------------------

# 11. Scenario-Specific Evaluation

Do not evaluate only one aggregate fraud score. Evaluate detection of
the three simulator scenarios separately.

## Scenario 1

Question:

> Can the transaction model identify obvious high-value fraud?

## Scenario 2

Question:

> Can the system detect a compromised terminal over time?

Metrics should include:

-   terminal risk escalation
-   detection rate
-   detection delay
-   false alerts during normal periods
-   risk recovery after compromise ends

## Scenario 3

Question:

> Can the system detect a compromised customer's abnormal spending
> behavior?

Metrics should include:

-   customer risk escalation
-   detection rate
-   detection delay
-   false alerts
-   risk recovery after the compromise ends

------------------------------------------------------------------------

# 12. Behavioral Anomaly Engine

The ML model is not the entire risk system. Build a separate
behavioral/anomaly layer. It should compare current behavior against
historical baselines. Examples:

### Customer anomaly

Current amount vs customer historical amount.

### Customer velocity anomaly

Current transaction frequency vs customer's normal frequency.

### Terminal anomaly

Current terminal transaction volume vs terminal baseline.

### Terminal fraud anomaly

Recent terminal fraud rate vs historical terminal fraud rate.

### Temporal anomaly

Current behavior vs normal time-of-day behavior. Initially use
interpretable statistical/baseline methods. Do not immediately introduce
an unnecessary complex anomaly-detection algorithm.

------------------------------------------------------------------------

# 13. Risk Aggregator

Create a dedicated risk aggregation layer. Inputs:

-   transaction ML probability
-   customer behavioral anomaly
-   terminal behavioral anomaly
-   temporal/spike signals

Output:

-   final risk score
-   risk level
-   evidence/signals

Conceptual output:

    transaction_risk
    customer_risk
    terminal_risk
    behavioral_risk
    final_risk
    risk_level
    top_reasons

Do NOT choose arbitrary weights permanently. Start with a transparent
aggregation strategy, evaluate it, and tune thresholds/weights on
validation data.

------------------------------------------------------------------------

# 14. Risk Levels

Initial conceptual levels:

-   LOW
-   MEDIUM
-   HIGH
-   CRITICAL

Initial numeric thresholds are placeholders only. Determine final
thresholds using validation data and false-positive/fraud-loss
tradeoffs.

------------------------------------------------------------------------

# 15. Alert / Policy Engine

Do not let the LLM directly execute financial actions. The policy engine
should restrict available responses. Example action set:

-   ALLOW
-   MONITOR
-   STEP_UP_VERIFICATION
-   TEMPORARY_REVIEW
-   ESCALATE

The LLM can recommend an action. A deterministic policy validator
decides whether that recommendation is allowed. Architecture:

    Risk Engine
        ↓
    Policy Engine
        ↓
    Allowed actions
        ↓
    AI explanation/recommendation

------------------------------------------------------------------------

# 16. AI Risk Analyst

The LLM is NOT the primary fraud classifier. The ML/statistical risk
engine determines risk. The LLM receives structured evidence and
produces:

1.  risk explanation
2.  summary of what changed
3.  evidence supporting the alert
4.  recommended bounded action
5.  confidence/caveats when appropriate

Example input:

    {
      "terminal_risk": 0.91,
      "transaction_risk": 0.83,
      "fraud_rate_change": 12.4,
      "velocity_change": 3.2,
      "recent_suspicious_transactions": 31
    }

The LLM should reason only from supplied evidence. Do not allow it to
invent transaction facts, statistics, thresholds, or actions.

------------------------------------------------------------------------

# 17. Explainability

For every risk alert, show the underlying evidence. Example:

    Terminal Risk: 91/100

    Evidence:
    - Fraud rate increased from baseline
    - Transaction velocity is 2.8× baseline
    - 31 suspicious transactions detected
    - Unusual customer activity increased
    - Pattern persisted for multiple time windows

For the ML model, use feature importance / SHAP where useful.
Explainability should be grounded in actual model inputs and computed
statistics.

------------------------------------------------------------------------

# 18. Concept Drift

This is a core part of the project. The Handbook's compromised-terminal
and compromised-customer scenarios are temporary. Therefore the system
must not permanently mark an entity as bad. Desired behavior:

    NORMAL
      ↓
    RISK RISING
      ↓
    HIGH / COMPROMISED
      ↓
    RISK DECLINES
      ↓
    NORMAL

Track risk over time. Demonstrate that the system can recover after the
temporary fraud scenario ends.

------------------------------------------------------------------------

# 19. Backend Architecture

Use:

-   Python
-   FastAPI
-   Pydantic
-   PostgreSQL

Recommended logical services/modules:

    app/
    ├── api/
    │   ├── transactions
    │   ├── customers
    │   ├── terminals
    │   ├── alerts
    │   └── analyst
    │
    ├── services/
    │   ├── feature_service
    │   ├── fraud_model_service
    │   ├── anomaly_service
    │   ├── risk_service
    │   ├── alert_service
    │   ├── policy_service
    │   └── analyst_service
    │
    ├── models/
    ├── schemas/
    ├── db/
    └── config/

Keep data science/training code separate from production inference
services.

------------------------------------------------------------------------

# 20. Database Design

Minimum logical tables:

## customers

-   customer_id
-   profile fields if available

## terminals

-   terminal_id
-   profile fields if available

## transactions

-   transaction_id
-   timestamp
-   customer_id
-   terminal_id
-   amount
-   fraud label
-   fraud scenario

## transaction_features

-   transaction_id
-   generated behavioral features

## risk_scores

-   transaction_id
-   customer_id
-   terminal_id
-   transaction risk
-   customer risk
-   terminal risk
-   final risk
-   risk level

## alerts

-   alert_id
-   customer_id
-   terminal_id
-   severity
-   reason
-   evidence
-   recommendation
-   status
-   timestamp

## audit_logs

Record:

-   alert
-   risk score
-   evidence
-   AI explanation
-   recommended action
-   policy decision
-   timestamp

------------------------------------------------------------------------

# 21. Frontend

Preferred: **React + Vite** If frontend time becomes a major constraint,
use Streamlit as a fallback. The dashboard should have four primary
views.

## View 1 --- Risk Overview

Show:

-   total transactions
-   fraud rate
-   active alerts
-   high-risk terminals
-   high-risk customers
-   fraud/risk trend

## View 2 --- Terminal Risk

Show:

-   terminal risk score
-   baseline vs current fraud rate
-   transaction velocity
-   suspicious transaction count
-   customer activity
-   risk timeline
-   explanation

## View 3 --- Customer Risk

Show:

-   customer risk score
-   normal amount
-   current amount behavior
-   transaction frequency
-   terminal usage
-   anomalies
-   explanation

## View 4 --- Alert Investigation

Show:

-   alert severity
-   entity
-   risk score
-   evidence
-   timeline
-   AI explanation
-   recommended action
-   policy decision
-   audit history

------------------------------------------------------------------------

# 22. Demo / Replay Engine

Do not attempt to process 1.75M transactions live during the demo. Build
a chronological replay mechanism. Conceptually:

    Historical transactions
            ↓
    Replay chronologically
            ↓
    Feature engine
            ↓
    Risk engine
            ↓
    Alerts
            ↓
    Dashboard

Allow the demo to accelerate time. The replay should make the transition
from normal behavior to fraud visible.

------------------------------------------------------------------------

# 23. Required Demo Scenarios

## Demo 1 --- Normal operation

Show:

-   normal transaction flow
-   low risk
-   stable terminal behavior

## Demo 2 --- High-value fraud

Show:

-   abnormal transaction amount
-   elevated transaction risk
-   explanation
-   alert

## Demo 3 --- Compromised terminal

This should be the main demo. Show:

    Normal terminal
          ↓
    Compromise begins
          ↓
    Fraud activity increases
          ↓
    Terminal risk rises
          ↓
    Alert
          ↓
    AI explanation
          ↓
    Defensive recommendation
          ↓
    Compromise ends
          ↓
    Risk returns toward normal

## Demo 4 --- Compromised customer

Show:

    Normal customer spending
          ↓
    Compromise
          ↓
    Abnormal amounts
          ↓
    Customer risk rises
          ↓
    Explanation
          ↓
    Recovery

------------------------------------------------------------------------

# 24. MVP Scope

The MVP is complete when the following work:

1.  Handbook data loaded.
2.  Chronological train/validation/test split created.
3.  Historical feature engine implemented.
4.  Logistic Regression baseline implemented.
5.  XGBoost model implemented.
6.  Precision/Recall/F1/PR-AUC evaluation implemented.
7.  Customer behavioral risk implemented.
8.  Terminal behavioral risk implemented.
9.  Final risk aggregation implemented.
10. Alerts generated.
11. Dashboard displays risk and alerts.
12. At least one end-to-end fraud scenario replay works.

------------------------------------------------------------------------

# 25. Stretch Goals

Only after MVP works:

1.  SHAP explanations.
2.  LLM Risk Analyst.
3.  Bounded action recommendations.
4.  Concept-drift visualization.
5.  Alert history/audit trail.
6.  Multi-scenario replay.
7.  External validation using another public dataset.
8.  More sophisticated anomaly detection.
9.  Better UI/polish.
10. Real-time streaming simulation.

Do NOT start stretch goals before the MVP is functional.

------------------------------------------------------------------------

# 26. Explicitly Out of Scope

Do not spend development time on:

-   real Razorpay production payment integration
-   real customer PII
-   real merchant credentials
-   deep learning
-   graph neural networks
-   Kafka/Kubernetes/microservices unless a concrete need appears
-   complex autonomous agents
-   hundreds of hand-written fraud rules
-   attempting to reproduce a production payment gateway
-   claiming synthetic data is real-world Razorpay data

------------------------------------------------------------------------

# 27. Development Order

Claude should execute in this exact sequence.

## Phase 1 --- Repository + Data

1.  Create project structure.
2.  Create configuration.
3.  Download/read Handbook data.
4.  Validate all files.
5.  Inspect profile availability.
6.  Build raw → processed pipeline.
7.  Preserve raw data unchanged.

## Phase 2 --- Data Understanding

1.  Calculate complete dataset statistics.
2.  Verify date range.
3.  Verify fraud count/rate.
4.  Verify scenario distribution.
5.  Verify customer/terminal counts.
6.  Analyze fraud by scenario.
7.  Analyze temporal distribution.

Create a short `DATASET_REPORT.md`.

## Phase 3 --- Feature Engineering

1.  Implement chronological historical features.
2.  Implement customer features.
3.  Implement terminal features.
4.  Implement transaction features.
5.  Add leakage tests.
6.  Persist feature datasets.

Create `FEATURE_SPEC.md`.

## Phase 4 --- ML Baseline

1.  Logistic Regression.
2.  Evaluation.
3.  Error analysis.
4.  Save model and preprocessing.

## Phase 5 --- Main Model

1.  XGBoost.
2.  Validation tuning.
3.  Threshold selection.
4.  Evaluation.
5.  Compare against baseline.
6.  Save final candidate model.

Create `MODEL_REPORT.md`.

## Phase 6 --- Behavioral Risk

1.  Customer baseline engine.
2.  Terminal baseline engine.
3.  Temporal anomaly calculations.
4.  Risk aggregation.
5.  Scenario-specific evaluation.

## Phase 7 --- Backend

1.  Database.
2.  Models/schemas.
3.  Risk APIs.
4.  Alert APIs.
5.  Replay APIs.
6.  Policy engine.

## Phase 8 --- Dashboard

1.  Overview.
2.  Terminal risk.
3.  Customer risk.
4.  Alert investigation.
5.  Replay visualization.

## Phase 9 --- AI Analyst

Only now add the LLM.

1.  Structured evidence object.
2.  Prompt.
3.  JSON response schema.
4.  Explanation.
5.  Recommendation.
6.  Policy validation.
7.  Fallback when LLM unavailable.

## Phase 10 --- Final Demo + Hardening

1.  Run all scenarios.
2.  Measure detection delay.
3.  Test false positives.
4.  Test model failure.
5.  Test LLM failure.
6.  Test missing/insufficient history.
7.  Test API failures.
8.  Verify audit logs.
9.  Freeze test results.
10. Prepare final presentation/demo.

------------------------------------------------------------------------

# 28. Failure Handling Requirements

The system must continue functioning when individual components fail.

## ML unavailable

Fallback:

-   rule/statistical behavioral signals
-   mark model status unavailable
-   do not silently invent a score

## LLM unavailable

Risk engine continues. Display structured evidence instead of an AI
explanation.

## Insufficient history

Do not produce a misleading anomaly score. Return:

> Insufficient historical baseline.

## Missing fields

Degrade gracefully and reduce confidence where appropriate.

## API/database failure

Return controlled errors and preserve auditability.

------------------------------------------------------------------------

# 29. Engineering Principles

1.  **Interpretability over unnecessary complexity.**
2.  **No temporal leakage.**
3.  **ML decides risk; LLM explains risk.**
4.  **LLM does not directly execute financial actions.**
5.  **Every alert must have evidence.**
6.  **Every action recommendation must pass policy validation.**
7.  **Synthetic data must be clearly labeled as synthetic.**
8.  **Keep raw and processed datasets separate.**
9.  **Make every major component independently testable.**
10. **Build MVP before stretch features.**

------------------------------------------------------------------------

# 30. What Claude Must NOT Do

Do not:

-   redesign the track
-   replace the Handbook dataset without evidence of a blocker
-   immediately add another dataset
-   start with an LLM agent
-   start with a complex anomaly algorithm
-   train on randomly shuffled temporal data
-   use future information in historical features
-   report accuracy as the main fraud metric
-   fabricate data meanings
-   fabricate business metrics
-   claim simulated data is Razorpay production data
-   build unnecessary infrastructure
-   spend time polishing UI before the core pipeline works

If a requirement is ambiguous, first check this document and the project
source material.

------------------------------------------------------------------------

# 31. Definition of Done

The project is considered technically complete when we can demonstrate:

    Historical payment stream
            ↓
    Feature engineering
            ↓
    Transaction ML risk
            +
    Customer behavioral risk
            +
    Terminal behavioral risk
            ↓
    Unified risk score
            ↓
    Alert
            ↓
    Evidence
            ↓
    AI explanation
            ↓
    Bounded action recommendation
            ↓
    Audit log
            ↓
    Dashboard

And we can replay:

1.  normal behavior,
2.  high-value fraud,
3.  compromised terminal,
4.  compromised customer,

while showing measurable detection performance.

------------------------------------------------------------------------

# 32. Final Product Story

The final presentation should NOT be:

> "We trained XGBoost on a fraud dataset."

It should be:

> **"Merchant Risk Sentinel continuously learns transaction, customer,**
> **and terminal behavior. It detects both suspicious payments and**
> **emerging merchant/customer-level fraud patterns, quantifies the
> risk,** **explains the evidence, and recommends bounded defensive
> actions."**

The ML model is one component. The **risk intelligence system** is the
product.

------------------------------------------------------------------------

# 33. Cross-Project Development Rules

These rules apply to **every phase** and must be followed throughout the
entire project. Phase-specific instructions will be supplied separately
at the start of each phase and may refine these rules when necessary.

## 33.1 Source of Truth

-   This document is the baseline project specification.
-   The current phase instructions are the immediate implementation
    specification for that phase.
-   The official Fraud Detection Handbook documentation/source is the
    authority for dataset semantics.
-   If a later phase discovers a genuine technical blocker or evidence
    that a design assumption is incorrect, stop and report the blocker
    before silently redesigning the architecture.
-   Do not repeatedly revisit already-frozen product decisions without
    evidence.

## 33.2 Inspect Before Implementing

Before changing an existing repository:

1.  inspect the current file tree;
2.  inspect existing configuration and dependencies;
3.  inspect existing models/schemas/services;
4.  understand existing data flow;
5.  reuse working components where appropriate.

Do not overwrite working code merely to impose a preferred structure.

## 33.3 Minimal, Modular Changes

-   Prefer small, isolated modules.
-   Keep acquisition, preprocessing, feature engineering, training,
    inference, behavioral scoring, APIs, and UI separated.
-   Avoid unnecessary dependencies and infrastructure.
-   Keep interfaces between modules explicit.
-   Do not duplicate logic across notebooks and production modules.

## 33.4 Reproducibility

Every experiment must record:

-   dataset/source version or date range;
-   feature version;
-   train/validation/test boundaries;
-   model type and important hyperparameters;
-   threshold;
-   evaluation metrics;
-   random seed where randomness exists.

A result must be reproducible from the repository.

## 33.5 Data Provenance and Immutability

-   Never modify raw source files in place.
-   Maintain raw → processed → feature → model-data lineage.
-   Document source repository/URL, retrieval date, and transformations.
-   Do not commit large raw datasets or secrets to Git.
-   Prefer Parquet for processed/feature data where practical.

## 33.6 Leakage Prevention --- Non-Negotiable

Never use:

-   `TX_FRAUD` as an input feature;
-   `TX_FRAUD_SCENARIO` as an input feature;
-   future transactions;
-   future customer/terminal statistics;
-   post-event labels;
-   any feature unavailable at the moment a transaction would be scored.

Historical fraud-rate features are allowed only when they use labels
already known **before** the transaction being scored. Every historical
feature must have a documented "as-of" timestamp.

## 33.7 Cold Starts and Missing History

Explicitly handle:

-   first transaction for a customer;
-   first transaction for a terminal;
-   new customer-terminal relationship;
-   insufficient history;
-   zero variance;
-   missing profile data.

Never silently substitute future information.

## 33.8 Testing

Every phase must add appropriate tests.

At minimum:

-   deterministic transformation/unit tests;
-   data/schema validation;
-   temporal leakage tests;
-   API tests once APIs exist;
-   model smoke tests once models exist;
-   end-to-end tests for the final pipeline.

A phase is not complete merely because code runs once in a notebook.

## 33.9 Observability and Auditability

Important operations should produce structured logs/metadata.

Record where relevant:

-   pipeline/model version;
-   timestamp;
-   transaction/entity identifier;
-   risk output;
-   evidence used;
-   alert/action decision;
-   errors/fallbacks.

Never log secrets or API keys.

## 33.10 Security and Secrets

-   Use environment variables for credentials.
-   Provide `.env.example`; never commit `.env`.
-   Validate external inputs.
-   Never expose database credentials through APIs.
-   Treat LLM output as untrusted text.
-   Never allow an LLM response to directly execute a financial action.

## 33.11 No Silent Assumptions

When an assumption is required:

1.  identify it;
2.  verify it from the source if possible;
3.  document it;
4.  isolate it so it can be changed later.

Do not invent missing dataset semantics.

## 33.12 Documentation

Each major phase should leave concise documentation covering:

-   what was built;
-   inputs/outputs;
-   important design decisions;
-   known limitations;
-   tests/results;
-   how to run it.

Do not document functionality that has not been implemented.

# 34. Critical Data / ML Clarifications

## 34.1 Labels Are Ground Truth, Not Features

`TX_FRAUD` and `TX_FRAUD_SCENARIO` are for training/evaluation analysis
only. They must never be predictive inputs.

## 34.2 Historical Fraud Features

A terminal/customer historical fraud rate may be used only from outcomes
known before the current transaction.

For replay/live semantics:

``` text
transactions before T
        ↓
known historical state
        ↓
build features at T
        ↓
score T
        ↓
later reveal/store T outcome
        ↓
update future state
```

This ordering must be enforced in both implementation and tests.

## 34.3 Scenario Labels

`TX_FRAUD_SCENARIO` may be used for:

-   scenario-specific evaluation;
-   simulator analysis;
-   debugging;
-   demo ground truth.

It is not a model input.

## 34.4 Entity IDs

Do not treat `CUSTOMER_ID` or `TERMINAL_ID` as ordinary continuous
numeric variables. They identify entities and are primarily used for
historical aggregation/entity lookup.

## 34.5 Deployment-Available Features

If a feature cannot exist at scoring time, it cannot be used merely
because it improves offline metrics.

Maintain a clear distinction between:

-   labels;
-   scoring-time features;
-   post-transaction outcomes.

# 35. Requirements Traceability

Maintain a simple mapping from the Track 2 requirements to implemented
evidence.

At minimum:

  -----------------------------------------------------------------------
  Requirement             Implementation          Evidence
  ----------------------- ----------------------- -----------------------
  Transaction fraud       ML risk engine          metrics/report
  detection                                       

  Customer behavioral     customer baseline       scenario-3 evaluation
  risk                    engine                  

  Merchant/terminal risk  terminal risk engine    scenario-2 evaluation

  Temporal/emerging fraud replay + temporal       detection-delay results
                          features                

  Explainability          evidence + model        alert examples
                          explanations            

  AI assistance           Risk Analyst            grounded structured
                                                  outputs

  Defensive               policy engine           policy tests
  recommendation                                  

  Auditability            alerts/audit logs       persisted records

  Demonstrable product    dashboard/replay        end-to-end demo
  -----------------------------------------------------------------------

Do not claim a requirement is satisfied merely because a component
exists. Demonstrate it with a test, metric, screenshot, or replay
result.

# 36. Model / Risk Governance

Retain:

-   model version;
-   feature version;
-   threshold version;
-   training date;
-   evaluation period;
-   configuration version.

When a model changes, previous risk results must remain attributable to
the model version that produced them.

Store component signals with risk results so a score can be
reconstructed later.

# 37. Calibration and Threshold Selection

Do not automatically describe a model output as a calibrated
probability.

Document whether the output is:

-   raw model score;
-   calibrated probability;
-   normalized risk score.

Do not call `0.80` an "80% probability" unless calibration supports that
interpretation.

Final alert thresholds must be selected using validation data, not the
held-out test set.

# 38. Risk Aggregation Requirements

Keep component scores separately:

``` text
transaction_risk
customer_behavior_risk
terminal_behavior_risk
temporal_risk
final_risk
```

The dashboard must be able to show why final risk increased.

Avoid a black-box second model that hides component reasoning unless a
measured experiment justifies it.

# 39. Replay / Streaming Semantics

The replay engine must preserve chronological ordering and emulate
information available at each point in time.

For each replayed transaction:

``` text
1. Read current transaction.
2. Build features from prior state only.
3. Score transaction.
4. Update entity risk/state.
5. Generate alert if appropriate.
6. Only then make the transaction outcome available to future state.
```

This is mandatory for a credible demo.

# 40. Performance / Resource Rules

The full dataset is \~1.75M transactions, so development must be
memory-conscious.

Prefer:

-   chunked reads where needed;
-   Parquet for processed data;
-   vectorized operations;
-   sorted/indexed timestamps;
-   efficient rolling/grouped calculations;
-   persisted intermediate datasets.

Do not repeatedly recompute expensive historical features from raw data.
Profile slow operations before introducing architectural complexity.

# 41. LLM Safety and Reliability

The Risk Analyst receives only structured, computed evidence.

It must not:

-   calculate authoritative risk metrics itself;
-   invent missing evidence;
-   alter model scores;
-   override deterministic policy;
-   execute actions;
-   claim access to data it did not receive.

Use a strict structured response schema and validate it.

If the LLM is unavailable or invalid, return deterministic risk evidence
and a safe fallback.

# 42. Final Quality Gate

Before declaring the project complete:

-   [ ] raw data is reproducible and traceable;
-   [ ] profile/transaction schemas are validated;
-   [ ] no temporal leakage exists;
-   [ ] labels are never model features;
-   [ ] cold-start behavior is defined;
-   [ ] chronological evaluation is used;
-   [ ] baseline and main model are compared;
-   [ ] PR-AUC/F1/recall/precision are reported;
-   [ ] threshold selection uses validation only;
-   [ ] scenarios 1/2/3 are evaluated separately;
-   [ ] customer/terminal risk is explainable;
-   [ ] concept-drift/recovery behavior is demonstrated;
-   [ ] LLM output is grounded and schema-validated;
-   [ ] policy actions are deterministic/bounded;
-   [ ] audit records are persisted;
-   [ ] failure fallbacks work;
-   [ ] critical logic has tests;
-   [ ] dashboard and replay work end-to-end;
-   [ ] final test set remains untouched until evaluation is frozen.

# 43. Development Roadmap

The phase list below is the **stable roadmap**, not the phase-specific
instruction set. Exact instructions for the active phase will be
supplied separately at the start of that phase.

## Phase 1 --- Repository + Data

Create structure/configuration, acquire and validate Handbook data,
locate profile information, build raw → processed flow, and preserve raw
data.

## Phase 2 --- Data Understanding

Produce complete dataset statistics, verify distributions, analyze
scenarios and temporal behavior, and create `DATASET_REPORT.md`.

## Phase 3 --- Feature Engineering

Implement chronological transaction/customer/terminal features, leakage
tests, persistence, and `FEATURE_SPEC.md`.

## Phase 4 --- ML Baseline

Implement Logistic Regression, evaluation, error analysis, and
model/preprocessing persistence.

## Phase 5 --- Main Model

Implement XGBoost, validation tuning, threshold selection, comparison,
and `MODEL_REPORT.md`.

## Phase 6 --- Behavioral Risk

Implement customer baseline, terminal baseline, temporal anomaly
calculations, risk aggregation, and scenario evaluation.

## Phase 7 --- Backend

Implement PostgreSQL schema, FastAPI models/schemas, risk APIs, alert
APIs, replay APIs, and policy engine.

## Phase 8 --- Dashboard

Implement overview, terminal risk, customer risk, alert investigation,
and replay visualization.

## Phase 9 --- AI Analyst

Add structured evidence, prompt/schema, explanation, recommendation,
validation, and fallback.

## Phase 10 --- Final Demo + Hardening

Run all scenarios, measure detection delay and false positives, test
failures, verify audit logs, freeze results, and prepare the final demo.

# 44. Final Product Story

The final presentation should NOT be:

> "We trained XGBoost on a fraud dataset."

It should be:

> **"Merchant Risk Sentinel continuously learns transaction, customer,
> and terminal behavior. It detects both suspicious payments and
> emerging merchant/customer-level fraud patterns, quantifies the risk,
> explains the evidence, and recommends bounded defensive actions."**

The ML model is one component. The **risk intelligence system** is the
product.

# 45. Phase Handoff Protocol

At the beginning of **every phase**, phase-specific instructions will be
supplied separately.

Claude must begin each phase with a short **Phase Start Check**:

1.  current repository state;
2.  relevant existing files/modules;
3.  inputs available from previous phase;
4.  expected outputs for this phase;
5.  applicable cross-project constraints;
6.  blockers or assumptions requiring confirmation.

Then implement the phase.

At the end of each phase, Claude must provide a **Phase Completion
Report**:

-   files created/modified;
-   functionality implemented;
-   tests executed and results;
-   metrics/results where applicable;
-   known limitations;
-   decisions made;
-   artifacts produced;
-   exact recommended next phase;
-   blockers that must be resolved before continuing.

Do not automatically begin the next phase unless explicitly instructed.

# 46. Working Principle

Build the project **evidence-first**:

``` text
Source
  ↓
Understand
  ↓
Validate
  ↓
Implement
  ↓
Test
  ↓
Measure
  ↓
Decide
  ↓
Integrate
```

Never:

``` text
Assume
  ↓
Build
  ↓
Hope
```

When a component does not improve the product measurably or is not
required for the Track 2 story, prefer the simpler implementation.

The goal is a **credible, explainable, demonstrable risk-intelligence
product**, not maximum technical complexity.
