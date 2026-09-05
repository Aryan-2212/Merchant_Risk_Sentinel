# Merchant Risk Sentinel

**Track 2 — AI Risk Manager**

Merchant Risk Sentinel is an explainable AI-assisted risk intelligence system designed to detect and investigate suspicious activity across transactions, customers, payment terminals, and changing behavior over time.

The system is built around the idea that fraud risk does not always come from a single suspicious transaction. A transaction may look relatively normal by itself while the customer's behavior is changing, a payment terminal is showing abnormal activity, or risk is increasing across connected entities.

Merchant Risk Sentinel combines these signals into a unified risk assessment and provides an investigation workflow for understanding why the risk increased and what action should be taken.

**Important:** All data used in this project is simulated. The primary benchmark comes from the Fraud Detection Handbook's public simulated dataset, supplemented by a simulated recent operational transaction stream. This project does not use or represent real Razorpay production traffic.

---

## Project Objective

The objective of Merchant Risk Sentinel is to build a risk-management system that goes beyond traditional transaction-level fraud classification.

The system evaluates:

* Individual transaction risk
* Customer behavioral risk
* Terminal behavioral risk
* Changes in behavior over time
* Relationships between customers and terminals
* Combined risk across multiple signals

These signals are then passed through a deterministic risk aggregation and policy layer before reaching the AI Risk Analyst.

The AI is used for explanation and investigation assistance rather than making autonomous financial decisions.

---

## Core Architecture

Merchant Risk Sentinel follows a modular risk-intelligence architecture:

Data
↓
Feature Engineering
↓
Transaction ML Risk + Customer Behavioral Risk + Terminal Behavioral Risk
↓
Risk Aggregation
↓
Alert / Policy Engine
↓
AI Risk Analyst
↓
Dashboard / Investigation / Audit Trail

The important design decision is that these components remain separate.

The transaction model does not determine the entire system's risk on its own.

Customer and terminal behavior are evaluated independently, and the resulting signals are combined by the risk aggregation layer.

The AI Risk Analyst receives the evidence produced by these systems and explains it to the analyst.

---

## Transaction-Level Risk

The transaction-level risk component uses machine learning to evaluate individual transactions.

The final system uses XGBoost as the primary transaction model, with Logistic Regression used as an interpretable baseline.

The model produces a transaction risk score which is then combined with behavioral risk from the customer and terminal associated with that transaction.

The selected XGBoost threshold is 0.970.

---

## Customer Behavioral Risk

Customer behavior is evaluated against historical customer activity.

The system looks for changes such as:

* Changes in transaction frequency
* Changes in transaction amounts
* Changes in elevated-risk activity
* Changes in severity rate
* Changes in overall behavioral patterns

Customer risk is treated as a changing state rather than a permanent classification.

A customer can move through states such as:

NORMAL → RISK RISING → HIGH RISK → RECOVERY → NORMAL

This allows the system to represent temporary periods of elevated risk and subsequent recovery.

---

## Terminal Behavioral Risk

Payment terminals are evaluated independently using historical terminal behavior.

The system can detect changes such as:

* Increased transaction velocity
* Increased elevated-risk transaction rates
* Abnormal activity concentration
* Significant deviation from historical terminal behavior

For example, a terminal may have historically processed a relatively stable percentage of elevated-risk transactions and then experience a sudden increase.

This change becomes behavioral evidence for the terminal's current risk state.

Terminal behavior is therefore not treated as proof that a terminal is permanently malicious.

---

## Temporal Risk Detection

Temporal analysis is a core part of Merchant Risk Sentinel.

The system compares current activity with historical baselines and tracks how risk evolves over time.

Behavioral states can move between:

NORMAL
↓
RISK RISING
↓
HIGH RISK
↓
RECOVERY
↓
NORMAL

This allows the system to identify emerging risk while also recognizing recovery.

The objective is to detect changes in behavior rather than permanently label customers or terminals based on one historical event.

---

## Temporal Integrity

Temporal correctness is one of the most important technical requirements of the system.

Historical features must only use information that would genuinely have been available at the time the transaction was scored.

The predictive pipeline does not use:

* Future transactions
* Future customer statistics
* Future terminal statistics
* Future labels
* TX_FRAUD as a predictive feature
* TX_FRAUD_SCENARIO as a predictive feature

The benchmark evaluation uses chronological train, validation, and test periods rather than a random train/test split.

This ensures that historical behavioral features do not accidentally include future information.

---

## Risk Aggregation

The system does not treat the transaction model as the final decision-maker.

Instead, the following components are evaluated independently:

Transaction ML Risk
+
Customer Behavioral Risk
+
Terminal Behavioral Risk
↓
Unified Risk Assessment

This approach allows the system to identify cases where a transaction-level model alone may not provide enough context.

For example, a transaction can have a relatively moderate ML score while both the customer and terminal are experiencing significant behavioral changes.

The combined evidence can therefore produce a higher overall risk assessment.

---

## Deterministic Policy Engine

After risk aggregation, the system applies a deterministic policy layer.

The available defensive actions are:

* ALLOW
* MONITOR
* STEP_UP_VERIFICATION
* TEMPORARY_REVIEW
* ESCALATE

The policy engine is responsible for determining which action is allowed for a given risk level.

The AI Risk Analyst does not override this policy.

This separation ensures that the language model remains an explanation and advisory component rather than becoming an autonomous financial decision-maker.

---

## AI Risk Analyst

The AI Risk Analyst provides an investigation layer on top of the existing risk system.

The AI receives structured evidence that has already been calculated by the system.

This can include:

* Transaction risk
* Customer behavioral state
* Terminal behavioral state
* Contributing risk signals
* Historical deviations
* Model information
* Policy action
* Relevant transaction context

The AI can then:

* Summarize the situation
* Explain why the risk increased
* Highlight important evidence
* Provide a bounded advisory recommendation

The AI cannot:

* Independently determine fraud
* Change the risk score
* Override deterministic policy
* Execute financial actions
* Invent evidence or metrics

If the AI service is unavailable, the system can fall back to deterministic evidence and explanation so that the underlying risk pipeline continues functioning.

---

## Explainability

Every important risk assessment is supported by computed evidence.

For example, an explanation may identify that risk increased because:

* Transaction amount deviated from the customer's historical baseline
* Terminal transaction velocity increased
* Terminal elevated-risk rate increased
* Customer behavioral risk increased

The AI does not calculate these metrics itself.

It receives the structured evidence produced by the risk system and converts that information into an analyst-friendly explanation.

This keeps the explanation grounded in the actual system state.

---

## Dataset

The primary dataset is the Fraud Detection Handbook's public simulated benchmark dataset.

It provides the foundation for:

* Transaction-level fraud detection
* Customer behavioral analysis
* Terminal behavioral analysis
* Temporal analysis
* Scenario-specific evaluation

The benchmark contains 1,754,155 transactions.

The chronological dataset split used by the project is:

Train: 1,169,723 transactions
Validation: 296,559 transactions
Test: 287,873 transactions

The dataset is simulated benchmark data and must not be interpreted as real Razorpay production data.

---

## Model Evaluation

The project evaluates fraud detection using Precision, Recall, F1, PR-AUC, ROC-AUC, and False Positive Rate rather than relying primarily on accuracy.

### Logistic Regression Baseline

Precision: 0.348
Recall: 0.733
F1: 0.472
PR-AUC: 0.412
ROC-AUC: 0.962

### XGBoost

Precision: 0.772
Recall: 0.663
F1: 0.713
PR-AUC: 0.763
ROC-AUC: 0.981

Selected threshold: 0.970

XGBoost provides a stronger overall balance between precision and recall on the chronological test period.

---

## Scenario Performance

The model behaves differently across the fraud scenarios in the benchmark.

| Scenario   | Logistic Recall | XGBoost Recall |
| ---------- | --------------: | -------------: |
| Scenario 1 |           32.6% |          84.7% |
| Scenario 2 |           75.6% |          59.3% |

This difference is important because it demonstrates why Merchant Risk Sentinel does not rely only on transaction-level machine learning.

Behavioral and temporal signals provide additional context that can capture patterns that a transaction classifier may not capture consistently across scenarios.

---

## Terminal Behavioral Evaluation

Terminal behavioral analysis provides another layer of detection.

For the Scenario-2 evaluation:

* Fraud transaction detection: 90.4%
* Terminal-level recall: 98.3%
* Compromised terminals detected: 351 / 357
* Terminal-level precision: 11.75%

These metrics describe terminal-level behavioral detection and should not be interpreted as transaction-level fraud classification metrics.

The purpose of this component is to identify terminals whose behavior has changed significantly from their historical baseline.

---

## Recent Simulated Operational Stream

To demonstrate how the system behaves on evolving activity, Merchant Risk Sentinel includes a deterministic simulated recent transaction stream.

The stream covers:

August 15, 2026 to September 4, 2026

It contains:

41,610 simulated transactions

The stream is designed to demonstrate:

* Changing customer activity
* Changing terminal activity
* Increasing and decreasing risk
* Behavioral state transitions
* Chronological processing
* Recent operational investigation
* Network changes over time

The recent stream is simulated and does not represent real payment traffic.

---

## Continuous Simulated Ingestion

Merchant Risk Sentinel also supports continuously generated simulated transactions.

Each new transaction is processed through the same core risk pipeline:

New Simulated Transaction
↓
Feature Engineering
↓
Transaction ML Risk
↓
Customer Behavioral Risk
↓
Terminal Behavioral Risk
↓
Risk Aggregation
↓
Deterministic Policy
↓
Alert / Audit
↓
Network Update

This allows the system to demonstrate how risk changes as new activity arrives.

The implementation intentionally uses a lightweight in-process approach instead of introducing unnecessary infrastructure such as Kafka, Kubernetes, Redis, or Celery.

---

## Entity Network

The Entity Network provides relationship context between customers and payment terminals.

A simplified representation is:

Customer
├── Terminal
├── Terminal
└── Terminal

This allows an analyst to investigate whether risk is:

* Isolated to one customer
* Concentrated around a specific terminal
* Appearing across multiple connected entities

The network provides additional investigation context while the underlying transaction, customer, and terminal risk engines remain responsible for producing the risk signals.

---

## Chronological Replay

The Replay functionality provides a reproducible way to observe risk evolution.

The simulated recent stream can be replayed chronologically so that transactions are processed in their original order.

As the replay progresses:

* Historical features remain temporally valid
* Customer behavior changes
* Terminal behavior changes
* Risk states evolve
* Alerts are generated
* The network changes with incoming activity

This makes it possible to observe transitions such as:

NORMAL → RISK RISING → HIGH RISK → RECOVERY

rather than viewing risk as a static label.

---

## Dashboard

Merchant Risk Sentinel includes an investigation-oriented dashboard designed to provide both an operational overview and detailed investigation capabilities.

### Overview

Provides a high-level view of:

* Current risk
* Active alerts
* High-risk activity
* Behavioral movement
* Risk trends
* Recent high-risk transactions

### Alerts

Provides an operational record of risk events, including:

* Severity
* Transaction
* Customer
* Terminal
* Policy action
* Status
* Transaction time

### Transaction Explorer

Allows analysts to search and investigate transactions using transaction-level risk and contextual information.

### Transaction Detail

Provides detailed information about an individual transaction, including model risk, behavioral context, evidence, and policy outcome.

### Customers

Provides customer-level behavioral monitoring and historical activity.

### Customer Detail

Provides customer risk state, behavioral evidence, historical activity, transactions, and connected entities.

### Terminals

Provides terminal-level behavioral monitoring.

### Terminal Detail

Provides terminal risk state, behavioral evidence, timeline, transactions, and connected customers.

### Entity Network

Provides relationship-based investigation between customers and terminals.

### Live Network

Displays continuously arriving simulated activity and changing network relationships.

### Replay

Provides chronological replay of the simulated operational stream.

### System Health

Provides visibility into the operational state of the system.

---

## Example Investigation

One of the main demonstrations of the system is a situation where the transaction-level ML score alone would not necessarily trigger the highest response.

For example:

Transaction ML Risk
+
Customer Behavioral Risk
+
Terminal Behavioral Risk
↓
Unified Critical Risk
↓
Deterministic Policy
↓
Escalation
↓
AI Explanation

This demonstrates why the system is designed as a risk-intelligence platform rather than only a fraud classifier.

The transaction model evaluates the transaction itself.

The customer engine evaluates the customer's behavior.

The terminal engine evaluates the terminal's behavior.

The aggregation layer combines these independent signals.

The policy engine determines the allowed action.

The AI Risk Analyst then explains the evidence.

---

## Project Structure

The repository is organized around the different components of the risk pipeline.

Merchant_Risk_Sentinel/

src/mrs/
├── api/
├── analyst/
├── data/
├── features/
├── live/
├── models/
├── profiles/
├── risk/
└── ...

scripts/
tests/
docs/
frontend/
models/
external/
data/

Key areas include:

**src/mrs/** — Core application and risk-intelligence logic.

**src/mrs/api/** — Backend APIs and application routes.

**src/mrs/analyst/** — AI Risk Analyst, evidence construction, and structured schemas.

**src/mrs/data/** — Data ingestion and simulated stream handling.

**src/mrs/features/** — Temporal and behavioral feature engineering.

**src/mrs/live/** — Continuous simulated transaction ingestion.

**src/mrs/models/** — Model loading and inference.

**src/mrs/profiles/** — Customer and terminal profiles.

**src/mrs/risk/** — Risk scoring, aggregation, and policy logic.

**frontend/** — Dashboard application.

**scripts/** — Runnable data and operational scripts.

**tests/** — Automated test suite.

**docs/** — Architecture, evaluation, and development reports.

**models/** — Versioned model artifacts.

**external/fraud_detection_handbook/** — Isolated external simulator code.

**data/** — Local data files, which are not committed to the repository.

---

## Setup

Create the Python environment:

python3.12 -m venv .venv

Activate the environment:

source .venv/bin/activate

Install dependencies:

pip install -r requirements.txt

The repository also contains requirements.lock.txt for reproducible dependency versions.

---

## Running Tests

Run the backend test suite with:

.venv/bin/pytest -q

The final verified backend suite contains:

640 / 640 tests passing

The frontend TypeScript check and production build were also verified successfully.

The test suite covers areas including:

* Data assumptions
* Feature transformations
* Temporal leakage
* Behavioral risk logic
* Risk aggregation
* Policy decisions
* Model inference
* API behavior
* Continuous ingestion
* End-to-end backend behavior

---

## Data Provenance

The primary benchmark data originates from the Fraud Detection Handbook's simulated data project.

The external simulator code is kept isolated under:

external/fraud_detection_handbook/

This code is not directly imported by the core src/mrs application.

Data provenance and attribution information are maintained in the repository's relevant documentation and notice files.

---

## Engineering Principles

Merchant Risk Sentinel follows several core engineering principles.

### Temporal Correctness

Historical features must only use information available at scoring time.

### Explainability

Risk assessments must be supported by measurable evidence.

### Separation of Responsibilities

Transaction ML, customer behavior, terminal behavior, risk aggregation, policy, AI explanation, and presentation remain separate components.

### Deterministic Decisions

The AI layer does not control financial or risk decisions.

### Reproducibility

Important transformations, model versions, thresholds, and evaluation periods are traceable.

### Controlled Complexity

The system avoids unnecessary infrastructure and model complexity when a simpler solution is sufficient.

---

## Limitations

Merchant Risk Sentinel is a research and demonstration system.

The main limitations are:

* The primary benchmark dataset is simulated.
* The recent operational stream is simulated.
* Continuous ingestion generates simulated transactions rather than processing real payments.
* The system is not connected to real Razorpay payment infrastructure.
* Behavioral risk indicates abnormal behavior and is not proof of fraud.
* The AI Risk Analyst provides explanation and advisory recommendations rather than autonomous decisions.
* Entity relationships provide investigation context and do not independently establish fraud.

These limitations are intentional and should be considered when interpreting the results.

---

## Demo Flow

The recommended demonstration follows an investigation workflow:

Overview
↓
Customer / Transaction Investigation
↓
Terminal Investigation
↓
Alerts + AI Risk Analyst
↓
Entity Network
↓
Live Network
↓
Chronological Replay
↓
System Health

The purpose of the demo is to show how multiple independent risk signals come together into one explainable operational decision.

---

## Why Merchant Risk Sentinel?

Traditional fraud detection often focuses on one question:

"Is this transaction fraudulent?"

Merchant Risk Sentinel looks at a broader question:

"What is changing, where is the risk coming from, how are the entities connected, and what evidence supports the decision?"

The system combines:

Transaction ML
+
Customer Behavior
+
Terminal Behavior
+
Temporal Context
+
Entity Relationships
↓
Unified Risk
↓
Deterministic Policy
↓
Evidence
↓
AI Explanation

This is the central idea behind Merchant Risk Sentinel.

It is not simply an XGBoost fraud classifier.

It is an AI-assisted risk manager designed to help identify evolving risk, connect the relevant evidence, explain what changed, and provide a controlled next step.

---

## Final Note

Merchant Risk Sentinel was built for the **Track 2 — AI Risk Manager** challenge with a focus on:

* Measurable risk detection
* Customer and terminal behavioral intelligence
* Temporal correctness
* Explainable risk assessment
* Evidence-grounded AI assistance
* Deterministic policy
* Operational investigation
* Reproducible analysis

The system is designed so that the risk engine computes the evidence and decision, while the AI layer helps the analyst understand and act on that information.

**Detect the change. Connect the signals. Explain the risk.**
