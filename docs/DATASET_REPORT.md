# Dataset Report — Phase 2 (Data Understanding)

Per Dev Plan §27/§43 (Phase 2) and CLAUDE.md §13. All figures below are reproducible by
running `.venv/bin/python scripts/04_dataset_report.py` against the Phase 1 processed
layer (`data/processed/transactions/*.parquet`), and are also pinned as regression
assertions in `tests/test_splits.py`.

## 1. Overall summary

| Metric | Value |
|---|---|
| Transactions | 1,754,155 |
| Date range | 2018-04-01 00:00:31 – 2018-09-30 23:59:57 |
| Unique customers (active) | 4,990 |
| Unique terminals (active) | 10,000 |
| Fraud count | 14,681 |
| Fraud rate | 0.8369% |

These reconcile with the Dev Plan §2 headline figures and with the raw/processed layer
statistics already measured in Phase 1 (`docs/PHASE1_REPORT.md`) — this section is a
confirmation, not a new finding.

## 2. Monthly distribution

| Month | Transactions | Frauds | Fraud rate |
|---|---:|---:|---:|
| 2018-04 | 288,062 | 1,702 | 0.5908% |
| 2018-05 | 297,115 | 2,639 | 0.8882% |
| 2018-06 | 287,618 | 2,504 | 0.8706% |
| 2018-07 | 296,928 | 2,620 | 0.8824% |
| 2018-08 | 296,559 | 2,669 | 0.9000% |
| 2018-09 | 287,873 | 2,547 | 0.8848% |

Transaction volume is stable month to month (~288k–297k). Fraud rate is **not** stable
across the whole range: April sits well below the other five months (0.59% vs.
0.87–0.90%). This is investigated in §3.

## 3. Finding: a ramp-up transient in the first ~2–3 weeks

Daily fraud count for the first 15 days of the dataset:

| Date | Transactions | Frauds | Fraud rate |
|---|---:|---:|---:|
| 2018-04-01 | 9,488 | 3 | 0.0316% |
| 2018-04-02 | 9,583 | 13 | 0.1357% |
| 2018-04-03 | 9,747 | 15 | 0.1539% |
| 2018-04-04 | 9,530 | 18 | 0.1889% |
| 2018-04-05 | 9,651 | 22 | 0.2280% |
| 2018-04-06 | 9,539 | 27 | 0.2830% |
| 2018-04-07 | 9,438 | 39 | 0.4132% |
| 2018-04-10 | 9,672 | 58 | 0.5997% |
| 2018-04-15 | 9,478 | 52 | 0.5486% |

Fraud count rises roughly monotonically from 3 on day one to the ~40–90/day steady state
seen for the rest of the dataset (daily fraud count over the full 183 days: min 3, median
84, max 110 — see script output). Transaction *volume* has no such ramp — daily
transaction count is flat (mean 9,586, std 97) from day one. The explanation is
structural, not a data quality problem: scenario-2/3 compromise episodes are triggered
with random start dates spread across the whole 183-day window (§5), so on day one almost
none have started yet, and the number of *concurrently active* compromises only reaches a
steady state after enough of them have had a chance to begin. This is consistent with the
episode-length evidence in §5 (median terminal episode ≈2 days, max 27 days) — it takes
several weeks for the population of overlapping compromise windows to saturate.

**Why this matters for the split (§7):** the transient is confined to April, and April is
placed entirely inside `train`. It affects the *training* distribution (a naive baseline
built from April alone would understate fraud rate) but not validation or test, both of
which sit inside the stable regime. This is flagged as a Phase 3 consideration — any
customer/terminal historical baseline computed from early-April history should be treated
as having a shorter, less representative lookback window, i.e. a cold-start case, not
silently trusted (Dev Plan §33.7).

## 4. Scenario distribution

Overall `TX_FRAUD_SCENARIO` counts:

| Scenario | Count | Meaning |
|---|---:|---|
| 0 | 1,739,474 | genuine |
| 1 | 973 | high-value fraud (`TX_AMOUNT > 220`) |
| 2 | 9,077 | compromised terminal |
| 3 | 4,631 | compromised customer |

By month (fraud rows only):

| Month | Scenario 1 | Scenario 2 | Scenario 3 |
|---|---:|---:|---:|
| 2018-04 | 168 | 931 | 603 |
| 2018-05 | 141 | 1,634 | 864 |
| 2018-06 | 180 | 1,596 | 728 |
| 2018-07 | 171 | 1,588 | 861 |
| 2018-08 | 169 | 1,692 | 808 |
| 2018-09 | 144 | 1,636 | 767 |

Scenario 1 counts are roughly flat across months (~140–180/month) — consistent with it
being a stateless per-transaction rule with no ramp-up. Scenarios 2 and 3 show the same
April dip as the aggregate fraud rate (§3), for the same reason.

**Carried forward from Phase 1, independently reconfirmed here:** every scenario-1-labeled
row has `TX_AMOUNT > 220` (minimum observed: 220.02), and every genuine (`TX_FRAUD=0`) row
has `TX_AMOUNT ≤ 219.98` — the ceiling is exactly the injected rule's threshold, not a
coincidence of the simulator's amount distribution. Also reconfirmed: 2,285 transactions
have `TX_AMOUNT > 220` and `TX_FRAUD=1` but are labeled scenario 2 or 3, not scenario 1 —
direct evidence of the sequential-overwrite behavior (a transaction matching the
scenario-1 rule gets its scenario label overwritten if it also falls inside an active
scenario-2/3 compromise window). Any scenario-specific evaluation in later phases must
treat `TX_FRAUD_SCENARIO` as "last rule applied", not as a clean partition of fraud rows.

## 5. Compromised entities and episode structure

| Metric | Value |
|---|---|
| Terminals ever labeled scenario 2 | 357 of 10,000 |
| Customers ever labeled scenario 3 | 487 of 4,990 active |

Episode length is measured as the length (in days) of each contiguous run of
fraud-labeled days for a given entity — a proxy for how long a compromise stays *visible*
in the data, not the simulator's true internal compromise-window length (which can run
longer than what's observed if, on some day inside the true window, none of that
terminal's/customer's transactions happened to be sampled as fraudulent).

| | Terminal (scenario 2) | Customer (scenario 3) |
|---|---:|---:|
| Episodes | 2,353 | 1,621 |
| Median length | 2 days | 1 day |
| Mean length | 2.45 days | 1.95 days |
| Max length | 27 days | 14 days |

A single manually-inspected example (terminal 293) showed a 27-day contiguous block
(2018-05-12 to 2018-06-07) followed by four shorter, separate episodes clustered in
September — i.e. the same terminal was compromised more than once, at different times.
This directly supports the Dev Plan §12/§18 concept-drift requirement: an entity's risk
state must be able to rise, recover, and rise again, not be a permanent label.

## 6. Temporal patterns (hour-of-day, day-of-week)

Transaction volume follows a diurnal curve — low overnight (≈15k/hour at 00:00–01:00),
rising to a midday peak (≈126k–129k/hour, 10:00–13:00), and back down by night — while
**fraud rate is close to flat across hours** (0.77%–0.98%, no hour standing out as
systematically riskier once volume is accounted for). Day-of-week shows the same pattern:
transaction volume and fraud rate are both roughly flat across weekdays (0.79%–0.87%),
with Sunday (dow=6) slightly higher in volume (simulator's weekend effect, per the
Handbook) but not markedly different in fraud rate. **Conclusion:** hour/day-of-week are
useful for describing normal *volume* seasonality (relevant to Phase 3's velocity
features), but neither hour nor weekday functions as a strong standalone fraud signal in
this simulated dataset — consistent with the fraud scenarios being amount- and
entity-compromise-driven rather than time-of-day-driven.

## 7. Chronological train/validation/test split — decision

**Decision: adopt the Dev Plan §6 initial proposal unchanged.**

| Split | Range | Transactions | Frauds | Fraud rate | Scenario 1/2/3 |
|---|---|---:|---:|---:|---|
| train | 2018-04-01 – 2018-07-31 | 1,169,723 | 9,465 | 0.8092% | 660 / 5,749 / 3,056 |
| validation | 2018-08-01 – 2018-08-31 | 296,559 | 2,669 | 0.9000% | 169 / 1,692 / 808 |
| test | 2018-09-01 – 2018-09-30 | 287,873 | 2,547 | 0.8848% | 144 / 1,636 / 767 |

**Rationale, per Dev Plan §6's instruction to inspect the actual distribution before
finalizing:**

1. **Not materially uneven.** The only distributional anomaly found (§3) is April's
   ramp-up, and April falls entirely inside `train`. Validation (0.90%) and test (0.88%)
   both sit inside the post-ramp-up stable band (0.87%–0.90%, matching June/July), so the
   proposal does not put an unrepresentative period into evaluation.
2. **Strict chronological order, no overlap.** `train` ends 2018-07-31, `validation` runs
   2018-08-01 to 2018-08-31, `test` runs 2018-09-01 to 2018-09-30 — enforced
   programmatically (`mrs.data.splits.validate_split_boundaries`, run at import time) and
   covers the full verified Phase 1 date range with no gap.
3. **Sufficient minority-class volume per split for scenario-specific evaluation**
   (Dev Plan §11): every split has hundreds to low-thousands of examples of each of the
   three fraud scenarios — enough to compute per-scenario precision/recall/detection-delay
   without relying on single-digit sample counts.
4. **Test set integrity.** No model or threshold decision has been made yet; the test
   range is recorded here only as a boundary definition, not touched for any tuning.

No adjustment to the Dev Plan's initial boundaries was necessary. Implementation:
`src/mrs/data/splits.py` (`SPLIT_BOUNDARIES`, `assign_split()`); regression-pinned by
`tests/test_splits.py` against the exact row/fraud counts in the table above.

## 8. What Phase 3 must account for

- April's first ~2–3 weeks have a materially lower fraud rate and fewer active
  compromise episodes than the rest of the dataset — treat early-April historical
  baselines as a cold-start case, not a representative lookback window.
- `TX_FRAUD_SCENARIO` is overwritten sequentially (1→2→3); do not use it as a mutually
  exclusive partition when building or evaluating scenario-specific features.
- Entities can be compromised more than once, at different times, separated by normal
  periods — behavioral baselines must support rise/recover/rise-again, not a one-shot
  flag.
- Hour-of-day and day-of-week carry real *volume* seasonality (useful for velocity
  features) but weak standalone fraud signal on their own.

## 9. Files produced

- `src/mrs/data/dataset_stats.py` — reusable statistics functions (no I/O).
- `src/mrs/data/splits.py` — frozen split boundaries, `assign_split()`, leakage-guarding
  validation.
- `scripts/04_dataset_report.py` — prints every figure in this report from the processed
  dataset.
- `tests/test_dataset_stats.py` — 11 unit tests against synthetic frames (no dataset
  required).
- `tests/test_splits.py` — 10 tests: boundary validation, leakage-guard checks, and (data-
  marked) regression pins against the real processed dataset's measured split counts.

## 10. Tests

```
.venv/bin/pytest -q
54 passed in 2.78s
```

(33 from Phase 1, unchanged and still passing — no regression — plus 21 new Phase 2
tests.)

## 11. Limitations

- Compromise "episode length" (§5) is a measured proxy from labeled-day contiguity, not
  the simulator's ground-truth internal window parameter — stated explicitly rather than
  assumed exact (Dev Plan §33.11).
- This report characterizes the dataset as a whole; it does not yet build or evaluate any
  feature or model. That is Phase 3.
- No new raw or processed data was generated. All numbers derive from the existing Phase
  1 processed layer.

## Blockers

None.

## Recommended next phase

Phase 3 — Feature Engineering: chronological historical features (transaction-,
customer-, and terminal-level), leakage tests, feature persistence, and `FEATURE_SPEC.md`,
built against the `train`/`validation`/`test` boundaries decided in §7.
