"""Simulated Recent Operational Stream (Dev Plan addendum, approved out-of-band).

A deterministic, 21-day, ~38k-transaction demo dataset layered on TOP of -- never
replacing -- the frozen Fraud Detection Handbook benchmark (mrs.config.EXPECTED_*,
mrs.data.splits). It exists to demonstrate how customer/terminal behavioral risk
evolves over a recent operating period, something the frozen 2018 benchmark cannot show
live because its evaluation is already fixed. This is simulated operational/demo data,
generated entirely by this module; it is NOT real Razorpay production traffic and must
never be presented as such.

Design, matched to the rest of the pipeline so the recent stream needs no bespoke
downstream code:

- Same raw/processed schema as the benchmark (mrs.data.schema.RAW_COLUMNS,
  PROCESSED_DTYPES) -- validated with the SAME validator (validate_processed_frame),
  not a new one.
- Reuses real, existing CUSTOMER_ID/TERMINAL_ID values (a sampled subset of the
  5,000/10,000 already in data/reference/*.parquet and already in the customers/
  terminals DB tables) -- no new entities are invented, so the recent stream can be
  persisted without touching those reference tables again.
- TX_FRAUD/TX_FRAUD_SCENARIO are simulation annotations only, exactly like the
  benchmark's own labels: never fed to the model (mrs.models.dataset.get_feature_matrix
  restricts to the registry's feature columns regardless of what this module
  generates), and used by the terminal behavioral engine only as strictly-prior
  aggregates -- the same leakage-safe construction the benchmark already relies on
  (mrs.features._temporal), not a special case.
- TRANSACTION_ID starts at mrs.config.RECENT_STREAM_TX_ID_OFFSET (2,000,000,000),
  far above the benchmark's max (1,754,154), so the two id spaces can never collide.
- split="recent" (mrs.config.RECENT_STREAM_SPLIT_LABEL) is attached via
  mrs.features.build.build_feature_frame's split_override parameter, never via
  mrs.data.splits.assign_split -- the frozen train/validation/test boundaries never see
  a 2026 timestamp.

Behavioral narrative (Dev Plan "NORMAL -> RISK_RISING -> HIGH_RISK -> RECOVERY"):
a deterministically-sampled subset of the chosen customers/terminals is put into an
"elevated" cohort with a randomized (seeded) start day in week 2, a first phase whose
signal strength lands in the RISK_RISING band, then a second phase whose signal
strength lands in the HIGH_RISK band; a further subset of that cohort recovers
(reverts to baseline) in the final days, while the rest stay elevated through day 21 --
so the demo can show both a full recovery arc and a persisting episode, never
permanently labeling every elevated entity as malicious.

Customer signal: transaction amount inflated relative to the customer's OWN emerging
baseline (drives mrs.behavioral.customer's customer_amount_zscore). Terminal signal:
injected fraud-labeled transactions raising the terminal's OWN recent-vs-historical
fraud rate (drives mrs.behavioral.terminal's terminal_fraud_rate_deviation) -- the same
two signals the frozen behavioral engines already consume, unmodified.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from mrs import config
from mrs.data.schema import RAW_COLUMNS, normalize_dtypes, validate_processed_frame

#: Cohort mix for the sampled customers/terminals (Dev Plan narrative: most entities stay
#: normal; a minority rises and either recovers or stays elevated through day 21).
_COHORT_LABELS = ("normal", "elevated_recover", "elevated_sustain")
_COHORT_PROBS = (0.80, 0.12, 0.08)

#: Elevated window shape, in day numbers (1-indexed, 21-day stream).
_START_DAY_RANGE = (8, 10)  # inclusive -- week 2
_RISING_PHASE_LEN = 3  # days spent in the RISK_RISING-strength phase before HIGH_RISK
_RECOVERY_DAY_RANGE = (17, 19)  # inclusive -- when "elevated_recover" entities revert

#: Customer amount-inflation multipliers (customer_hist_amount_std ~= 0.5 *
#: customer_hist_amount_mean for this dataset -- verified against
#: data/reference/customer_profiles.parquet -- so a multiplier m gives roughly
#: z ~= 2*(m-1); see module-level comment in generate_recent_stream).
_CUSTOMER_RISING_MULTIPLIER = (2.0, 2.6)  # z ~= 2.0-3.2 (RISING_THRESHOLD=2.0)
_CUSTOMER_HIGH_MULTIPLIER = (3.2, 4.2)  # z ~= 4.4-6.4 (HIGH_RISK_THRESHOLD=4.0)

#: Terminal attack-injection intensity per active day (count of injected transactions,
#: fraction of them flagged fraud). terminal_hist_fraud_rate is ~0 pre-activation (no
#: background fraud noise is injected into organic transactions), so recent_rate alone
#: drives the deviation past RISING_THRESHOLD=0.05 / HIGH_RISK_THRESHOLD=0.15.
_TERMINAL_RISING_ATTACK = (10, 14)
_TERMINAL_RISING_FRAUD_PROB = 0.5
_TERMINAL_HIGH_ATTACK = (18, 26)
_TERMINAL_HIGH_FRAUD_PROB = 0.9
#: Scenario code stamped on injected-fraud rows -- mirrors the Handbook's own
#: compromised-terminal scenario code (Dev Plan Sec 2); a simulation annotation only,
#: never fed to any model.
_INJECTED_FRAUD_SCENARIO = 2


def _elevated_phase(day: int, start_day: int, recovery_day: int | None) -> str:
    """'normal' | 'rising' | 'high' for one entity on one day number (1-indexed)."""
    if day < start_day:
        return "normal"
    if recovery_day is not None and day >= recovery_day:
        return "normal"
    if day < start_day + _RISING_PHASE_LEN:
        return "rising"
    return "high"


def _assign_cohort(rng: np.random.Generator, ids: np.ndarray) -> pd.DataFrame:
    labels = rng.choice(_COHORT_LABELS, size=len(ids), p=_COHORT_PROBS)
    start_days = rng.integers(_START_DAY_RANGE[0], _START_DAY_RANGE[1] + 1, size=len(ids))
    recovery_days = rng.integers(_RECOVERY_DAY_RANGE[0], _RECOVERY_DAY_RANGE[1] + 1, size=len(ids))
    return pd.DataFrame(
        {
            "id": ids,
            "cohort": labels,
            "start_day": start_days,
            # Only meaningful for cohort == "elevated_recover"; ignored otherwise.
            "recovery_day": recovery_days,
        }
    ).set_index("id")


def generate_recent_stream(seed: int = config.RECENT_STREAM_SEED) -> pd.DataFrame:
    """Deterministically generate the Simulated Recent Operational Stream.

    Same seed -> byte-identical output (every random draw goes through the one
    `np.random.default_rng(seed)` created here). Returns a frame with exactly
    mrs.data.schema.RAW_COLUMNS, in mrs.data.schema.PROCESSED_DTYPES, chronologically
    ordered, already passed through validate_processed_frame -- ready to hand to
    mrs.features.build.build_feature_frame(..., split_override="recent") exactly like
    any other processed-layer frame.
    """
    rng = np.random.default_rng(seed)

    customer_profiles = pd.read_parquet(config.REFERENCE_DIR / "customer_profiles.parquet")
    terminal_profiles = pd.read_parquet(config.REFERENCE_DIR / "terminal_profiles.parquet")

    customer_ids = rng.choice(
        customer_profiles["CUSTOMER_ID"].to_numpy(), size=config.RECENT_STREAM_N_CUSTOMERS, replace=False
    )
    terminal_ids = rng.choice(
        terminal_profiles["TERMINAL_ID"].to_numpy(), size=config.RECENT_STREAM_N_TERMINALS, replace=False
    )

    cust_profile = customer_profiles.set_index("CUSTOMER_ID").loc[customer_ids]
    cust_cohort = _assign_cohort(rng, customer_ids)
    term_cohort = _assign_cohort(rng, terminal_ids)

    start_date = pd.Timestamp(config.RECENT_STREAM_START_DATE)
    weights = cust_profile["mean_nb_tx_per_day"].to_numpy().clip(min=0.05)
    base_p = weights / weights.sum()

    day_frames: list[pd.DataFrame] = []
    for day in range(1, config.RECENT_STREAM_DAYS + 1):
        day_index0 = day - 1
        day_date = start_date + pd.Timedelta(days=int(day_index0))

        # ---- organic transactions (every customer, weighted by their own real velocity)
        phases = np.array(
            [
                _elevated_phase(
                    day,
                    int(cust_cohort.loc[cid, "start_day"]),
                    int(cust_cohort.loc[cid, "recovery_day"]) if cust_cohort.loc[cid, "cohort"] == "elevated_recover" else None,
                )
                if cust_cohort.loc[cid, "cohort"] != "normal"
                else "normal"
                for cid in customer_ids
            ]
        )
        multiplier = np.ones(len(customer_ids))
        rising_mask = phases == "rising"
        high_mask = phases == "high"
        multiplier[rising_mask] = rng.uniform(*_CUSTOMER_RISING_MULTIPLIER, size=rising_mask.sum())
        multiplier[high_mask] = rng.uniform(*_CUSTOMER_HIGH_MULTIPLIER, size=high_mask.sum())
        # Elevated customers also transact more often that day (frequency deviation).
        day_p = base_p * np.where(phases == "normal", 1.0, 2.5)
        day_p = day_p / day_p.sum()

        n_organic = config.RECENT_STREAM_TX_PER_DAY
        drawn_idx = rng.choice(len(customer_ids), size=n_organic, p=day_p)
        drawn_customer_ids = customer_ids[drawn_idx]
        mean_amounts = cust_profile["mean_amount"].to_numpy()[drawn_idx]
        std_amounts = cust_profile["std_amount"].to_numpy()[drawn_idx]
        drawn_multiplier = multiplier[drawn_idx]

        raw_amount = rng.normal(loc=mean_amounts, scale=std_amounts)
        amounts = np.clip(raw_amount, 0.5, None) * drawn_multiplier

        # Terminal choice is drawn from the same fixed, sampled terminal pool used for
        # the compromised-terminal cohort (not each customer's own full
        # available_terminals list, which spans thousands of the 10,000 real
        # terminals) -- concentrating the ~38k-transaction demo volume onto ~100
        # terminals so each one plausibly reaches MIN_TERMINAL_HISTORY within the
        # 21-day window and can actually show a behavioral-state transition, rather
        # than the volume being diluted across too many terminals to assess any one
        # of them.
        terminals_for_draw = rng.choice(terminal_ids, size=n_organic)

        organic = pd.DataFrame(
            {
                "CUSTOMER_ID": drawn_customer_ids,
                "TERMINAL_ID": terminals_for_draw,
                "TX_AMOUNT": amounts,
                "TX_FRAUD": 0,
                "TX_FRAUD_SCENARIO": 0,
            }
        )

        # ---- injected attack transactions (active compromised terminals only)
        attack_frames = []
        for tid in terminal_ids:
            cohort = term_cohort.loc[tid, "cohort"]
            if cohort == "normal":
                continue
            phase = _elevated_phase(
                day,
                int(term_cohort.loc[tid, "start_day"]),
                int(term_cohort.loc[tid, "recovery_day"]) if cohort == "elevated_recover" else None,
            )
            if phase == "normal":
                continue
            attack_lo, attack_hi = _TERMINAL_RISING_ATTACK if phase == "rising" else _TERMINAL_HIGH_ATTACK
            fraud_prob = _TERMINAL_RISING_FRAUD_PROB if phase == "rising" else _TERMINAL_HIGH_FRAUD_PROB
            n_attack = int(rng.integers(attack_lo, attack_hi + 1))
            attack_customers = rng.choice(customer_ids, size=n_attack, replace=True)
            is_fraud = rng.random(n_attack) < fraud_prob
            attack_frames.append(
                pd.DataFrame(
                    {
                        "CUSTOMER_ID": attack_customers,
                        "TERMINAL_ID": tid,
                        "TX_AMOUNT": rng.uniform(220.0, 320.0, size=n_attack),
                        "TX_FRAUD": is_fraud.astype(int),
                        "TX_FRAUD_SCENARIO": np.where(is_fraud, _INJECTED_FRAUD_SCENARIO, 0),
                    }
                )
            )

        day_df = pd.concat([organic, *attack_frames], ignore_index=True) if attack_frames else organic
        n_today = len(day_df)

        seconds_in_day = rng.integers(0, 86_400, size=n_today)
        order = np.argsort(seconds_in_day, kind="stable")
        day_df = day_df.iloc[order].reset_index(drop=True)
        seconds_in_day = seconds_in_day[order]

        day_df["TX_TIME_SECONDS"] = day_index0 * 86_400 + seconds_in_day
        day_df["TX_TIME_DAYS"] = day_index0
        day_df["TX_DATETIME"] = day_date + pd.to_timedelta(seconds_in_day, unit="s")

        day_frames.append(day_df)

    full = pd.concat(day_frames, ignore_index=True)
    full["TRANSACTION_ID"] = config.RECENT_STREAM_TX_ID_OFFSET + np.arange(len(full))
    full = full[list(RAW_COLUMNS)]
    full = normalize_dtypes(full)

    validate_processed_frame(full, source="recent_stream")
    return full
