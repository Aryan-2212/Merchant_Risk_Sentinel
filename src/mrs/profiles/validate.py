"""Validate reproduced profiles against the published transaction data.

The reproduction is only trustworthy if it is checked against ground truth we did not
generate. Every (CUSTOMER_ID, TERMINAL_ID) pair observed in the transactions must be
reachable from the reproduced customer_profiles.available_terminals, because the
simulator only ever picks terminals via random.choice(available_terminals) and no fraud
scenario rewrites TERMINAL_ID (verified by reading add_frauds() in the source notebook).
If this check fails, the reproduction must not be trusted or persisted.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class ProfileValidationResult:
    passed: bool
    problems: list[str] = field(default_factory=list)

    def raise_if_failed(self) -> None:
        if not self.passed:
            raise ValueError(
                "Profile reproduction failed validation against transaction data:\n  "
                + "\n  ".join(self.problems)
            )


def validate_profiles_against_transactions(
    customer_profiles: pd.DataFrame,
    transactions: pd.DataFrame,
) -> ProfileValidationResult:
    """Check the reproduced profiles are consistent with the observed transactions.

    ``transactions`` must contain at least CUSTOMER_ID and TERMINAL_ID.
    """
    problems: list[str] = []

    available_by_customer = customer_profiles.set_index("CUSTOMER_ID")["available_terminals"]

    observed_pairs = transactions[["CUSTOMER_ID", "TERMINAL_ID"]].drop_duplicates()
    observed_by_customer = observed_pairs.groupby("CUSTOMER_ID")["TERMINAL_ID"].apply(set)

    containment_failures = 0
    example_failures: list[str] = []
    for customer_id, observed_terminals in observed_by_customer.items():
        if customer_id not in available_by_customer.index:
            containment_failures += 1
            if len(example_failures) < 5:
                example_failures.append(f"CUSTOMER_ID {customer_id}: not in reproduced profiles")
            continue
        allowed = set(available_by_customer.loc[customer_id])
        missing = observed_terminals - allowed
        if missing:
            containment_failures += 1
            if len(example_failures) < 5:
                example_failures.append(
                    f"CUSTOMER_ID {customer_id}: used terminals {sorted(missing)[:5]} "
                    f"not in reproduced available_terminals ({len(allowed)} available)"
                )

    if containment_failures:
        problems.append(
            f"{containment_failures} of {len(observed_by_customer)} customers used a "
            f"terminal outside their reproduced available_terminals. Examples:\n    "
            + "\n    ".join(example_failures)
        )

    transacting_customers = set(observed_by_customer.index)
    zero_terminal_customers = set(
        customer_profiles.loc[customer_profiles["nb_terminals"] == 0, "CUSTOMER_ID"]
    )
    violating = zero_terminal_customers & transacting_customers
    if violating:
        problems.append(
            f"{len(violating)} customers with nb_terminals=0 nonetheless transacted: "
            f"{sorted(violating)[:5]}"
        )

    return ProfileValidationResult(passed=not problems, problems=problems)
