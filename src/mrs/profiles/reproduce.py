"""Reproduce customer/terminal profile tables using the published simulator parameters.

Parameters below are copied from the notebook's own invocation
(``generate_dataset(n_customers=5000, n_terminals=10000, nb_days=183,
start_date="2018-04-01", r=5)``) — the exact call that produced the published dataset.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from external.fraud_detection_handbook.simulator_profiles import (
    generate_customer_profiles_table,
    generate_terminal_profiles_table,
    get_list_terminals_within_radius,
)

N_CUSTOMERS = 5000
N_TERMINALS = 10000
RADIUS = 5
CUSTOMER_RANDOM_STATE = 0
TERMINAL_RANDOM_STATE = 1


def reproduce_profiles() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Regenerate the customer and terminal profile tables.

    Returns (customer_profiles, terminal_profiles). customer_profiles carries
    available_terminals (list[int]) and nb_terminals (int), matching the notebook's
    generate_dataset().
    """
    customer_profiles = generate_customer_profiles_table(
        N_CUSTOMERS, random_state=CUSTOMER_RANDOM_STATE
    )
    terminal_profiles = generate_terminal_profiles_table(
        N_TERMINALS, random_state=TERMINAL_RANDOM_STATE
    )

    x_y_terminals = terminal_profiles[["x_terminal_id", "y_terminal_id"]].values.astype(float)
    customer_profiles["available_terminals"] = customer_profiles.apply(
        lambda row: get_list_terminals_within_radius(row, x_y_terminals=x_y_terminals, r=RADIUS),
        axis=1,
    )
    customer_profiles["nb_terminals"] = customer_profiles["available_terminals"].apply(len)

    return customer_profiles, terminal_profiles
