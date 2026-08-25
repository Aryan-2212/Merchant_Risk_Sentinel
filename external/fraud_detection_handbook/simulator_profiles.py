"""Verbatim port of the Handbook's profile-generation functions.

Source: Fraud-Detection-Handbook/fraud-detection-handbook,
Chapter_3_GettingStarted/SimulatedDataset.ipynb. See NOTICE.md for license and scope.

Reproduced exactly as published, including using the legacy global
``numpy.random`` state (``np.random.seed`` / ``np.random.uniform``) rather than a
``Generator`` instance — this must match bit-for-bit to reproduce the published
dataset, and NumPy guarantees stream compatibility for this legacy API across versions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def generate_customer_profiles_table(n_customers: int, random_state: int = 0) -> pd.DataFrame:
    np.random.seed(random_state)

    customer_id_properties = []

    for customer_id in range(n_customers):
        x_customer_id = np.random.uniform(0, 100)
        y_customer_id = np.random.uniform(0, 100)

        mean_amount = np.random.uniform(5, 100)  # Arbitrary (but sensible) value
        std_amount = mean_amount / 2  # Arbitrary (but sensible) value

        mean_nb_tx_per_day = np.random.uniform(0, 4)  # Arbitrary (but sensible) value

        customer_id_properties.append(
            [
                customer_id,
                x_customer_id,
                y_customer_id,
                mean_amount,
                std_amount,
                mean_nb_tx_per_day,
            ]
        )

    customer_profiles_table = pd.DataFrame(
        customer_id_properties,
        columns=[
            "CUSTOMER_ID",
            "x_customer_id",
            "y_customer_id",
            "mean_amount",
            "std_amount",
            "mean_nb_tx_per_day",
        ],
    )

    return customer_profiles_table


def generate_terminal_profiles_table(n_terminals: int, random_state: int = 0) -> pd.DataFrame:
    np.random.seed(random_state)

    terminal_id_properties = []

    for terminal_id in range(n_terminals):
        x_terminal_id = np.random.uniform(0, 100)
        y_terminal_id = np.random.uniform(0, 100)

        terminal_id_properties.append([terminal_id, x_terminal_id, y_terminal_id])

    terminal_profiles_table = pd.DataFrame(
        terminal_id_properties, columns=["TERMINAL_ID", "x_terminal_id", "y_terminal_id"]
    )

    return terminal_profiles_table


def get_list_terminals_within_radius(customer_profile, x_y_terminals, r) -> list[int]:
    x_y_customer = customer_profile[["x_customer_id", "y_customer_id"]].values.astype(float)

    squared_diff_x_y = np.square(x_y_customer - x_y_terminals)

    dist_x_y = np.sqrt(np.sum(squared_diff_x_y, axis=1))

    available_terminals = list(np.where(dist_x_y < r)[0])

    return available_terminals
