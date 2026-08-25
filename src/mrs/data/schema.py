"""Schema definition and validation for Handbook transaction data.

The raw files carry four columns as ``object`` dtype holding plain Python ints
(``CUSTOMER_ID``, ``TERMINAL_ID``, ``TX_TIME_SECONDS``, ``TX_TIME_DAYS``) — an artefact
of how the simulator assembled them via ``groupby.apply``. The raw layer keeps them
exactly as published; normalisation to real integer dtypes happens on the way into the
processed layer.
"""

from __future__ import annotations

import pandas as pd

#: Columns of the raw transaction table, in published order (Dev Plan §4.3).
RAW_COLUMNS: tuple[str, ...] = (
    "TRANSACTION_ID",
    "TX_DATETIME",
    "CUSTOMER_ID",
    "TERMINAL_ID",
    "TX_AMOUNT",
    "TX_TIME_SECONDS",
    "TX_TIME_DAYS",
    "TX_FRAUD",
    "TX_FRAUD_SCENARIO",
)

#: Ground-truth labels. NEVER permitted as model inputs (Dev Plan §33.6, §34.1).
#: Downstream feature code should import this and subtract it, so the rule is enforced
#: by construction rather than by remembering it.
LABEL_COLUMNS: frozenset[str] = frozenset({"TX_FRAUD", "TX_FRAUD_SCENARIO"})

#: Identifier columns. Not continuous numerics — used for entity lookup and historical
#: aggregation only (Dev Plan §34.4).
ID_COLUMNS: frozenset[str] = frozenset({"TRANSACTION_ID", "CUSTOMER_ID", "TERMINAL_ID"})

#: Target dtypes for the processed layer. Widths chosen against observed value ranges:
#: 5,000 customers and 10,000 terminals fit int32; 183 days fits int16; labels are
#: small non-negative integers.
PROCESSED_DTYPES: dict[str, str] = {
    "TRANSACTION_ID": "int64",
    "TX_DATETIME": "datetime64[ns]",
    "CUSTOMER_ID": "int32",
    "TERMINAL_ID": "int32",
    "TX_AMOUNT": "float64",
    "TX_TIME_SECONDS": "int64",
    "TX_TIME_DAYS": "int16",
    "TX_FRAUD": "int8",
    "TX_FRAUD_SCENARIO": "int8",
}

#: Values the simulator can emit. Scenario 0 means genuine.
VALID_FRAUD_VALUES: frozenset[int] = frozenset({0, 1})
VALID_SCENARIO_VALUES: frozenset[int] = frozenset({0, 1, 2, 3})


class SchemaError(ValueError):
    """Raised when data does not match the documented Handbook schema."""


def feature_candidate_columns(columns: list[str]) -> list[str]:
    """Return ``columns`` with label columns removed.

    Provided so later phases have one obvious way to exclude labels rather than each
    module re-deriving the exclusion list.
    """
    return [c for c in columns if c not in LABEL_COLUMNS]


def validate_raw_frame(df: pd.DataFrame, source: str) -> None:
    """Validate a single raw daily file. Raises :class:`SchemaError` on any violation.

    ``source`` is used only to make failures identifiable.
    """
    actual = tuple(df.columns)
    if actual != RAW_COLUMNS:
        raise SchemaError(
            f"{source}: column mismatch.\n  expected {RAW_COLUMNS}\n  actual   {actual}"
        )

    if df.empty:
        raise SchemaError(f"{source}: file contains zero transactions")

    null_counts = df.isna().sum()
    if null_counts.any():
        offenders = null_counts[null_counts > 0].to_dict()
        raise SchemaError(f"{source}: unexpected nulls {offenders}")

    if not pd.api.types.is_datetime64_any_dtype(df["TX_DATETIME"]):
        raise SchemaError(
            f"{source}: TX_DATETIME must be datetime, got {df['TX_DATETIME'].dtype}"
        )

    if not pd.api.types.is_numeric_dtype(df["TX_AMOUNT"]):
        raise SchemaError(f"{source}: TX_AMOUNT must be numeric, got {df['TX_AMOUNT'].dtype}")

    # The four object-dtype columns must hold values losslessly castable to int.
    for column in ("CUSTOMER_ID", "TERMINAL_ID", "TX_TIME_SECONDS", "TX_TIME_DAYS"):
        try:
            df[column].astype("int64")
        except (TypeError, ValueError) as exc:
            raise SchemaError(f"{source}: {column} is not integer-castable: {exc}") from exc

    _validate_label_domains(df, source)

    if (df["TX_AMOUNT"] < 0).any():
        raise SchemaError(f"{source}: negative TX_AMOUNT present")

    seconds = df["TX_TIME_SECONDS"].astype("int64")
    if not seconds.is_monotonic_increasing:
        raise SchemaError(f"{source}: rows are not ordered by TX_TIME_SECONDS")

    days = df["TX_TIME_DAYS"].astype("int64").unique()
    if len(days) != 1:
        raise SchemaError(f"{source}: expected exactly one TX_TIME_DAYS value, got {days}")

    tx_dates = df["TX_DATETIME"].dt.normalize().unique()
    if len(tx_dates) != 1:
        raise SchemaError(f"{source}: expected exactly one calendar date, got {len(tx_dates)}")


def _validate_label_domains(df: pd.DataFrame, source: str) -> None:
    fraud_values = set(df["TX_FRAUD"].astype("int64").unique())
    if not fraud_values <= VALID_FRAUD_VALUES:
        raise SchemaError(f"{source}: TX_FRAUD outside {sorted(VALID_FRAUD_VALUES)}: {fraud_values}")

    scenario_values = set(df["TX_FRAUD_SCENARIO"].astype("int64").unique())
    if not scenario_values <= VALID_SCENARIO_VALUES:
        raise SchemaError(
            f"{source}: TX_FRAUD_SCENARIO outside {sorted(VALID_SCENARIO_VALUES)}: {scenario_values}"
        )

    # A genuine transaction must carry scenario 0, and a fraud must carry a nonzero
    # scenario. This is the consistency guarantee the simulator's add_frauds() provides.
    genuine = df["TX_FRAUD"].astype("int64") == 0
    if (df.loc[genuine, "TX_FRAUD_SCENARIO"].astype("int64") != 0).any():
        raise SchemaError(f"{source}: TX_FRAUD=0 rows with nonzero TX_FRAUD_SCENARIO")
    if (df.loc[~genuine, "TX_FRAUD_SCENARIO"].astype("int64") == 0).any():
        raise SchemaError(f"{source}: TX_FRAUD=1 rows with TX_FRAUD_SCENARIO=0")


def normalize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Cast a validated raw frame to the processed layer's declared dtypes.

    Returns a new frame; the input is not modified.
    """
    out = df.copy()
    for column, dtype in PROCESSED_DTYPES.items():
        if out[column].dtype != dtype:
            out[column] = out[column].astype(dtype)
    return out


def validate_processed_frame(df: pd.DataFrame, source: str = "processed") -> None:
    """Validate the assembled processed dataset."""
    actual = tuple(df.columns)
    if actual != RAW_COLUMNS:
        raise SchemaError(f"{source}: column mismatch.\n  expected {RAW_COLUMNS}\n  actual {actual}")

    for column, expected in PROCESSED_DTYPES.items():
        if str(df[column].dtype) != expected:
            raise SchemaError(
                f"{source}: {column} dtype is {df[column].dtype}, expected {expected}"
            )

    if not df["TX_DATETIME"].is_monotonic_increasing:
        raise SchemaError(f"{source}: not in chronological order (Dev Plan §5)")

    if df["TRANSACTION_ID"].duplicated().any():
        count = int(df["TRANSACTION_ID"].duplicated().sum())
        raise SchemaError(f"{source}: {count} duplicate TRANSACTION_ID values")

    _validate_label_domains(df, source)
