"""Leakage-safe temporal aggregation primitives.

Every function here was validated against small synthetic datasets before being trusted
(Dev Plan §33.6 is non-negotiable; "looks correct" is not evidence). The properties proven
in tests/test_features_temporal.py, and relied on by every function below:

- Rolling time-windows use ``closed="left"``, giving the half-open interval [t-window, t):
  the left boundary (exactly window-ago) is included, the current row's own timestamp is
  never included, even when another row shares that exact timestamp.
- Expanding ("all prior history") statistics use ``shift(1)`` before ``expanding()``, which
  removes the current row from its own aggregate before the expanding window ever sees it.
- Duplicate timestamps within one entity are broken deterministically by TRANSACTION_ID
  (verified globally unique in Phase 1's schema validation). This is an arbitrary but
  documented, deterministic choice -- there is no other ordering signal available for two
  transactions recorded at the exact same instant.
- pandas' rolling().count()/expanding().sum() return NaN (not 0) over an empty window;
  callers that want a true "0 = no history" count must pass fillna_zero=True.
- Realignment back to the caller's row order never trusts pandas index labels (duplicate
  timestamps make index labels non-unique) -- it uses an explicit positional column, `_pos`.

Nothing here reads TX_FRAUD or TX_FRAUD_SCENARIO by name; callers choose which column to
aggregate, and mrs.data.schema.LABEL_COLUMNS still governs what may reach a model.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

#: Deterministic global ordering used everywhere in this module. TRANSACTION_ID breaks
#: ties among transactions sharing an identical TX_DATETIME (Dev Plan §33.11 -- documented
#: choice, not a silent assumption).
_ORDER_COLUMNS = ["TX_DATETIME", "TRANSACTION_ID"]


def sort_canonical(df: pd.DataFrame) -> pd.DataFrame:
    """Sort by (TX_DATETIME, TRANSACTION_ID) with a fresh RangeIndex.

    This is the one canonical chronological order the whole feature layer uses. Do not
    trust incoming row order -- Phase 2 analysis found at least one pair of same-timestamp
    rows whose on-disk order disagreed with TRANSACTION_ID order.
    """
    return df.sort_values(_ORDER_COLUMNS).reset_index(drop=True)


def _with_positions(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["_pos"] = np.arange(len(out))
    return out


def _realign(work: pd.DataFrame, values: np.ndarray, n_rows: int) -> np.ndarray:
    """Scatter `values` (in `work`'s row order) back to `_pos` order (0..n_rows-1)."""
    work = work.reset_index(drop=True)
    work = work[["_pos"]].assign(_val=values)
    ordered = work.set_index("_pos").sort_index()["_val"]
    if len(ordered) != n_rows or not (ordered.index == np.arange(n_rows)).all():
        raise AssertionError("realignment lost or misplaced rows")
    return ordered.to_numpy()


def rolling_count(
    df: pd.DataFrame,
    id_col: str,
    window: str,
    *,
    time_col: str = "TX_DATETIME",
) -> np.ndarray:
    """Count of same-entity rows in the half-open window [t-window, t), current row excluded.

    Returns an array aligned to `df`'s current row order (df is not required to be
    pre-sorted). Duplicate-timestamp-safe and entity-isolated by construction: the window
    boundary is a timestamp comparison, never a row-position comparison.
    """
    positioned = _with_positions(df)
    work = positioned.sort_values([id_col] + _ORDER_COLUMNS).set_index(time_col)
    rolled = work.groupby(id_col, sort=False).rolling(
        window, closed="left", min_periods=0
    )["_pos"].count()
    values = rolled.reset_index(drop=True).to_numpy()
    return _realign(work.reset_index(drop=True), values, len(df))


def rolling_sum(
    df: pd.DataFrame,
    id_col: str,
    value_col: str,
    window: str,
    *,
    time_col: str = "TX_DATETIME",
) -> np.ndarray:
    """Sum of `value_col` over same-entity rows in [t-window, t), current row excluded.

    Used for e.g. recent fraud count (value_col='TX_FRAUD'). Empty windows yield 0.0, not
    NaN -- a "recent fraud count" of zero when nothing has happened recently is a real
    answer, not a missing one.
    """
    positioned = _with_positions(df)
    work = positioned.sort_values([id_col] + _ORDER_COLUMNS).set_index(time_col)
    rolled = work.groupby(id_col, sort=False).rolling(
        window, closed="left", min_periods=0
    )[value_col].sum()
    values = rolled.reset_index(drop=True).fillna(0.0).to_numpy()
    return _realign(work.reset_index(drop=True), values, len(df))


def expanding_prior(
    df: pd.DataFrame,
    id_col: str,
    value_col: str,
    stat: str,
    *,
    time_col: str = "TX_DATETIME",
) -> np.ndarray:
    """Expanding statistic over ALL strictly-prior same-entity rows (current row excluded).

    stat in {"mean", "std", "count", "sum"}. "mean"/"std" return NaN for a cold-start
    entity (no prior rows) or, for "std", a single prior row (variance undefined) -- this
    is a true "not yet knowable" NaN, distinct from a real zero. "count"/"sum" return 0.0
    for a cold-start entity, since "zero prior transactions" and "zero prior frauds" are
    real, known answers, not missing ones.
    """
    if stat not in {"mean", "std", "count", "sum"}:
        raise ValueError(f"unsupported stat: {stat!r}")

    work = _with_positions(df).sort_values([id_col] + _ORDER_COLUMNS).reset_index(drop=True)

    shifted = work.groupby(id_col)[value_col].shift(1)
    grouped_expanding = shifted.groupby(work[id_col]).expanding()
    result = getattr(grouped_expanding, stat)().reset_index(level=0, drop=True)
    values = result.to_numpy()

    if stat in {"count", "sum"}:
        values = np.nan_to_num(values, nan=0.0)

    return _realign(work, values, len(df))


def time_since_previous(
    df: pd.DataFrame,
    id_col: str,
    *,
    time_col: str = "TX_DATETIME",
) -> np.ndarray:
    """Seconds since this entity's previous transaction; NaN for a cold-start entity.

    Duplicate-timestamp-safe: for two rows tied at the same instant, the later
    (TRANSACTION_ID-broken) row sees a gap of exactly 0.0 seconds -- a real, correct answer
    (two transactions truly recorded at the same instant), not an artifact.
    """
    work = _with_positions(df).sort_values([id_col] + _ORDER_COLUMNS).reset_index(drop=True)
    prev_time = work.groupby(id_col)[time_col].shift(1)
    delta_seconds = (work[time_col] - prev_time).dt.total_seconds().to_numpy()
    return _realign(work, delta_seconds, len(df))


def first_occurrence_and_prior_unique_count(
    df: pd.DataFrame,
    id_col: str,
    other_col: str,
) -> tuple[np.ndarray, np.ndarray]:
    """For each row, is this the first-ever (id_col, other_col) pairing, and how many
    distinct `other_col` values has `id_col` seen strictly before this row?

    Both current-row-safe: a row's own first-occurrence status is never counted toward its
    own "prior unique count" (enforced via shift(1) before the cumulative sum).
    """
    work = _with_positions(df).sort_values([id_col] + _ORDER_COLUMNS).reset_index(drop=True)
    is_first = (~work.duplicated([id_col, other_col], keep="first")).astype(int)
    shifted_first = is_first.groupby(work[id_col]).shift(1).fillna(0)
    prior_unique_count = shifted_first.groupby(work[id_col]).cumsum().to_numpy()

    is_first_realigned = _realign(work, is_first.to_numpy(), len(df))
    count_realigned = _realign(work, prior_unique_count, len(df))
    return is_first_realigned, count_realigned


def first_seen_timestamp(
    df: pd.DataFrame,
    id_col: str,
    *,
    time_col: str = "TX_DATETIME",
) -> np.ndarray:
    """This entity's earliest transaction timestamp, as of the whole dataset.

    Safe to compute without shifting: an entity's minimum timestamp over the full dataset
    equals its minimum timestamp restricted to "up to and including the current row",
    because every not-yet-seen future row has a strictly later (or, at worst, tied)
    timestamp and can therefore never lower an already-established minimum. Verified by a
    dedicated leakage test (mutating a later row's timestamp must not change an earlier
    row's value here).
    """
    positioned = _with_positions(df)
    first_seen = positioned.groupby(id_col)[time_col].transform("min")
    return _realign(positioned, first_seen.to_numpy(), len(df))
