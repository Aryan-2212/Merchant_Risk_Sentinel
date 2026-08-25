"""Chronological train/validation/test split boundaries (Phase 2, Dev Plan §6).

Phase 1 deliberately left these undefined (see ``mrs.config``). They are decided here,
from the actual daily/monthly transaction and fraud distributions measured in
``docs/DATASET_REPORT.md``, not assumed in advance.

Evidence for the boundaries below (full detail in DATASET_REPORT.md):

- Monthly fraud rate is stable from May onward (0.0087-0.0090); April is measurably
  lower (0.0059) because of a ramp-up transient in its first ~2-3 weeks (compromise
  episodes need time to start once the simulator begins). April is entirely inside
  train, so this transient affects only baseline/training data, not evaluation.
- August (proposed validation) and September (proposed test) both sit inside the
  stable-rate regime and closely match the June/July train months, so the split is not
  "materially uneven" in the sense Dev Plan §6 warns about.
- Every split has enough transactions of every fraud scenario for scenario-specific
  evaluation (Dev Plan §11): validation has 169/1,692/808 scenario-1/2/3 frauds, test has
  144/1,636/767.

The Dev Plan's initial proposal (train=Apr-Jul, validation=Aug, test=Sep) is therefore
adopted unchanged — the data did not show a reason to deviate from it.
"""

from __future__ import annotations

import pandas as pd

#: (start_date, end_date) inclusive, ISO format. Order matters: this is iterated to
#: build cutoffs, and validated to be strictly chronological and non-overlapping.
SPLIT_BOUNDARIES: dict[str, tuple[str, str]] = {
    "train": ("2018-04-01", "2018-07-31"),
    "validation": ("2018-08-01", "2018-08-31"),
    "test": ("2018-09-01", "2018-09-30"),
}

#: Split names in chronological order. The test set must remain untouched until
#: model/threshold decisions are frozen (Dev Plan §6, §37).
SPLIT_ORDER: tuple[str, ...] = ("train", "validation", "test")


class SplitError(ValueError):
    """Raised when the split boundaries or their application are inconsistent."""


def validate_split_boundaries() -> None:
    """Check the boundaries are strictly chronological and non-overlapping.

    Static check on the constants themselves — no data required. Called at import time
    so a bad edit to SPLIT_BOUNDARIES fails immediately rather than silently leaking.
    """
    previous_end: pd.Timestamp | None = None
    for name in SPLIT_ORDER:
        start_str, end_str = SPLIT_BOUNDARIES[name]
        start, end = pd.Timestamp(start_str), pd.Timestamp(end_str)
        if start > end:
            raise SplitError(f"{name}: start {start_str} is after end {end_str}")
        if previous_end is not None and start <= previous_end:
            raise SplitError(
                f"{name} starts at {start_str}, which does not come strictly after "
                f"the previous split's end {previous_end.date()}"
            )
        previous_end = end


validate_split_boundaries()


def assign_split(tx_datetime: pd.Series) -> pd.Series:
    """Label each timestamp with its split name ('train'/'validation'/'test').

    Raises :class:`SplitError` if any timestamp falls outside every configured range,
    rather than silently dropping or mislabeling it.
    """
    labels = pd.Series(pd.NA, index=tx_datetime.index, dtype="object")
    dates = tx_datetime.dt.normalize()
    for name in SPLIT_ORDER:
        start_str, end_str = SPLIT_BOUNDARIES[name]
        in_range = (dates >= pd.Timestamp(start_str)) & (dates <= pd.Timestamp(end_str))
        labels = labels.where(~in_range, name)

    unassigned = labels.isna()
    if unassigned.any():
        example_dates = tx_datetime[unassigned].dt.date.unique()[:5]
        raise SplitError(
            f"{int(unassigned.sum())} timestamps fall outside all configured split "
            f"boundaries, e.g. {list(example_dates)}"
        )
    return labels


def split_date_range(name: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Inclusive (start, end) timestamps for one split."""
    start_str, end_str = SPLIT_BOUNDARIES[name]
    return pd.Timestamp(start_str), pd.Timestamp(end_str)
