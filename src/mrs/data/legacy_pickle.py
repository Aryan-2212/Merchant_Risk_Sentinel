"""Read-only loader for the Handbook's legacy pandas pickles.

The published ``.pkl`` files were written by pandas 1.x and reference
``pandas.core.indexes.numeric.Int64Index``, a class removed in pandas 2.0. pandas 2.x
still carries ``pandas.compat.pickle_compat``, which remaps that class to
``pandas.Index`` on read, so ``pd.read_pickle`` succeeds unaided on the pinned stack
(verified on pandas 2.3.3 / numpy 2.5.2).

:func:`read_legacy_pickle` tries that path first and falls back to an explicit
compatibility unpickler if a future dependency bump removes the shim. Either way the
file is opened ``"rb"`` and never rewritten — the raw layer stays byte-identical to the
source (Dev Plan §33.5).
"""

from __future__ import annotations

import io
import pickle
from pathlib import Path

import pandas as pd

#: Classes referenced by the legacy pickles that moved or were removed, mapped to their
#: modern equivalents. Both former numeric index types collapse into pandas.Index, which
#: is what pandas' own compat layer does.
_CLASS_RELOCATIONS: dict[tuple[str, str], tuple[str, str]] = {
    ("pandas.core.indexes.numeric", "Int64Index"): ("pandas.core.indexes.base", "Index"),
    ("pandas.core.indexes.numeric", "Float64Index"): ("pandas.core.indexes.base", "Index"),
    ("pandas.core.indexes.numeric", "UInt64Index"): ("pandas.core.indexes.base", "Index"),
}


class _CompatUnpickler(pickle.Unpickler):
    """Unpickler that redirects relocated pandas/numpy classes."""

    def find_class(self, module: str, name: str):  # noqa: D102 - stdlib override
        relocated = _CLASS_RELOCATIONS.get((module, name))
        if relocated is not None:
            module, name = relocated
        elif module.startswith("numpy.core"):
            # numpy.core became numpy._core in NumPy 2.0. NumPy ships an unpickling
            # shim, but redirect explicitly in case it is ever dropped.
            candidate = module.replace("numpy.core", "numpy._core", 1)
            try:
                return super().find_class(candidate, name)
            except (ImportError, AttributeError):
                pass
        return super().find_class(module, name)


class LegacyPickleError(RuntimeError):
    """Raised when a raw file cannot be read by any supported compatibility path."""


def read_legacy_pickle(path: str | Path) -> pd.DataFrame:
    """Load one Handbook daily transaction file as a DataFrame.

    The file is only ever read. Raises :class:`LegacyPickleError` if every path fails,
    rather than returning a partial or silently substituted result.
    """
    path = Path(path)
    try:
        return pd.read_pickle(path)
    except Exception as primary:  # noqa: BLE001 - deliberately broad, we re-raise below
        try:
            with path.open("rb") as handle:
                payload = handle.read()
            frame = _CompatUnpickler(io.BytesIO(payload)).load()
        except Exception as fallback:  # noqa: BLE001
            raise LegacyPickleError(
                f"Could not read {path.name}.\n"
                f"  pd.read_pickle failed: {primary!r}\n"
                f"  compatibility unpickler failed: {fallback!r}"
            ) from fallback

        if not isinstance(frame, pd.DataFrame):
            raise LegacyPickleError(
                f"{path.name}: expected a DataFrame, got {type(frame).__name__}"
            ) from primary
        return frame
