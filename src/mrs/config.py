"""Project paths and data-source constants.

Scope note (Phase 1): this module records only what Phase 1 has *verified* — where data
lives, where it came from, and the temporal coverage observed in the source. It
deliberately contains no train/validation/test split boundaries; a placeholder here would
have quietly become the answer while the caveat beside it stopped being read.

Update (Phase 2): those split boundaries have now been decided from the actual dataset
distribution (Dev Plan §6) and live in :mod:`mrs.data.splits`, not here — kept separate
because "where data lives" and "how it is evaluated" are different concerns.
"""

from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------- paths

REPO_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = Path(os.environ.get("MRS_DATA_DIR", REPO_ROOT / "data"))

#: Immutable original Handbook .pkl files. Never written to after download.
RAW_DIR = DATA_DIR / "raw"

#: Normalised, chronologically ordered Parquet built from RAW_DIR.
PROCESSED_DIR = DATA_DIR / "processed"

#: Entity reference tables (customer/terminal profiles), written only after validation.
REFERENCE_DIR = DATA_DIR / "reference"

#: Provenance record for the raw layer: per-file SHA-256, size, source URL, retrieval
#: timestamp, and the upstream commit the files were fetched from.
RAW_MANIFEST_PATH = RAW_DIR / "MANIFEST.json"

PROCESSED_TRANSACTIONS_DIR = PROCESSED_DIR / "transactions"

# --------------------------------------------------------------------- data source

#: Public simulated benchmark data. This is NOT Razorpay production traffic and must
#: never be described as such (Dev Plan §2, §26).
SOURCE_REPO = "Fraud-Detection-Handbook/simulated-data-raw"
SOURCE_BRANCH = "main"
SOURCE_TREE_API = f"https://api.github.com/repos/{SOURCE_REPO}/git/trees/{SOURCE_BRANCH}?recursive=1"
SOURCE_COMMIT_API = f"https://api.github.com/repos/{SOURCE_REPO}/commits/{SOURCE_BRANCH}"
SOURCE_FILE_URL = (
    f"https://raw.githubusercontent.com/{SOURCE_REPO}/{SOURCE_BRANCH}/data/{{filename}}"
)

#: Temporal coverage advertised by the source and verified during Phase 1 download.
EXPECTED_START_DATE = "2018-04-01"
EXPECTED_END_DATE = "2018-09-30"
EXPECTED_FILE_COUNT = 183


def ensure_dirs() -> None:
    """Create the data directories if they do not exist. Never deletes anything."""
    for directory in (RAW_DIR, PROCESSED_DIR, REFERENCE_DIR):
        directory.mkdir(parents=True, exist_ok=True)
