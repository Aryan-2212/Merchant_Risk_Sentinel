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

#: Phase 3 feature layer output (mrs.features.build.build_feature_frame), partitioned by
#: split. Read-only from Phase 4 onward -- Phase 4 never regenerates or edits it.
FEATURES_DIR = DATA_DIR / "features"

#: Trained model artifacts (Phase 4+): one subdirectory per model version, each holding
#: a serialized pipeline plus a metadata.json recording feature/split/threshold lineage
#: (Dev Plan §33.4, §36). Lives at the repo root, parallel to data/, since these are
#: model outputs rather than dataset stages.
MODELS_DIR = REPO_ROOT / "models"

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

# ------------------------------------------------------------- recent operational stream

#: Simulated Recent Operational Stream (see mrs.data.recent_stream). NOT the frozen
#: benchmark dataset above, NOT real Razorpay production data -- a separate, clearly
#: demarcated, deterministically-generated 21-day demo stream, reusing existing
#: customer/terminal IDs so it flows through the same architecture.
#:
#: RECENT_STREAM_START_DATE is only the DEFAULT start date, read by
#: mrs.data.recent_stream.generate_recent_stream when its caller does not pass its own
#: `start_date` argument -- it is a configuration value, not something hard-coded at
#: every call site. Either way the value used is always a fixed constant for that run
#: (never computed from "today"), so a given (seed, start_date) pair stays reproducible
#: regardless of when it is generated. RECENT_STREAM_END_DATE is documentation only
#: (start + RECENT_STREAM_DAYS - 1 days) -- it is not read by the generator, which
#: derives the end of the window from start_date + RECENT_STREAM_DAYS.
RECENT_STREAM_START_DATE = "2026-08-15"
RECENT_STREAM_END_DATE = "2026-09-04"
RECENT_STREAM_DAYS = 21
RECENT_STREAM_TX_PER_DAY = 1_800
RECENT_STREAM_SEED = 20260814
RECENT_STREAM_SPLIT_LABEL = "recent"
#: Transaction IDs for the recent stream start here -- far above the frozen benchmark's
#: max TRANSACTION_ID (1,754,154), so the two can never collide, in Postgres or anywhere
#: else, regardless of how the benchmark dataset itself is re-derived.
RECENT_STREAM_TX_ID_OFFSET = 2_000_000_000
#: How many of the existing 5,000 customers / 10,000 terminals participate in the
#: recent stream (a demo-sized subset, not the full population -- Dev Plan §26: no
#: unnecessary scale for a 21-day, ~37.8k-transaction demo).
RECENT_STREAM_N_CUSTOMERS = 250
RECENT_STREAM_N_TERMINALS = 100

# --------------------------------------------------------------- simulated live stream

#: Continuous Simulated Live Stream (see mrs.live.continuous). A THIRD split, distinct
#: from both the frozen benchmark (train/validation/test) and the fixed 21-day recent
#: stream ("recent") -- new transactions generated at real wall-clock "now" timestamps,
#: one every LIVE_STREAM_DEFAULT_INTERVAL_SECONDS while a producer is running, reusing
#: the same real customer/terminal profiles and the identical scoring pipeline. NOT
#: real payment traffic, NOT real-time production data -- a deterministic-per-tick,
#: but not deterministic-across-runs, simulated demonstration (unlike the recent
#: stream, this is not meant to be byte-reproducible: its whole purpose is to feel
#: "alive" each time it runs, so it is not seeded from a fixed constant).
LIVE_STREAM_SPLIT_LABEL = "live"
#: Far above both the benchmark's max id (1,754,154) and the recent stream's own
#: range (2,000,000,000 + up to ~50,000), so all three id spaces can never collide.
LIVE_STREAM_TX_ID_OFFSET = 3_000_000_000
LIVE_STREAM_DEFAULT_INTERVAL_SECONDS = 2.0


def ensure_dirs() -> None:
    """Create the data directories if they do not exist. Never deletes anything."""
    for directory in (RAW_DIR, PROCESSED_DIR, REFERENCE_DIR):
        directory.mkdir(parents=True, exist_ok=True)
