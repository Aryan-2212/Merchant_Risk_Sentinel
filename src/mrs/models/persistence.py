"""Persist/load a trained model pipeline plus its lineage metadata (Dev Plan Sec 36).

Every save is a new version directory, never overwritten in place, so a previous model
version's artifacts stay attributable to the risk results it produced.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
from sklearn.pipeline import Pipeline

MODEL_FILENAME = "model.joblib"
METADATA_FILENAME = "metadata.json"


def save_model(pipeline: Pipeline, metadata: dict, version_dir: Path) -> None:
    if version_dir.exists():
        raise FileExistsError(
            f"save_model: {version_dir} already exists; refusing to overwrite a saved "
            "model version (Dev Plan Sec 36: prior model versions must stay attributable "
            "to the risk results they produced). Use a new version directory."
        )
    version_dir.mkdir(parents=True)
    joblib.dump(pipeline, version_dir / MODEL_FILENAME)
    with open(version_dir / METADATA_FILENAME, "w") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)


def load_model(version_dir: Path) -> tuple[Pipeline, dict]:
    pipeline = joblib.load(version_dir / MODEL_FILENAME)
    with open(version_dir / METADATA_FILENAME) as f:
        metadata = json.load(f)
    return pipeline, metadata
