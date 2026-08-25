"""Acquire the Fraud Detection Handbook raw dataset and record its provenance.

Idempotent: a file already present whose SHA-256 matches the manifest is left alone.
Downloaded files are set read-only (0o444) so the raw layer cannot be modified in place
by accident (Dev Plan §33.5).
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import requests

from mrs import config

_TIMEOUT = 60
_CHUNK = 1 << 16


@dataclass(frozen=True)
class RemoteFile:
    """One daily file as advertised by the upstream git tree."""

    filename: str
    git_blob_sha: str
    size_bytes: int

    @property
    def url(self) -> str:
        return config.SOURCE_FILE_URL.format(filename=self.filename)


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_blob_sha_of(path: Path) -> str:
    """Compute the git blob SHA-1 so local bytes can be checked against upstream."""
    payload = path.read_bytes()
    digest = hashlib.sha1()
    digest.update(f"blob {len(payload)}\0".encode())
    digest.update(payload)
    return digest.hexdigest()


def list_remote_files(session: requests.Session) -> list[RemoteFile]:
    """Enumerate the daily .pkl files in the upstream repository."""
    response = session.get(config.SOURCE_TREE_API, timeout=_TIMEOUT)
    response.raise_for_status()
    tree = response.json().get("tree", [])

    files = [
        RemoteFile(
            filename=Path(entry["path"]).name,
            git_blob_sha=entry["sha"],
            size_bytes=int(entry["size"]),
        )
        for entry in tree
        if entry["type"] == "blob"
        and entry["path"].startswith("data/")
        and entry["path"].endswith(".pkl")
    ]
    return sorted(files, key=lambda f: f.filename)


def resolve_upstream_commit(session: requests.Session) -> str:
    """Return the commit SHA the raw layer is pinned to, for reproducibility."""
    response = session.get(config.SOURCE_COMMIT_API, timeout=_TIMEOUT)
    response.raise_for_status()
    return response.json()["sha"]


def _make_writable(path: Path) -> None:
    if path.exists():
        path.chmod(path.stat().st_mode | stat.S_IWUSR)


def _make_read_only(path: Path) -> None:
    path.chmod(0o444)


def _download_one(session: requests.Session, remote: RemoteFile, destination: Path) -> None:
    with session.get(remote.url, timeout=_TIMEOUT, stream=True) as response:
        response.raise_for_status()
        temporary = destination.with_suffix(destination.suffix + ".part")
        with temporary.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=_CHUNK):
                handle.write(chunk)

    actual_blob = git_blob_sha_of(temporary)
    if actual_blob != remote.git_blob_sha:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(
            f"{remote.filename}: content does not match upstream.\n"
            f"  expected git blob {remote.git_blob_sha}\n  got              {actual_blob}"
        )

    _make_writable(destination)
    os.replace(temporary, destination)
    _make_read_only(destination)


def load_manifest() -> dict | None:
    if not config.RAW_MANIFEST_PATH.exists():
        return None
    return json.loads(config.RAW_MANIFEST_PATH.read_text())


def download_raw_dataset(*, force: bool = False) -> dict:
    """Download all daily files and write the provenance manifest.

    Returns the manifest. Files already present with a matching SHA-256 are skipped
    unless ``force`` is set.
    """
    config.ensure_dirs()
    session = requests.Session()

    remote_files = list_remote_files(session)
    if not remote_files:
        raise RuntimeError("Upstream repository listed no .pkl files")

    commit = resolve_upstream_commit(session)
    previous = load_manifest() or {}
    previous_files = {entry["filename"]: entry for entry in previous.get("files", [])}

    entries: list[dict] = []
    downloaded = skipped = 0

    for remote in remote_files:
        destination = config.RAW_DIR / remote.filename
        prior = previous_files.get(remote.filename)

        reusable = (
            not force
            and destination.exists()
            and prior is not None
            and prior.get("git_blob_sha") == remote.git_blob_sha
            and prior.get("sha256") == sha256_of(destination)
        )

        if reusable:
            skipped += 1
            entries.append(prior)
            continue

        _download_one(session, remote, destination)
        downloaded += 1
        entries.append(
            {
                "filename": remote.filename,
                "sha256": sha256_of(destination),
                "git_blob_sha": remote.git_blob_sha,
                "size_bytes": destination.stat().st_size,
                "source_url": remote.url,
                "retrieved_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }
        )

    manifest = {
        "source_repo": config.SOURCE_REPO,
        "source_branch": config.SOURCE_BRANCH,
        "source_commit": commit,
        "description": (
            "Fraud Detection Handbook simulated benchmark dataset. Public synthetic "
            "data — not real payment traffic (Dev Plan §2)."
        ),
        "file_count": len(entries),
        "total_bytes": sum(entry["size_bytes"] for entry in entries),
        "manifest_written_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "files": entries,
    }

    config.RAW_MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    _make_writable(config.RAW_MANIFEST_PATH)
    config.RAW_MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n")

    manifest["_downloaded"] = downloaded
    manifest["_skipped"] = skipped
    return manifest


def verify_raw_integrity() -> list[str]:
    """Re-hash every raw file against the manifest. Returns a list of problems."""
    manifest = load_manifest()
    if manifest is None:
        return ["manifest is missing; run scripts/01_download_raw.py"]

    problems: list[str] = []
    for entry in manifest["files"]:
        path = config.RAW_DIR / entry["filename"]
        if not path.exists():
            problems.append(f"{entry['filename']}: missing")
            continue
        if sha256_of(path) != entry["sha256"]:
            problems.append(f"{entry['filename']}: SHA-256 changed since download")
    return problems
