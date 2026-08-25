#!/usr/bin/env python
"""Download the Fraud Detection Handbook raw dataset and write its provenance manifest.

Idempotent — safe to re-run; already-downloaded files with a matching SHA-256 are
skipped. Run with: .venv/bin/python scripts/01_download_raw.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mrs.data.download import download_raw_dataset  # noqa: E402


def main() -> None:
    manifest = download_raw_dataset()
    print(f"source commit:   {manifest['source_commit']}")
    print(f"files in manifest: {manifest['file_count']}")
    print(f"downloaded now:  {manifest['_downloaded']}")
    print(f"already present: {manifest['_skipped']}")
    print(f"total bytes:     {manifest['total_bytes']:,}")


if __name__ == "__main__":
    main()
