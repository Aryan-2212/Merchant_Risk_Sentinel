#!/usr/bin/env python
"""Phase 8 Step 1: apply the Postgres schema (DDL only -- no data).

Creates the seven tables defined in mrs.db.models against the database named by
MRS_DATABASE_URL (default: the local merchant_risk_sentinel database). Idempotent --
safe to re-run; existing tables are left untouched (mrs.db.base.create_all).

Does not read or write any Phase 1-7 data. Database population (loading transactions/
features/risk scores) is a separate, later step.

Run with: .venv/bin/python scripts/11_init_db_schema.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mrs.db import Base, create_all, get_database_url, get_engine  # noqa: E402


def main() -> None:
    url = get_database_url()
    print(f"Applying schema to: {url}")
    engine = get_engine()
    create_all(engine)
    table_names = sorted(Base.metadata.tables.keys())
    print(f"Schema applied ({len(table_names)} tables): {', '.join(table_names)}")


if __name__ == "__main__":
    main()
