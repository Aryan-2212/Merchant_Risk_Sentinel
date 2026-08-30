"""Phase 8 backend persistence layer (Dev Plan §19/§20; approved Phase 8 Step 1).

This package is a storage layer only. It reads nothing from and writes nothing into
mrs.data/mrs.features/mrs.models/mrs.behavioral/mrs.risk -- those Phase 1-7 modules
remain unmodified and unaware this package exists. A later step (database population,
not part of Step 1) will read their already-computed outputs and write rows here.
"""

from __future__ import annotations

from mrs.db import models  # noqa: F401  registers all ORM models on Base.metadata
from mrs.db import populate
from mrs.db.base import Base, create_all, drop_all
from mrs.db.engine import get_database_url, get_engine, get_session_factory

__all__ = [
    "Base",
    "create_all",
    "drop_all",
    "get_database_url",
    "get_engine",
    "get_session_factory",
    "models",
    "populate",
]
