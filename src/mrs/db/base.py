"""SQLAlchemy declarative base + schema application (Dev Plan §19; approved Phase 8
Step 1 decisions: SQLAlchemy 2.x, no Alembic/migrations framework).

The schema is applied directly via ``Base.metadata.create_all()``, which is idempotent
(``checkfirst=True``: existing tables are left untouched, missing ones are created).
That is the entire DDL mechanism for this project -- there is deliberately no migration
history/versioning tool, per the approved decision to avoid infrastructure this
single-developer local demo does not need (Dev Plan §26).
"""

from __future__ import annotations

from sqlalchemy import Engine, MetaData
from sqlalchemy.orm import DeclarativeBase

#: Explicit constraint-naming convention so FK/unique/index names are predictable and
#: readable (in psql \d output and IntegrityError messages) instead of driver-assigned
#: defaults that vary by dialect.
_NAMING_CONVENTION = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


class Base(DeclarativeBase):
    metadata = MetaData(naming_convention=_NAMING_CONVENTION)


def create_all(engine: Engine) -> None:
    """Apply the schema. Safe to call repeatedly (checkfirst=True)."""
    Base.metadata.create_all(engine, checkfirst=True)


def drop_all(engine: Engine) -> None:
    """Drop every table this project defines. Never used against production data --
    exists for test teardown."""
    Base.metadata.drop_all(engine, checkfirst=True)
