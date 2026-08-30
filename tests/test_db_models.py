"""Tests for mrs.db -- Phase 8 Step 1 backend schema (Dev Plan §20).

Two groups:

1. Metadata-only tests (no database needed): verify table names, columns, types,
   nullability, primary keys, foreign keys, and unique constraints by introspecting
   Base.metadata directly. These run everywhere, including machines without a local
   Postgres instance.
2. Live-Postgres integration tests (require_database fixture, skipped when Postgres is
   unreachable): apply the schema to a real database, verify foreign-key and
   unique-constraint enforcement actually holds at the database level, verify
   JSONB/ARRAY columns round-trip, then drop everything this module created so the
   database is left empty afterward (Step 1 is schema only -- population is a later,
   separately-approved step).
"""

from __future__ import annotations

import datetime as dt

import pytest
from sqlalchemy import inspect
from sqlalchemy.exc import IntegrityError

from mrs.db.base import Base, create_all, drop_all
from mrs.db.models import Alert, AuditLog, Customer, RiskScore, Terminal, Transaction, TransactionFeatures

EXPECTED_TABLES = {
    "customers",
    "terminals",
    "transactions",
    "transaction_features",
    "risk_scores",
    "alerts",
    "audit_logs",
}


# --------------------------------------------------------------------- metadata-only


def test_all_seven_tables_registered():
    assert set(Base.metadata.tables.keys()) == EXPECTED_TABLES


def test_customers_primary_key_and_columns():
    table = Base.metadata.tables["customers"]
    assert [c.name for c in table.primary_key.columns] == ["customer_id"]
    expected_columns = {
        "customer_id",
        "x_customer_id",
        "y_customer_id",
        "mean_amount",
        "std_amount",
        "mean_nb_tx_per_day",
        "nb_terminals",
        "available_terminals",
        "created_at",
    }
    assert set(table.columns.keys()) == expected_columns


def test_terminals_primary_key_and_columns():
    table = Base.metadata.tables["terminals"]
    assert [c.name for c in table.primary_key.columns] == ["terminal_id"]
    assert set(table.columns.keys()) == {"terminal_id", "x_terminal_id", "y_terminal_id", "created_at"}


def test_transactions_primary_key_and_foreign_keys():
    table = Base.metadata.tables["transactions"]
    assert [c.name for c in table.primary_key.columns] == ["transaction_id"]

    fk_targets = {(fk.column.table.name, fk.column.name) for fk in table.foreign_keys}
    assert fk_targets == {("customers", "customer_id"), ("terminals", "terminal_id")}

    # tx_fraud/tx_fraud_scenario are stored (Dev Plan §34.3 demo/eval ground truth) but
    # never read by mrs.models/mrs.features -- this test only asserts they exist as
    # columns, not that anything downstream may treat them as a feature.
    expected_columns = {
        "transaction_id",
        "tx_datetime",
        "customer_id",
        "terminal_id",
        "tx_amount",
        "tx_time_seconds",
        "tx_time_days",
        "tx_fraud",
        "tx_fraud_scenario",
        "split",
        "created_at",
    }
    assert set(table.columns.keys()) == expected_columns


def test_transactions_has_chronological_and_lookup_indexes():
    table = Base.metadata.tables["transactions"]
    indexed_columns = {tuple(c.name for c in idx.columns) for idx in table.indexes}
    assert ("tx_datetime",) in indexed_columns, "replay requires an index on tx_datetime"
    assert ("customer_id",) in indexed_columns
    assert ("terminal_id",) in indexed_columns


def test_transaction_features_primary_key_is_transaction_id_fk():
    table = Base.metadata.tables["transaction_features"]
    assert [c.name for c in table.primary_key.columns] == ["transaction_id"]
    fk_targets = {(fk.column.table.name, fk.column.name) for fk in table.foreign_keys}
    assert fk_targets == {("transactions", "transaction_id")}
    assert set(table.columns.keys()) == {
        "transaction_id",
        "feature_version",
        "features",
        "created_at",
    }


def test_risk_scores_primary_key_foreign_keys_and_denormalized_columns():
    table = Base.metadata.tables["risk_scores"]
    assert [c.name for c in table.primary_key.columns] == ["transaction_id"]
    fk_targets = {(fk.column.table.name, fk.column.name) for fk in table.foreign_keys}
    assert fk_targets == {
        ("transactions", "transaction_id"),
        ("customers", "customer_id"),
        ("terminals", "terminal_id"),
    }
    expected_columns = {
        "transaction_id",
        "customer_id",
        "terminal_id",
        "transaction_risk",
        "transaction_risk_severity",
        "terminal_risk_state",
        "terminal_risk_severity",
        "customer_risk_state",
        "customer_risk_severity",
        "unified_risk_level",
        "contributing_signals",
        "model_version",
        "transaction_risk_threshold",
        "feature_version",
        "computed_at",
    }
    assert set(table.columns.keys()) == expected_columns


def test_risk_scores_component_columns_are_nullable():
    """Unavailable components are a real, expected value (Dev Plan §28), not a defect."""
    table = Base.metadata.tables["risk_scores"]
    for name in (
        "transaction_risk",
        "transaction_risk_severity",
        "terminal_risk_state",
        "terminal_risk_severity",
        "customer_risk_state",
        "customer_risk_severity",
    ):
        assert table.columns[name].nullable, f"{name} must be nullable"
    assert not table.columns["unified_risk_level"].nullable


def test_risk_scores_has_lookup_indexes():
    table = Base.metadata.tables["risk_scores"]
    indexed_columns = {tuple(c.name for c in idx.columns) for idx in table.indexes}
    assert ("unified_risk_level",) in indexed_columns
    assert ("customer_id",) in indexed_columns
    assert ("terminal_id",) in indexed_columns


def test_alerts_unique_transaction_id_and_foreign_keys():
    table = Base.metadata.tables["alerts"]
    assert [c.name for c in table.primary_key.columns] == ["alert_id"]
    fk_targets = {(fk.column.table.name, fk.column.name) for fk in table.foreign_keys}
    assert fk_targets == {
        ("risk_scores", "transaction_id"),
        ("customers", "customer_id"),
        ("terminals", "terminal_id"),
    }
    unique_column_sets = {tuple(c.name for c in uc.columns) for uc in table.constraints if hasattr(uc, "columns")}
    assert ("transaction_id",) in unique_column_sets or any(
        idx.unique and tuple(c.name for c in idx.columns) == ("transaction_id",) for idx in table.indexes
    )


def test_audit_logs_columns_and_nullable_links():
    table = Base.metadata.tables["audit_logs"]
    assert [c.name for c in table.primary_key.columns] == ["audit_id"]
    assert table.columns["transaction_id"].nullable
    assert table.columns["alert_id"].nullable
    assert not table.columns["event_type"].nullable
    assert not table.columns["payload"].nullable


def test_no_cascade_deletes_defined():
    """No ON DELETE CASCADE anywhere (Dev Plan §33.5: the dataset is immutable)."""
    for table in Base.metadata.tables.values():
        for fk in table.foreign_keys:
            assert fk.ondelete in (None, "RESTRICT", "NO ACTION"), (
                f"{table.name}.{fk.parent.name} -> unexpected ondelete={fk.ondelete!r}"
            )


# ------------------------------------------------------------------ live-Postgres
# db_engine/db_session fixtures live in conftest.py (shared with tests/test_db_populate.py).


def _seed_customer_terminal_transaction(session, *, transaction_id=1, customer_id=1, terminal_id=1):
    session.add(Customer(
        customer_id=customer_id, x_customer_id=1.0, y_customer_id=1.0, mean_amount=50.0,
        std_amount=10.0, mean_nb_tx_per_day=2.0, nb_terminals=3, available_terminals=[terminal_id, 2, 3],
    ))
    session.add(Terminal(terminal_id=terminal_id, x_terminal_id=1.0, y_terminal_id=1.0))
    session.flush()
    session.add(Transaction(
        transaction_id=transaction_id, tx_datetime=dt.datetime(2018, 4, 1, 12, 0, 0),
        customer_id=customer_id, terminal_id=terminal_id, tx_amount=42.5, tx_time_seconds=100,
        tx_time_days=0, tx_fraud=0, tx_fraud_scenario=0, split="train",
    ))
    session.flush()


def test_schema_applies_and_round_trips_array_and_jsonb(db_session):
    _seed_customer_terminal_transaction(db_session)
    db_session.add(TransactionFeatures(
        transaction_id=1, feature_version="phase3_v1", features={"tx_amount": 42.5, "tx_hour": 12},
    ))
    db_session.add(RiskScore(
        transaction_id=1, customer_id=1, terminal_id=1, transaction_risk=0.12,
        transaction_risk_severity=0, terminal_risk_state="NORMAL", terminal_risk_severity=0,
        customer_risk_state="NORMAL", customer_risk_severity=0, unified_risk_level="LOW",
        contributing_signals=[], model_version="xgboost_v1", transaction_risk_threshold=0.97,
        feature_version="phase3_v1",
    ))
    db_session.commit()

    customer = db_session.get(Customer, 1)
    assert customer.available_terminals == [1, 2, 3]

    features = db_session.get(TransactionFeatures, 1)
    assert features.features == {"tx_amount": 42.5, "tx_hour": 12}

    risk = db_session.get(RiskScore, 1)
    assert risk.unified_risk_level == "LOW"
    assert risk.contributing_signals == []


def test_transaction_features_rejects_unknown_transaction_id(db_session):
    with pytest.raises(IntegrityError):
        db_session.add(TransactionFeatures(transaction_id=999, feature_version="phase3_v1", features={}))
        db_session.commit()


def test_risk_score_rejects_unknown_customer_id(db_session):
    _seed_customer_terminal_transaction(db_session)
    with pytest.raises(IntegrityError):
        db_session.add(RiskScore(
            transaction_id=1, customer_id=999, terminal_id=1, unified_risk_level="LOW",
            contributing_signals=[], model_version="xgboost_v1", transaction_risk_threshold=0.97,
            feature_version="phase3_v1",
        ))
        db_session.commit()


def test_alert_transaction_id_uniqueness_enforced(db_session):
    _seed_customer_terminal_transaction(db_session)
    db_session.add(RiskScore(
        transaction_id=1, customer_id=1, terminal_id=1, unified_risk_level="CRITICAL",
        contributing_signals=["terminal_behavioral_risk: HIGH_RISK"], model_version="xgboost_v1",
        transaction_risk_threshold=0.97, feature_version="phase3_v1",
    ))
    db_session.commit()

    db_session.add(Alert(
        transaction_id=1, customer_id=1, terminal_id=1, severity="CRITICAL",
        reason="terminal_behavioral_risk: HIGH_RISK", evidence={"terminal_risk_state": "HIGH_RISK"},
        status="OPEN",
    ))
    db_session.commit()

    with pytest.raises(IntegrityError):
        db_session.add(Alert(
            transaction_id=1, customer_id=1, terminal_id=1, severity="CRITICAL",
            reason="duplicate", evidence={}, status="OPEN",
        ))
        db_session.commit()


def test_audit_log_allows_null_transaction_and_alert(db_session):
    db_session.add(AuditLog(event_type="SYSTEM_STARTUP", payload={"note": "schema initialized"}))
    db_session.commit()

    logs = db_session.query(AuditLog).all()
    assert len(logs) == 1
    assert logs[0].transaction_id is None
    assert logs[0].alert_id is None


def test_create_all_is_idempotent(db_engine):
    # Calling create_all a second time against an already-applied schema must not raise.
    create_all(db_engine)
    inspector = inspect(db_engine)
    assert set(inspector.get_table_names()) == EXPECTED_TABLES
