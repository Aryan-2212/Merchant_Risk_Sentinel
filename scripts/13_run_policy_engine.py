#!/usr/bin/env python
"""Phase 8 Step 3: run the deterministic policy engine against the real database.

Reads the already-populated risk_scores table (Phase 8 Step 2 -- itself the
materialized, unmodified Phase 5+6+7 pipeline output), evaluates
mrs.policy.rules.evaluate for every row, and persists alerts/audit_logs via
mrs.policy.engine.apply_policy. Computes no risk signal of its own. Safe to re-run
(idempotent -- see mrs.policy.engine's module docstring); a second run is expected to
report n_newly_decided=0 and leave alerts/audit_logs row counts unchanged.

Run with: .venv/bin/python scripts/13_run_policy_engine.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sqlalchemy import func, select  # noqa: E402

from mrs.db.engine import get_database_url, get_engine  # noqa: E402
from mrs.db.models import Alert, AuditLog  # noqa: E402
from mrs.policy.engine import apply_policy  # noqa: E402


def main() -> None:
    engine = get_engine()
    print(f"Database: {get_database_url()}")

    print()
    print("=" * 70)
    print("Applying deterministic policy to risk_scores")
    print("=" * 70)
    summary = apply_policy(engine)

    print(f"policy_version: {summary['policy_version']}")
    print(f"rows read (risk_scores): {summary['n_rows_read']:,}")
    print(f"newly decided this run: {summary['n_newly_decided']:,}")
    print(f"skipped (already decided): {summary['n_skipped_already_decided']:,}")
    print(f"alerts written this run: {summary['n_alerts_written']:,}")
    print(f"audit_log rows written this run: {summary['n_audit_written']:,}")
    print(f"elapsed: {summary['elapsed_seconds']:.2f}s")
    print()
    print("unified_risk_level distribution (all rows read):")
    for level, count in sorted(summary["level_counts"].items()):
        print(f"  {level:<22} {count:,}")
    print()
    print("bounded-action distribution (all rows read):")
    for action, count in sorted(summary["action_counts"].items()):
        print(f"  {action:<22} {count:,}")

    print()
    print("=" * 70)
    print("Ground-truth verification: query Postgres directly")
    print("=" * 70)
    with engine.connect() as conn:
        alert_count = conn.execute(select(func.count()).select_from(Alert.__table__)).scalar_one()
        audit_count = conn.execute(select(func.count()).select_from(AuditLog.__table__)).scalar_one()
        severity_rows = conn.execute(select(Alert.severity, func.count()).group_by(Alert.severity)).all()
        action_rows = conn.execute(
            select(Alert.recommended_action, func.count()).group_by(Alert.recommended_action)
        ).all()

    print(f"alerts table row count: {alert_count:,}")
    print(f"audit_logs table row count: {audit_count:,}")
    print("alerts by severity:")
    for severity, count in sorted(severity_rows):
        print(f"  {severity:<22} {count:,}")
    print("alerts by recommended_action:")
    for action, count in sorted(action_rows):
        print(f"  {action:<22} {count:,}")


if __name__ == "__main__":
    main()
