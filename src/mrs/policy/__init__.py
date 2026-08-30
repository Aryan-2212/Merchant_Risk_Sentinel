"""Deterministic policy/alert engine (Dev Plan §15; Phase 8 Step 3).

    risk_scores (Phase 5+6+7, already computed)
                    v
        mrs.policy.rules.evaluate  (pure, deterministic)
                    v
        mrs.policy.engine.apply_policy  (persistence: alerts + audit_logs)

No ML, no LLM, no new risk computation happens in this package -- it only maps an
already-computed unified_risk_level to a bounded action and records the decision.
"""

from __future__ import annotations

from mrs.policy import engine, rules

__all__ = ["engine", "rules"]
