"""AI Risk Analyst (Dev Plan §16/§41; Phase 8 Step 6).

    risk_scores + alerts + audit_logs (already computed, mrs.db)
                    v
        mrs.analyst.evidence.build_evidence  (pure, no computation)
                    v
        mrs.analyst.client.generate_explanation  (one structured LLM call,
                    v                              or deterministic fallback)
        AnalystResult

The analyst explains an already-decided risk assessment and policy action; it never
determines one. See mrs.analyst.client's module docstring for the full guardrail list.
"""

from __future__ import annotations
