"""Phase 8 Step 4: read-only FastAPI application over the Phase 8 database.

    risk_scores / alerts / audit_logs (already computed, mrs.db)
                    v
        mrs.api.schemas  (typed response views)
                    v
        mrs.api.routers.*  (thin: query + map, no computation)
                    v
        mrs.api.main:app

No route here calls into mrs.models/mrs.behavioral/mrs.risk/mrs.policy's decision
logic -- only mrs.db.models, read-only.
"""

from __future__ import annotations
