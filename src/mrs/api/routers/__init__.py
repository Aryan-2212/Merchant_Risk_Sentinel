"""Route modules for the Phase 8 Step 4 read API. Each router is thin: it queries
mrs.db.models directly (via the get_db dependency) and maps rows to mrs.api.schemas.
No router recomputes risk or makes a policy decision -- see mrs.risk/mrs.behavioral/
mrs.models for risk computation and mrs.policy for policy decisions.
"""
