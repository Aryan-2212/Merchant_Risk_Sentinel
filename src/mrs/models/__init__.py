"""Phase 4+ model training/evaluation layer.

Kept separate from mrs.features (which only ever produces leakage-safe inputs) and from
mrs.data (raw/processed acquisition). This package consumes the Phase 3 feature layer
read-only and never regenerates it.
"""
