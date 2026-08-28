"""Phase 7: Risk Aggregation.

Combines already-computed component signals (Phase 5 transaction ML risk, Phase 6
terminal behavioral risk, Phase 7 customer behavioral risk) into a unified, explainable
risk assessment. This package fits no model and computes no new feature -- it is a pure,
deterministic function over three already-computed component outputs.
"""
