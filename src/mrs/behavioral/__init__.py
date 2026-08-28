"""Phase 6: behavioral-risk engines.

Separate from mrs.models (the ML risk layer) and mrs.features (point-in-time feature
computation). Modules here consume already-computed Phase 3 features and add a stateful,
interpretable, non-ML interpretation layer on top -- they never fit a model and never
recompute a feature Phase 3 already provides.
"""
