"""Feature engineering (Phase 3): transaction, customer, and terminal behavioral features.

All temporal aggregation goes through mrs.features._temporal, the single place the
leakage-critical pandas patterns live (Dev Plan §33.6). Individual feature modules
(transaction.py, customer.py, terminal.py, relationship.py) build on those primitives
rather than re-deriving them.
"""
