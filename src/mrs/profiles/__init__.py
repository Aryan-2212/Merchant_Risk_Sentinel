"""Reproduction and validation of the Handbook's customer/terminal profile tables.

These tables are not published with the raw data (see
external/fraud_detection_handbook/NOTICE.md for the investigation). They are
regenerated from the official simulator's deterministic profile-generation code and
must pass validate.py before being persisted. If validation fails, no table is written.
"""
