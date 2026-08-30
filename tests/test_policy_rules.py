"""Tests for mrs.policy.rules -- Phase 8 Step 3 deterministic policy (Dev Plan §15).

Pure, no database, no real dataset -- small synthetic risk_scores-shaped dicts only.
"""

from __future__ import annotations

import pytest

from mrs.policy.rules import (
    ALLOW,
    ESCALATE,
    MONITOR,
    POLICY_VERSION,
    STEP_UP_VERIFICATION,
    TEMPORARY_REVIEW,
    PolicyDecision,
    build_reason,
    decide_action,
    evaluate,
)
from mrs.risk.aggregate import CRITICAL, HIGH, INSUFFICIENT_EVIDENCE, LOW, MEDIUM


def _row(**overrides) -> dict:
    row = {
        "transaction_id": 1,
        "customer_id": 10,
        "terminal_id": 20,
        "transaction_risk": 0.1,
        "transaction_risk_severity": 0,
        "terminal_risk_state": "NORMAL",
        "terminal_risk_severity": 0,
        "customer_risk_state": "NORMAL",
        "customer_risk_severity": 0,
        "unified_risk_level": LOW,
        "contributing_signals": [],
        "model_version": "xgboost_v1",
        "transaction_risk_threshold": 0.97,
        "feature_version": "phase3_v1",
    }
    row.update(overrides)
    return row


# --------------------------------------------------------------------- exact mapping


@pytest.mark.parametrize(
    "level,expected_action",
    [
        (LOW, ALLOW),
        (MEDIUM, MONITOR),
        (HIGH, STEP_UP_VERIFICATION),
        (CRITICAL, ESCALATE),
        (INSUFFICIENT_EVIDENCE, TEMPORARY_REVIEW),
    ],
)
def test_exact_action_mapping(level, expected_action):
    assert decide_action(level) == expected_action


def test_decide_action_rejects_unknown_level():
    with pytest.raises(ValueError):
        decide_action("NOT_A_REAL_LEVEL")


def test_decide_action_is_deterministic_pure_function():
    # Same input -> same output, every time; no hidden state.
    for _ in range(5):
        assert decide_action(CRITICAL) == ESCALATE
    # Action depends only on level, never on contributing_signals content.
    assert decide_action(HIGH) == decide_action(HIGH)


# --------------------------------------------------------------------- per-level evaluate


def test_low_risk_is_allow_and_not_alert_worthy():
    decision = evaluate(_row(unified_risk_level=LOW, contributing_signals=[]))
    assert decision.action == ALLOW
    assert decision.is_alert is False


def test_medium_risk_is_monitor_and_alert_worthy():
    decision = evaluate(
        _row(
            unified_risk_level=MEDIUM,
            terminal_risk_state="RISK_RISING",
            terminal_risk_severity=1,
            contributing_signals=["terminal_behavioral_risk: RISK_RISING"],
        )
    )
    assert decision.action == MONITOR
    assert decision.is_alert is True
    assert decision.reason == "terminal_behavioral_risk: RISK_RISING"


def test_high_risk_is_step_up_verification():
    decision = evaluate(
        _row(
            unified_risk_level=HIGH,
            transaction_risk=0.99,
            transaction_risk_severity=2,
            contributing_signals=["transaction_ml_risk >= 0.97"],
        )
    )
    assert decision.action == STEP_UP_VERIFICATION
    assert decision.is_alert is True


def test_critical_risk_is_escalate():
    decision = evaluate(
        _row(
            unified_risk_level=CRITICAL,
            transaction_risk_severity=2,
            terminal_risk_state="HIGH_RISK",
            terminal_risk_severity=2,
            contributing_signals=["transaction_ml_risk >= 0.97", "terminal_behavioral_risk: HIGH_RISK"],
        )
    )
    assert decision.action == ESCALATE
    assert decision.is_alert is True


def test_insufficient_evidence_is_temporary_review_not_allow():
    """Dev Plan Sec 28: absence of evidence must never be treated as calm -- ALLOW
    would be the wrong action here even though there is no elevated signal."""
    decision = evaluate(
        _row(
            unified_risk_level=INSUFFICIENT_EVIDENCE,
            transaction_risk=None,
            transaction_risk_severity=None,
            terminal_risk_state=None,
            terminal_risk_severity=None,
            customer_risk_state="INSUFFICIENT_HISTORY",
            customer_risk_severity=None,
            contributing_signals=[],
        )
    )
    assert decision.action == TEMPORARY_REVIEW
    assert decision.action != ALLOW
    assert decision.is_alert is True
    assert "insufficient history" in decision.reason.lower()


# --------------------------------------------------------------------- evidence sourcing


def test_transaction_ml_evidence_reflected_in_reason_and_evidence():
    decision = evaluate(
        _row(
            unified_risk_level=HIGH,
            transaction_risk=0.985,
            transaction_risk_severity=2,
            contributing_signals=["transaction_ml_risk >= 0.97"],
        )
    )
    assert "transaction_ml_risk" in decision.reason
    assert decision.evidence["transaction_risk"] == 0.985
    assert decision.evidence["transaction_risk_severity"] == 2


def test_terminal_behavioral_evidence_reflected_in_reason_and_evidence():
    decision = evaluate(
        _row(
            unified_risk_level=HIGH,
            terminal_risk_state="HIGH_RISK",
            terminal_risk_severity=2,
            contributing_signals=["terminal_behavioral_risk: HIGH_RISK"],
        )
    )
    assert "terminal_behavioral_risk" in decision.reason
    assert decision.evidence["terminal_risk_state"] == "HIGH_RISK"


def test_customer_behavioral_evidence_reflected_in_reason_and_evidence():
    decision = evaluate(
        _row(
            unified_risk_level=HIGH,
            customer_risk_state="HIGH_RISK",
            customer_risk_severity=2,
            contributing_signals=["customer_behavioral_risk: HIGH_RISK"],
        )
    )
    assert "customer_behavioral_risk" in decision.reason
    assert decision.evidence["customer_risk_state"] == "HIGH_RISK"


def test_multiple_contributing_signals_all_present_in_reason_and_evidence():
    signals = ["transaction_ml_risk >= 0.97", "terminal_behavioral_risk: HIGH_RISK"]
    decision = evaluate(_row(unified_risk_level=CRITICAL, contributing_signals=signals))
    assert decision.reason == "transaction_ml_risk >= 0.97; terminal_behavioral_risk: HIGH_RISK"
    assert decision.evidence["contributing_signals"] == signals


def test_no_fabricated_evidence_evidence_dict_is_exactly_the_source_fields():
    row = _row(
        unified_risk_level=CRITICAL,
        transaction_risk=0.99,
        transaction_risk_severity=2,
        terminal_risk_state="HIGH_RISK",
        terminal_risk_severity=2,
        customer_risk_state="NORMAL",
        customer_risk_severity=0,
        contributing_signals=["transaction_ml_risk >= 0.97", "terminal_behavioral_risk: HIGH_RISK"],
    )
    decision = evaluate(row)
    assert decision.evidence == {
        "unified_risk_level": CRITICAL,
        "contributing_signals": row["contributing_signals"],
        "transaction_risk": 0.99,
        "transaction_risk_severity": 2,
        "terminal_risk_state": "HIGH_RISK",
        "terminal_risk_severity": 2,
        "customer_risk_state": "NORMAL",
        "customer_risk_severity": 0,
        "transaction_risk_threshold": 0.97,
        "model_version": "xgboost_v1",
        "feature_version": "phase3_v1",
    }


def test_evaluate_requires_fields():
    with pytest.raises(ValueError):
        evaluate({"transaction_id": 1})


# --------------------------------------------------------------------- reason building


def test_build_reason_low_no_signals():
    assert build_reason(LOW, []) == "LOW: no elevated component signals"


def test_build_reason_insufficient_evidence_no_signals():
    reason = build_reason(INSUFFICIENT_EVIDENCE, [])
    assert "INSUFFICIENT_EVIDENCE" in reason
    assert "insufficient history" in reason.lower()


# --------------------------------------------------------------------- traceability


def test_policy_decision_carries_policy_version():
    decision = evaluate(_row())
    assert decision.policy_version == POLICY_VERSION
    assert isinstance(decision, PolicyDecision)
