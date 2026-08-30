"""Tests for mrs.analyst -- Phase 8 Step 6 AI Risk Analyst (Dev Plan §16/§41).

No test in this module makes a real network call: mrs.analyst.client._call_llm_raw is
monkeypatched to a stub in every generate_explanation test, so these run without an
ANTHROPIC_API_KEY and without network access. That is deliberate -- it lets us
directly simulate the failure modes Dev Plan §41 requires handling (unavailable,
invalid output, refusal) and the hallucination boundary Step 6 explicitly asks to be
tested, none of which are reliably reproducible against a real, non-deterministic LLM.
"""

from __future__ import annotations

import datetime as dt

import pydantic
import pytest

from mrs.analyst.client import (
    ANALYST_MODEL,
    _check_evidence_grounding,
    _fallback,
    generate_explanation,
)
from mrs.analyst.evidence import build_evidence
from mrs.analyst.schemas import AnalystEvidence, AnalystExplanation
from mrs.db.models import Alert, RiskScore, Transaction
from mrs.policy.rules import ALLOW, CRITICAL, ESCALATE, HIGH, INSUFFICIENT_EVIDENCE, LOW, MONITOR


def _evidence(**overrides) -> AnalystEvidence:
    fields = dict(
        transaction_id=1,
        tx_amount=42.5,
        tx_datetime=dt.datetime(2018, 4, 1, 12, 0, 0),
        customer_id=10,
        terminal_id=20,
        unified_risk_level=LOW,
        transaction_risk=0.1,
        transaction_risk_severity=0,
        terminal_risk_state="NORMAL",
        terminal_risk_severity=0,
        customer_risk_state="NORMAL",
        customer_risk_severity=0,
        contributing_signals=[],
        policy_action=ALLOW,
        policy_reason=None,
        policy_version="policy_v1",
        model_version="xgboost_v1",
        feature_version="phase3_v1",
        transaction_risk_threshold=0.97,
    )
    fields.update(overrides)
    return AnalystEvidence(**fields)


def _explanation(**overrides) -> AnalystExplanation:
    fields = dict(
        summary="Low risk transaction with no elevated component signals.",
        evidence_explanation="No contributing signals were present.",
        recommended_action=ALLOW,
        recommendation_rationale="All components are at severity 0.",
        confidence="high",
        caveats=[],
    )
    fields.update(overrides)
    return AnalystExplanation(**fields)


# --------------------------------------------------------------------------- evidence


def test_build_evidence_with_alert():
    tx = Transaction(
        transaction_id=1,
        tx_datetime=dt.datetime(2018, 4, 1, 12, 0, 0),
        customer_id=10,
        terminal_id=20,
        tx_amount=999.0,
        tx_time_seconds=0,
        tx_time_days=0,
        tx_fraud=1,
        tx_fraud_scenario=1,
        split="train",
    )
    risk = RiskScore(
        transaction_id=1,
        customer_id=10,
        terminal_id=20,
        transaction_risk=0.99,
        transaction_risk_severity=2,
        terminal_risk_state="HIGH_RISK",
        terminal_risk_severity=2,
        customer_risk_state="NORMAL",
        customer_risk_severity=0,
        unified_risk_level=CRITICAL,
        contributing_signals=["transaction_ml_risk >= 0.97", "terminal_behavioral_risk: HIGH_RISK"],
        model_version="xgboost_v1",
        transaction_risk_threshold=0.97,
        feature_version="phase3_v1",
    )
    alert = Alert(
        transaction_id=1,
        customer_id=10,
        terminal_id=20,
        severity=CRITICAL,
        reason="transaction_ml_risk >= 0.97; terminal_behavioral_risk: HIGH_RISK",
        evidence={},
        recommended_action=ESCALATE,
        status="OPEN",
    )

    evidence = build_evidence(tx, risk, alert, "policy_v1")

    assert evidence.transaction_id == 1
    assert evidence.tx_amount == 999.0
    assert evidence.unified_risk_level == CRITICAL
    assert evidence.contributing_signals == ["transaction_ml_risk >= 0.97", "terminal_behavioral_risk: HIGH_RISK"]
    assert evidence.policy_action == ESCALATE
    assert evidence.policy_reason == "transaction_ml_risk >= 0.97; terminal_behavioral_risk: HIGH_RISK"
    assert evidence.policy_version == "policy_v1"


def test_build_evidence_without_alert_defaults_to_allow():
    tx = Transaction(
        transaction_id=2,
        tx_datetime=dt.datetime(2018, 4, 1, 0, 0, 0),
        customer_id=1,
        terminal_id=1,
        tx_amount=5.0,
        tx_time_seconds=0,
        tx_time_days=0,
        tx_fraud=0,
        tx_fraud_scenario=0,
        split="train",
    )
    risk = RiskScore(
        transaction_id=2,
        customer_id=1,
        terminal_id=1,
        transaction_risk=0.01,
        transaction_risk_severity=0,
        terminal_risk_state="NORMAL",
        terminal_risk_severity=0,
        customer_risk_state="NORMAL",
        customer_risk_severity=0,
        unified_risk_level=LOW,
        contributing_signals=[],
        model_version="xgboost_v1",
        transaction_risk_threshold=0.97,
        feature_version="phase3_v1",
    )

    evidence = build_evidence(tx, risk, None, None)

    assert evidence.policy_action == ALLOW
    assert evidence.policy_reason is None
    assert evidence.policy_version is None


# ------------------------------------------------------------------- evidence grounding


def test_grounding_check_passes_clean_explanation():
    assert _check_evidence_grounding(_explanation()) is None


@pytest.mark.parametrize(
    "phrase",
    [
        "This transaction is fraud.",
        "We have confirmed fraud on this account.",
        "This is definitely fraud based on the pattern.",
        "This is a scam targeting the merchant.",
    ],
)
def test_grounding_check_flags_fraud_certainty_claims(phrase):
    violation = _check_evidence_grounding(_explanation(summary=phrase))
    assert violation is not None
    assert "fraud certainty" in violation


def test_grounding_check_flags_fraud_certainty_in_any_field():
    violation = _check_evidence_grounding(_explanation(recommendation_rationale="This transaction is fraudulent."))
    assert violation is not None


def test_grounding_check_catches_invalid_action_bypassing_schema():
    # Pydantic's Literal type already prevents this at construction time; simulate a
    # library/schema change loosening that constraint by bypassing validation via
    # model_construct, to prove the redundant runtime check independently catches it.
    bad = AnalystExplanation.model_construct(
        summary="ok",
        evidence_explanation="ok",
        recommended_action="DENY_TRANSACTION",  # not a bounded action
        recommendation_rationale="ok",
        confidence="high",
        caveats=[],
    )
    violation = _check_evidence_grounding(bad)
    assert violation is not None
    assert "not a bounded action" in violation


def test_schema_itself_rejects_invalid_action_at_construction():
    with pytest.raises(pydantic.ValidationError):
        AnalystExplanation(
            summary="ok",
            evidence_explanation="ok",
            recommended_action="DENY_TRANSACTION",
            recommendation_rationale="ok",
            confidence="high",
        )


def test_schema_rejects_invalid_confidence():
    with pytest.raises(pydantic.ValidationError):
        AnalystExplanation(
            summary="ok",
            evidence_explanation="ok",
            recommended_action=ALLOW,
            recommendation_rationale="ok",
            confidence="extremely-certain",  # not low/medium/high
        )


# ------------------------------------------------------------------------------ fallback


def test_fallback_mirrors_deterministic_policy_action_never_invents_one():
    evidence = _evidence(unified_risk_level=HIGH, policy_action=MONITOR, contributing_signals=["x"])
    result = _fallback(evidence, reason="test reason")

    assert result.is_fallback is True
    assert result.fallback_reason == "test reason"
    assert result.explanation.recommended_action == MONITOR  # mirrors evidence.policy_action exactly
    assert result.explanation.confidence == "low"
    assert "test reason" in result.explanation.caveats


def test_fallback_evidence_explanation_uses_only_supplied_signals():
    evidence = _evidence(contributing_signals=["terminal_behavioral_risk: HIGH_RISK"])
    result = _fallback(evidence, reason="x")
    assert result.explanation.evidence_explanation == "terminal_behavioral_risk: HIGH_RISK"


def test_fallback_handles_insufficient_evidence_level():
    evidence = _evidence(unified_risk_level=INSUFFICIENT_EVIDENCE, policy_action="TEMPORARY_REVIEW", contributing_signals=[])
    result = _fallback(evidence, reason="x")
    assert result.explanation.recommended_action == "TEMPORARY_REVIEW"
    assert "no elevated component signals" in result.explanation.evidence_explanation


# ------------------------------------------------------------------- generate_explanation


class _FakeResponse:
    def __init__(self, *, stop_reason="end_turn", parsed_output=None):
        self.stop_reason = stop_reason
        self.parsed_output = parsed_output


def test_generate_explanation_success_path(monkeypatch):
    good = _explanation(summary="Elevated risk driven by terminal behavior.")

    def fake_call(evidence):
        return _FakeResponse(stop_reason="end_turn", parsed_output=good)

    monkeypatch.setattr("mrs.analyst.client._call_llm_raw", fake_call)

    result = generate_explanation(_evidence())
    assert result.is_fallback is False
    assert result.fallback_reason is None
    assert result.explanation is good


def test_generate_explanation_falls_back_on_exception(monkeypatch):
    def fake_call(evidence):
        raise RuntimeError("simulated network failure")

    monkeypatch.setattr("mrs.analyst.client._call_llm_raw", fake_call)

    result = generate_explanation(_evidence())
    assert result.is_fallback is True
    assert "RuntimeError" in result.fallback_reason
    assert "simulated network failure" in result.fallback_reason


def test_generate_explanation_falls_back_on_refusal(monkeypatch):
    def fake_call(evidence):
        return _FakeResponse(stop_reason="refusal", parsed_output=None)

    monkeypatch.setattr("mrs.analyst.client._call_llm_raw", fake_call)

    result = generate_explanation(_evidence())
    assert result.is_fallback is True
    assert "refusal" in result.fallback_reason


def test_generate_explanation_falls_back_on_missing_parsed_output(monkeypatch):
    def fake_call(evidence):
        return _FakeResponse(stop_reason="end_turn", parsed_output=None)

    monkeypatch.setattr("mrs.analyst.client._call_llm_raw", fake_call)

    result = generate_explanation(_evidence())
    assert result.is_fallback is True
    assert "did not return structured output" in result.fallback_reason


def test_generate_explanation_falls_back_on_hallucinated_fraud_certainty(monkeypatch):
    """The key hallucination-boundary test: a SCHEMA-VALID response (the call itself
    succeeded, recommended_action is a real bounded action) is still rejected because
    its content violates the no-fraud-certainty rule -- proving the guardrail
    inspects content, not just call success/schema shape."""
    hallucinated = _explanation(
        summary="This transaction is confirmed fraud.",
        recommended_action=ALLOW,  # schema-valid action; the violation is in the text
    )

    def fake_call(evidence):
        return _FakeResponse(stop_reason="end_turn", parsed_output=hallucinated)

    monkeypatch.setattr("mrs.analyst.client._call_llm_raw", fake_call)

    result = generate_explanation(_evidence())
    assert result.is_fallback is True
    assert "evidence-grounding check failed" in result.fallback_reason
    assert "fraud certainty" in result.fallback_reason
    # The fallback explanation replaces the hallucinated one entirely.
    assert result.explanation is not hallucinated
    assert "confirmed fraud" not in result.explanation.summary.lower()


def test_generate_explanation_never_raises_regardless_of_failure_mode(monkeypatch):
    def fake_call(evidence):
        raise ValueError("anything at all")

    monkeypatch.setattr("mrs.analyst.client._call_llm_raw", fake_call)

    # Must not raise -- the caller (the API route) relies on this.
    result = generate_explanation(_evidence())
    assert result.is_fallback is True


def test_analyst_model_id_has_no_date_suffix():
    assert ANALYST_MODEL == "claude-opus-5"
