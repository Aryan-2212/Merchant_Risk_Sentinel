"""Tests for mrs.analyst -- Phase 8 Step 6 AI Risk Analyst (Dev Plan §16/§41).

No test in this module makes a real network call: mrs.analyst.client._call_llm_raw is
monkeypatched to a stub in every generate_explanation test, so these run without a
GEMINI_API_KEY and without network access. That is deliberate -- it lets us
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
    evidence = _evidence(terminal_risk_state="HIGH_RISK", contributing_signals=["terminal_behavioral_risk: HIGH_RISK"])
    result = _fallback(evidence, reason="x")
    # Human-readable, grounded in the supplied evidence -- never the raw
    # "field_name: STATE" / "field_name >= threshold" signal syntax verbatim.
    assert result.explanation.evidence_explanation == "Risk increased due to terminal behavioral state currently HIGH RISK."
    assert "terminal_behavioral_risk:" not in result.explanation.evidence_explanation


def test_fallback_handles_insufficient_evidence_level():
    evidence = _evidence(unified_risk_level=INSUFFICIENT_EVIDENCE, policy_action="TEMPORARY_REVIEW", contributing_signals=[])
    result = _fallback(evidence, reason="x")
    assert result.explanation.recommended_action == "TEMPORARY_REVIEW"
    assert "no elevated component signals" in result.explanation.evidence_explanation


# ------------------------------------------------------------------- generate_explanation


class _FakeCandidate:
    def __init__(self, *, finish_reason="STOP"):
        self.finish_reason = finish_reason


class _FakeResponse:
    def __init__(self, *, finish_reason="STOP", parsed=None, block_reason=None, candidates=True):
        self.candidates = [_FakeCandidate(finish_reason=finish_reason)] if candidates else []
        self.parsed = parsed
        self.prompt_feedback = _FakePromptFeedback(block_reason) if block_reason else None


class _FakePromptFeedback:
    def __init__(self, block_reason):
        self.block_reason = block_reason


def test_generate_explanation_success_path(monkeypatch):
    good = _explanation(summary="Elevated risk driven by terminal behavior.")

    def fake_call(evidence):
        return _FakeResponse(finish_reason="STOP", parsed=good)

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
    # Never the raw exception class/message -- only the pre-approved public category.
    assert result.fallback_reason == "AI explanation temporarily unavailable."
    assert "RuntimeError" not in result.fallback_reason
    assert "simulated network failure" not in result.fallback_reason


def test_generate_explanation_falls_back_on_refusal(monkeypatch):
    def fake_call(evidence):
        return _FakeResponse(finish_reason="SAFETY", parsed=None)

    monkeypatch.setattr("mrs.analyst.client._call_llm_raw", fake_call)

    result = generate_explanation(_evidence())
    assert result.is_fallback is True
    assert result.fallback_reason == "AI response was incomplete or blocked before completion."
    assert "SAFETY" not in result.fallback_reason
    assert "finish_reason" not in result.fallback_reason


def test_generate_explanation_falls_back_on_blocked_prompt(monkeypatch):
    def fake_call(evidence):
        return _FakeResponse(candidates=False, block_reason="SAFETY")

    monkeypatch.setattr("mrs.analyst.client._call_llm_raw", fake_call)

    result = generate_explanation(_evidence())
    assert result.is_fallback is True
    assert result.fallback_reason == "AI response was blocked by a content safety filter."
    assert "SAFETY" not in result.fallback_reason
    assert "block_reason" not in result.fallback_reason


def test_generate_explanation_falls_back_on_missing_parsed_output(monkeypatch):
    def fake_call(evidence):
        return _FakeResponse(finish_reason="STOP", parsed=None)

    monkeypatch.setattr("mrs.analyst.client._call_llm_raw", fake_call)

    result = generate_explanation(_evidence())
    assert result.is_fallback is True
    assert result.fallback_reason == "AI response could not be validated against the expected format."


def test_generate_explanation_falls_back_on_invalid_structured_action(monkeypatch):
    """A schema-bypassing invalid recommended_action (simulating a malformed/invalid
    structured response from the provider) must fall back, not crash or propagate."""
    bad = AnalystExplanation.model_construct(
        summary="ok",
        evidence_explanation="ok",
        recommended_action="DENY_TRANSACTION",
        recommendation_rationale="ok",
        confidence="high",
        caveats=[],
    )

    def fake_call(evidence):
        return _FakeResponse(finish_reason="STOP", parsed=bad)

    monkeypatch.setattr("mrs.analyst.client._call_llm_raw", fake_call)

    result = generate_explanation(_evidence())
    assert result.is_fallback is True
    assert result.explanation.recommended_action == "ALLOW"  # mirrors evidence.policy_action (default ALLOW)
    # The violation detail is human-composed (mrs.analyst.client._check_evidence_grounding),
    # not a raw provider exception, so surfacing it here is transparency, not a diagnostics leak.
    assert "not a bounded action" in result.fallback_reason


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
        return _FakeResponse(finish_reason="STOP", parsed=hallucinated)

    monkeypatch.setattr("mrs.analyst.client._call_llm_raw", fake_call)

    result = generate_explanation(_evidence())
    assert result.is_fallback is True
    assert "AI response was rejected" in result.fallback_reason
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


def test_analyst_model_defaults_to_flash_lite():
    assert ANALYST_MODEL == "gemini-3.5-flash-lite"


def test_analyst_model_configurable_via_env_var(monkeypatch):
    """GEMINI_MODEL must be able to override the default without a code change --
    ANALYST_MODEL is read at import time, so this reloads the module under a
    patched environment rather than asserting against the already-imported constant."""
    import importlib

    import mrs.analyst.client as client_module

    monkeypatch.setenv("GEMINI_MODEL", "gemini-2.5-pro")
    try:
        reloaded = importlib.reload(client_module)
        assert reloaded.ANALYST_MODEL == "gemini-2.5-pro"
    finally:
        monkeypatch.delenv("GEMINI_MODEL", raising=False)
        importlib.reload(client_module)  # restore the default for any test that runs after this one


# --------------------------------------------------------------------------- resilience


class _FakeAPIError(Exception):
    """Stands in for google.genai.errors.APIError/ClientError/ServerError without
    constructing the real thing (which needs a response_json shape) -- _is_retryable
    and _public_failure_reason only ever look at .code, so this is a faithful double."""

    def __init__(self, code: int, message: str = "simulated provider error detail"):
        self.code = code
        super().__init__(message)


@pytest.fixture(autouse=True)
def _no_real_sleep(monkeypatch):
    """Every retry test in this module exercises the real _RETRY_DELAY_SECONDS delay
    path; patch it to 0 so the suite doesn't actually sleep for it."""
    monkeypatch.setattr("mrs.analyst.client.time.sleep", lambda seconds: None)


def _patch_error_type(monkeypatch, cls: type[Exception]):
    """_is_retryable/_public_failure_reason check isinstance(exc, genai_errors.APIError);
    patch that reference so _FakeAPIError satisfies it without depending on the real
    google.genai.errors.APIError constructor signature."""
    monkeypatch.setattr("mrs.analyst.client.genai_errors.APIError", cls)


def test_gemini_429_rate_limit_retries_then_falls_back_with_sanitized_reason(monkeypatch):
    monkeypatch.setattr("mrs.analyst.client.genai_errors.APIError", _FakeAPIError)
    calls = []

    def fake_call(evidence):
        calls.append(1)
        raise _FakeAPIError(429, "429 RESOURCE_EXHAUSTED. quota exceeded, retry after 47 seconds")

    monkeypatch.setattr("mrs.analyst.client._call_llm_raw", fake_call)

    result = generate_explanation(_evidence())
    assert result.is_fallback is True
    assert result.fallback_reason == "AI explanation temporarily unavailable (rate limit reached)."
    # Critical rule: never expose the raw provider diagnostic text.
    assert "RESOURCE_EXHAUSTED" not in result.fallback_reason
    assert "quota" not in result.fallback_reason
    assert "retry after" not in result.fallback_reason
    # Bounded retry: exactly _MAX_ATTEMPTS attempts, not one, not unbounded.
    assert len(calls) == 2


def test_gemini_quota_exhaustion_is_handled_gracefully(monkeypatch):
    """Daily quota exhaustion surfaces as the same 429 category as rate limiting from
    this module's point of view (Gemini itself returns 429 RESOURCE_EXHAUSTED for
    both) -- the system must still degrade to a usable fallback either way."""
    monkeypatch.setattr("mrs.analyst.client.genai_errors.APIError", _FakeAPIError)

    def fake_call(evidence):
        raise _FakeAPIError(429, "GenerateRequestsPerDayPerProjectPerModel-FreeTier quota exceeded")

    monkeypatch.setattr("mrs.analyst.client._call_llm_raw", fake_call)

    result = generate_explanation(_evidence())
    assert result.is_fallback is True
    assert result.fallback_reason == "AI explanation temporarily unavailable (rate limit reached)."
    assert "FreeTier" not in result.fallback_reason
    assert result.explanation.recommended_action == ALLOW  # policy decision still produced


def test_gemini_transient_failure_succeeds_on_retry(monkeypatch):
    """Proves the retry path isn't just bookkeeping -- a transient failure that
    clears on the second attempt must produce a genuine (non-fallback) result."""
    monkeypatch.setattr("mrs.analyst.client.genai_errors.APIError", _FakeAPIError)
    good = _explanation(summary="Recovered after one retry.")
    calls = []

    def fake_call(evidence):
        calls.append(1)
        if len(calls) == 1:
            raise _FakeAPIError(503, "server temporarily overloaded")
        return _FakeResponse(finish_reason="STOP", parsed=good)

    monkeypatch.setattr("mrs.analyst.client._call_llm_raw", fake_call)

    result = generate_explanation(_evidence())
    assert result.is_fallback is False
    assert result.explanation is good
    assert len(calls) == 2


def test_gemini_timeout_is_retried_then_falls_back(monkeypatch):
    calls = []

    def fake_call(evidence):
        calls.append(1)
        raise TimeoutError("the request took too long")

    monkeypatch.setattr("mrs.analyst.client._call_llm_raw", fake_call)

    result = generate_explanation(_evidence())
    assert result.is_fallback is True
    assert result.fallback_reason == "AI explanation temporarily unavailable (request timed out)."
    assert "took too long" not in result.fallback_reason
    assert len(calls) == 2  # timeout is retryable


def test_gemini_network_failure_is_retried_then_falls_back(monkeypatch):
    calls = []

    def fake_call(evidence):
        calls.append(1)
        raise ConnectionError("DNS resolution failed for generativelanguage.googleapis.com")

    monkeypatch.setattr("mrs.analyst.client._call_llm_raw", fake_call)

    result = generate_explanation(_evidence())
    assert result.is_fallback is True
    assert result.fallback_reason == "AI explanation temporarily unavailable (network error)."
    assert "DNS" not in result.fallback_reason
    assert "googleapis.com" not in result.fallback_reason
    assert len(calls) == 2  # network errors are retryable


def test_missing_api_key_is_not_retried_and_handled_gracefully(monkeypatch):
    calls = []

    def fake_call(evidence):
        calls.append(1)
        raise ValueError(
            "No API key was provided. Please pass a valid API key. "
            "Learn how to create an API key at https://ai.google.dev/gemini-api/docs/api-key."
        )

    monkeypatch.setattr("mrs.analyst.client._call_llm_raw", fake_call)

    result = generate_explanation(_evidence())
    assert result.is_fallback is True
    assert result.fallback_reason == "AI explanation unavailable (no API key configured)."
    assert "ai.google.dev" not in result.fallback_reason
    # Retrying a missing API key can never succeed -- must fail fast, not retry.
    assert len(calls) == 1


def test_invalid_api_configuration_is_not_retried(monkeypatch):
    """A 401/403 (bad/revoked key, wrong project) is an auth problem, not a transient
    one -- retrying it wastes a call for no chance of success."""
    monkeypatch.setattr("mrs.analyst.client.genai_errors.APIError", _FakeAPIError)
    calls = []

    def fake_call(evidence):
        calls.append(1)
        raise _FakeAPIError(403, "PERMISSION_DENIED: API key not authorized for this project")

    monkeypatch.setattr("mrs.analyst.client._call_llm_raw", fake_call)

    result = generate_explanation(_evidence())
    assert result.is_fallback is True
    assert result.fallback_reason == "AI explanation unavailable (authentication issue)."
    assert "PERMISSION_DENIED" not in result.fallback_reason
    assert len(calls) == 1


def test_unexpected_exception_never_leaks_raw_text(monkeypatch):
    """A catch-all: whatever kind of exception the SDK might one day raise, the public
    fallback_reason must never contain the exception's own message or class name --
    this is the single most important assertion in this file (Dev Plan §10)."""
    secret_looking_detail = "google.api_core.exceptions.InternalServerError: upstream 502 at 10.0.4.17"

    def fake_call(evidence):
        raise Exception(secret_looking_detail)  # noqa: TRY002 -- deliberately generic/unclassified

    monkeypatch.setattr("mrs.analyst.client._call_llm_raw", fake_call)

    result = generate_explanation(_evidence())
    assert result.is_fallback is True
    assert secret_looking_detail not in result.fallback_reason
    assert "10.0.4.17" not in result.fallback_reason
    assert "google.api_core" not in result.fallback_reason
    for caveat in result.explanation.caveats:
        assert secret_looking_detail not in caveat


def test_risk_system_and_policy_decision_unaffected_by_ai_failure(monkeypatch):
    """The AI layer is explanation-only: whatever it does, the deterministic
    policy_action already decided must pass through to the fallback unchanged."""

    def fake_call(evidence):
        raise ConnectionError("simulated total AI outage")

    monkeypatch.setattr("mrs.analyst.client._call_llm_raw", fake_call)

    evidence = _evidence(unified_risk_level=CRITICAL, policy_action=ESCALATE, contributing_signals=["x"])
    result = generate_explanation(evidence)

    assert result.is_fallback is True
    # The deterministic policy decision is untouched by the AI outage.
    assert result.explanation.recommended_action == ESCALATE
