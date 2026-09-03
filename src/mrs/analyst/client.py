"""AI Risk Analyst: a single, non-agentic structured LLM call that explains an
already-computed risk assessment (Dev Plan §16/§41; Phase 8 Step 6).

    AnalystEvidence (mrs.analyst.evidence -- already-computed, read-only)
                    v
        client.models.generate_content(response_schema=AnalystExplanation)  (ONE call,
                    v                                    no tools, no loop, no agent)
        _check_evidence_grounding  (automated hallucination backstop)
                    v
        AnalystResult (explanation + is_fallback + fallback_reason)

This module never:
  - computes or modifies a risk score (mrs.models/mrs.behavioral/mrs.risk are not
    imported here, deliberately -- there is nothing for this module to recompute);
  - writes to the deterministic policy engine's decision (mrs.policy is only ever
    READ from, via the evidence it was given -- see mrs.analyst.evidence);
  - executes a tool, calls another service, or loops -- one request, one response.

Failure handling (Dev Plan §41: "If the LLM is unavailable or invalid, return
deterministic risk evidence and a safe fallback"): any exception from the API call,
a blocked/non-STOP finish_reason, a missing parsed output, or a grounding-check
violation all route to the same deterministic _fallback -- built entirely from
already-computed evidence fields, mirroring mrs.policy.rules.build_reason's own
no-fabrication discipline. The caller (mrs.api.routers.analyst) always gets a usable
AnalystExplanation and never has to handle an LLM exception itself.

Reliability (demo-hardening pass): a single transient failure (429/5xx/timeout) gets
one bounded retry (_MAX_ATTEMPTS) with a short fixed delay -- never an unbounded or
aggressive retry loop that would burn further into an already-tight daily quota. The
*public* fallback_reason returned to callers (and therefore ever shown in the UI) is
always one of a small set of pre-approved, operator-safe category strings -- never the
raw exception text, HTTP body, or provider-specific jargon (a 429 quota response
includes account-identifying billing/quota detail that must never reach the UI). The
full original exception is logged server-side (module `logger`) for developer
diagnosis; only its sanitized category crosses the function boundary.
"""

from __future__ import annotations

import logging
import os
import time

from google.genai import errors as genai_errors

from mrs.analyst.schemas import AnalystEvidence, AnalystExplanation, AnalystResult
from mrs.policy.rules import BOUNDED_ACTIONS

logger = logging.getLogger(__name__)

#: Configurable via GEMINI_MODEL so the deployed model can be changed (e.g. to work
#: around a quota-exhausted model) without a code change or redeploy. Defaults to a
#: Flash-Lite tier: cheaper/higher-quota than the default Flash tier, appropriate for
#: this single non-agentic structured call. Reads GEMINI_API_KEY or GOOGLE_API_KEY
#: from the environment (see _call_llm_raw); never hardcode credentials here.
ANALYST_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite")

#: Bounded retry for transient failures only (429/5xx/timeout/connection) -- 2 total
#: attempts, one short fixed delay between them. Deliberately NOT exponential/unbounded:
#: retrying a genuinely quota-exhausted key repeatedly would only add latency without
#: ever succeeding, and the Dev Plan explicitly calls out not making the quota problem
#: worse. Non-retryable failures (bad API key, malformed output, safety block) fail
#: immediately -- retrying those can never succeed.
_MAX_ATTEMPTS = 2
_RETRY_DELAY_SECONDS = 1.0
_RETRYABLE_HTTP_CODES = frozenset({429, 500, 502, 503, 504})

SYSTEM_PROMPT = """You are the Merchant Risk Sentinel AI Risk Analyst, a component that sits \
after a deterministic ML/behavioral risk pipeline and a deterministic policy engine (Dev Plan \
Sec 16). Both have ALREADY run and their output is provided to you below as structured \
evidence -- you are not computing risk, you are explaining it.

You will receive one JSON object describing one already-scored transaction: its unified risk \
level, transaction-ML risk score/severity, terminal and customer behavioral states/severities, \
the contributing signals that actually drove the risk level, and the deterministic policy \
action already decided.

Rules you must follow exactly:
1. Use ONLY the facts in the supplied evidence JSON. Never invent a number, date, entity, \
signal, or fact that is not present in that JSON.
2. Never assert that the transaction definitely IS or IS NOT fraudulent. You are explaining a \
risk assessment, not adjudicating fraud. Use language like "elevated risk", "consistent with", \
or "may indicate" -- never "is fraud" or "confirmed fraud".
3. Do not change, second-guess, or override the deterministic policy_action supplied in the \
evidence. You may independently RECOMMEND one of the five bounded actions (ALLOW, MONITOR, \
STEP_UP_VERIFICATION, TEMPORARY_REVIEW, ESCALATE) based on your own reading of the evidence, \
but it is advisory only -- the deterministic policy decision remains authoritative regardless \
of what you recommend.
4. If a field is null/missing in the evidence (e.g. an "unavailable" behavioral component), \
say so explicitly -- do not guess a value or treat missing as calm.
5. Keep the summary to 1-3 sentences. Be concrete about which supplied signals drove the \
assessment.
6. Write for a fraud-operations analyst, not a developer reading a debug log. The \
contributing_signals field contains raw internal syntax such as "transaction_ml_risk >= 0.97" \
or "terminal_behavioral_risk: HIGH_RISK" -- never quote that syntax verbatim in your response. \
Translate it into plain language instead, e.g. "elevated transaction-level ML risk" and \
"terminal behavioral state is currently HIGH RISK", citing the underlying score/threshold/state \
values as supporting numbers rather than the raw field names.

Respond only in the required structured format."""

#: Deliberately narrow, automatable check for the single most safety-critical
#: hallucination class this project names explicitly (Dev Plan Sec 10/16: the
#: analyst must not independently decide fraud). The system prompt carries the
#: broader no-fabrication instruction; this is the automated backstop for the one
#: rule checkable without a second LLM call.
_FRAUD_CERTAINTY_PHRASES = (
    "is fraud",
    "is fraudulent",
    "confirmed fraud",
    "definitely fraud",
    "100% fraud",
    "is a scam",
    "certainly fraudulent",
)


def _check_evidence_grounding(explanation: AnalystExplanation) -> str | None:
    """Returns a violation description, or None if the explanation passes."""
    text = f"{explanation.summary} {explanation.evidence_explanation} {explanation.recommendation_rationale}".lower()
    for phrase in _FRAUD_CERTAINTY_PHRASES:
        if phrase in text:
            return f"asserted fraud certainty ({phrase!r}), which the analyst must never do"

    # Redundant safety net alongside the Literal type on AnalystExplanation itself --
    # defends against a future schema/library change silently loosening that
    # constraint without this function being updated to match.
    if explanation.recommended_action not in BOUNDED_ACTIONS:
        return f"recommended_action {explanation.recommended_action!r} is not a bounded action"

    return None


def _describe_signal(signal: str, evidence: AnalystEvidence) -> str:
    """Translates one raw contributing_signals entry (mrs.risk.aggregate._signal_text
    syntax, e.g. "transaction_ml_risk >= 0.97") into an analyst-facing phrase for the
    deterministic fallback's evidence_explanation -- mirrors the phrasing the frontend
    itself uses for these same signals (AlertDetail.tsx behavioralFinding) so an
    analyst sees consistent language whether the LLM is available or not, and never
    sees raw field-name/operator syntax as the primary explanation."""
    if signal.startswith("transaction_ml_risk"):
        risk = evidence.transaction_risk
        threshold = evidence.transaction_risk_threshold
        if risk is not None:
            return f"elevated transaction-level ML risk (score {risk:.3f} vs threshold {threshold:.2f})"
        return "elevated transaction-level ML risk"
    if signal.startswith("terminal_behavioral_risk"):
        state = (evidence.terminal_risk_state or "unknown").replace("_", " ")
        return f"terminal behavioral state currently {state}"
    if signal.startswith("customer_behavioral_risk"):
        state = (evidence.customer_risk_state or "unknown").replace("_", " ")
        return f"customer behavioral state currently {state}"
    return signal


def _is_retryable(exc: Exception) -> bool:
    """True only for failures a second attempt might plausibly fix: rate limiting,
    server-side errors, and network/timeout errors. A bad API key, a malformed
    request, or a content-safety rejection will fail identically on retry, so those
    are deliberately excluded -- retrying them only adds latency for no benefit."""
    if isinstance(exc, genai_errors.APIError) and exc.code in _RETRYABLE_HTTP_CODES:
        return True
    return isinstance(exc, (TimeoutError, ConnectionError))


def _public_failure_reason(exc: Exception) -> str:
    """Maps any exception from the LLM call to one of a small set of pre-approved,
    operator-safe category strings -- never the raw exception text. A 429 response
    body, for example, carries account/billing/quota identifiers that must never
    reach the UI (Dev Plan §10: raw provider diagnostics are not risk intelligence).
    Callers should log the real exception themselves before calling this."""
    if isinstance(exc, genai_errors.APIError):
        if exc.code == 429:
            return "AI explanation temporarily unavailable (rate limit reached)."
        if exc.code in (401, 403):
            return "AI explanation unavailable (authentication issue)."
        if exc.code in _RETRYABLE_HTTP_CODES:
            return "AI explanation temporarily unavailable (service issue)."
        return "AI explanation temporarily unavailable."
    if isinstance(exc, ValueError) and "API key" in str(exc):
        return "AI explanation unavailable (no API key configured)."
    if isinstance(exc, TimeoutError):
        return "AI explanation temporarily unavailable (request timed out)."
    if isinstance(exc, ConnectionError):
        return "AI explanation temporarily unavailable (network error)."
    return "AI explanation temporarily unavailable."


def _call_with_retry(evidence: AnalystEvidence):
    """Up to _MAX_ATTEMPTS attempts at the real network call, retrying only
    transient failures (_is_retryable), with a short fixed delay between attempts.
    Re-raises the last exception if every attempt fails, for generate_explanation's
    own except clause to classify and log."""
    for attempt in range(1, _MAX_ATTEMPTS + 1):
        try:
            return _call_llm_raw(evidence)
        except Exception as exc:  # noqa: BLE001 -- classified by the caller
            if attempt >= _MAX_ATTEMPTS or not _is_retryable(exc):
                raise
            logger.warning("AI Risk Analyst call failed (attempt %d/%d), retrying: %r", attempt, _MAX_ATTEMPTS, exc)
            time.sleep(_RETRY_DELAY_SECONDS)


def _fallback(evidence: AnalystEvidence, *, reason: str) -> AnalystResult:
    """Deterministic explanation built entirely from already-computed evidence
    fields -- no LLM call, no risk computation. recommended_action always mirrors
    the already-decided deterministic policy action, never a different one."""
    if evidence.contributing_signals:
        phrases = [_describe_signal(s, evidence) for s in evidence.contributing_signals]
        evidence_explanation = "Risk increased due to " + "; ".join(phrases) + "."
    else:
        evidence_explanation = f"{evidence.unified_risk_level}: no elevated component signals."

    explanation = AnalystExplanation(
        summary=f"Transaction {evidence.transaction_id}: unified_risk_level={evidence.unified_risk_level}.",
        evidence_explanation=evidence_explanation,
        recommended_action=evidence.policy_action,
        recommendation_rationale=(
            "Deterministic fallback (AI Risk Analyst unavailable): mirrors the already-decided "
            "policy action rather than generating a new recommendation."
        ),
        confidence="low",
        caveats=["AI explanation unavailable; this is a deterministic fallback.", reason],
    )
    return AnalystResult(explanation=explanation, is_fallback=True, fallback_reason=reason)


def _call_llm_raw(evidence: AnalystEvidence):
    """The actual network call, isolated in its own function so tests can monkeypatch
    exactly this and nothing else (mrs.analyst.client._call_llm_raw)."""
    from google import genai
    from google.genai import types

    client = genai.Client()  # resolves GEMINI_API_KEY / GOOGLE_API_KEY from the environment
    return client.models.generate_content(
        model=ANALYST_MODEL,
        contents=evidence.model_dump_json(),
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            response_mime_type="application/json",
            response_schema=AnalystExplanation,
            # Flash-Lite tier models do not spend tokens on hidden "thinking" (unlike
            # a reasoning-tier model such as gemini-3.6-flash); the AnalystExplanation
            # JSON payload itself measured ~150-260 output tokens end to end, so 1024
            # is a comfortable margin, not a tight fit. If ANALYST_MODEL is overridden
            # via GEMINI_MODEL to a reasoning-tier model, this may need raising.
            max_output_tokens=1024,
        ),
    )


def generate_explanation(evidence: AnalystEvidence) -> AnalystResult:
    """One structured LLM call (with one bounded retry on transient failure), or a
    deterministic fallback on any failure. Never raises -- the caller always gets a
    usable AnalystResult, and fallback_reason is always one of the pre-approved
    operator-safe category strings from _public_failure_reason -- never a raw
    exception, HTTP body, or provider-specific error string (see module docstring).

    A single broad except is deliberate here, not sloppy: every failure mode below
    (network error, auth error, rate limit, malformed output) leads to the identical
    fallback behavior, so there is nothing a narrower exception chain would change.
    """
    try:
        response = _call_with_retry(evidence)
    except Exception as exc:  # noqa: BLE001 -- see docstring
        logger.warning("AI Risk Analyst call failed, using deterministic fallback: %r", exc, exc_info=True)
        return _fallback(evidence, reason=_public_failure_reason(exc))

    block_reason = getattr(response.prompt_feedback, "block_reason", None) if response.prompt_feedback else None
    if block_reason:
        logger.info("AI Risk Analyst request blocked (block_reason=%s), using deterministic fallback", block_reason)
        return _fallback(evidence, reason="AI response was blocked by a content safety filter.")

    if not response.candidates:
        logger.warning("AI Risk Analyst returned no candidates, using deterministic fallback")
        return _fallback(evidence, reason="AI explanation temporarily unavailable.")

    finish_reason = response.candidates[0].finish_reason
    if finish_reason is not None and finish_reason != "STOP":
        logger.info("AI Risk Analyst response incomplete (finish_reason=%s), using deterministic fallback", finish_reason)
        return _fallback(evidence, reason="AI response was incomplete or blocked before completion.")

    explanation = response.parsed
    if explanation is None:
        logger.warning("AI Risk Analyst returned unparseable structured output, using deterministic fallback")
        return _fallback(evidence, reason="AI response could not be validated against the expected format.")

    violation = _check_evidence_grounding(explanation)
    if violation is not None:
        logger.warning("AI Risk Analyst response failed grounding check (%s), using deterministic fallback", violation)
        return _fallback(evidence, reason=f"AI response was rejected: {violation}.")

    return AnalystResult(explanation=explanation, is_fallback=False, fallback_reason=None)
