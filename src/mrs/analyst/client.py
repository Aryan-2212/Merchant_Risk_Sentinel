"""AI Risk Analyst: a single, non-agentic structured LLM call that explains an
already-computed risk assessment (Dev Plan §16/§41; Phase 8 Step 6).

    AnalystEvidence (mrs.analyst.evidence -- already-computed, read-only)
                    v
        client.messages.parse(output_format=AnalystExplanation)  (ONE call, no tools,
                    v                                              no loop, no agent)
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
a refusal stop_reason, a missing parsed_output, or a grounding-check violation all
route to the same deterministic _fallback -- built entirely from already-computed
evidence fields, mirroring mrs.policy.rules.build_reason's own no-fabrication
discipline. The caller (mrs.api.routers.analyst) always gets a usable
AnalystExplanation and never has to handle an LLM exception itself.
"""

from __future__ import annotations

from mrs.analyst.schemas import AnalystEvidence, AnalystExplanation, AnalystResult
from mrs.policy.rules import BOUNDED_ACTIONS

#: ALWAYS claude-opus-5 unless a different model is explicitly requested -- no
#: date-suffixed variant, matching the current model roster.
ANALYST_MODEL = "claude-opus-5"

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


def _fallback(evidence: AnalystEvidence, *, reason: str) -> AnalystResult:
    """Deterministic explanation built entirely from already-computed evidence
    fields -- no LLM call, no risk computation. recommended_action always mirrors
    the already-decided deterministic policy action, never a different one."""
    if evidence.contributing_signals:
        evidence_explanation = "; ".join(evidence.contributing_signals)
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
    import anthropic

    client = anthropic.Anthropic()  # resolves ANTHROPIC_API_KEY from the environment
    return client.messages.parse(
        model=ANALYST_MODEL,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": evidence.model_dump_json()}],
        output_format=AnalystExplanation,
    )


def generate_explanation(evidence: AnalystEvidence) -> AnalystResult:
    """One structured LLM call, or a deterministic fallback on any failure. Never
    raises -- the caller always gets a usable AnalystResult.

    A single broad except is deliberate here, not sloppy: every failure mode below
    (network error, auth error, rate limit, refusal, malformed output) leads to the
    identical fallback behavior, so there is nothing a narrower exception chain would
    change -- the Anthropic SDK's own client already retries transient failures
    (max_retries default) before any exception reaches this function.
    """
    try:
        response = _call_llm_raw(evidence)
    except Exception as exc:  # noqa: BLE001 -- see docstring
        return _fallback(evidence, reason=f"LLM call failed: {exc.__class__.__name__}: {exc}")

    if response.stop_reason == "refusal":
        return _fallback(evidence, reason="LLM declined to respond (stop_reason=refusal)")

    explanation = response.parsed_output
    if explanation is None:
        return _fallback(evidence, reason="LLM did not return structured output")

    violation = _check_evidence_grounding(explanation)
    if violation is not None:
        return _fallback(evidence, reason=f"evidence-grounding check failed: {violation}")

    return AnalystResult(explanation=explanation, is_fallback=False, fallback_reason=None)
