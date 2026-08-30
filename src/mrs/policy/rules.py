"""Deterministic policy rules (Dev Plan §15; approved Phase 8 Step 3 architecture).

Pure functions only: given the already-computed Phase 7 risk-aggregation result
(never recomputed here), decide a bounded defensive action and build a reason/
evidence payload. No randomness, no I/O, no LLM. Every field this module writes into
`evidence` is copied verbatim from the caller-supplied risk_scores row -- nothing is
invented (Step 3 requirement: "Do not invent evidence").

Bounded action set (Dev Plan §15), mapped one action per unified_risk_level -- a
documented project decision (Dev Plan §33.11: the Plan lists the action vocabulary but
does not specify an exact mapping; this module is the single, isolated, versioned
place that decision lives, so it can be revisited without touching anything else):

    LOW                    -> ALLOW               no elevated component
    MEDIUM                 -> MONITOR              one component elevated, not severe
    HIGH                   -> STEP_UP_VERIFICATION exactly one severe component
    CRITICAL               -> ESCALATE             >=2 severe components, corroborated
    INSUFFICIENT_EVIDENCE  -> TEMPORARY_REVIEW      Dev Plan Sec 28: absence of evidence
                                                     is never treated as calm, so ALLOW
                                                     would be wrong; ESCALATE would
                                                     overstate a risk that is unknown,
                                                     not confirmed severe -- hold for
                                                     review instead.

The action mapping depends on unified_risk_level ONLY -- mrs.risk.aggregate has
already resolved precedence across components (severity-2-count / max-severity, Dev
Plan Sec 13/38) into that single level, so this module does not re-derive precedence;
it is deterministic and stateless by construction (same level -> same action, always).

Only alert-worthy decisions (action != ALLOW) become an `alerts` row -- see
mrs.policy.engine. Every transaction still gets a policy decision recorded in
audit_logs, whether or not it is alert-worthy (Dev Plan Sec 33.9 governance trail).
"""

from __future__ import annotations

from dataclasses import dataclass

from mrs.risk.aggregate import CRITICAL, HIGH, INSUFFICIENT_EVIDENCE, LOW, MEDIUM

ALLOW = "ALLOW"
MONITOR = "MONITOR"
STEP_UP_VERIFICATION = "STEP_UP_VERIFICATION"
TEMPORARY_REVIEW = "TEMPORARY_REVIEW"
ESCALATE = "ESCALATE"

BOUNDED_ACTIONS = frozenset({ALLOW, MONITOR, STEP_UP_VERIFICATION, TEMPORARY_REVIEW, ESCALATE})

#: Policy rules version (Dev Plan Sec 36 governance/traceability). Independent of
#: model_version/feature_version -- bump only when this mapping or reason logic
#: changes, so a stored decision stays attributable to the rules that produced it.
POLICY_VERSION = "policy_v1"

_LEVEL_TO_ACTION: dict[str, str] = {
    LOW: ALLOW,
    MEDIUM: MONITOR,
    HIGH: STEP_UP_VERIFICATION,
    CRITICAL: ESCALATE,
    INSUFFICIENT_EVIDENCE: TEMPORARY_REVIEW,
}

_KNOWN_LEVELS = frozenset(_LEVEL_TO_ACTION)

#: Fields evaluate() requires on its input dict (a risk_scores row, or an equivalent
#: mapping with the same keys).
REQUIRED_FIELDS = ("transaction_id", "unified_risk_level", "contributing_signals")

#: Every evidence field is copied from these risk_scores keys, verbatim, or is None
#: when the source row does not have it -- never fabricated.
_EVIDENCE_SOURCE_FIELDS = (
    "transaction_risk",
    "transaction_risk_severity",
    "terminal_risk_state",
    "terminal_risk_severity",
    "customer_risk_state",
    "customer_risk_severity",
    "transaction_risk_threshold",
    "model_version",
    "feature_version",
)


@dataclass(frozen=True)
class PolicyDecision:
    transaction_id: int
    unified_risk_level: str
    action: str
    reason: str
    evidence: dict
    is_alert: bool
    policy_version: str = POLICY_VERSION


def decide_action(unified_risk_level: str) -> str:
    """The one-to-one, deterministic level -> bounded-action mapping. Raises on any
    value mrs.risk.aggregate does not itself produce -- a defensive check, since a
    silently-accepted unknown level would otherwise fall through with no action."""
    if unified_risk_level not in _KNOWN_LEVELS:
        raise ValueError(f"mrs.policy.rules.decide_action: unrecognized unified_risk_level: {unified_risk_level!r}")
    return _LEVEL_TO_ACTION[unified_risk_level]


def build_reason(unified_risk_level: str, contributing_signals: list[str]) -> str:
    """A deterministic, human-readable summary built ONLY from already-computed
    contributing_signals (Dev Plan aggregate.py's own evidence) -- never invented."""
    if contributing_signals:
        return "; ".join(contributing_signals)
    if unified_risk_level == INSUFFICIENT_EVIDENCE:
        return "INSUFFICIENT_EVIDENCE: one or more risk components unavailable (insufficient history)"
    return f"{unified_risk_level}: no elevated component signals"


def evaluate(risk_score_row: dict) -> PolicyDecision:
    """risk_score_row: a mapping with at least REQUIRED_FIELDS, matching
    mrs.db.models.RiskScore's columns (or mrs.risk.aggregate.aggregate_risk's output
    row, pre-persistence, with customer_id/terminal_id attached). Computes nothing new
    about risk -- purely a deterministic function of already-computed fields.
    """
    missing = [f for f in REQUIRED_FIELDS if f not in risk_score_row]
    if missing:
        raise ValueError(f"mrs.policy.rules.evaluate: missing fields: {missing}")

    level = risk_score_row["unified_risk_level"]
    signals = list(risk_score_row["contributing_signals"] or [])
    action = decide_action(level)
    reason = build_reason(level, signals)
    evidence = {"unified_risk_level": level, "contributing_signals": signals}
    evidence.update({field: risk_score_row.get(field) for field in _EVIDENCE_SOURCE_FIELDS})

    return PolicyDecision(
        transaction_id=risk_score_row["transaction_id"],
        unified_risk_level=level,
        action=action,
        reason=reason,
        evidence=evidence,
        is_alert=action != ALLOW,
    )
