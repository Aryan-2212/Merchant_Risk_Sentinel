import type { PolicyAction, RiskLevel } from "../../lib/types";
import { ActionBadge } from "./ActionBadge";
import { RiskBadge } from "./RiskBadge";
import "./EvidenceChain.css";

const SIGNAL_LABELS: Record<string, string> = {
  transaction_ml_risk: "Transaction ML risk",
  terminal_behavioral_risk: "Terminal behavioral risk",
  customer_behavioral_risk: "Customer behavioral risk",
};

/** contributing_signals entries look like "terminal_behavioral_risk: HIGH_RISK" or
 * "transaction_ml_risk >= 0.97" (mrs.risk.aggregate._signal_text) -- split the known
 * signal-name prefix out for a readable label, but fall back to the raw string
 * verbatim if the shape doesn't match (never silently drop evidence). */
function splitSignal(signal: string): { label: string; rest: string } {
  const sepIndex = signal.search(/[:]|(?= >=)/);
  if (sepIndex === -1) return { label: signal, rest: "" };
  const prefix = signal.slice(0, sepIndex).trim();
  const rest = signal.slice(sepIndex).replace(/^:\s*/, "").trim();
  return { label: SIGNAL_LABELS[prefix] ?? prefix, rest };
}

interface EvidenceChainProps {
  contributingSignals: string[];
  unifiedRiskLevel: RiskLevel;
  deterministicAction: PolicyAction;
}

/**
 * SIGNAL -> EVIDENCE -> RISK -> POLICY -> ACTION, built only from
 * risk_scores.contributing_signals / unified_risk_level and the policy engine's own
 * already-decided action -- no claim here is generated in frontend code.
 */
export function EvidenceChain({ contributingSignals, unifiedRiskLevel, deterministicAction }: EvidenceChainProps) {
  const hasSignals = contributingSignals.length > 0;

  return (
    <div className="chain">
      <div className="chain-node">
        <span className="chain-step">Signals</span>
        {hasSignals ? (
          <ul className="chain-signals">
            {contributingSignals.map((signal) => {
              const { label, rest } = splitSignal(signal);
              return (
                <li key={signal}>
                  <span className="chain-signal-label">{label}</span>
                  {rest && <span className="chain-signal-rest mono">{rest}</span>}
                </li>
              );
            })}
          </ul>
        ) : (
          <p className="chain-empty">
            {unifiedRiskLevel === "INSUFFICIENT_EVIDENCE"
              ? "No component signal has enough history to evaluate."
              : "No component crossed an elevated severity threshold."}
          </p>
        )}
      </div>

      <div className="chain-arrow" aria-hidden="true">
        ↓
      </div>

      <div className="chain-node chain-node-inline">
        <span className="chain-step">Unified risk</span>
        <RiskBadge level={unifiedRiskLevel} />
      </div>

      <div className="chain-arrow" aria-hidden="true">
        ↓
      </div>

      <div className="chain-node chain-node-inline">
        <span className="chain-step">Policy decision</span>
        <ActionBadge action={deterministicAction} />
      </div>
    </div>
  );
}
