import type { RiskScoreOut } from "../../lib/types";
import { formatScore } from "../../lib/format";
import { severityColor } from "../../lib/riskColor";
import { RiskBadge } from "./RiskBadge";
import { StateBadge } from "./StateBadge";
import "./RiskDecomposition.css";

/** Three lit/unlit segments representing severity 0/1/2 -- null (unavailable) renders
 * all segments dim rather than "0 segments lit" (which would visually read as calm). */
function SeverityMeter({ severity }: { severity: number | null }) {
  const color = severityColor(severity);
  return (
    <div className="sev-meter" role="img" aria-label={severity === null ? "unavailable" : `severity ${severity} of 2`}>
      {[0, 1, 2].map((i) => (
        <span
          key={i}
          className="sev-seg"
          style={{
            background: severity !== null && i <= severity ? color : "var(--border-strong)",
            opacity: severity === null ? 0.35 : 1,
          }}
        />
      ))}
    </div>
  );
}

interface RowProps {
  label: string;
  value: React.ReactNode;
  severity: number | null;
  detail?: React.ReactNode;
}

function Row({ label, value, severity, detail }: RowProps) {
  return (
    <div className="decomp-row">
      <div className="decomp-label">{label}</div>
      <div className="decomp-value">{value}</div>
      <SeverityMeter severity={severity} />
      {detail && <div className="decomp-detail">{detail}</div>}
    </div>
  );
}

/**
 * The core MRS component: shows the independent component signals that feed
 * mrs.risk.aggregate.aggregate_risk, then the unified result -- never a single
 * "the model said X" number. Only fields the API actually returns are rendered;
 * there is deliberately no fourth "temporal" row here (no numeric temporal score
 * exists in the backend -- see EvidenceChain for how temporal/pattern evidence is
 * represented instead, via contributing_signals).
 */
export function RiskDecomposition({ risk }: { risk: RiskScoreOut }) {
  return (
    <div className="decomp">
      <div className="decomp-header">
        <h3>Risk decomposition</h3>
        <span className="decomp-caption">component signals → unified risk</span>
      </div>

      <div className="decomp-rows">
        <Row
          label="Transaction ML"
          value={<span className="mono">{formatScore(risk.transaction_risk)}</span>}
          severity={risk.transaction_risk_severity}
          detail={<span className="decomp-threshold">threshold {risk.transaction_risk_threshold.toFixed(2)}</span>}
        />
        <Row
          label="Terminal behavior"
          value={<StateBadge state={risk.terminal_risk_state} />}
          severity={risk.terminal_risk_severity}
        />
        <Row
          label="Customer behavior"
          value={<StateBadge state={risk.customer_risk_state} />}
          severity={risk.customer_risk_severity}
        />
      </div>

      <div className="decomp-result">
        <span className="decomp-result-label">Unified risk</span>
        <RiskBadge level={risk.unified_risk_level} />
      </div>
    </div>
  );
}
