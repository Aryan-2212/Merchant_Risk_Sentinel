import { useQuery } from "@tanstack/react-query";
import { api } from "../../lib/api";
import { Loading, ErrorBlock } from "../common/States";
import { Icon } from "../common/Icon";
import "./AnalystPanel.css";

/**
 * Wraps GET /transactions/{id}/analyst -- the only analyst endpoint, not a second
 * chat/analyst implementation. Styled as a "System Log / AI Analyst Synthesis"
 * console (approved reference: terminal_investigation_dark -- DESIGN.md's "Decision
 * Console" component: monospace, terminal-output aesthetic, distinct background tint,
 * signals this is a synthesized interpretation layer, not raw database evidence).
 * Visually separates the LLM's advisory opinion from the policy engine's
 * authoritative decision (Dev Plan Sec 16/41), and never disguises a deterministic
 * fallback as a successful AI generation -- the reference's "System Log" framing
 * fits our fallback path especially well: a deterministic fallback IS a system log
 * entry, not a masked failure.
 */
export function AnalystPanel({ transactionId, actions }: { transactionId: number; actions?: React.ReactNode }) {
  const analyst = useQuery({
    queryKey: ["analyst", transactionId],
    queryFn: () => api.getTransactionAnalyst(transactionId),
  });

  if (analyst.isLoading) return <Loading label="Consulting AI Risk Analyst…" />;
  if (analyst.isError) return <ErrorBlock error={analyst.error} onRetry={() => analyst.refetch()} />;
  const data = analyst.data!;
  const severe = data.unified_risk_level === "CRITICAL";
  const critical = severe || data.unified_risk_level === "HIGH";

  return (
    <div className="analyst">
      <div className="analyst-header">
        <Icon name="memory" size={18} className={`analyst-header-icon ${data.is_fallback ? "" : "analyst-pulse"}`} />
        <span className="analyst-title">
          System Log <span className="analyst-title-sep">//</span> AI Analyst Synthesis
        </span>
        {data.is_fallback && <span className="analyst-tag-fallback">DETERMINISTIC FALLBACK</span>}
      </div>

      <div className="analyst-log">
        <p className="analyst-line">
          <span className="analyst-prompt">&gt;</span> ANALYZING TRANSACTION {transactionId}…
        </p>

        {data.is_fallback ? (
          <p className="analyst-line analyst-line-muted">
            <span className="analyst-prompt">&gt;</span> AI explanation temporarily unavailable. Showing
            system-generated evidence instead.
            {/* fallback_reason is always one of a small set of pre-approved, sanitized category
                strings (mrs.analyst.client._public_failure_reason) -- never a raw provider
                error/exception, so it's safe to surface here. */}
            {data.fallback_reason && <span className="analyst-detail"> {data.fallback_reason}</span>}
          </p>
        ) : (
          <p className="analyst-line analyst-line-tertiary">
            <span className="analyst-prompt">&gt;</span> {data.evidence_explanation}
          </p>
        )}

        <p className={`analyst-line ${severe ? "analyst-line-severe" : critical ? "analyst-line-critical" : "analyst-line-tertiary"}`}>
          <span className="analyst-prompt">&gt;</span> {data.summary}
        </p>

        <p className="analyst-line">
          <span className="analyst-prompt">&gt;</span> ADVISORY RECOMMENDATION:{" "}
          <span className="analyst-recommendation">{data.recommended_action.replaceAll("_", " ")}</span> --{" "}
          {data.recommendation_rationale}
        </p>

        {/* Fallback caveats always duplicate the "AI analyst unavailable" line + fallback_reason
            above (see mrs.analyst.client._fallback) -- only render genuine LLM-returned caveats
            here so the console doesn't repeat the same fallback reason twice. */}
        {!data.is_fallback &&
          data.caveats.map((c) => (
            <p className="analyst-line analyst-line-muted" key={c}>
              <span className="analyst-prompt">&gt;</span> {c}
            </p>
          ))}

        <p className="analyst-line analyst-cursor">
          <span className="analyst-prompt">&gt;</span> _
        </p>
      </div>

      <p className="analyst-caption">
        Advisory only — confidence: {data.confidence}. The policy decision is authoritative regardless of this
        recommendation.
      </p>

      {actions && <div className="analyst-actions">{actions}</div>}
    </div>
  );
}
