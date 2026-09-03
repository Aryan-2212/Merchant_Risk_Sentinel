import { useQuery } from "@tanstack/react-query";
import { Link, useParams } from "react-router-dom";
import { api } from "../lib/api";
import { formatDateTime, formatIsoZ } from "../lib/format";
import { Loading, ErrorBlock, EmptyState } from "../components/common/States";
import { RiskBadge } from "../components/risk/RiskBadge";
import { ActionBadge, ACTION_COLOR_CLASS } from "../components/risk/ActionBadge";
import { Icon } from "../components/common/Icon";
import { BackLink } from "../components/common/BackLink";
import { behavioralFinding } from "../lib/behavioralNarrative";
import type { EntityDeviation, PolicyAction } from "../lib/types";
import "./AlertDetail.css";

interface SynthesisBullet {
  kind: "transaction" | "terminal" | "customer" | "other";
  text: string;
  detail: string | null;
  severity: number | null;
}

/** Real current-vs-baseline rate of this entity's OWN transactions landing at this
 * system's severity-2 tier (GET /terminals or /customers/{id}/deviation --
 * mrs.api.lookups.entity_deviation_rates), formatted as the compact "Current ·
 * Baseline · deviation" line. Note this is a real, computed, but DIFFERENT quantity
 * from terminal_fraud_rate_deviation (the raw Phase 3 feature the behavioral state
 * machine's own transitions threshold against, which isn't persisted per-transaction
 * and so isn't available here) -- it corroborates the state finding above (how often
 * this entity has recently been flagged our own highest severity) without claiming to
 * be the literal input that flipped its state. Returns null (never a fabricated 0%)
 * when there isn't yet a current-window rate to show, e.g. still loading or the
 * entity has no transactions in the recent window. */
function formatRateDeviation(dev: EntityDeviation | undefined): string | null {
  if (!dev || dev.current_rate === null) return null;
  const current = `${(dev.current_rate * 100).toFixed(1)}%`;
  if (dev.baseline_rate === null) {
    return `Current: ${current} high-severity (${dev.recent_window_days}d, n=${dev.current_transaction_count}) · no established baseline yet`;
  }
  const baseline = `${(dev.baseline_rate * 100).toFixed(1)}%`;
  const deltaPp = (dev.current_rate - dev.baseline_rate) * 100;
  const sign = deltaPp >= 0 ? "+" : "";
  return `Current: ${current} (${dev.recent_window_days}d) · Baseline: ${baseline} (${dev.baseline_window_days}d) · ${sign}${deltaPp.toFixed(1)}pp`;
}

/** Builds the "why risk increased" bullets from the alert's own real evidence dict
 * (mrs.policy.rules.evaluate's payload -- never invented). Each contributing_signals
 * entry looks like "transaction_ml_risk >= 0.97" or "terminal_behavioral_risk:
 * HIGH_RISK" (mrs.risk.aggregate._signal_text) -- translated into analyst-facing
 * language here, never exposing that raw syntax directly. terminal/customer rows'
 * `detail` is left null and filled in by the caller once the live deviation query
 * resolves (see formatRateDeviation), since that real metric isn't in this evidence
 * dict. */
function buildSynthesis(evidence: Record<string, unknown>): SynthesisBullet[] {
  const signals = (evidence.contributing_signals as string[] | undefined) ?? [];
  return signals.map((signal) => {
    if (signal.startsWith("transaction_ml_risk")) {
      const risk = evidence.transaction_risk as number | null;
      const threshold = evidence.transaction_risk_threshold as number | undefined;
      return {
        kind: "transaction" as const,
        text: "Elevated transaction-level ML risk detected.",
        detail:
          risk !== null && risk !== undefined && threshold !== undefined
            ? `Score: ${risk.toFixed(3)} · Threshold: ${threshold.toFixed(2)}`
            : null,
        severity: evidence.transaction_risk_severity as number | null,
      };
    }
    if (signal.startsWith("terminal_behavioral_risk")) {
      return {
        kind: "terminal" as const,
        text: behavioralFinding("Terminal", evidence.terminal_risk_state as string | null),
        detail: null,
        severity: evidence.terminal_risk_severity as number | null,
      };
    }
    if (signal.startsWith("customer_behavioral_risk")) {
      return {
        kind: "customer" as const,
        text: behavioralFinding("Customer", evidence.customer_risk_state as string | null),
        detail: null,
        severity: evidence.customer_risk_severity as number | null,
      };
    }
    return { kind: "other" as const, text: signal, detail: null, severity: null };
  });
}

function downloadEvidence(alertId: number, data: unknown) {
  const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `alert_${alertId}_evidence.json`;
  a.click();
  URL.revokeObjectURL(url);
}

/**
 * Alert investigation workspace (approved reference: alert_workspace_dark). Two
 * reference elements imply direct enforcement ("Block Terminal", "Dismiss Alert")
 * that this system cannot perform -- it is a read-only API with no write/mutate
 * endpoints, and the policy engine's action is a bounded recommendation category,
 * not a literal trigger (Dev Plan Sec 10/11). Replaced with a read-only display of
 * the real AI advisory recommendation and the real authoritative policy decision.
 * "Triggering Events Log" implied multiple raw events per alert; this system's
 * alerts are 1:1 with a single transaction (mrs.db.models.Alert.transaction_id is
 * UNIQUE), so that slot shows real recent alerts at the same terminal instead.
 */
export function AlertDetail() {
  const { id } = useParams<{ id: string }>();
  const alertId = Number(id);

  const alert = useQuery({
    queryKey: ["alert", alertId],
    queryFn: () => api.getAlert(alertId),
    enabled: Number.isFinite(alertId),
  });
  const analyst = useQuery({
    queryKey: ["analyst", alert.data?.transaction_id],
    queryFn: () => api.getTransactionAnalyst(alert.data!.transaction_id),
    enabled: alert.data !== undefined,
  });
  const related = useQuery({
    queryKey: ["alerts", "related", alert.data?.terminal_id],
    queryFn: () => api.listAlerts({ terminal_id: alert.data!.terminal_id, limit: 8 }),
    enabled: alert.data !== undefined,
  });
  const terminalDeviation = useQuery({
    queryKey: ["terminal-deviation", alert.data?.terminal_id],
    queryFn: () => api.getTerminalDeviation(alert.data!.terminal_id),
    enabled: alert.data !== undefined,
  });
  const customerDeviation = useQuery({
    queryKey: ["customer-deviation", alert.data?.customer_id],
    queryFn: () => api.getCustomerDeviation(alert.data!.customer_id),
    enabled: alert.data !== undefined,
  });

  if (!Number.isFinite(alertId)) return <EmptyState>Invalid alert ID.</EmptyState>;
  if (alert.isLoading) return <Loading label="Loading alert…" />;
  if (alert.isError) return <ErrorBlock error={alert.error} onRetry={() => alert.refetch()} />;
  const a = alert.data!;
  const bullets = buildSynthesis(a.evidence).map((b) => {
    // The deviation metric measures this entity's OWN severity-2 classification rate,
    // so it only lines up with a HIGH_RISK (severity 2) finding -- attaching it under
    // a RISK_RISING/RECOVERY (severity 1) finding would pair a severity-1 claim with a
    // severity-2 statistic, which tends to read as near-zero/contradictory even though
    // it's real data, simply because it's the wrong tier for that finding.
    if (b.kind === "terminal" && b.severity === 2) return { ...b, detail: b.detail ?? formatRateDeviation(terminalDeviation.data) };
    if (b.kind === "customer" && b.severity === 2) return { ...b, detail: b.detail ?? formatRateDeviation(customerDeviation.data) };
    return b;
  });

  return (
    <div className="page ad">
      <BackLink to="/alerts" label="Back to Alerts" />

      <div className="ad-header-row">
        <div className="ad-badges">
          <span className={`ad-severity-pill ad-severity-${a.severity.toLowerCase()}`}>SEVERITY: {a.severity}</span>
          <span className="ad-ts-pill mono">TS: {formatIsoZ(a.created_at)}</span>
        </div>
        <button className="btn" onClick={() => downloadEvidence(a.alert_id, a)}>
          <Icon name="download" size={14} /> Export Evidence
        </button>
      </div>

      <div className="ad-title-row">
        <h1 className="ad-title">ALERT_{a.alert_id}</h1>
        <span className="ad-target">
          Target:{" "}
          <Link className="link-id" to={`/terminals/${a.terminal_id}`}>
            TERM_{a.terminal_id}
          </Link>
        </span>
      </div>

      <div className="ad-grid">
        <div className="ad-main">
          <section className="card ad-synthesis">
            <h2 className="ad-panel-title">
              <Icon name="memory" size={18} className="ad-panel-icon" />
              System Synthesis: Why Risk Increased
            </h2>

            {bullets.length === 0 ? (
              <p className="ad-empty-note">No elevated component signals -- this decision was ALLOW.</p>
            ) : (
              <ul className="ad-bullets">
                {bullets.map((b, i) => (
                  <li key={i}>
                    <span className={`ad-bullet-dot ${b.severity === 2 ? "ad-dot-high" : b.severity === 1 ? "ad-dot-medium" : "ad-dot-neutral"}`} />
                    <div>
                      <p className="ad-bullet-text">{b.text}</p>
                      {b.detail && (
                        <p
                          className={`ad-bullet-detail mono ${b.severity === 2 ? "ad-bullet-detail-high" : b.severity === 1 ? "ad-bullet-detail-medium" : ""}`}
                        >
                          {b.detail}
                        </p>
                      )}
                    </div>
                  </li>
                ))}
              </ul>
            )}

            {analyst.isLoading && <p className="ad-synthesis-narrative-loading">Interpreting evidence…</p>}
            {analyst.data && (
              <div className="ad-synthesis-narrative-block">
                <span className={`ad-synthesis-narrative-label ${analyst.data.is_fallback ? "ad-synthesis-narrative-label-fallback" : ""}`}>
                  {analyst.data.is_fallback ? "Deterministic interpretation" : "AI interpretation"}
                </span>
                <p className="ad-synthesis-narrative">{analyst.data.evidence_explanation}</p>
              </div>
            )}
          </section>

          <section className="card ad-related">
            <div className="ad-related-header">
              <h2 className="ad-panel-title">Related Terminal Activity</h2>
              {related.data && <span className="ad-related-caption">Showing last {related.data.items.length} alerts</span>}
            </div>
            {related.isLoading && <Loading label="Loading related activity…" />}
            {related.isError && <ErrorBlock error={related.error} onRetry={() => related.refetch()} />}
            {related.data && related.data.items.length === 0 && <EmptyState>No other alerts at this terminal.</EmptyState>}
            {related.data && related.data.items.length > 0 && (
              <table className="data-table ad-related-table">
                <thead>
                  <tr>
                    <th>Created</th>
                    <th>Transaction</th>
                    <th>Severity</th>
                    <th>Action</th>
                    <th>Status</th>
                  </tr>
                </thead>
                <tbody>
                  {related.data.items.map((r) => (
                    <tr key={r.alert_id} data-clickable="true">
                      <td>
                        <Link className="link-id" to={`/alerts/${r.alert_id}`}>
                          {formatDateTime(r.created_at)}
                        </Link>
                      </td>
                      <td>TX_{r.transaction_id}</td>
                      <td>
                        <RiskBadge level={r.severity} size="sm" />
                      </td>
                      <td>{r.recommended_action && <ActionBadge action={r.recommended_action} />}</td>
                      <td>{r.status}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </section>
        </div>

        <div className="card ad-side">
          <h2 className="ad-panel-title">Recommended Action</h2>

          {analyst.isLoading && <Loading label="Consulting AI Risk Analyst…" />}
          {analyst.data && (
            <p className="ad-confidence">
              AI advisory confidence:{" "}
              <span className={`ad-confidence-value ${ACTION_COLOR_CLASS[analyst.data.recommended_action as PolicyAction]}`}>
                {analyst.data.confidence}
              </span>
            </p>
          )}

          <div className="ad-action-btn ad-action-btn-policy">
            <Icon name="gavel" size={16} />
            <span className="ad-action-btn-label">Policy decision: {a.recommended_action?.replaceAll("_", " ") ?? "ALLOW"}</span>
          </div>

          {analyst.data && (
            <div className={`ad-action-btn ad-action-btn-analyst ${analyst.data.is_fallback ? "ad-action-btn-fallback" : ""}`}>
              <Icon name="smart_toy" size={16} />
              <div>
                <span className="ad-action-btn-label">AI advisory: {analyst.data.recommended_action.replaceAll("_", " ")}</span>
                {analyst.data.is_fallback && <span className="ad-action-btn-sub">deterministic fallback</span>}
              </div>
            </div>
          )}
          {analyst.data && <p className="ad-advisory-rationale">{analyst.data.recommendation_rationale}</p>}

          <p className="ad-authoritative-note">The policy decision above is authoritative. The AI recommendation is advisory only.</p>

          <div className="ad-status-box">
            <span className="ad-panel-title-sm">Policy Execution Status</span>
            <div className="ad-status-content">
              <Icon name="receipt_long" size={20} className="ad-status-icon" />
              <p className="mono">
                Status: {a.status}.{a.policy_version && ` Decided under ${a.policy_version}.`}
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
