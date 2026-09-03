import { useQuery } from "@tanstack/react-query";
import { Link, useParams } from "react-router-dom";
import { api } from "../lib/api";
import { useRiskHistory } from "../lib/useRiskHistory";
import { Loading, ErrorBlock, EmptyState } from "../components/common/States";
import { RiskBadge } from "../components/risk/RiskBadge";
import { StateBadge } from "../components/risk/StateBadge";
import { ZigzagTimeline } from "../components/risk/ZigzagTimeline";
import { AnalystPanel } from "../components/analyst/AnalystPanel";
import { Icon } from "../components/common/Icon";
import { BackLink } from "../components/common/BackLink";
import "./EntityDetail.css";
import "./TerminalDetail.css";

const STATE_LABEL: Record<string, string> = {
  NORMAL: "Normal",
  RISK_RISING: "Risk Rising",
  RECOVERY: "Recovery",
  HIGH_RISK: "High Risk",
  INSUFFICIENT_HISTORY: "Insufficient History",
};

function downloadAnalystLog(terminalId: number, text: string) {
  const blob = new Blob([text], { type: "text/plain" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `terminal_${terminalId}_analyst_log.txt`;
  a.click();
  URL.revokeObjectURL(url);
}

/**
 * Terminal Investigation (approved reference: terminal_investigation_dark). Two
 * reference elements conflict with this system's real safety boundaries and are
 * deliberately NOT reproduced as literal interactive controls:
 * - "Isolate Terminal" implied an executable enforcement action; this system has no
 *   such capability and the AI may not execute actions (Dev Plan Sec 10/41). Replaced
 *   with a read-only display of the actual deterministic policy decision.
 * - "Fraud Rate" implied a raw ground-truth fraud percentage; replaced with this
 *   system's own computed terminal_risk_severity rate (never tx_fraud).
 */
export function TerminalDetail() {
  const { id } = useParams<{ id: string }>();
  const terminalId = Number(id);

  const terminal = useQuery({
    queryKey: ["terminal", terminalId],
    queryFn: () => api.getTerminal(terminalId),
    enabled: Number.isFinite(terminalId),
  });
  const deviation = useQuery({
    queryKey: ["terminal-deviation", terminalId],
    queryFn: () => api.getTerminalDeviation(terminalId),
    enabled: Number.isFinite(terminalId),
  });
  const history = useRiskHistory(terminalId, api.getTerminalRiskHistory, "terminal-risk");

  const { items, total, offset, setOffset, pageSize } = history;
  const onLastPage = offset + items.length >= total;
  const latest = onLastPage ? items[items.length - 1] : undefined;
  const latestTxId = latest?.transaction_id;

  const latestDetail = useQuery({
    queryKey: ["transaction", latestTxId],
    queryFn: () => api.getTransaction(latestTxId!),
    enabled: latestTxId !== undefined,
  });
  const latestAnalyst = useQuery({
    queryKey: ["analyst", latestTxId],
    queryFn: () => api.getTransactionAnalyst(latestTxId!),
    enabled: latestTxId !== undefined,
  });

  if (!Number.isFinite(terminalId)) return <EmptyState>Invalid terminal ID.</EmptyState>;
  if (terminal.isLoading) return <Loading label="Loading terminal…" />;
  if (terminal.isError) return <ErrorBlock error={terminal.error} onRetry={() => terminal.refetch()} />;

  const t = terminal.data!;
  const dev = deviation.data;
  const currentRatePct = dev?.current_rate !== null && dev?.current_rate !== undefined ? dev.current_rate * 100 : null;
  const deviationPp =
    dev?.current_rate !== null && dev?.current_rate !== undefined && dev?.baseline_rate !== null && dev?.baseline_rate !== undefined
      ? (dev.current_rate - dev.baseline_rate) * 100
      : null;

  const currentVelocity = dev ? dev.current_transaction_count / dev.recent_window_days : null;
  const baselineVelocity = dev ? dev.baseline_transaction_count / dev.baseline_window_days : null;
  const velocityRatio = currentVelocity !== null && baselineVelocity !== null && baselineVelocity > 0 ? currentVelocity / baselineVelocity : null;

  const deterministicAction = latestDetail.data?.alert?.recommended_action ?? "ALLOW";
  const isCritical = latest?.unified_risk_level === "CRITICAL";

  return (
    <div className="page td">
      <BackLink to="/terminals" label="Back to Terminals" />

      <header className="td-header">
        <div>
          <h1 className="td-title">
            <Icon name="terminal" size={22} className="td-title-icon" />
            Terminal Investigation
          </h1>
          <p className="td-subtitle mono">
            ID: TERM_{t.terminal_id} | Status:{" "}
            <span
              className={
                latest?.terminal_risk_state === "HIGH_RISK" ? (isCritical ? "td-subtitle-severe" : "td-subtitle-critical") : ""
              }
            >
              {latest?.terminal_risk_state ? STATE_LABEL[latest.terminal_risk_state].toUpperCase() : "UNSCORED"}
            </span>
          </p>
        </div>
        <div className="td-score-box">
          <div className="td-score-col">
            <span className="field-label">Current elevated rate</span>
            <div
              className={`td-score ${
                currentRatePct !== null && currentRatePct >= 30 ? (isCritical ? "td-score-severe" : "td-score-critical") : ""
              }`}
            >
              {currentRatePct === null ? "—" : currentRatePct.toFixed(0)}
            </div>
          </div>
          <div className="td-score-bar-col">
            <div className="td-score-bar-labels">
              <span>Low</span>
              <span className="td-score-bar-labels-high">High</span>
            </div>
            <div className="td-score-bar-track">
              <div className="td-score-bar-fill" style={{ width: `${currentRatePct ?? 0}%` }} />
            </div>
          </div>
        </div>
      </header>

      <div className="td-bento">
        <section className="card td-timeline-card">
          <h2 className="td-section-title">
            <Icon name="history" size={18} className="td-section-icon" />
            Risk Timeline
          </h2>
          {history.isLoading && <Loading label="Loading risk history…" />}
          {history.isError && <ErrorBlock error={history.error} onRetry={history.refetch} />}
          {items.length === 0 && !history.isLoading && <EmptyState>Insufficient historical baseline.</EmptyState>}
          {items.length > 0 && (
            <ZigzagTimeline points={items.map((r) => ({ transactionId: r.transaction_id, state: r.terminal_risk_state }))} />
          )}
        </section>

        <div className="td-right-stack">
          <section className="card td-evidence-card">
            <div className="td-evidence-header">
              <h2 className="td-section-title">
                <Icon name="monitoring" size={18} className="td-section-icon" />
                Behavioral Evidence
              </h2>
              <span className="td-evidence-caption">Current vs Baseline</span>
            </div>

            {deviation.isLoading && <Loading label="Loading behavioral evidence…" />}
            {deviation.isError && <ErrorBlock error={deviation.error} onRetry={() => deviation.refetch()} />}

            {dev && (
              <div className="td-metrics">
                <div className="td-metric">
                  <div className="td-metric-stripe td-metric-stripe-tertiary" />
                  <span className="field-label">Transaction Velocity</span>
                  <div className="td-metric-row">
                    <span className="td-metric-value">{velocityRatio === null ? "—" : `${velocityRatio.toFixed(1)}x`}</span>
                    {velocityRatio !== null && (
                      <span className="td-metric-tag td-metric-tag-tertiary">
                        <Icon name={velocityRatio >= 1 ? "trending_up" : "trending_down"} size={14} />
                        {velocityRatio >= 1 ? "Above" : "Below"} Baseline
                      </span>
                    )}
                  </div>
                  <div className="td-metric-footer">
                    <span>Baseline: {baselineVelocity === null ? "—" : `${baselineVelocity.toFixed(1)}/day`}</span>
                    <span>Current: {currentVelocity === null ? "—" : `${currentVelocity.toFixed(1)}/day`}</span>
                  </div>
                </div>

                <div className={`td-metric ${deviationPp !== null && deviationPp > 5 && isCritical ? "td-metric-glow" : ""}`}>
                  <div
                    className={`td-metric-stripe ${
                      deviationPp !== null && deviationPp > 0
                        ? isCritical
                          ? "td-metric-stripe-severe"
                          : "td-metric-stripe-critical"
                        : "td-metric-stripe-tertiary"
                    }`}
                  />
                  <span className="field-label">Severity Rate Change</span>
                  <div className="td-metric-row">
                    <span
                      className={`td-metric-value ${
                        deviationPp !== null && deviationPp > 0 ? (isCritical ? "td-metric-value-severe" : "td-metric-value-critical") : ""
                      }`}
                    >
                      {deviationPp === null ? "—" : `${deviationPp >= 0 ? "+" : ""}${deviationPp.toFixed(1)}pp`}
                    </span>
                    {deviationPp !== null && deviationPp > 5 && (
                      <span className={`td-metric-tag ${isCritical ? "td-metric-tag-severe" : "td-metric-tag-critical"}`}>
                        <Icon name="priority_high" size={14} />
                        Critical
                      </span>
                    )}
                  </div>
                  <div className="td-metric-footer">
                    <span>Baseline: {dev.baseline_rate === null ? "—" : `${(dev.baseline_rate * 100).toFixed(1)}%`}</span>
                    <span>Current: {dev.current_rate === null ? "—" : `${(dev.current_rate * 100).toFixed(1)}%`}</span>
                  </div>
                </div>
              </div>
            )}
          </section>

          {latestTxId !== undefined ? (
            <AnalystPanel
              transactionId={latestTxId}
              actions={
                <>
                  {/* Same visual slot/weight as the reference's "Isolate Terminal" button,
                      but this system has no enforcement capability and the AI may not
                      execute actions -- so this is a read-only display of the actual
                      deterministic policy decision, not a clickable control. */}
                  <div className={`td-policy-btn td-policy-btn-${deterministicAction.toLowerCase()}`}>
                    <Icon name="shield" size={16} />
                    Policy: {deterministicAction.replaceAll("_", " ")}
                  </div>
                  <button
                    className="btn"
                    disabled={!latestAnalyst.data}
                    onClick={() => {
                      if (!latestAnalyst.data) return;
                      const a = latestAnalyst.data;
                      const log = [
                        `Terminal TERM_${terminalId} -- AI Risk Analyst log`,
                        `Transaction: TX_${latestTxId}`,
                        `Unified risk: ${a.unified_risk_level}`,
                        `Deterministic policy action: ${a.deterministic_action}`,
                        `Fallback: ${a.is_fallback}${a.fallback_reason ? ` (${a.fallback_reason})` : ""}`,
                        "",
                        `Evidence: ${a.evidence_explanation}`,
                        `Assessment: ${a.summary}`,
                        `Advisory recommendation: ${a.recommended_action} -- ${a.recommendation_rationale}`,
                        `Confidence: ${a.confidence}`,
                        a.caveats.length ? `Caveats: ${a.caveats.join("; ")}` : "",
                      ]
                        .filter(Boolean)
                        .join("\n");
                      downloadAnalystLog(terminalId, log);
                    }}
                  >
                    <Icon name="download" size={14} /> Export Log
                  </button>
                </>
              }
            />
          ) : (
            <EmptyState>No scored transactions yet -- nothing for the AI analyst to explain.</EmptyState>
          )}
        </div>
      </div>

      {items.length > 0 && (
        <div className="section">
          <span className="section-title">Scored transactions</span>
          <div className="table-scroll">
            <table className="data-table">
              <thead>
                <tr>
                  <th>Transaction</th>
                  <th>Unified risk</th>
                  <th>Terminal state</th>
                  <th>Customer</th>
                </tr>
              </thead>
              <tbody>
                {[...items].reverse().map((r) => (
                  <tr key={r.transaction_id}>
                    <td>
                      <Link className="link-id" to={`/transactions/${r.transaction_id}`}>
                        TX_{r.transaction_id}
                      </Link>
                    </td>
                    <td>
                      <RiskBadge level={r.unified_risk_level} size="sm" />
                    </td>
                    <td>
                      <StateBadge state={r.terminal_risk_state} />
                    </td>
                    <td>
                      <Link className="link-id" to={`/customers/${r.customer_id}`}>
                        CUST_{r.customer_id}
                      </Link>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div className="pagination">
            <span>
              {total.toLocaleString()} total scored transaction{total === 1 ? "" : "s"}
            </span>
            <button className="btn" disabled={offset === 0} onClick={() => setOffset(Math.max(0, offset - pageSize))}>
              Older
            </button>
            <button className="btn" disabled={offset + pageSize >= total} onClick={() => setOffset(offset + pageSize)}>
              Newer
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
