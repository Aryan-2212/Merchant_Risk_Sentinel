import { useQuery } from "@tanstack/react-query";
import { Link, useParams } from "react-router-dom";
import { api } from "../lib/api";
import { formatAmount } from "../lib/format";
import { useRiskHistory } from "../lib/useRiskHistory";
import { behavioralFinding } from "../lib/behavioralNarrative";
import { Loading, ErrorBlock, EmptyState } from "../components/common/States";
import { RiskBadge } from "../components/risk/RiskBadge";
import { StateBadge } from "../components/risk/StateBadge";
import { BehavioralTimeline } from "../components/risk/BehavioralTimeline";
import { Icon } from "../components/common/Icon";
import { BackLink } from "../components/common/BackLink";
import "./EntityDetail.css";
import "./CustomerDetail.css";

const STATE_LABEL: Record<string, string> = {
  NORMAL: "Normal",
  RISK_RISING: "Risk Rising",
  RECOVERY: "Recovery",
  HIGH_RISK: "High Risk",
  INSUFFICIENT_HISTORY: "Insufficient History",
};

function downloadEvidence(customerId: number, data: unknown) {
  const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `customer_${customerId}_evidence.json`;
  a.click();
  URL.revokeObjectURL(url);
}

/**
 * Customer Investigation (approved reference: customer_investigation_dark). Several
 * reference elements have no real backend equivalent in this system and are
 * deliberately NOT reproduced as literal fabricated data:
 * - Contact info (email/phone) and "Account Age" -- the Handbook dataset carries no
 *   customer PII or account-creation date at all (Dev Plan Sec 2: simulated benchmark
 *   data only). Omitted rather than invented.
 * - "Freeze Account" / "Resolve Case" implied executable case-management actions this
 *   read-only API cannot perform (no write/mutate endpoints exist -- same boundary as
 *   AlertDetail's "Block Terminal"/"Dismiss Alert" and TerminalDetail's "Isolate
 *   Terminal"). Replaced with a real, working "Export Evidence" download, the same
 *   pattern AlertDetail already uses for its own real data.
 * - "Chargeback Rate" -- this dataset has no chargeback concept. The equivalent real,
 *   computed metric is this customer's own severity-2 rate vs its baseline (GET
 *   /customers/{id}/deviation -- mrs.api.lookups.entity_deviation_rates), the same
 *   quantity TerminalDetail's "Current Elevated Rate" / "Severity Rate Change" cards
 *   already use for terminals -- reused here for a customer, never invented anew.
 * - "Linked Methods" (cards) and the three-item CVV/IP/shipping checklist -- no
 *   payment-method or device/IP data exists in this dataset. Omitted; the reference's
 *   "Behavioral Evidence" panel instead shows this customer's real behavioral-state
 *   timeline (BehavioralTimeline, already used on this page).
 * - "4 Devices / Across 2 regions" -- no device/session data exists. The nearest real,
 *   already-available equivalent is nb_terminals (how many distinct terminals this
 *   customer has transacted at), shown honestly as "Terminals used".
 */
export function CustomerDetail() {
  const { id } = useParams<{ id: string }>();
  const customerId = Number(id);

  const customer = useQuery({
    queryKey: ["customer", customerId],
    queryFn: () => api.getCustomer(customerId),
    enabled: Number.isFinite(customerId),
  });
  const deviation = useQuery({
    queryKey: ["customer-deviation", customerId],
    queryFn: () => api.getCustomerDeviation(customerId),
    enabled: Number.isFinite(customerId),
  });
  const history = useRiskHistory(customerId, api.getCustomerRiskHistory, "customer-risk");

  if (!Number.isFinite(customerId)) return <EmptyState>Invalid customer ID.</EmptyState>;
  if (customer.isLoading) return <Loading label="Loading customer…" />;
  if (customer.isError) return <ErrorBlock error={customer.error} onRetry={() => customer.refetch()} />;

  const c = customer.data!;
  const { items, total, offset, setOffset, pageSize } = history;
  // items are ordered oldest-first within the current (most-recent-by-default) page,
  // so the last item is the most recent state only when this is the final page.
  const onLastPage = offset + items.length >= total;
  const current = onLastPage ? items[items.length - 1] : undefined;
  const state = current?.customer_risk_state;
  const isCritical = current?.unified_risk_level === "CRITICAL";
  const isElevated = state === "HIGH_RISK" || state === "RISK_RISING";

  const dev = deviation.data;
  const currentRatePct = dev?.current_rate !== null && dev?.current_rate !== undefined ? dev.current_rate * 100 : null;
  const deviationPp =
    dev?.current_rate !== null && dev?.current_rate !== undefined && dev?.baseline_rate !== null && dev?.baseline_rate !== undefined
      ? (dev.current_rate - dev.baseline_rate) * 100
      : null;

  return (
    <div className="page cd">
      <BackLink to="/customers" label="Back to Customers" />

      <header className="cd-header">
        <div>
          <h1 className="cd-title">
            <Icon name="group" size={22} className="cd-title-icon" />
            Customer Investigation
          </h1>
          <p className="cd-subtitle mono">
            ID: CUST_{c.customer_id} | Status:{" "}
            <span className={state === "HIGH_RISK" ? (isCritical ? "cd-subtitle-severe" : "cd-subtitle-critical") : ""}>
              {state ? STATE_LABEL[state].toUpperCase() : "UNSCORED"}
            </span>
            {isElevated && (
              <span className={`cd-flag-badge ${isCritical ? "cd-flag-badge-severe" : ""}`}>
                <Icon name="warning" size={12} /> Flagged
              </span>
            )}
          </p>
        </div>
        <button className="btn" onClick={() => downloadEvidence(c.customer_id, { customer: c, deviation: dev, recent_history: items })}>
          <Icon name="download" size={14} /> Export Evidence
        </button>
      </header>

      <div className="cd-kpi-row">
        <div className="cd-kpi-card">
          <span className="field-label">Current elevated rate</span>
          <div className={`cd-kpi-value ${currentRatePct !== null && currentRatePct >= 30 ? (isCritical ? "cd-kpi-value-severe" : "cd-kpi-value-critical") : ""}`}>
            {currentRatePct === null ? "—" : currentRatePct.toFixed(0)}
            <span className="cd-kpi-value-suffix">/100</span>
          </div>
          <div className="cd-kpi-bar-track">
            <div
              className={`cd-kpi-bar-fill ${currentRatePct !== null && currentRatePct >= 30 ? (isCritical ? "cd-kpi-bar-fill-severe" : "cd-kpi-bar-fill-critical") : ""}`}
              style={{ width: `${currentRatePct ?? 0}%` }}
            />
          </div>
        </div>

        <div className="cd-kpi-card">
          <span className="field-label">Baseline avg. amount</span>
          <div className="cd-kpi-value mono">{formatAmount(c.mean_amount)}</div>
          <span className="cd-kpi-footnote">{c.mean_nb_tx_per_day.toFixed(2)} tx/day avg.</span>
        </div>

        <div className={`cd-kpi-card ${deviationPp !== null && deviationPp > 5 ? (isCritical ? "cd-kpi-card-severe" : "cd-kpi-card-critical") : ""}`}>
          <span className="field-label">Severity rate change</span>
          <div className={`cd-kpi-value ${deviationPp !== null && deviationPp > 0 ? (isCritical ? "cd-kpi-value-severe" : "cd-kpi-value-critical") : ""}`}>
            {deviationPp === null ? "—" : `${deviationPp >= 0 ? "+" : ""}${deviationPp.toFixed(1)}pp`}
          </div>
          <span className="cd-kpi-footnote">
            {deviationPp !== null && deviationPp > 5 ? (
              <span className="cd-kpi-warn">
                <Icon name="priority_high" size={12} /> vs 30-day baseline
              </span>
            ) : (
              "vs 30-day baseline"
            )}
          </span>
        </div>

        <div className="cd-kpi-card">
          <span className="field-label">Terminals used</span>
          <div className="cd-kpi-value mono">{c.nb_terminals}</div>
          <span className="cd-kpi-footnote">across scored history</span>
        </div>
      </div>

      <div className={`cd-risk-summary ${isElevated ? (isCritical ? "cd-risk-summary-severe" : "cd-risk-summary-critical") : ""}`}>
        <Icon name={isElevated ? "error" : "info"} size={20} className="cd-risk-summary-icon" />
        <div className="cd-risk-summary-body">
          <h2 className="cd-risk-summary-title">Risk Summary</h2>
          <p>{behavioralFinding("Customer", state)}</p>
        </div>
        <a href="#cd-activity" className="btn">
          View Details
        </a>
      </div>

      <div className="cd-bento">
        <section className="card cd-activity-card" id="cd-activity">
          <h2 className="cd-section-title">
            <Icon name="receipt_long" size={18} className="cd-section-icon" />
            Recent Activity
          </h2>

          {history.isLoading && <Loading label="Loading risk history…" />}
          {history.isError && <ErrorBlock error={history.error} onRetry={history.refetch} />}
          {items.length === 0 && !history.isLoading && (
            <EmptyState>Insufficient historical baseline -- no scored transactions yet.</EmptyState>
          )}

          {items.length > 0 && (
            <>
              <div className="table-scroll">
                <table className="data-table">
                  <thead>
                    <tr>
                      <th>Transaction</th>
                      <th>Unified risk</th>
                      <th>Customer state</th>
                      <th>Terminal</th>
                    </tr>
                  </thead>
                  <tbody>
                    {[...items].reverse().map((r) => (
                      <tr key={r.transaction_id}>
                        <td>
                          <Link className="link-id mono cd-tx-link" to={`/transactions/${r.transaction_id}`}>
                            <Icon name="shopping_cart" size={14} className="cd-tx-icon" />
                            TX_{r.transaction_id}
                          </Link>
                        </td>
                        <td>
                          <RiskBadge level={r.unified_risk_level} size="sm" />
                        </td>
                        <td>
                          <StateBadge state={r.customer_risk_state} />
                        </td>
                        <td>
                          <Link className="link-id mono" to={`/terminals/${r.terminal_id}`}>
                            TERM_{r.terminal_id}
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
            </>
          )}
        </section>

        <section className="card cd-evidence-card">
          <h2 className="cd-section-title">
            <Icon name="monitoring" size={18} className="cd-section-icon" />
            Behavioral Evidence
          </h2>
          {items.length === 0 && !history.isLoading && <EmptyState>No behavioral history yet.</EmptyState>}
          {items.length > 0 && (
            <BehavioralTimeline points={items.map((r) => ({ transactionId: r.transaction_id, state: r.customer_risk_state }))} />
          )}
        </section>
      </div>
    </div>
  );
}
