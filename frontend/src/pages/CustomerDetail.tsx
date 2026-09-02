import { useQuery } from "@tanstack/react-query";
import { Link, useParams } from "react-router-dom";
import { api } from "../lib/api";
import { formatAmount } from "../lib/format";
import { useRiskHistory } from "../lib/useRiskHistory";
import { Loading, ErrorBlock, EmptyState } from "../components/common/States";
import { RiskBadge } from "../components/risk/RiskBadge";
import { StateBadge } from "../components/risk/StateBadge";
import { BehavioralTimeline } from "../components/risk/BehavioralTimeline";
import "./EntityDetail.css";

export function CustomerDetail() {
  const { id } = useParams<{ id: string }>();
  const customerId = Number(id);

  const customer = useQuery({
    queryKey: ["customer", customerId],
    queryFn: () => api.getCustomer(customerId),
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

  return (
    <div className="page">
      <div className="page-header">
        <div>
          <h1 className="page-title">Customer #{c.customer_id}</h1>
          <p className="page-subtitle">Behavioral risk state, not a permanent fraud label.</p>
        </div>
        {current?.customer_risk_state && <StateBadge state={current.customer_risk_state} />}
      </div>

      <div className="card entity-meta">
        <div>
          <span className="field-label">Baseline avg. amount</span>
          <p className="mono">{formatAmount(c.mean_amount)}</p>
        </div>
        <div>
          <span className="field-label">Amount std. dev.</span>
          <p className="mono">{formatAmount(c.std_amount)}</p>
        </div>
        <div>
          <span className="field-label">Avg. tx / day</span>
          <p className="mono">{c.mean_nb_tx_per_day.toFixed(2)}</p>
        </div>
        <div>
          <span className="field-label">Terminals used</span>
          <p className="mono">{c.nb_terminals}</p>
        </div>
      </div>

      <div className="section">
        <span className="section-title">Behavioral state over time{!onLastPage && offset > 0 ? " (this page)" : ""}</span>
        {history.isLoading && <Loading label="Loading risk history…" />}
        {history.isError && <ErrorBlock error={history.error} onRetry={history.refetch} />}
        {items.length === 0 && !history.isLoading && (
          <EmptyState>Insufficient historical baseline -- no scored transactions yet.</EmptyState>
        )}
        {items.length > 0 && (
          <div className="card">
            <BehavioralTimeline
              points={items.map((r) => ({ transactionId: r.transaction_id, state: r.customer_risk_state }))}
            />
          </div>
        )}
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
                  <th>Customer state</th>
                  <th>Terminal</th>
                </tr>
              </thead>
              <tbody>
                {[...items].reverse().map((r) => (
                  <tr key={r.transaction_id}>
                    <td>
                      <Link className="link-id mono" to={`/transactions/${r.transaction_id}`}>
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
        </div>
      )}
    </div>
  );
}
