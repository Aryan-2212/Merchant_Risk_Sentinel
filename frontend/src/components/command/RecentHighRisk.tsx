import { Link } from "react-router-dom";
import type { ReplayItemOut } from "../../lib/types";
import { formatAmount, formatDateTimeCompact } from "../../lib/format";
import { EmptyState } from "../common/States";
import { Icon } from "../common/Icon";
import "./RecentHighRisk.css";

/** GET /stats/recent-activity?levels=HIGH,CRITICAL -- filtered server-side (sampling
 * the last N transactions overall and filtering client-side can legitimately return
 * zero rows, since elevated transactions are ~1.5% of the stream). The client-side
 * filter below is redundant defense-in-depth, not the real guarantee. */
export function RecentHighRisk({ items }: { items: ReplayItemOut[] }) {
  const highRisk = items.filter((i) => i.risk_score?.unified_risk_level === "HIGH" || i.risk_score?.unified_risk_level === "CRITICAL").slice(0, 4);

  return (
    <div className="rhr">
      <div className="rhr-header">
        <Icon name="warning" size={18} className="rhr-header-icon" />
        <span className="rhr-title">Recent High Risk</span>
      </div>

      {highRisk.length === 0 ? (
        <EmptyState>No HIGH or CRITICAL activity in the recent window.</EmptyState>
      ) : (
        <ol className="rhr-list">
          {highRisk.map(({ transaction, risk_score }) => {
            const critical = risk_score?.unified_risk_level === "CRITICAL";
            return (
              <li key={transaction.transaction_id} className="rhr-item">
                <span className={`rhr-marker ${critical ? "rhr-marker-critical" : "rhr-marker-high"}`} />
                <Link to={`/transactions/${transaction.transaction_id}`} className="rhr-body">
                  <div className="rhr-row">
                    <span className={`rhr-id mono ${critical ? "rhr-id-critical" : "rhr-id-high"}`}>
                      TX_{transaction.transaction_id}
                    </span>
                    <span className="rhr-time mono">{formatDateTimeCompact(transaction.tx_datetime)}</span>
                  </div>
                  <p className="rhr-desc">{risk_score?.contributing_signals[0] ?? "Elevated risk detected"}</p>
                  <p className="rhr-detail">
                    {formatAmount(transaction.tx_amount)} · CUST_{transaction.customer_id} → TERM_{transaction.terminal_id}
                  </p>
                </Link>
              </li>
            );
          })}
        </ol>
      )}
    </div>
  );
}
