import { Link } from "react-router-dom";
import type { ReplayItemOut } from "../../lib/types";
import { formatAmount, formatDateTimeCompact } from "../../lib/format";
import { EmptyState } from "../common/States";
import { Icon } from "../common/Icon";
import "./RecentHighRisk.css";

/** Translates one contributing_signals entry (mrs.risk.aggregate._signal_text syntax,
 * e.g. "transaction_ml_risk >= 0.97" or "terminal_behavioral_risk: HIGH_RISK") into a
 * short analyst-facing phrase for this compact feed item -- never the raw
 * field-name/operator syntax verbatim (mirrors the tone of mrs.analyst.client's own
 * deterministic-fallback phrasing, kept brief here for a single feed line). */
function describeSignal(signal: string): string {
  if (signal.startsWith("transaction_ml_risk")) return "Elevated transaction-level ML risk";
  if (signal.startsWith("terminal_behavioral_risk")) {
    const state = signal.split(":")[1]?.trim().replace(/_/g, " ");
    return state ? `Terminal behavioral state: ${state}` : "Elevated terminal behavioral risk";
  }
  if (signal.startsWith("customer_behavioral_risk")) {
    const state = signal.split(":")[1]?.trim().replace(/_/g, " ");
    return state ? `Customer behavioral state: ${state}` : "Elevated customer behavioral risk";
  }
  return signal;
}

/** True dataset-source labels for this panel's contextual tag -- computed from
 * transaction.split (already returned by every ReplayItemOut), never assumed. "recent"
 * is the Simulated Recent Operational Stream (mrs.data.recent_stream); every other
 * split value ("train"/"validation"/"test") is the frozen Apr-Sep 2018 benchmark. */
const DATASET_LABEL = {
  recent: "SIMULATED RECENT ACTIVITY · AUG 15 – SEP 04, 2026",
  benchmark: "HISTORICAL BENCHMARK · APR–SEP 2018",
} as const;

/** GET /stats/recent-activity?levels=HIGH,CRITICAL -- filtered server-side (sampling
 * the last N transactions overall and filtering client-side can legitimately return
 * zero rows, since elevated transactions are ~1.5% of the stream). The client-side
 * filter below is redundant defense-in-depth, not the real guarantee. */
export function RecentHighRisk({ items }: { items: ReplayItemOut[] }) {
  const highRisk = items.filter((i) => i.risk_score?.unified_risk_level === "HIGH" || i.risk_score?.unified_risk_level === "CRITICAL").slice(0, 4);

  // Once any Simulated Recent Operational Stream data exists, this feed (sorted
  // newest-first) is dominated by it -- 2026 transactions always sort after every 2018
  // one, so a blanket "which dataset is this" label at the panel level is accurate
  // without labeling each row individually. Computed from real split values, not
  // assumed: falls back to the historical label whenever no shown row is actually
  // "recent" (e.g. before the recent stream is ever ingested).
  const hasRecent = highRisk.some((i) => i.transaction.split === "recent");
  const datasetLabel = hasRecent ? DATASET_LABEL.recent : DATASET_LABEL.benchmark;

  return (
    <div className="rhr">
      <div className="rhr-header">
        <div className="rhr-header-main">
          <Icon name="warning" size={18} className="rhr-header-icon" />
          <span className="rhr-title">Recent High Risk</span>
        </div>
        {highRisk.length > 0 && <span className="rhr-dataset-label">{datasetLabel}</span>}
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
                  <p className="rhr-desc">
                    {risk_score?.contributing_signals[0] ? describeSignal(risk_score.contributing_signals[0]) : "Elevated risk detected"}
                  </p>
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
