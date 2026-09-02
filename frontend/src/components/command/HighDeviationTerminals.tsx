import { Link } from "react-router-dom";
import type { EntityAtRiskRow } from "../../lib/types";
import { EmptyState } from "../common/States";
import { Icon } from "../common/Icon";
import "./HighDeviationTerminals.css";

function formatPct(rate: number): string {
  return `${(rate * 100).toFixed(1)}%`;
}

function severityDotClass(severity: number | null): string {
  if (severity === 2) return "hdt-dot-high";
  if (severity === 1) return "hdt-dot-medium";
  return "hdt-dot-low";
}

function deviationIcon(pp: number): string | null {
  if (pp >= 8) return "keyboard_double_arrow_up";
  if (pp >= 2) return "arrow_upward";
  return null;
}

function toCsv(rows: EntityAtRiskRow[]): string {
  const header = "terminal_id,state,baseline_rate,current_rate,deviation_pp,recent_transactions,last_activity";
  const lines = rows.map((r) => {
    const dev = r.baseline_rate === null ? "" : ((r.current_rate - r.baseline_rate) * 100).toFixed(2);
    return [
      `TERM_${r.entity_id}`,
      r.risk_state,
      r.baseline_rate === null ? "" : (r.baseline_rate * 100).toFixed(2),
      (r.current_rate * 100).toFixed(2),
      dev,
      r.recent_transaction_count,
      r.last_activity,
    ].join(",");
  });
  return [header, ...lines].join("\n");
}

function downloadCsv(rows: EntityAtRiskRow[]) {
  const blob = new Blob([toCsv(rows)], { type: "text/csv" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "high_deviation_terminals.csv";
  a.click();
  URL.revokeObjectURL(url);
}

/**
 * "High Deviation Terminals" (approved reference). baseline_rate/current_rate are
 * each terminal's own fraction of transactions at severity 2 (mrs.risk.aggregate),
 * recent 7-day window vs. the 30 days before it -- GET /stats/terminals-at-risk.
 * Never the ground-truth tx_fraud label, and never a fabricated "location" (this
 * system has no real terminal address data, only synthetic 2D coordinates).
 */
export function HighDeviationTerminals({ rows }: { rows: EntityAtRiskRow[] }) {
  return (
    <div className="hdt">
      <div className="hdt-header">
        <span className="hdt-title">High Deviation Terminals</span>
        <button className="hdt-export" onClick={() => downloadCsv(rows)} disabled={rows.length === 0}>
          Export <Icon name="download" size={14} />
        </button>
      </div>

      {rows.length === 0 ? (
        <EmptyState>No terminals currently show elevated behavioral risk.</EmptyState>
      ) : (
        <table className="data-table hdt-table">
          <thead>
            <tr>
              <th>Terminal ID</th>
              <th>State</th>
              <th>Baseline rate</th>
              <th>Current rate</th>
              <th>Deviation</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => {
              const dev = row.baseline_rate === null ? null : (row.current_rate - row.baseline_rate) * 100;
              const devIcon = dev === null ? null : deviationIcon(Math.abs(dev));
              return (
                <tr key={row.entity_id}>
                  <td>
                    <Link to={`/terminals/${row.entity_id}`} className="hdt-id-cell">
                      <span className={`hdt-dot ${severityDotClass(row.risk_severity)}`} />
                      <span>TERM_{row.entity_id}</span>
                    </Link>
                  </td>
                  <td className="hdt-state">{row.risk_state.replaceAll("_", " ").toLowerCase()}</td>
                  <td className="hdt-num">{row.baseline_rate === null ? "—" : formatPct(row.baseline_rate)}</td>
                  <td className={`hdt-num ${row.current_rate >= 0.3 ? "hdt-num-critical" : ""}`}>
                    {formatPct(row.current_rate)}
                  </td>
                  <td className={`hdt-num hdt-dev ${dev !== null && dev > 0 ? "hdt-num-critical" : ""}`}>
                    {dev === null ? "—" : `${dev >= 0 ? "+" : ""}${dev.toFixed(1)}%`}
                    {devIcon && <Icon name={devIcon} size={14} />}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      )}
    </div>
  );
}
