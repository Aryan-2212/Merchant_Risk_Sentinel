import { useQuery } from "@tanstack/react-query";
import { api } from "../lib/api";
import { formatCount } from "../lib/format";
import "./SystemHealth.css";

type Status = "ok" | "degraded" | "unknown";

function Indicator({ status }: { status: Status }) {
  return <span className={`health-dot health-${status}`} aria-hidden="true" />;
}

/**
 * Every row here is backed by an actual query result, shown as the "detail" column
 * so a claim like "Operational" is never asserted without the number behind it. Rows
 * with no real signal available (e.g. the feature engine, which has no API route) are
 * omitted rather than invented -- Dev Plan Sec 18/24.
 */
export function SystemHealth() {
  const health = useQuery({ queryKey: ["health"], queryFn: api.health, retry: 0 });
  const stats = useQuery({ queryKey: ["stats", "overview"], queryFn: api.overviewStats });
  const bounds = useQuery({ queryKey: ["replay-bounds"], queryFn: api.replayBounds });

  const rows: { name: string; status: Status; detail: string }[] = [
    {
      name: "Database",
      status: health.isSuccess ? "ok" : health.isError ? "degraded" : "unknown",
      detail: health.isSuccess ? "connected" : health.isError ? "unreachable" : "checking…",
    },
    {
      name: "Risk engine (transaction ML + behavioral + aggregation)",
      status: stats.data ? (stats.data.total_risk_scores > 0 ? "ok" : "degraded") : "unknown",
      detail: stats.data ? `${formatCount(stats.data.total_risk_scores)} risk scores persisted` : "checking…",
    },
    {
      name: "Policy engine",
      status: stats.data ? "ok" : "unknown",
      detail: stats.data ? `${formatCount(stats.data.total_alerts)} alerts decided` : "checking…",
    },
    {
      name: "Replay engine",
      status: bounds.isSuccess ? "ok" : bounds.isError ? "degraded" : "unknown",
      detail: bounds.data
        ? `${formatCount(bounds.data.total_transactions)} transactions available`
        : bounds.isError
          ? "unreachable"
          : "checking…",
    },
    {
      name: "AI risk analyst",
      status: health.data ? (health.data.ai_analyst_configured ? "ok" : "degraded") : "unknown",
      detail: health.data
        ? health.data.ai_analyst_configured
          ? "GEMINI_API_KEY configured -- live explanations"
          : "no API key configured -- deterministic fallback only"
        : "checking…",
    },
  ];

  return (
    <div className="page">
      <div className="page-header">
        <div>
          <h1 className="page-title">System health</h1>
          <p className="page-subtitle">Genuine, queried system state -- nothing here is a simulated metric.</p>
        </div>
      </div>

      <div className="card">
        <table className="data-table health-table">
          <tbody>
            {rows.map((row) => (
              <tr key={row.name}>
                <td>
                  <Indicator status={row.status} />
                  {row.name}
                </td>
                <td className="health-detail">{row.detail}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
