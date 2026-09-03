import { useQuery } from "@tanstack/react-query";
import { api } from "../lib/api";
import { formatCount } from "../lib/format";
import { Icon } from "../components/common/Icon";
import "./SystemHealth.css";

type Status = "ok" | "degraded" | "unknown";

const STATUS_LABEL: Record<Status, string> = {
  ok: "Operational",
  degraded: "Degraded",
  unknown: "Checking…",
};

/**
 * Every row here is backed by an actual query result, shown as the "detail" line so a
 * claim like "Operational" is never asserted without the number behind it. Rows with
 * no real signal available (e.g. the feature engine, which has no API route) are
 * omitted rather than invented -- Dev Plan Sec 18/24. Icons are reused from where each
 * concept already appears elsewhere (gavel = policy engine on AlertDetail, smart_toy =
 * AI analyst on AnalystPanel, history = replay in the sidebar) rather than invented.
 */
export function SystemHealth() {
  const health = useQuery({ queryKey: ["health"], queryFn: api.health, retry: 0 });
  const stats = useQuery({ queryKey: ["stats", "overview"], queryFn: api.overviewStats });
  const bounds = useQuery({ queryKey: ["replay-bounds"], queryFn: api.replayBounds });

  const rows: { name: string; icon: string; status: Status; detail: string }[] = [
    {
      name: "Database",
      icon: "storage",
      status: health.isSuccess ? "ok" : health.isError ? "degraded" : "unknown",
      detail: health.isSuccess ? "Connected" : health.isError ? "Unreachable" : "Checking…",
    },
    {
      name: "Risk Engine",
      icon: "insights",
      status: stats.data ? (stats.data.total_risk_scores > 0 ? "ok" : "degraded") : "unknown",
      detail: stats.data ? `${formatCount(stats.data.total_risk_scores)} risk scores persisted` : "Checking…",
    },
    {
      name: "Policy Engine",
      icon: "gavel",
      status: stats.data ? "ok" : "unknown",
      detail: stats.data ? `${formatCount(stats.data.total_alerts)} alerts decided` : "Checking…",
    },
    {
      name: "Replay Engine",
      icon: "history",
      status: bounds.isSuccess ? "ok" : bounds.isError ? "degraded" : "unknown",
      detail: bounds.data ? `${formatCount(bounds.data.total_transactions)} transactions available` : bounds.isError ? "Unreachable" : "Checking…",
    },
    {
      name: "AI Risk Analyst",
      icon: "smart_toy",
      status: health.data ? (health.data.ai_analyst_configured ? "ok" : "degraded") : "unknown",
      detail: health.data
        ? health.data.ai_analyst_configured
          ? "GEMINI_API_KEY configured -- live explanations"
          : "No API key configured -- deterministic fallback only"
        : "Checking…",
    },
  ];

  return (
    <div className="page">
      <div className="page-header">
        <div>
          <h1 className="page-title">
            <Icon name="monitor_heart" size={22} className="sh-title-icon" />
            System Health
          </h1>
          <p className="page-subtitle">Genuine, queried system state -- nothing here is a simulated metric.</p>
        </div>
      </div>

      <div className="sh-grid">
        {rows.map((row) => (
          <div className="card sh-card" key={row.name}>
            <div className="sh-card-header">
              <Icon name={row.icon} size={18} className="sh-card-icon" />
              <span className="sh-card-name">{row.name}</span>
            </div>
            <span className={`sh-status-pill sh-status-${row.status}`}>
              <span className={`sh-status-dot sh-status-dot-${row.status}`} aria-hidden="true" />
              {STATUS_LABEL[row.status]}
            </span>
            <p className="sh-card-detail mono">{row.detail}</p>
          </div>
        ))}
      </div>
    </div>
  );
}
