import { Link } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { api } from "../../lib/api";
import { useHealth } from "../../lib/hooks";
import { Icon } from "../common/Icon";
import "./Topbar.css";

function formatRecentContext(start?: string, end?: string) {
  if (!start || !end) return "Recent stream · not ingested";
  const fmt = new Intl.DateTimeFormat("en-GB", { day: "2-digit", month: "short", year: "numeric" });
  return `Recent simulated stream · ${fmt.format(new Date(start))}–${fmt.format(new Date(end))}`;
}

export function Topbar() {
  const { data: health, isError } = useHealth();
  const stats = useQuery({ queryKey: ["stats", "overview"], queryFn: api.overviewStats, staleTime: 30_000 });
  const recent = useQuery({ queryKey: ["recent", "bounds"], queryFn: api.recentBounds, staleTime: 30_000, retry: false });

  const connected = !isError && health?.status === "ok";
  const openAlerts = stats.data?.alert_status_counts["OPEN"] ?? 0;
  const recentLoaded = !recent.isError && !!recent.data;

  return (
    <header className="topbar">
      <div className="topbar-left">
        <span className="topbar-wordmark">Risk Sentinel</span>
        <div className="topbar-sep" />
        <span className="topbar-context">Benchmark · Apr–Sep 2018</span>
        <div className="topbar-sep" />
        <span className="topbar-context" title={recentLoaded ? `${recent.data.total_transactions.toLocaleString()} recent transactions` : undefined}>
          {formatRecentContext(recent.data?.min_tx_datetime, recent.data?.max_tx_datetime)}
        </span>
        {recentLoaded && <span className="topbar-stream-dot" aria-label="Recent stream available" />}
      </div>

      <div className="topbar-actions">
        <Link to="/alerts" className="topbar-icon-btn" aria-label={`${openAlerts} open alerts`} title={`${openAlerts} open alerts`}>
          <Icon name="notifications" size={20} />
          {openAlerts > 0 && <span className="topbar-icon-dot" />}
        </Link>
        <Link to="/system" className="topbar-icon-btn" aria-label="System health" title={connected ? "System operational" : "API unreachable"}>
          <Icon name="settings" size={20} />
          <span className={`topbar-icon-status ${connected ? "status-ok" : "status-down"}`} />
        </Link>
      </div>
    </header>
  );
}
