import { Link } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { api } from "../../lib/api";
import { useHealth } from "../../lib/hooks";
import { Icon } from "../common/Icon";
import "./Topbar.css";

export function Topbar() {
  const { data: health, isError } = useHealth();
  const stats = useQuery({ queryKey: ["stats", "overview"], queryFn: api.overviewStats, staleTime: 30_000 });

  const connected = !isError && health?.status === "ok";
  const openAlerts = stats.data?.alert_status_counts["OPEN"] ?? 0;

  return (
    <header className="topbar">
      <div className="topbar-left">
        <span className="topbar-wordmark">Merchant Risk Sentinel</span>
        <div className="topbar-sep" />
        {/* Names both datasets the app can show, rather than asserting only one
            (Apr-Sep 2018) -- several pages (Transactions, Replay) can display either
            the frozen benchmark or the Simulated Recent Operational Stream, and a
            single hard-coded claim here would contradict whichever one is on screen.
            Never framed as a live/real-time feed -- "Simulated" is explicit. Kept
            short (.topbar-context is white-space: nowrap, no overflow handling) so it
            doesn't crowd the topbar on narrower widths. */}
        <span className="topbar-context">Benchmark 2018 · Recent (Simulated) 2026</span>
      </div>

      {/* Global "jump to ID" search was removed here -- it duplicated the Customers/
          Terminals per-page ID lookups and, being global, had no way to disambiguate
          which entity type a bare number meant. Direct lookup by ID still works via
          those pages' own search boxes (functional, page-scoped, not removed). */}
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
