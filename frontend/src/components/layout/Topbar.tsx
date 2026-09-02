import { useState } from "react";
import { useNavigate, Link } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { api } from "../../lib/api";
import { useHealth } from "../../lib/hooks";
import { Icon } from "../common/Icon";
import "./Topbar.css";

function resolveSearch(raw: string): string | null {
  const value = raw.trim().toLowerCase();
  if (!value) return null;
  const custom = value.match(/^c(?:ustomer)?\s*#?(\d+)$/);
  if (custom) return `/customers/${custom[1]}`;
  const term = value.match(/^t(?:erminal)?\s*#?(\d+)$/);
  if (term) return `/terminals/${term[1]}`;
  const tx = value.match(/^(?:tx)?\s*#?(\d+)$/);
  if (tx) return `/transactions/${tx[1]}`;
  return null;
}

export function Topbar() {
  const [query, setQuery] = useState("");
  const navigate = useNavigate();
  const { data: health, isError } = useHealth();
  const stats = useQuery({ queryKey: ["stats", "overview"], queryFn: api.overviewStats, staleTime: 30_000 });

  function onSubmit(e: React.FormEvent) {
    e.preventDefault();
    const path = resolveSearch(query);
    if (path) {
      navigate(path);
      setQuery("");
    }
  }

  const connected = !isError && health?.status === "ok";
  const openAlerts = stats.data?.alert_status_counts["OPEN"] ?? 0;

  return (
    <header className="topbar">
      <div className="topbar-left">
        <span className="topbar-wordmark">Risk Sentinel</span>
        <div className="topbar-sep" />
        {/* Real data span, not a live "last 24h" claim -- this is historical replay
            data (Apr-Sep 2018), never framed as a real-time feed. */}
        <span className="topbar-context">Historical dataset · Apr–Sep 2018</span>
      </div>

      <form className="topbar-search" onSubmit={onSubmit}>
        <Icon name="search" size={16} className="topbar-search-icon" />
        <input
          type="search"
          placeholder="TX ID, C-id, or T-id…"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          aria-label="Jump to transaction, customer, or terminal by ID"
        />
      </form>

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
