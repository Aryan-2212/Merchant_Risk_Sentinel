import { useQuery } from "@tanstack/react-query";
import { api } from "../lib/api";
import { Loading, ErrorBlock } from "../components/common/States";
import { KpiStrip } from "../components/command/KpiStrip";
import { HighDeviationTerminals } from "../components/command/HighDeviationTerminals";
import { RiskSignalBreakdown } from "../components/command/RiskSignalBreakdown";
import { RecentHighRisk } from "../components/command/RecentHighRisk";
import { RiskActivityChart } from "../components/risk/RiskActivityChart";
import "./Overview.css";

export function Overview() {
  const stats = useQuery({ queryKey: ["stats", "overview"], queryFn: api.overviewStats });
  const activity = useQuery({ queryKey: ["stats", "risk-activity", 14], queryFn: () => api.riskActivity(14) });
  const feed = useQuery({
    queryKey: ["stats", "recent-activity", "high-risk"],
    queryFn: () => api.recentActivity(6, "HIGH,CRITICAL"),
  });
  const terminals = useQuery({ queryKey: ["stats", "terminals-at-risk"], queryFn: () => api.terminalsAtRisk(6) });

  // Gated on just the two queries the above-the-fold KPI strip needs, not all four --
  // previously this covered every query on the page, so once the KPI strip and one
  // panel had already rendered, a slower-loading panel (recent activity/terminals)
  // kept this full-page "Loading command center…" banner showing ABOVE already-live
  // content, reading as if the page were stuck. Each remaining panel below now shows
  // its own inline loading state instead (same pattern as the rest of the app, e.g.
  // AlertDetail's per-section Loading).
  const kpiLoading = stats.isLoading || activity.isLoading;

  return (
    <div className="page page-wide">
      {/* The approved reference goes straight from the topbar into the KPI row --
          no page-title block. Kept for screen readers only, not visually rendered,
          so this still matches the reference pixel-for-pixel above the fold. */}
      <h1 className="visually-hidden">Overview</h1>

      {stats.isError && <ErrorBlock error={stats.error} onRetry={() => stats.refetch()} />}
      {kpiLoading && !stats.isError && <Loading label="Loading command center…" />}

      {stats.data && activity.data && <KpiStrip stats={stats.data} activity={activity.data} />}

      <div className="ov-grid">
        <div className="card ov-main">
          {terminals.isLoading && <Loading label="Loading high deviation terminals…" />}
          {terminals.isError && <ErrorBlock error={terminals.error} onRetry={() => terminals.refetch()} />}
          {terminals.data && <HighDeviationTerminals rows={terminals.data} />}

          <div className="ov-sub-grid">
            <div className="ov-sub-panel">
              <span className="section-title">Risk Signal Breakdown</span>
              {activity.isError && <ErrorBlock error={activity.error} onRetry={() => activity.refetch()} />}
              {activity.data && <RiskSignalBreakdown data={activity.data} />}
            </div>
            <div className="ov-sub-panel">
              <span className="section-title">Risk Trend / Behavioral Shift</span>
              {activity.data && <RiskActivityChart data={activity.data} />}
            </div>
          </div>
        </div>

        <div className="card ov-side">
          {feed.isLoading && <Loading label="Loading recent high-risk activity…" />}
          {feed.isError && <ErrorBlock error={feed.error} onRetry={() => feed.refetch()} />}
          {feed.data && <RecentHighRisk items={feed.data} />}
        </div>
      </div>
    </div>
  );
}
