import type { OverviewStats, RiskActivityPoint } from "../../lib/types";
import { formatCount } from "../../lib/format";
import { Icon } from "../common/Icon";
import "./KpiStrip.css";

interface Props {
  stats: OverviewStats;
  activity: RiskActivityPoint[];
}

/**
 * Compact KPI row (approved reference: 4 bordered tiles, icon top-right, large
 * score-display figure). The reference's "Overall Risk Index /100" has no backing in
 * this system -- Phase 7's risk aggregation is deliberately rule/state-based
 * specifically to avoid a fabricated blended score (mrs.risk.aggregate). In its
 * place: the real, derived elevated-risk rate. "Risk Velocity" is a real
 * day-over-day change in elevated activity (from the same series driving the trend
 * chart), not a live "/hr" feed -- this is historical replay data, not a live stream.
 */
export function KpiStrip({ stats, activity }: Props) {
  const elevated = (stats.risk_level_counts.HIGH ?? 0) + (stats.risk_level_counts.CRITICAL ?? 0);
  const elevatedRate = stats.total_risk_scores > 0 ? (elevated / stats.total_risk_scores) * 100 : 0;
  const openAlerts = stats.alert_status_counts["OPEN"] ?? stats.total_alerts;

  let velocity: number | null = null;
  if (activity.length >= 2) {
    const last = activity[activity.length - 1];
    const prev = activity[activity.length - 2];
    velocity = prev.elevated_transactions === 0 ? null : ((last.elevated_transactions - prev.elevated_transactions) / prev.elevated_transactions) * 100;
  }
  const elevatedIsHigh = elevatedRate >= 1;

  return (
    <div className="kpi-strip">
      <div className="kpi-tile">
        <div className="kpi-tile-head">
          <span className="kpi-label">Elevated risk rate</span>
          <Icon name="trending_up" size={16} className={elevatedIsHigh ? "kpi-icon-critical" : "kpi-icon"} />
        </div>
        <div className="kpi-value-row">
          <span className={`kpi-value mono ${elevatedIsHigh ? "kpi-value-critical" : ""}`}>{elevatedRate.toFixed(1)}</span>
          <span className="kpi-unit mono">%</span>
        </div>
      </div>

      <div className="kpi-tile">
        <div className="kpi-tile-head">
          <span className="kpi-label">Active alerts</span>
          <Icon name="notifications_active" size={16} className="kpi-icon-primary" />
        </div>
        <div className="kpi-value-row">
          <span className="kpi-value mono">{formatCount(openAlerts)}</span>
        </div>
      </div>

      <div className="kpi-tile">
        <div className="kpi-tile-head">
          <span className="kpi-label">Suspect terminals</span>
          <Icon name="terminal" size={16} className="kpi-icon-tertiary" />
        </div>
        <div className="kpi-value-row">
          <span className="kpi-value mono kpi-value-tertiary">{formatCount(stats.terminals_at_risk)}</span>
          <span className="kpi-unit mono">flagged</span>
        </div>
      </div>

      <div className="kpi-tile">
        <div className="kpi-tile-head">
          <span className="kpi-label">Risk velocity</span>
          <Icon name="speed" size={16} className="kpi-icon" />
        </div>
        <div className="kpi-value-row">
          {velocity === null ? (
            <span className="kpi-value mono kpi-value-muted">—</span>
          ) : (
            <>
              <span className="kpi-value mono">
                {velocity >= 0 ? "+" : ""}
                {velocity.toFixed(1)}
              </span>
              <span className="kpi-unit mono">% vs prior day</span>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
