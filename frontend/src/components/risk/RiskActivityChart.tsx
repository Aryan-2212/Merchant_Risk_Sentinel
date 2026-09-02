import { useMemo, useState } from "react";
import type { RiskActivityPoint } from "../../lib/types";
import { formatCount } from "../../lib/format";
import "./RiskActivityChart.css";

const WIDTH = 480;
const HEIGHT = 180;
const PAD_L = 30;
const PAD_B = 22;
const PAD_T = 10;
const PAD_R = 8;

function formatAxisDate(iso: string): string {
  const d = new Date(iso + "T00:00:00");
  return d.toLocaleDateString(undefined, { month: "short", day: "numeric" });
}

/**
 * Daily count of transactions whose unified_risk_level was HIGH/CRITICAL (GET
 * /stats/risk-activity) -- the single real "how much elevated risk activity"
 * trend line the approved reference calls for. No narrative point labels ("Velocity
 * Spike", "Fraud Shift") -- this system has no event-naming log to honestly attribute
 * a specific day's peak to a specific cause; the hover tooltip surfaces the real date
 * and count instead of a fabricated label.
 */
export function RiskActivityChart({ data }: { data: RiskActivityPoint[] }) {
  const [hoverIdx, setHoverIdx] = useState<number | null>(null);

  const { pathD, maxY, plotH, points } = useMemo(() => {
    const plotW = WIDTH - PAD_L - PAD_R;
    const plotH = HEIGHT - PAD_T - PAD_B;
    const maxY = Math.max(1, ...data.map((d) => d.elevated_transactions));
    const n = Math.max(1, data.length - 1);
    const xAt = (i: number) => PAD_L + (i / n) * plotW;
    const yAt = (v: number) => PAD_T + plotH - (v / maxY) * plotH;
    const points = data.map((d, i) => ({ x: xAt(i), y: yAt(d.elevated_transactions) }));
    const pathD = points.map((p, i) => `${i === 0 ? "M" : "L"}${p.x},${p.y}`).join(" ");
    return { pathD, maxY, plotH, points };
  }, [data]);

  if (data.length === 0) {
    return <div className="ract-empty">No risk activity in this window.</div>;
  }

  const n = Math.max(1, data.length - 1);
  const xAt = (i: number) => PAD_L + (i / n) * (WIDTH - PAD_L - PAD_R);

  function onMove(e: React.MouseEvent<SVGSVGElement>) {
    const rect = e.currentTarget.getBoundingClientRect();
    const relX = ((e.clientX - rect.left) / rect.width) * WIDTH;
    const idx = Math.round(((relX - PAD_L) / (WIDTH - PAD_L - PAD_R)) * n);
    setHoverIdx(Math.min(data.length - 1, Math.max(0, idx)));
  }

  const hovered = hoverIdx !== null ? data[hoverIdx] : null;

  return (
    <div className="ract">
      <div className="ract-range">
        {formatAxisDate(data[0].date)} – {formatAxisDate(data[data.length - 1].date)}
      </div>
      <svg
        viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
        onMouseMove={onMove}
        onMouseLeave={() => setHoverIdx(null)}
        role="img"
        aria-label="Daily count of HIGH or CRITICAL risk transactions"
      >
        {[0, 0.5, 1].map((f) => (
          <line key={f} className="ract-grid" x1={PAD_L} x2={WIDTH - PAD_R} y1={PAD_T + plotH * f} y2={PAD_T + plotH * f} />
        ))}
        <text x={2} y={PAD_T + 4} className="ract-axis-label">
          {formatCount(maxY)}
        </text>
        <text x={2} y={PAD_T + plotH} className="ract-axis-label">
          0
        </text>

        <path d={pathD} className="ract-line" />

        {hoverIdx !== null && (
          <>
            <line x1={xAt(hoverIdx)} x2={xAt(hoverIdx)} y1={PAD_T} y2={PAD_T + plotH} className="ract-crosshair" />
            <circle cx={points[hoverIdx].x} cy={points[hoverIdx].y} r={3.5} className="ract-dot" />
          </>
        )}

        <text x={PAD_L} y={HEIGHT - 4} className="ract-axis-label">
          {formatAxisDate(data[0].date)}
        </text>
        <text x={WIDTH - PAD_R} y={HEIGHT - 4} textAnchor="end" className="ract-axis-label">
          {formatAxisDate(data[data.length - 1].date)}
        </text>
      </svg>

      {hovered && (
        <div className="ract-tooltip">
          <strong>{formatAxisDate(hovered.date)}</strong>
          <span>{formatCount(hovered.elevated_transactions)} HIGH/CRITICAL transactions</span>
          <span className="ract-tooltip-total">{formatCount(hovered.total_scored)} scored that day</span>
        </div>
      )}
    </div>
  );
}
