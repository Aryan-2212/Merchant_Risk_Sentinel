import { useMemo } from "react";
import type { RiskActivityPoint } from "../../lib/types";
import "./RiskSignalBreakdown.css";

interface Signal {
  label: string;
  changePct: number | null;
}

/** Splits the fetched window in half and compares each component's elevated-count
 * sum between halves -- a real period-over-period change, computed client-side from
 * the same GET /stats/risk-activity series driving the trend chart. No fourth
 * "Temporal" row: mrs.risk.aggregate has no independent temporal risk component
 * (documented architectural decision, not an oversight) -- three real signals only. */
function computeSignals(data: RiskActivityPoint[]): Signal[] {
  const mid = Math.floor(data.length / 2);
  const prior = data.slice(0, mid);
  const recent = data.slice(mid);

  function change(key: keyof RiskActivityPoint): number | null {
    const priorSum = prior.reduce((s, d) => s + Number(d[key]), 0);
    const recentSum = recent.reduce((s, d) => s + Number(d[key]), 0);
    if (priorSum === 0) return recentSum === 0 ? 0 : null;
    return ((recentSum - priorSum) / priorSum) * 100;
  }

  return [
    { label: "Transaction", changePct: change("transaction_high") },
    { label: "Customer", changePct: change("customer_high") },
    { label: "Terminal", changePct: change("terminal_high") },
  ];
}

export function RiskSignalBreakdown({ data }: { data: RiskActivityPoint[] }) {
  const signals = useMemo(() => computeSignals(data), [data]);
  const max = Math.max(1, ...signals.map((s) => Math.abs(s.changePct ?? 0)));

  return (
    <div className="rsb">
      {signals.map((s) => {
        const pct = s.changePct;
        const width = pct === null ? 0 : Math.min(100, (Math.abs(pct) / max) * 100);
        const rising = pct !== null && pct > 0.5;
        const falling = pct !== null && pct < -0.5;
        return (
          <div className="rsb-row" key={s.label}>
            <div className="rsb-row-head">
              <span className="rsb-label">{s.label}</span>
              <span className={`rsb-change mono ${rising ? "rsb-up" : falling ? "rsb-down" : ""}`}>
                {pct === null ? "n/a" : `${pct >= 0 ? "+" : ""}${pct.toFixed(0)}%`}
              </span>
            </div>
            <div className="rsb-track">
              <div
                className={`rsb-fill ${rising ? "rsb-up" : falling ? "rsb-down" : "rsb-flat"}`}
                style={{ width: `${Math.max(2, width)}%` }}
              />
            </div>
          </div>
        );
      })}
    </div>
  );
}
