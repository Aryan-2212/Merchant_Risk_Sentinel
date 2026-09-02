import type { Point } from "./BehavioralTimeline";
import { toSegments } from "./BehavioralTimeline";
import "./ZigzagTimeline.css";

const STATE_LABEL: Record<string, string> = {
  NORMAL: "Normal",
  RISK_RISING: "Risk Rising",
  RECOVERY: "Recovery",
  HIGH_RISK: "High Risk",
  INSUFFICIENT_HISTORY: "Insufficient History",
};

function stateClass(state: string | null): string {
  if (state === "HIGH_RISK") return "zt-high";
  if (state === "RISK_RISING" || state === "RECOVERY") return "zt-medium";
  return "zt-neutral";
}

/**
 * Approved reference (terminal_investigation_dark): alternating left/right timeline
 * of real behavioral state transitions -- reuses BehavioralTimeline's segmentation
 * (real risk-history data, oldest-collapsed-into-segments), never a fabricated
 * narrative ("Fraud Spike Detected"). Newest segment first, marked "Current"; the
 * oldest segment is the entity's first scored transaction, marked "Baseline".
 */
export function ZigzagTimeline({ points }: { points: Point[] }) {
  if (points.length === 0) return null;
  const segments = [...toSegments(points)].reverse();

  return (
    <div className="zt">
      <div className="zt-line" />
      {segments.map((seg, i) => {
        const isCurrent = i === 0;
        const isBaseline = i === segments.length - 1 && segments.length > 1;
        const side = i % 2 === 0 ? "left" : "right";
        const content = (
          <div className={`zt-content ${isBaseline ? "zt-baseline" : ""}`}>
            <span className={`zt-tag ${stateClass(seg.state)}`}>{isCurrent ? "Current" : isBaseline ? "Baseline" : ""}</span>
            <span className="zt-desc">{STATE_LABEL[seg.state ?? "INSUFFICIENT_HISTORY"]}</span>
            <span className="zt-meta">
              {seg.count} transaction{seg.count === 1 ? "" : "s"} · #{seg.fromTx}
              {seg.toTx !== seg.fromTx ? `–${seg.toTx}` : ""}
            </span>
          </div>
        );
        return (
          <div className={`zt-row zt-${side}`} key={`${seg.fromTx}-${i}`}>
            {side === "left" ? content : <div className="zt-content-empty" />}
            <div className={`zt-node ${stateClass(seg.state)} ${isCurrent ? "zt-node-flagged" : ""}`} />
            {side === "right" ? content : <div className="zt-content-empty" />}
          </div>
        );
      })}
    </div>
  );
}
