import type { BehavioralState } from "../../lib/types";
import { StateBadge } from "./StateBadge";
import "./BehavioralTimeline.css";

export interface Point {
  transactionId: number;
  state: BehavioralState | null;
}

export interface Segment {
  state: BehavioralState | null;
  count: number;
  fromTx: number;
  toTx: number;
}

/** Collapses consecutive same-state points into segments so a long history reads as
 * "NORMAL for 40 tx -> RISK_RISING for 3 tx -> ..." rather than one badge per row --
 * the state-transition story Dev Plan Sec 8/14 asks for, built only from the
 * chronologically-ordered states the API already returns (never re-derived). Exported
 * so other views (e.g. Terminal Investigation's zigzag timeline) can reuse the exact
 * same segmentation instead of re-deriving it. */
export function toSegments(points: Point[]): Segment[] {
  const segments: Segment[] = [];
  for (const point of points) {
    const last = segments[segments.length - 1];
    if (last && last.state === point.state) {
      last.count += 1;
      last.toTx = point.transactionId;
    } else {
      segments.push({ state: point.state, count: 1, fromTx: point.transactionId, toTx: point.transactionId });
    }
  }
  return segments;
}

export function BehavioralTimeline({ points }: { points: Point[] }) {
  if (points.length === 0) return null;
  const segments = toSegments(points);

  return (
    <div className="btimeline" role="img" aria-label="Behavioral state transitions over time, oldest first">
      {segments.map((seg, i) => (
        <div className="btimeline-seg" key={`${seg.fromTx}-${i}`}>
          {i > 0 && <span className="btimeline-arrow">→</span>}
          <div className="btimeline-chip">
            <StateBadge state={seg.state} />
            <span className="btimeline-count mono">
              ×{seg.count} · #{seg.fromTx}
              {seg.toTx !== seg.fromTx ? `–${seg.toTx}` : ""}
            </span>
          </div>
        </div>
      ))}
    </div>
  );
}
