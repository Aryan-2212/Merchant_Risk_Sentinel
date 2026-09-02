import type { BehavioralState } from "../../lib/types";
import "./badges.css";

const LABEL: Record<BehavioralState, string> = {
  NORMAL: "Normal",
  RISK_RISING: "Risk rising",
  RECOVERY: "Recovery",
  HIGH_RISK: "High risk",
  INSUFFICIENT_HISTORY: "Insufficient history",
};

const CLASS: Record<BehavioralState, string> = {
  NORMAL: "state-normal",
  RISK_RISING: "state-rising",
  RECOVERY: "state-recovery",
  HIGH_RISK: "state-high",
  INSUFFICIENT_HISTORY: "state-insufficient",
};

export function StateBadge({ state }: { state: BehavioralState | null }) {
  if (state === null) {
    return <span className="state-badge state-insufficient">Unavailable</span>;
  }
  return <span className={`state-badge ${CLASS[state]}`}>{LABEL[state]}</span>;
}
