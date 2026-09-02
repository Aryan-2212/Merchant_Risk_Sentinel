import type { RiskLevel } from "../../lib/types";
import "./badges.css";

const LABEL: Record<RiskLevel, string> = {
  LOW: "Low",
  MEDIUM: "Medium",
  HIGH: "High",
  CRITICAL: "Critical",
  INSUFFICIENT_EVIDENCE: "Insufficient evidence",
};

const CLASS: Record<RiskLevel, string> = {
  LOW: "risk-low",
  MEDIUM: "risk-medium",
  HIGH: "risk-high",
  CRITICAL: "risk-critical",
  INSUFFICIENT_EVIDENCE: "risk-unknown",
};

export function RiskBadge({ level, size = "md" }: { level: RiskLevel; size?: "sm" | "md" }) {
  return (
    <span className={`badge ${CLASS[level]} ${size === "sm" ? "badge-sm" : ""}`}>
      <span className="badge-dot" aria-hidden="true" />
      {LABEL[level]}
    </span>
  );
}
