import type { PolicyAction } from "../../lib/types";
import "./badges.css";

const LABEL: Record<PolicyAction, string> = {
  ALLOW: "Allow",
  MONITOR: "Monitor",
  STEP_UP_VERIFICATION: "Step-up verification",
  TEMPORARY_REVIEW: "Temporary review",
  ESCALATE: "Escalate",
};

const CLASS: Record<PolicyAction, string> = {
  ALLOW: "action-allow",
  MONITOR: "action-monitor",
  STEP_UP_VERIFICATION: "action-step-up",
  TEMPORARY_REVIEW: "action-review",
  ESCALATE: "action-escalate",
};

/** Text-only counterpart to CLASS above (color, no badge chrome) -- same severity
 * mapping, for inline words that need to read as urgent without becoming a pill. */
export const ACTION_COLOR_CLASS: Record<PolicyAction, string> = {
  ALLOW: "action-color-allow",
  MONITOR: "action-color-monitor",
  STEP_UP_VERIFICATION: "action-color-step-up",
  TEMPORARY_REVIEW: "action-color-review",
  ESCALATE: "action-color-escalate",
};

export function ActionBadge({ action }: { action: PolicyAction }) {
  return <span className={`action-badge ${CLASS[action]}`}>{LABEL[action]}</span>;
}
