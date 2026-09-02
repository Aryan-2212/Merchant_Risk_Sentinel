import type { BehavioralState } from "./types";

/** Severity (0/1/2, or null for "unavailable") -> the fixed status-palette color used
 * everywhere risk is drawn: RiskDecomposition's meters, the entity network's node
 * fills, badges. One mapping, reused, so a color never means something different in
 * two places. */
export function severityColor(severity: number | null): string {
  if (severity === null) return "var(--risk-unknown)";
  if (severity === 0) return "var(--risk-low)";
  if (severity === 1) return "var(--risk-medium)";
  return "var(--risk-high)";
}

const STATE_SEVERITY: Record<BehavioralState, number | null> = {
  NORMAL: 0,
  RISK_RISING: 1,
  RECOVERY: 1,
  HIGH_RISK: 2,
  INSUFFICIENT_HISTORY: null,
};

export function stateColor(state: BehavioralState | null): string {
  if (state === null) return severityColor(null);
  return severityColor(STATE_SEVERITY[state]);
}
