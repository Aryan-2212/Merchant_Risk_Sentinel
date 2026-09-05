// Mirrors src/mrs/api/schemas.py field-for-field. Nothing here is invented: every
// type below matches a Pydantic response model actually returned by the Phase 8
// backend. Do not add fields the API does not return.

export type RiskLevel = "LOW" | "MEDIUM" | "HIGH" | "CRITICAL" | "INSUFFICIENT_EVIDENCE";

export type BehavioralState = "NORMAL" | "RISK_RISING" | "RECOVERY" | "HIGH_RISK" | "INSUFFICIENT_HISTORY";

export type PolicyAction = "ALLOW" | "MONITOR" | "STEP_UP_VERIFICATION" | "TEMPORARY_REVIEW" | "ESCALATE";

// alerts.severity uses the same string set as unified_risk_level, but LOW never
// appears (mrs.policy.rules: ALLOW decisions never become an alert row).
export type AlertSeverity = Exclude<RiskLevel, "LOW">;

export interface CustomerOut {
  customer_id: number;
  x_customer_id: number;
  y_customer_id: number;
  mean_amount: number;
  std_amount: number;
  mean_nb_tx_per_day: number;
  nb_terminals: number;
  available_terminals: number[];
}

export interface TerminalOut {
  terminal_id: number;
  x_terminal_id: number;
  y_terminal_id: number;
}

export interface TransactionOut {
  transaction_id: number;
  tx_datetime: string;
  customer_id: number;
  terminal_id: number;
  tx_amount: number;
  split: string;
}

export interface RiskScoreOut {
  transaction_id: number;
  customer_id: number;
  terminal_id: number;
  transaction_risk: number | null;
  transaction_risk_severity: number | null;
  terminal_risk_state: BehavioralState | null;
  terminal_risk_severity: number | null;
  customer_risk_state: BehavioralState | null;
  customer_risk_severity: number | null;
  unified_risk_level: RiskLevel;
  contributing_signals: string[];
  model_version: string;
  transaction_risk_threshold: number;
  feature_version: string;
  computed_at: string;
}

export interface AlertSummaryOut {
  alert_id: number;
  transaction_id: number;
  customer_id: number;
  terminal_id: number;
  severity: AlertSeverity;
  recommended_action: PolicyAction | null;
  status: string;
  /** Batch row-insertion time, identical across alerts loaded in the same run --
   * not when the activity happened. Prefer tx_datetime for display. */
  created_at: string;
  /** When the alerting transaction actually occurred. */
  tx_datetime: string | null;
}

export interface AlertDetailOut extends AlertSummaryOut {
  reason: string;
  evidence: Record<string, unknown>;
  policy_version: string | null;
}

export interface TransactionDetailOut {
  transaction: TransactionOut;
  risk_score: RiskScoreOut | null;
  alert: AlertDetailOut | null;
  policy_version: string | null;
}

export interface PaginatedAlerts {
  items: AlertSummaryOut[];
  total: number;
  limit: number;
  offset: number;
}

export interface ReplayItemOut {
  transaction: TransactionOut;
  risk_score: RiskScoreOut | null;
  alert: AlertSummaryOut | null;
}

export interface ReplayPage {
  items: ReplayItemOut[];
  count: number;
  next_cursor: string | null;
}

export interface ReplayBounds {
  min_tx_datetime: string;
  max_tx_datetime: string;
  total_transactions: number;
}

export interface AnalystResponseOut {
  transaction_id: number;
  unified_risk_level: RiskLevel;
  deterministic_action: PolicyAction;
  policy_version: string | null;
  summary: string;
  evidence_explanation: string;
  recommended_action: string;
  recommendation_rationale: string;
  confidence: string;
  caveats: string[];
  is_fallback: boolean;
  fallback_reason: string | null;
  analyst_model: string | null;
}

export interface PaginatedRiskHistory {
  items: RiskScoreOut[];
  total: number;
  limit: number;
  offset: number;
}

export interface AuditLogOut {
  audit_id: number;
  transaction_id: number | null;
  alert_id: number | null;
  event_type: string;
  payload: Record<string, unknown>;
  model_version: string | null;
  created_at: string;
}

export interface OverviewStats {
  total_transactions: number;
  total_customers: number;
  total_terminals: number;
  total_risk_scores: number;
  total_alerts: number;
  risk_level_counts: Partial<Record<RiskLevel, number>>;
  alert_action_counts: Partial<Record<PolicyAction, number>>;
  alert_status_counts: Record<string, number>;
  customers_at_risk: number;
  terminals_at_risk: number;
  risk_exposure_amount: number;
}

export interface HealthOut {
  status: string;
  ai_analyst_configured: boolean;
}

export interface RiskActivityPoint {
  date: string;
  transaction_high: number;
  customer_high: number;
  terminal_high: number;
  elevated_transactions: number;
  total_scored: number;
}

export interface EntityDeviation {
  entity_type: EntityType;
  entity_id: number;
  current_rate: number | null;
  baseline_rate: number | null;
  current_transaction_count: number;
  baseline_transaction_count: number;
  recent_window_days: number;
  baseline_window_days: number;
}

export interface EntityAtRiskRow {
  entity_type: EntityType;
  entity_id: number;
  risk_state: BehavioralState;
  risk_severity: number | null;
  current_rate: number;
  baseline_rate: number | null;
  recent_transaction_count: number;
  last_activity: string;
}

export type EntityType = "customer" | "terminal";

export interface NetworkNode {
  id: string;
  entity_type: EntityType;
  entity_id: number;
  risk_state: BehavioralState | null;
  risk_severity: number | null;
  is_focus: boolean;
}

export interface NetworkEdge {
  source: string;
  target: string;
  weight: number;
}

export interface NetworkGraph {
  nodes: NetworkNode[];
  edges: NetworkEdge[];
  focus_ids: string[];
  /** Only set when the live_window query param was used -- the single most recent
   * transaction_id in that window, so the newest arrival can be highlighted without
   * a second request. null for the default (unwindowed) graph. */
  latest_transaction_id: number | null;
}

/** GET/POST /live/* -- the Continuous Simulated Live Stream's real backend-thread
 * state (mrs.live.manager). `running` reflects the actual producer, not a client-side
 * toggle, so a reload or a second tab always sees the truth. */
export interface LiveStreamStatus {
  running: boolean;
  interval_seconds: number;
  n_generated: number;
  last_transaction_id: number | null;
  last_tx_datetime: string | null;
  started_at: string | null;
  error: string | null;
}
