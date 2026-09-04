import type {
  AlertDetailOut,
  AnalystResponseOut,
  AuditLogOut,
  CustomerOut,
  EntityAtRiskRow,
  EntityDeviation,
  EntityType,
  HealthOut,
  NetworkGraph,
  OverviewStats,
  PaginatedAlerts,
  PaginatedRiskHistory,
  ReplayBounds,
  ReplayItemOut,
  ReplayPage,
  RiskActivityPoint,
  TerminalOut,
  TransactionDetailOut,
} from "./types";

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";

/** Thrown for any non-2xx response; callers surface `.status`/`.detail` to the user
 * rather than a generic failure so a 404 ("not found") reads differently from a 500
 * ("risk data could not be loaded"). */
export class ApiError extends Error {
  status: number;
  detail: string;

  constructor(status: number, detail: string) {
    super(detail);
    this.name = "ApiError";
    this.status = status;
    this.detail = detail;
  }
}

async function request<T>(path: string, params?: Record<string, string | number | boolean | undefined>): Promise<T> {
  const url = new URL(API_BASE + path);
  if (params) {
    for (const [key, value] of Object.entries(params)) {
      if (value !== undefined) url.searchParams.set(key, String(value));
    }
  }
  let res: Response;
  try {
    res = await fetch(url.toString());
  } catch {
    throw new ApiError(0, "Could not reach the Merchant Risk Sentinel API.");
  }
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      if (typeof body?.detail === "string") detail = body.detail;
    } catch {
      // response body wasn't JSON -- keep statusText
    }
    throw new ApiError(res.status, detail);
  }
  return res.json() as Promise<T>;
}

export const api = {
  health: () => request<HealthOut>("/health"),

  overviewStats: () => request<OverviewStats>("/stats/overview"),
  riskActivity: (days = 30) => request<RiskActivityPoint[]>("/stats/risk-activity", { days }),
  recentActivity: (limit = 20, levels?: string) => request<ReplayItemOut[]>("/stats/recent-activity", { limit, levels }),
  entityNetwork: (focus?: { type: EntityType; id: number }) =>
    request<NetworkGraph>("/stats/network", focus ? { focus_type: focus.type, focus_id: focus.id } : undefined),
  terminalsAtRisk: (limit = 8) => request<EntityAtRiskRow[]>("/stats/terminals-at-risk", { limit }),

  getTransaction: (id: number) => request<TransactionDetailOut>(`/transactions/${id}`),
  getTransactionAudit: (id: number) => request<AuditLogOut[]>(`/transactions/${id}/audit`),
  getTransactionAnalyst: (id: number) => request<AnalystResponseOut>(`/transactions/${id}/analyst`),

  getCustomer: (id: number) => request<CustomerOut>(`/customers/${id}`),
  getCustomerRiskHistory: (id: number, limit = 50, offset = 0) =>
    request<PaginatedRiskHistory>(`/customers/${id}/risk`, { limit, offset }),
  getCustomerDeviation: (id: number) => request<EntityDeviation>(`/customers/${id}/deviation`),

  getTerminal: (id: number) => request<TerminalOut>(`/terminals/${id}`),
  getTerminalRiskHistory: (id: number, limit = 50, offset = 0) =>
    request<PaginatedRiskHistory>(`/terminals/${id}/risk`, { limit, offset }),
  getTerminalDeviation: (id: number) => request<EntityDeviation>(`/terminals/${id}/deviation`),

  listAlerts: (params: {
    status?: string;
    severity?: string;
    customer_id?: number;
    terminal_id?: number;
    start?: string;
    end?: string;
    limit?: number;
    offset?: number;
  }) => request<PaginatedAlerts>("/alerts", params),
  getAlert: (id: number) => request<AlertDetailOut>(`/alerts/${id}`),

  replayBounds: () => request<ReplayBounds>("/replay/bounds"),
  replayTransactions: (params: {
    after_cursor?: string;
    start?: string;
    end?: string;
    customer_id?: number;
    terminal_id?: number;
    desc?: boolean;
    limit?: number;
  }) => request<ReplayPage>("/replay/transactions", params),
};
