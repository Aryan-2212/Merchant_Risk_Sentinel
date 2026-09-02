import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { useNavigate } from "react-router-dom";
import { api } from "../lib/api";
import { formatDateTime } from "../lib/format";
import { Loading, ErrorBlock, EmptyState } from "../components/common/States";
import { RiskBadge } from "../components/risk/RiskBadge";
import { ActionBadge } from "../components/risk/ActionBadge";
import type { AlertSeverity } from "../lib/types";

const SEVERITIES: AlertSeverity[] = ["CRITICAL", "HIGH", "MEDIUM", "INSUFFICIENT_EVIDENCE"];
const PAGE_SIZE = 25;

export function Alerts() {
  const [severity, setSeverity] = useState<AlertSeverity | "">("");
  const [status, setStatus] = useState<string>("");
  const [offset, setOffset] = useState(0);
  const navigate = useNavigate();

  const alerts = useQuery({
    queryKey: ["alerts", { severity, status, offset }],
    queryFn: () =>
      api.listAlerts({
        severity: severity || undefined,
        status: status || undefined,
        limit: PAGE_SIZE,
        offset,
      }),
    placeholderData: (prev) => prev,
  });

  function updateFilter(setter: (v: string) => void, value: string) {
    setter(value);
    setOffset(0);
  }

  const total = alerts.data?.total ?? 0;
  const from = total === 0 ? 0 : offset + 1;
  const to = Math.min(offset + PAGE_SIZE, total);

  return (
    <div className="page">
      <div className="page-header">
        <div>
          <h1 className="page-title">Alerts</h1>
          <p className="page-subtitle">Deterministic policy decisions that required more than ALLOW.</p>
        </div>
      </div>

      <div className="toolbar">
        <div>
          <label className="field-label" htmlFor="severity-filter">
            Severity
          </label>
          <br />
          <select
            id="severity-filter"
            className="select"
            value={severity}
            onChange={(e) => updateFilter((v) => setSeverity(v as AlertSeverity | ""), e.target.value)}
          >
            <option value="">All</option>
            {SEVERITIES.map((s) => (
              <option key={s} value={s}>
                {s}
              </option>
            ))}
          </select>
        </div>
        <div>
          <label className="field-label" htmlFor="status-filter">
            Status
          </label>
          <br />
          <select
            id="status-filter"
            className="select"
            value={status}
            onChange={(e) => updateFilter(setStatus, e.target.value)}
          >
            <option value="">All</option>
            <option value="OPEN">Open</option>
          </select>
        </div>
      </div>

      {alerts.isLoading && <Loading label="Loading alerts…" />}
      {alerts.isError && <ErrorBlock error={alerts.error} onRetry={() => alerts.refetch()} />}

      {alerts.data && alerts.data.items.length === 0 && <EmptyState>No alerts match these filters.</EmptyState>}

      {alerts.data && alerts.data.items.length > 0 && (
        <>
          <div className="table-scroll">
            <table className="data-table">
              <thead>
                <tr>
                  <th>Severity</th>
                  <th>Transaction</th>
                  <th>Customer</th>
                  <th>Terminal</th>
                  <th>Action</th>
                  <th>Status</th>
                  <th>Created</th>
                </tr>
              </thead>
              <tbody>
                {alerts.data.items.map((alert) => (
                  <tr
                    key={alert.alert_id}
                    data-clickable="true"
                    onClick={() => navigate(`/alerts/${alert.alert_id}`)}
                  >
                    <td>
                      <RiskBadge level={alert.severity} size="sm" />
                    </td>
                    <td>TX_{alert.transaction_id}</td>
                    <td>CUST_{alert.customer_id}</td>
                    <td>TERM_{alert.terminal_id}</td>
                    <td>{alert.recommended_action && <ActionBadge action={alert.recommended_action} />}</td>
                    <td>{alert.status}</td>
                    <td className="mono">{formatDateTime(alert.created_at)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <div className="pagination">
            <span>
              {from}–{to} of {total.toLocaleString()}
            </span>
            <button className="btn" disabled={offset === 0} onClick={() => setOffset(Math.max(0, offset - PAGE_SIZE))}>
              Previous
            </button>
            <button className="btn" disabled={to >= total} onClick={() => setOffset(offset + PAGE_SIZE)}>
              Next
            </button>
          </div>
        </>
      )}
    </div>
  );
}
