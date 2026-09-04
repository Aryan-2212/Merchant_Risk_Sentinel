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
  // Draft values live in the inputs; only an Apply press moves them into the query,
  // so a half-typed date never fires a request (same convention as Transactions).
  const [startDraft, setStartDraft] = useState("");
  const [endDraft, setEndDraft] = useState("");
  const [range, setRange] = useState<{ start: string; end: string }>({ start: "", end: "" });
  const [offset, setOffset] = useState(0);
  const navigate = useNavigate();

  const alerts = useQuery({
    queryKey: ["alerts", { severity, status, range, offset }],
    queryFn: () =>
      api.listAlerts({
        severity: severity || undefined,
        status: status || undefined,
        start: range.start ? new Date(range.start).toISOString() : undefined,
        end: range.end ? new Date(range.end).toISOString() : undefined,
        limit: PAGE_SIZE,
        offset,
      }),
    placeholderData: (prev) => prev,
  });

  function updateFilter(setter: (v: string) => void, value: string) {
    setter(value);
    setOffset(0);
  }

  function applyRange(event: React.FormEvent) {
    event.preventDefault();
    setRange({ start: startDraft, end: endDraft });
    setOffset(0);
  }

  function clearRange() {
    setStartDraft("");
    setEndDraft("");
    setRange({ start: "", end: "" });
    setOffset(0);
  }

  const rangeActive = range.start !== "" || range.end !== "";

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

        <form className="toolbar-range" onSubmit={applyRange}>
          <div>
            <label className="field-label" htmlFor="alert-start">
              From
            </label>
            <br />
            <input
              id="alert-start"
              type="datetime-local"
              className="text-input"
              value={startDraft}
              onChange={(e) => setStartDraft(e.target.value)}
            />
          </div>
          <div>
            <label className="field-label" htmlFor="alert-end">
              To
            </label>
            <br />
            <input
              id="alert-end"
              type="datetime-local"
              className="text-input"
              value={endDraft}
              onChange={(e) => setEndDraft(e.target.value)}
            />
          </div>
          <button className="btn btn-primary" type="submit">
            Apply
          </button>
          {rangeActive && (
            <button className="btn" type="button" onClick={clearRange}>
              Clear
            </button>
          )}
        </form>
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
                  <th>Transaction time</th>
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
                    <td className="mono">{alert.tx_datetime ? formatDateTime(alert.tx_datetime) : "—"}</td>
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
