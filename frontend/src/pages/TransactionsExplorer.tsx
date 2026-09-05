import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { useNavigate, Link } from "react-router-dom";
import { api } from "../lib/api";
import { formatAmount, formatDateTime } from "../lib/format";
import { Loading, ErrorBlock, EmptyState } from "../components/common/States";
import { RiskBadge } from "../components/risk/RiskBadge";
import { ActionBadge } from "../components/risk/ActionBadge";

const PAGE_SIZE = 50;

type Source = "benchmark" | "recent";

const DATASET_CONTEXT: Record<Source, string> = {
  benchmark: "HISTORICAL DATASET · APR–SEP 2018",
  recent: "SIMULATED RECENT OPERATIONAL STREAM · AUG–SEP 2026",
};

/** ISO instant -> the local-time value a <input type="datetime-local"> accepts
 * ("YYYY-MM-DDTHH:mm"), truncated to minutes (that input has no seconds field). */
function toDatetimeLocalValue(iso: string): string {
  const d = new Date(iso);
  const pad = (n: number) => String(n).padStart(2, "0");
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}T${pad(d.getHours())}:${pad(d.getMinutes())}`;
}

/**
 * Browses the same chronological streams Replay plays back (GET /replay/transactions
 * for the frozen benchmark, GET /recent/transactions for the Simulated Recent
 * Operational Stream -- see mrs.data.recent_stream) -- there is no separate "list
 * transactions" endpoint for either, and both routers already serve their own full
 * stream, so this page reuses them rather than duplicating pagination logic. Replay
 * owns pacing/playback; this page is the static investigation table over the same
 * data, with an explicit source toggle (mirroring Replay.tsx's) so the two datasets
 * are never silently conflated and the recent stream isn't a hidden/undiscoverable URL.
 */
export function TransactionsExplorer() {
  const [source, setSource] = useState<Source>("benchmark");
  const [start, setStart] = useState("");
  const [end, setEnd] = useState("");
  const [rangeError, setRangeError] = useState<string | null>(null);
  const [cursorStack, setCursorStack] = useState<(string | undefined)[]>([undefined]);
  const navigate = useNavigate();

  const cursor = cursorStack[cursorStack.length - 1];

  const bounds = useQuery({
    queryKey: ["explorer-bounds", source],
    queryFn: () => (source === "benchmark" ? api.replayBounds() : api.recentBounds()),
    retry: false,
  });

  const page = useQuery({
    queryKey: ["explorer", source, { start, end, cursor }],
    queryFn: () => {
      const fetchPage = source === "benchmark" ? api.replayTransactions : api.recentTransactions;
      return fetchPage({
        after_cursor: cursor,
        start: start ? new Date(start).toISOString() : undefined,
        end: end ? new Date(end).toISOString() : undefined,
        limit: PAGE_SIZE,
      });
    },
    placeholderData: (prev) => prev,
  });

  function selectSource(next: Source) {
    if (next === source) return;
    setSource(next);
    setStart("");
    setEnd("");
    setRangeError(null);
    setCursorStack([undefined]);
  }

  /** Fills the from/to fields with the active source's own valid bounds -- the
   * fastest way to reach a working range without hand-typing into a locale-ordered
   * <input type="datetime-local">, which is exactly what produced the reported
   * "Please enter valid date and time" browser error when typed in day/month/year
   * order. This does not bypass validation: the values written here are the real
   * GET .../bounds values, always valid and always start <= end. */
  function fillActiveRange() {
    if (!bounds.data) return;
    setStart(toDatetimeLocalValue(bounds.data.min_tx_datetime));
    setEnd(toDatetimeLocalValue(bounds.data.max_tx_datetime));
    setRangeError(null);
  }

  function applyFilters(e: React.FormEvent) {
    e.preventDefault();
    if (start && end && new Date(start) > new Date(end)) {
      setRangeError("From must be on or before To.");
      return;
    }
    setRangeError(null);
    setCursorStack([undefined]);
  }

  return (
    <div className="page">
      <div className="page-header">
        <div>
          <h1 className="page-title">Transactions</h1>
          <p className="page-subtitle">{DATASET_CONTEXT[source]}</p>
        </div>
      </div>

      <div className="toolbar" role="group" aria-label="Data source">
        <button
          className={`btn ${source === "benchmark" ? "btn-primary" : ""}`}
          type="button"
          onClick={() => selectSource("benchmark")}
          aria-pressed={source === "benchmark"}
        >
          Benchmark Dataset
        </button>
        <button
          className={`btn ${source === "recent" ? "btn-primary" : ""}`}
          type="button"
          onClick={() => selectSource("recent")}
          aria-pressed={source === "recent"}
        >
          Recent Simulated Stream
        </button>
      </div>

      <form className="toolbar" onSubmit={applyFilters}>
        <div>
          <label className="field-label" htmlFor="start">
            From
          </label>
          <br />
          <input
            id="start"
            type="datetime-local"
            className="text-input"
            value={start}
            min={bounds.data ? toDatetimeLocalValue(bounds.data.min_tx_datetime) : undefined}
            max={bounds.data ? toDatetimeLocalValue(bounds.data.max_tx_datetime) : undefined}
            onChange={(e) => setStart(e.target.value)}
          />
        </div>
        <div>
          <label className="field-label" htmlFor="end">
            To
          </label>
          <br />
          <input
            id="end"
            type="datetime-local"
            className="text-input"
            value={end}
            min={bounds.data ? toDatetimeLocalValue(bounds.data.min_tx_datetime) : undefined}
            max={bounds.data ? toDatetimeLocalValue(bounds.data.max_tx_datetime) : undefined}
            onChange={(e) => setEnd(e.target.value)}
          />
        </div>
        <button className="btn btn-primary" type="submit">
          Apply
        </button>
        {bounds.data && (
          <button className="btn" type="button" onClick={fillActiveRange}>
            Use full {source === "benchmark" ? "benchmark" : "recent"} range
          </button>
        )}
      </form>
      {rangeError && <p className="field-error">{rangeError}</p>}

      {page.isLoading && <Loading label="Loading transactions…" />}
      {page.isError && <ErrorBlock error={page.error} onRetry={() => page.refetch()} />}
      {page.data && page.data.items.length === 0 && <EmptyState>No transactions in this range.</EmptyState>}

      {page.data && page.data.items.length > 0 && (
        <>
          <div className="table-scroll">
            <table className="data-table">
              <thead>
                <tr>
                  <th>Transaction</th>
                  <th>Time</th>
                  <th>Amount</th>
                  <th>Customer</th>
                  <th>Terminal</th>
                  <th>Risk</th>
                  <th>Action</th>
                </tr>
              </thead>
              <tbody>
                {page.data.items.map(({ transaction, risk_score, alert }) => (
                  <tr
                    key={transaction.transaction_id}
                    data-clickable="true"
                    onClick={() => navigate(`/transactions/${transaction.transaction_id}`)}
                  >
                    <td>
                      <Link className="link-id mono" to={`/transactions/${transaction.transaction_id}`} onClick={(e) => e.stopPropagation()}>
                        TX_{transaction.transaction_id}
                      </Link>
                    </td>
                    <td>{formatDateTime(transaction.tx_datetime)}</td>
                    <td>{formatAmount(transaction.tx_amount)}</td>
                    <td>
                      {/* stopPropagation: the row itself navigates to this transaction; a
                          nested Link must win over that when clicked directly, not fire both. */}
                      <Link className="link-id" to={`/customers/${transaction.customer_id}`} onClick={(e) => e.stopPropagation()}>
                        CUST_{transaction.customer_id}
                      </Link>
                    </td>
                    <td>
                      <Link className="link-id" to={`/terminals/${transaction.terminal_id}`} onClick={(e) => e.stopPropagation()}>
                        TERM_{transaction.terminal_id}
                      </Link>
                    </td>
                    <td>{risk_score ? <RiskBadge level={risk_score.unified_risk_level} size="sm" /> : "—"}</td>
                    <td>{alert?.recommended_action ? <ActionBadge action={alert.recommended_action} /> : "—"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <div className="pagination">
            <span>{page.data.items.length} rows this page</span>
            <button className="btn" disabled={cursorStack.length <= 1} onClick={() => setCursorStack((s) => s.slice(0, -1))}>
              Previous
            </button>
            <button
              className="btn"
              disabled={!page.data.next_cursor}
              onClick={() => setCursorStack((s) => [...s, page.data!.next_cursor ?? undefined])}
            >
              Next
            </button>
          </div>
        </>
      )}
    </div>
  );
}
