import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { useNavigate, Link } from "react-router-dom";
import { api } from "../lib/api";
import { formatAmount, formatDateTime } from "../lib/format";
import { Loading, ErrorBlock, EmptyState } from "../components/common/States";
import { RiskBadge } from "../components/risk/RiskBadge";
import { ActionBadge } from "../components/risk/ActionBadge";

const PAGE_SIZE = 50;

/**
 * Browses the same chronological stream Replay plays back (GET /replay/transactions)
 * -- there is no separate "list transactions" endpoint, and Dev Plan Sec 22's replay
 * module is explicit that it already serves the full historical stream, so this page
 * reuses it rather than duplicating pagination logic. Replay owns pacing/playback;
 * this page is the static investigation table over the same data.
 */
export function TransactionsExplorer() {
  const [start, setStart] = useState("");
  const [end, setEnd] = useState("");
  const [cursorStack, setCursorStack] = useState<(string | undefined)[]>([undefined]);
  const navigate = useNavigate();

  const cursor = cursorStack[cursorStack.length - 1];

  const page = useQuery({
    queryKey: ["explorer", { start, end, cursor }],
    queryFn: () =>
      api.replayTransactions({
        after_cursor: cursor,
        start: start ? new Date(start).toISOString() : undefined,
        end: end ? new Date(end).toISOString() : undefined,
        limit: PAGE_SIZE,
      }),
    placeholderData: (prev) => prev,
  });

  function applyFilters(e: React.FormEvent) {
    e.preventDefault();
    setCursorStack([undefined]);
  }

  return (
    <div className="page">
      <div className="page-header">
        <div>
          <h1 className="page-title">Transactions</h1>
          <p className="page-subtitle">Chronological stream of scored transactions.</p>
        </div>
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
            onChange={(e) => setEnd(e.target.value)}
          />
        </div>
        <button className="btn btn-primary" type="submit">
          Apply
        </button>
      </form>

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
