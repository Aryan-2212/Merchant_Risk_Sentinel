import { useCallback, useEffect, useRef, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { useNavigate } from "react-router-dom";
import { api } from "../lib/api";
import type { ReplayItemOut } from "../lib/types";
import { formatAmount, formatCount, formatDateTime, formatDateTimeCompact } from "../lib/format";
import { Loading, ErrorBlock } from "../components/common/States";
import { RiskBadge } from "../components/risk/RiskBadge";
import { ActionBadge } from "../components/risk/ActionBadge";
import "./Replay.css";

const SPEEDS = [1, 5, 20, 100] as const;
const TICK_MS = 900;
const FEED_CAP = 120;

/**
 * Historical replay / demo mode -- not a live production stream (Dev Plan Sec 22).
 * The backend owns chronological ordering and cursor semantics (GET /replay/bounds,
 * GET /replay/transactions); this page only owns pacing and presentation. "Speed"
 * controls how many already-computed rows are revealed per tick, not a server-side
 * delay -- there is none to control.
 */
export function Replay() {
  const navigate = useNavigate();
  const bounds = useQuery({ queryKey: ["replay-bounds"], queryFn: api.replayBounds });

  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState<(typeof SPEEDS)[number]>(5);
  const [feed, setFeed] = useState<ReplayItemOut[]>([]);
  const [error, setError] = useState<unknown>(null);
  const [atEnd, setAtEnd] = useState(false);

  const cursorRef = useRef<string | undefined>(undefined);
  const inFlightRef = useRef(false);

  const reset = useCallback(() => {
    setPlaying(false);
    setFeed([]);
    setError(null);
    setAtEnd(false);
    cursorRef.current = undefined;
  }, []);

  useEffect(() => {
    if (!playing) return;
    let cancelled = false;

    async function tick() {
      if (inFlightRef.current) return;
      inFlightRef.current = true;
      try {
        const page = await api.replayTransactions({ after_cursor: cursorRef.current, limit: speed });
        if (cancelled) return;
        setFeed((prev) => [...page.items].reverse().concat(prev).slice(0, FEED_CAP));
        cursorRef.current = page.next_cursor ?? cursorRef.current;
        if (!page.next_cursor) {
          setAtEnd(true);
          setPlaying(false);
        }
      } catch (err) {
        if (!cancelled) {
          setError(err);
          setPlaying(false);
        }
      } finally {
        inFlightRef.current = false;
      }
    }

    tick();
    const id = window.setInterval(tick, TICK_MS);
    return () => {
      cancelled = true;
      window.clearInterval(id);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [playing, speed]);

  return (
    <div className="page">
      <div className="page-header">
        <div>
          <h1 className="page-title">Replay</h1>
          <p className="page-subtitle">Historical replay of simulated benchmark data -- not a live stream.</p>
        </div>
      </div>

      {bounds.isLoading && <Loading label="Loading replay bounds…" />}
      {bounds.isError && <ErrorBlock error={bounds.error} onRetry={() => bounds.refetch()} />}

      {bounds.data && (
        <div className="card replay-controls">
          <div className="replay-buttons">
            <button className="btn btn-primary" onClick={() => setPlaying((p) => !p)} disabled={atEnd && !playing}>
              {playing ? "Pause" : "Play"}
            </button>
            <button
              className="btn"
              onClick={() => {
                reset();
              }}
            >
              Reset
            </button>
            <div className="replay-speeds" role="group" aria-label="Playback speed">
              {SPEEDS.map((s) => (
                <button
                  key={s}
                  className={`btn replay-speed-btn ${speed === s ? "active" : ""}`}
                  onClick={() => setSpeed(s)}
                >
                  {s}×
                </button>
              ))}
            </div>
          </div>
          <div className="replay-range">
            <span>{formatDateTime(bounds.data.min_tx_datetime)}</span>
            <span className="replay-range-total mono">{formatCount(bounds.data.total_transactions)} transactions</span>
            <span>{formatDateTime(bounds.data.max_tx_datetime)}</span>
          </div>
        </div>
      )}

      {error !== null && <ErrorBlock error={error} onRetry={() => setError(null)} />}
      {atEnd && <p className="replay-end-note">Reached the end of the historical stream.</p>}

      <div className="replay-feed">
        {feed.length === 0 && !playing && (
          <div className="state-block state-empty">Press Play to begin replaying the historical transaction stream.</div>
        )}
        {feed.map((item) => (
          <div
            key={item.transaction.transaction_id}
            className="replay-row"
            onClick={() => navigate(`/transactions/${item.transaction.transaction_id}`)}
          >
            <span className="replay-row-time mono">{formatDateTimeCompact(item.transaction.tx_datetime)}</span>
            <span className="replay-row-id mono">TX_{item.transaction.transaction_id}</span>
            <span className="replay-row-amount mono">{formatAmount(item.transaction.tx_amount)}</span>
            <span className="replay-row-entities mono">
              CUST_{item.transaction.customer_id} · TERM_{item.transaction.terminal_id}
            </span>
            {item.risk_score ? <RiskBadge level={item.risk_score.unified_risk_level} size="sm" /> : <span>—</span>}
            {item.alert?.recommended_action ? <ActionBadge action={item.alert.recommended_action} /> : <span />}
          </div>
        ))}
      </div>
    </div>
  );
}
