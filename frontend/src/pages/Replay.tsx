import { useCallback, useEffect, useRef, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { useNavigate } from "react-router-dom";
import { api } from "../lib/api";
import type { ReplayItemOut } from "../lib/types";
import { formatAmount, formatCount, formatDateTime, formatDateTimeCompact } from "../lib/format";
import { Loading, ErrorBlock, EmptyState } from "../components/common/States";
import { RiskBadge } from "../components/risk/RiskBadge";
import { ActionBadge } from "../components/risk/ActionBadge";
import { Icon } from "../components/common/Icon";
import "./Replay.css";

const SPEEDS = [1, 5, 20, 100] as const;
const TICK_MS = 900;
const FEED_CAP = 120;

type Source = "benchmark" | "recent";

const SOURCE_LABEL: Record<Source, string> = {
  benchmark: "Fraud Detection Handbook (Apr–Sep 2018) — frozen benchmark",
  recent: "Simulated Recent Operational Stream (Aug–Sep 2026) — demo data, not real transactions",
};

/**
 * Historical replay / demo mode -- not a live production stream (Dev Plan Sec 22).
 * The backend owns chronological ordering and cursor semantics (GET /replay/bounds,
 * GET /replay/transactions, or their /recent/* siblings for the Simulated Recent
 * Operational Stream -- see mrs.data.recent_stream); this page only owns pacing,
 * presentation, and which of the two already-separate backend streams to read from.
 * "Speed" controls how many already-computed rows are revealed per tick, not a
 * server-side delay -- there is none to control.
 */
export function Replay() {
  const navigate = useNavigate();
  const [source, setSource] = useState<Source>("benchmark");
  const bounds = useQuery({
    queryKey: ["replay-bounds", source],
    queryFn: () => (source === "benchmark" ? api.replayBounds() : api.recentBounds()),
    retry: false,
  });

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

  const selectSource = useCallback(
    (next: Source) => {
      if (next === source) return;
      setSource(next);
      reset();
    },
    [source, reset],
  );

  useEffect(() => {
    if (!playing) return;
    let cancelled = false;

    async function tick() {
      if (inFlightRef.current) return;
      inFlightRef.current = true;
      try {
        const fetchPage = source === "benchmark" ? api.replayTransactions : api.recentTransactions;
        const page = await fetchPage({ after_cursor: cursorRef.current, limit: speed });
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
  }, [playing, speed, source]);

  return (
    <div className="page">
      <div className="page-header">
        <div>
          <h1 className="page-title">Replay</h1>
          <p className="page-subtitle">{SOURCE_LABEL[source]} -- not a live stream.</p>
        </div>
      </div>

      <div className="replay-buttons" role="group" aria-label="Data source">
        <button
          className={`btn replay-speed-btn ${source === "benchmark" ? "active" : ""}`}
          onClick={() => selectSource("benchmark")}
          aria-pressed={source === "benchmark"}
        >
          Benchmark Dataset
        </button>
        <button
          className={`btn replay-speed-btn ${source === "recent" ? "active" : ""}`}
          onClick={() => selectSource("recent")}
          aria-pressed={source === "recent"}
        >
          Recent Simulated Stream
        </button>
      </div>

      {bounds.isLoading && <Loading label="Loading replay bounds…" />}
      {bounds.isError && <ErrorBlock error={bounds.error} onRetry={() => bounds.refetch()} />}

      {bounds.data && (
        <div className="card replay-controls">
          <div className="replay-buttons">
            <button className="btn btn-primary replay-play-btn" onClick={() => setPlaying((p) => !p)} disabled={atEnd && !playing}>
              <Icon name={playing ? "pause" : "play_arrow"} size={16} />
              {playing ? "Pause" : "Play"}
            </button>
            <button className="btn" onClick={reset}>
              <Icon name="restart_alt" size={14} />
              Reset
            </button>
            <div className="replay-speeds" role="group" aria-label="Playback speed">
              {SPEEDS.map((s) => (
                <button
                  key={s}
                  className={`btn replay-speed-btn ${speed === s ? "active" : ""}`}
                  onClick={() => setSpeed(s)}
                  aria-pressed={speed === s}
                >
                  {s}×
                </button>
              ))}
            </div>
            {playing && (
              <span className="replay-live-indicator">
                <span className="replay-live-dot" aria-hidden="true" />
                Replaying
              </span>
            )}
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

      <section className="card replay-stream-card">
        <h2 className="replay-section-title">
          <Icon name="history" size={18} className="replay-section-icon" />
          Transaction Stream
        </h2>

        {feed.length > 0 && (
          <div className="replay-row replay-row-head" aria-hidden="true">
            <span>Timestamp</span>
            <span>Transaction</span>
            <span>Amount</span>
            <span>Customer · Terminal</span>
            <span>Risk</span>
            <span>Action</span>
          </div>
        )}

        <div className="replay-feed">
          {feed.length === 0 && !playing && (
            <EmptyState>Press Play to begin replaying the historical transaction stream.</EmptyState>
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
      </section>
    </div>
  );
}
