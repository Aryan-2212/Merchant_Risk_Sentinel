import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Link, useSearchParams } from "react-router-dom";
import { api } from "../lib/api";
import type { EntityType, NetworkNode } from "../lib/types";
import { behavioralFinding } from "../lib/behavioralNarrative";
import { formatAmount, formatDateTimeCompact } from "../lib/format";
import { Loading, ErrorBlock, EmptyState } from "../components/common/States";
import { BackLink } from "../components/common/BackLink";
import { StateBadge } from "../components/risk/StateBadge";
import { EntityNetworkGraph } from "../components/network/EntityNetworkGraph";
import { AnalystPanel } from "../components/analyst/AnalystPanel";
import { Icon } from "../components/common/Icon";
import "./Network.css";

/** Rolling window size for GET /stats/network?live_window=N -- reused as-is for both
 * the graph itself and the "recent transactions" side panel's implicit scope. Small
 * enough to stay readable as a force-directed graph, large enough to reliably show a
 * terminal-centric cluster (multiple customers sharing one terminal) within it. */
const LIVE_WINDOW = 30;
/** Within the 1-2s range the Simulated Live Stream calls for -- polling, not
 * WebSockets/SSE, matching the existing architecture (Replay already polls its own
 * bounds/transactions on demand rather than holding a server-pushed connection). */
const LIVE_POLL_MS = 1500;

type NetworkMode = "investigate" | "live";

function detailPath(type: EntityType, id: number): string {
  return type === "terminal" ? `/terminals/${id}` : `/customers/${id}`;
}

function idLabel(type: EntityType, id: number): string {
  return type === "terminal" ? `TERM_${id}` : `CUST_${id}`;
}

function downloadGraph(label: string, data: unknown) {
  const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `network_${label}.json`;
  a.click();
  URL.revokeObjectURL(url);
}

/**
 * Entity Network investigation (approved reference: the "Investigation: T-XXXX
 * Cluster" design, matching this project's own entity_network_analytical_graph Stitch
 * export). Graph rendering reuses the existing EntityNetworkGraph component and its
 * real GET /stats/network data verbatim -- this page only adds the surrounding
 * investigation chrome (focus entity snapshot, connected-entities list, recent
 * transactions, AI synthesis) around it.
 *
 * Reference elements with no real backend equivalent were adapted rather than
 * invented:
 * - "First Seen"/geographic "Cross-region activity" -- no onboarding date or real
 *   geography exists in this dataset (x/y coordinates are anonymized, not geographic).
 *   Omitted rather than fabricated.
 * - "Shared Methods" (payment methods) -- no payment-method data exists in this
 *   dataset (established already on CustomerDetail). Replaced with real linked-entity
 *   counts from this same graph.
 * - The reference's "Recommendations" / ">> RECOMMENDATION: Isolate..." console text
 *   is a fabricated enforcement narrative this read-only system cannot act on and did
 *   not compute. Replaced with the real AI Risk Analyst (AnalystPanel, the same
 *   component TerminalDetail/AlertDetail/TransactionDetail already use) bound to the
 *   focus entity's most recent scored transaction -- genuine evidence-grounded
 *   synthesis, never an invented trace log.
 * - "2h ago" relative timestamps assume a live clock; this is historical replay data
 *   (Dev Plan Sec 22), so timestamps are shown as their real absolute date/time
 *   instead, matching Replay's own "never a live stream" convention.
 */
export function Network() {
  const queryClient = useQueryClient();
  const [searchParams, setSearchParams] = useSearchParams();
  const [showElevatedOnly, setShowElevatedOnly] = useState(false);
  // Bumped to discard hand-dragged node positions and re-run the force layout.
  const [layoutKey, setLayoutKey] = useState(0);

  // SIMULATED LIVE STREAM: a second graph-canvas mode, off by default. "investigate"
  // is the page's original, unmodified behavior (global, unwindowed, URL-driven focus
  // via searchParams); "live" polls GET /stats/network?live_window=N instead -- a
  // rolling recent-activity view of the Continuous Simulated Live Stream
  // (mrs.live.manager/mrs.live.continuous), never real production traffic. "Start
  // Simulation"/"Pause" call the real backend control plane (GET/POST /live/*) --
  // there is no separate manual script to run; the button IS the producer's on/off
  // switch. `livePlaying` reflects the actual backend thread state (liveStatus.data
  // .running), not a client-side-only toggle, so a reload or a second tab always
  // agrees with reality.
  const [mode, setMode] = useState<NetworkMode>("investigate");
  // Which API a selected node's "Recent Transactions" panel should read from -- set
  // at click time to whichever mode was active, so drilling into a node found via the
  // Live canvas correctly shows its Continuous Simulated Live Stream activity
  // (GET /live/transactions) rather than the frozen 2018 benchmark or the fixed
  // 21-day recent stream, either of which would come back empty for an entity whose
  // activity is only in the "live" split (mrs.api.routers.live, a thin sibling of
  // mrs.api.routers.recent, permanently scoped to split=="live").
  const [focusSource, setFocusSource] = useState<"benchmark" | "recent" | "live">("benchmark");

  const typeParam = searchParams.get("type");
  const idParam = searchParams.get("id");
  const focus: { type: EntityType; id: number } | undefined =
    (typeParam === "customer" || typeParam === "terminal") && idParam && Number.isFinite(Number(idParam))
      ? { type: typeParam, id: Number(idParam) }
      : undefined;

  const liveStatus = useQuery({
    queryKey: ["live-status"],
    queryFn: api.liveStreamStatus,
    enabled: mode === "live",
    refetchInterval: mode === "live" ? LIVE_POLL_MS : false,
  });
  const livePlaying = liveStatus.data?.running ?? false;

  const startLive = useMutation({
    mutationFn: () => api.startLiveStream(),
    onSuccess: (status) => queryClient.setQueryData(["live-status"], status),
  });
  const stopLive = useMutation({
    mutationFn: () => api.stopLiveStream(),
    onSuccess: (status) => queryClient.setQueryData(["live-status"], status),
  });

  const graph = useQuery({
    queryKey: mode === "live" ? ["network-live", LIVE_WINDOW] : ["network", focus],
    queryFn: () => (mode === "live" ? api.entityNetwork(undefined, LIVE_WINDOW) : api.entityNetwork(focus)),
    refetchInterval: mode === "live" && livePlaying ? LIVE_POLL_MS : false,
  });

  const deviation = useQuery({
    queryKey: ["network-focus-deviation", focus],
    queryFn: () => (focus!.type === "terminal" ? api.getTerminalDeviation(focus!.id) : api.getCustomerDeviation(focus!.id)),
    enabled: focus !== undefined,
  });

  const recentTx = useQuery({
    queryKey: ["network-recent-tx", focus, focusSource],
    queryFn: () => {
      const params = {
        [focus!.type === "terminal" ? "terminal_id" : "customer_id"]: focus!.id,
        desc: true,
        limit: 5,
      };
      if (focusSource === "live") return api.liveTransactions(params);
      if (focusSource === "recent") return api.recentTransactions(params);
      return api.replayTransactions(params);
    },
    enabled: focus !== undefined,
  });

  function selectNode(node: NetworkNode) {
    setFocusSource(mode === "live" ? "live" : "benchmark");
    setSearchParams({ type: node.entity_type, id: String(node.entity_id) });
  }

  function selectMode(next: NetworkMode) {
    if (next === mode) return;
    setMode(next);
    // Deliberately does NOT stop the backend producer -- it is genuinely continuous
    // (mrs.live.manager runs server-side, independent of which page/mode any client
    // is currently viewing) and only the Start/Pause button controls it. Switching
    // canvas mode only changes what this page polls/displays.
    setSearchParams({}); // leaving "investigate" (or entering it) drops any stale focus
  }

  function toggleLive() {
    if (livePlaying) {
      stopLive.mutate();
    } else {
      startLive.mutate();
    }
  }

  if (graph.isLoading) return <Loading label="Loading entity network…" />;
  if (graph.isError) return <ErrorBlock error={graph.error} onRetry={() => graph.refetch()} />;
  const data = graph.data!;

  const focusNode = focus ? data.nodes.find((n) => n.is_focus && n.entity_type === focus.type && n.entity_id === focus.id) : undefined;
  const connected = data.nodes.filter((n) => !n.is_focus);
  const weightById = new Map<string, number>();
  for (const e of data.edges) {
    if (focusNode) {
      if (e.source === focusNode.id) weightById.set(e.target, e.weight);
      if (e.target === focusNode.id) weightById.set(e.source, e.weight);
    }
  }
  const maxWeight = Math.max(1, ...connected.map((n) => weightById.get(n.id) ?? 0));
  const linkedCustomers = connected.filter((n) => n.entity_type === "customer").length;
  const linkedTerminals = connected.filter((n) => n.entity_type === "terminal").length;

  const displayGraph = showElevatedOnly
    ? {
        ...data,
        nodes: data.nodes.filter((n) => n.is_focus || n.risk_state === "HIGH_RISK" || n.risk_state === "RISK_RISING"),
        edges: data.edges.filter((e) => {
          const ids = new Set(
            data.nodes.filter((n) => n.is_focus || n.risk_state === "HIGH_RISK" || n.risk_state === "RISK_RISING").map((n) => n.id),
          );
          return ids.has(e.source) && ids.has(e.target);
        }),
      }
    : data;

  const latestTxId = recentTx.data?.items[0]?.transaction.transaction_id;
  const currentRatePct = deviation.data?.current_rate !== null && deviation.data?.current_rate !== undefined ? deviation.data.current_rate * 100 : null;
  const deviationPp =
    deviation.data?.current_rate !== null &&
    deviation.data?.current_rate !== undefined &&
    deviation.data?.baseline_rate !== null &&
    deviation.data?.baseline_rate !== undefined
      ? (deviation.data.current_rate - deviation.data.baseline_rate) * 100
      : null;

  return (
    <div className="page page-wide net">
      {focus && <BackLink to="/network" label="Back to Network" />}
      <div className="net-header">
        <div>
          <h1 className="page-title">
            {mode === "live" ? "Entity Network — Live" : focus ? `Investigation: ${idLabel(focus.type, focus.id)} Cluster` : "Entity Network"}
          </h1>
          <p className="page-subtitle">
            {mode === "live"
              ? "SIMULATED LIVE STREAM · Continuously generated demo transactions from real customer/terminal profiles -- not real production traffic."
              : focus
                ? "Analyzing cross-entity propagation path -- real customer ↔ terminal relationships derived from actual shared transactions."
                : "Showing the most severe currently at-risk terminals and customers -- select a node to investigate its connections."}
          </p>
        </div>
        <div className="net-header-actions">
          <div className="toolbar" role="group" aria-label="Network canvas mode">
            <button
              className={`btn ${mode === "investigate" ? "btn-primary" : ""}`}
              onClick={() => selectMode("investigate")}
              aria-pressed={mode === "investigate"}
            >
              Investigate
            </button>
            <button
              className={`btn ${mode === "live" ? "btn-primary" : ""}`}
              onClick={() => selectMode("live")}
              aria-pressed={mode === "live"}
            >
              <Icon name="bolt" size={14} />
              Live
            </button>
          </div>
          {mode === "live" && (
            <>
              <button
                className="btn btn-primary replay-play-btn"
                onClick={toggleLive}
                disabled={startLive.isPending || stopLive.isPending}
              >
                <Icon name={livePlaying ? "pause" : "play_arrow"} size={16} />
                {livePlaying ? "Pause" : "Start Simulation"}
              </button>
              <span className="replay-live-indicator">
                <span className="replay-live-dot" aria-hidden="true" />
                {livePlaying ? "Live" : "Paused"}
              </span>
              {liveStatus.data && liveStatus.data.n_generated > 0 && (
                <span className="net-side-caption">{liveStatus.data.n_generated} generated this session</span>
              )}
              {liveStatus.data?.error && <span className="field-error">{liveStatus.data.error}</span>}
            </>
          )}
          {mode === "investigate" && (
            <button
              className={`btn ${showElevatedOnly ? "btn-primary" : ""}`}
              onClick={() => setShowElevatedOnly((v) => !v)}
              aria-pressed={showElevatedOnly}
            >
              <Icon name="filter_alt" size={14} />
              {showElevatedOnly ? "Elevated Only" : "Filter Nodes"}
            </button>
          )}
          <button className="btn" onClick={() => setLayoutKey((k) => k + 1)}>
            <Icon name="refresh" size={14} />
            Reset Layout
          </button>
          <button className="btn" onClick={() => downloadGraph(focus ? idLabel(focus.type, focus.id) : mode === "live" ? "live" : "overview", data)}>
            <Icon name="download" size={14} />
            Export Graph
          </button>
        </div>
      </div>

        <section className={`card net-canvas-card${focus ? " net-canvas-card-focused" : ""}`}>
          {focusNode && (
            <aside className="net-snapshot" aria-label="Selected entity summary">
              <div className="net-snapshot-header">
                <Icon name={focus!.type === "terminal" ? "terminal" : "group"} size={16} />
                <Link className="link-id mono" to={detailPath(focus!.type, focus!.id)}>
                  {idLabel(focus!.type, focus!.id)}
                </Link>
              </div>
              <span className="net-snapshot-group-label">Entity Info</span>
              <dl className="net-snapshot-rows">
                <div>
                  <dt>Type</dt>
                  <dd>{focus!.type === "terminal" ? "Terminal" : "Customer"}</dd>
                </div>
                <div>
                  <dt>Status</dt>
                  <dd>
                    <StateBadge state={focusNode.risk_state} />
                  </dd>
                </div>
              </dl>

              <span className="net-snapshot-group-label">Behavioral Signals</span>
              <dl className="net-snapshot-rows">
                <div>
                  <dt>Recent txns (7d)</dt>
                  <dd className="mono">{deviation.data ? deviation.data.current_transaction_count : "—"}</dd>
                </div>
                <div>
                  <dt>Elevated rate</dt>
                  <dd className="mono">{currentRatePct === null ? "—" : `${currentRatePct.toFixed(0)}%`}</dd>
                </div>
                <div>
                  <dt>Severity change</dt>
                  <dd className="mono">{deviationPp === null ? "—" : `${deviationPp >= 0 ? "+" : ""}${deviationPp.toFixed(1)}pp`}</dd>
                </div>
              </dl>

              <span className="net-snapshot-group-label">Connections</span>
              <dl className="net-snapshot-rows">
                <div>
                  <dt>Linked customers</dt>
                  <dd className="mono">{linkedCustomers}</dd>
                </div>
                <div>
                  <dt>Linked terminals</dt>
                  <dd className="mono">{linkedTerminals}</dd>
                </div>
              </dl>
            </aside>
          )}
          {mode === "live" && data.latest_transaction_id !== null && (
            <p className="net-side-caption">
              Just arrived:{" "}
              <Link className="link-id mono" to={`/transactions/${data.latest_transaction_id}`}>
                TX_{data.latest_transaction_id}
              </Link>{" "}
              -- its customer and terminal are outlined below.
            </p>
          )}
          <div className="net-graph-stage">
            <EntityNetworkGraph
            graph={displayGraph}
            selectedId={focusNode?.id}
            onSelect={selectNode}
            legend={!focus}
            resetKey={layoutKey}
          />
          </div>

          {focus && (
            <aside className="net-topology" aria-label="Network topology legend">
              <div className="net-topology-header">
                <Icon name="hub" size={14} />
                Network Topology
              </div>
              <ul className="net-topology-rows">
                <li>
                  <Icon name="group" size={13} />
                  Customer Node
                </li>
                <li>
                  <Icon name="terminal" size={13} />
                  Terminal Node
                </li>
                <li>
                  <span className="net-topology-dot" />
                  Transaction Edge
                </li>
              </ul>
              <p className="net-topology-hint">Drag any node to rearrange the layout.</p>
              <ul className="net-topology-rows net-topology-links">
                <li>
                  <span className="net-topology-line" style={{ background: "var(--risk-high)" }} />
                  High Risk Link
                </li>
                <li>
                  <span className="net-topology-line" style={{ background: "var(--risk-medium)" }} />
                  Risk Rising Link
                </li>
                <li>
                  <span className="net-topology-line" style={{ background: "var(--risk-low)" }} />
                  Normal Link
                </li>
              </ul>
            </aside>
          )}
        </section>

        <div className="net-panels">
          <section className="card net-side-card">
            <h2 className="net-side-title">
              <Icon name="hub" size={16} className="net-side-icon" />
              Connected Entities
            </h2>
            <p className="net-side-caption">
              {focus ? `Real entities sharing transactions with ${idLabel(focus.type, focus.id)}.` : "Click a node to investigate its connections."}
            </p>
            {connected.length === 0 ? (
              <EmptyState>No connected entities in this window.</EmptyState>
            ) : (
              <ul className="net-connected-list">
                {connected.map((n) => {
                  const weight = weightById.get(n.id) ?? 0;
                  return (
                    <li key={n.id} className="net-connected-row">
                      <div className="net-connected-head">
                        <span className="net-connected-id">
                          <span className="net-connected-avatar">
                            <Icon name={n.entity_type === "terminal" ? "terminal" : "group"} size={13} />
                          </span>
                          <Link className="link-id mono" to={detailPath(n.entity_type, n.entity_id)}>
                            {idLabel(n.entity_type, n.entity_id)}
                          </Link>
                        </span>
                        <StateBadge state={n.risk_state} />
                      </div>
                      {focus && (
                        <>
                          <div className="net-connected-footer">
                            <span>Shared Txns</span>
                            <span className="mono">{weight}</span>
                          </div>
                          <div className="net-connected-bar-track">
                            <div className="net-connected-bar-fill" style={{ width: `${(weight / maxWeight) * 100}%` }} />
                          </div>
                        </>
                      )}
                    </li>
                  );
                })}
              </ul>
            )}
          </section>

          {focus && (
            <section className="card net-side-card">
              <div className="net-side-header-row">
                <h2 className="net-side-title">
                  <Icon name="receipt_long" size={16} className="net-side-icon" />
                  Recent Transactions
                </h2>
                <Link className="link-id" to="/transactions">
                  View All
                </Link>
              </div>
              {/* Real source label, not assumed: this entity was drilled into from the
                  Live canvas, so its transactions here come from GET /live/transactions
                  (mrs.api.routers.live, split=="live") -- never implied as production
                  traffic. */}
              {focusSource === "live" && <p className="net-side-caption">SIMULATED LIVE STREAM transactions</p>}
              {recentTx.isLoading && <Loading label="Loading recent transactions…" />}
              {recentTx.isError && <ErrorBlock error={recentTx.error} onRetry={() => recentTx.refetch()} />}
              {recentTx.data && recentTx.data.items.length === 0 && <EmptyState>No transactions for this entity yet.</EmptyState>}
              {recentTx.data && recentTx.data.items.length > 0 && (
                <ul className="net-tx-list">
                  {recentTx.data.items.map((item) => (
                    <li key={item.transaction.transaction_id} className="net-tx-row">
                      <Link className="link-id mono" to={`/transactions/${item.transaction.transaction_id}`}>
                        TX_{item.transaction.transaction_id}
                      </Link>
                      <span className="net-tx-route mono">
                        CUST_{item.transaction.customer_id} &rarr; TERM_{item.transaction.terminal_id}
                      </span>
                      <span className="net-tx-amount mono">{formatAmount(item.transaction.tx_amount)}</span>
                      <span className="net-tx-time mono">{formatDateTimeCompact(item.transaction.tx_datetime)}</span>
                    </li>
                  ))}
                </ul>
              )}
            </section>
          )}
        </div>

      {focus && (
        <section className="card net-analyst-card">
          <h2 className="net-side-title net-analyst-heading">
            <Icon name="psychology" size={16} className="net-side-icon" />
            Analyst Synthesis
          </h2>
          {latestTxId !== undefined ? (
            <AnalystPanel transactionId={latestTxId} />
          ) : (
            <p className="net-side-caption net-analyst-heading">
              {focusNode
                ? behavioralFinding(focus.type === "terminal" ? "Terminal" : "Customer", focusNode.risk_state)
                : "No scored transactions yet for this entity."}
            </p>
          )}
        </section>
      )}
    </div>
  );
}
