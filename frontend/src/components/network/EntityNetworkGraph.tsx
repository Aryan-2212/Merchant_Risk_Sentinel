import { useEffect, useMemo, useRef, useState } from "react";
import { forceCenter, forceCollide, forceLink, forceManyBody, forceSimulation, forceX, forceY } from "d3-force";
import type { NetworkGraph, NetworkNode } from "../../lib/types";
import "./EntityNetworkGraph.css";

/** Fallback viewBox used only until the container has been measured (and in
 * non-DOM environments without ResizeObserver, e.g. unit tests). */
const WIDTH = 560;
const HEIGHT = 380;

interface SimNode extends NetworkNode {
  x: number;
  y: number;
  vx?: number;
  vy?: number;
  /** d3-force honours fx/fy as a fixed position -- used to pin a single
   * investigation target to the centre of the canvas. */
  fx?: number;
  fy?: number;
  /** Radius in viewBox units, sized from this entity's real shared-transaction
   * volume (see sizeNodes). */
  radius: number;
}

interface SimLink {
  source: string | SimNode;
  target: string | SimNode;
  weight: number;
}

const FOCUS_RADIUS = 22;
const MIN_RADIUS = 10;
const MAX_RADIUS = 20;

/** A node's size encodes how much real transaction volume it carries: the sum of the
 * weights (shared transaction counts) of its edges, square-rooted so area rather than
 * radius scales with volume. Purely a rendering of data the graph already returned --
 * it invents no metric and changes no risk value. */
function sizeNodes(graph: NetworkGraph): Map<string, number> {
  const totals = new Map<string, number>();
  for (const edge of graph.edges) {
    totals.set(edge.source, (totals.get(edge.source) ?? 0) + edge.weight);
    totals.set(edge.target, (totals.get(edge.target) ?? 0) + edge.weight);
  }
  const max = Math.max(1, ...totals.values());
  const radii = new Map<string, number>();
  for (const node of graph.nodes) {
    if (node.is_focus) {
      radii.set(node.id, FOCUS_RADIUS);
      continue;
    }
    const share = Math.sqrt((totals.get(node.id) ?? 0) / max);
    radii.set(node.id, MIN_RADIUS + (MAX_RADIUS - MIN_RADIUS) * share);
  }
  return radii;
}

function nodeRadius(node: SimNode): number {
  return node.radius;
}

/** Total real shared-transaction count across one entity's edges. */
function sharedTotal(graph: NetworkGraph, id: string): number {
  return graph.edges.reduce((sum, e) => (e.source === id || e.target === id ? sum + e.weight : sum), 0);
}

/** A gentle arc rather than a straight line, matching the reference's curved
 * connections -- it also separates the two directions of a pair visually. */
function edgePath(s: SimNode, t: SimNode): string {
  const dx = t.x - s.x;
  const dy = t.y - s.y;
  const distance = Math.hypot(dx, dy) || 1;
  const bow = distance * 0.12;
  // Control point offset perpendicular to the straight run between the two nodes.
  const cx = (s.x + t.x) / 2 + (-dy / distance) * bow;
  const cy = (s.y + t.y) / 2 + (dx / distance) * bow;
  return `M${s.x},${s.y} Q${cx},${cy} ${t.x},${t.y}`;
}

function nodeGlyph(node: NetworkNode): string {
  return node.entity_type === "terminal" ? "T" : "C";
}

function nodeLabel(node: NetworkNode): string {
  return node.entity_type === "terminal" ? `TERM_${node.entity_id}` : `CUST_${node.entity_id}`;
}

/* Reads the shared severity tokens directly, so a node's colour is always the exact
 * colour StateBadge gives that same state in the side panel -- grey when calm,
 * near-white once rising, red at high risk.
 *
 * RECOVERY groups with NORMAL, matching StateBadge, rather than with RISK_RISING as
 * raw severity 1 would imply. */
function graphStateColor(node: NetworkNode): string {
  switch (node.risk_state) {
    case "HIGH_RISK":
      return "var(--risk-high)";
    case "RISK_RISING":
      return "var(--risk-medium)";
    case "NORMAL":
    case "RECOVERY":
      return "var(--risk-low)";
    default:
      return "var(--risk-unknown)";
  }
}

/** A link's colour is driven by whichever endpoint carries the higher real computed
 * risk state -- the same value used for that node's own fill, never an invented
 * "link risk tier". */
function linkColor(s: SimNode, t: SimNode): string {
  const rank = (node: SimNode) => (node.risk_state === "HIGH_RISK" ? 2 : node.risk_state === "RISK_RISING" ? 1 : 0);
  return graphStateColor(rank(s) >= rank(t) ? s : t);
}

interface Props {
  graph: NetworkGraph;
  selectedId?: string;
  onSelect: (node: NetworkNode) => void;
  /** Suppress the built-in bottom legend bar -- used on Network.tsx, which renders a
   * richer floating "Network Topology" panel of its own instead. */
  legend?: boolean;
  /** Change this value to discard any hand-arranged node positions and re-run the
   * force layout from scratch. */
  resetKey?: number;
}

/**
 * The Command Center's signature Entity Risk Network -- a real, bounded neighborhood
 * graph from GET /stats/network (real customer<->terminal edges derived from actual
 * transactions, never fabricated relationships). Force-directed layout via d3-force;
 * this component only drives an SVG from its simulated positions, it does not
 * recompute risk or relationships itself.
 */
export function EntityNetworkGraph({ graph, selectedId, onSelect, legend = true, resetKey = 0 }: Props) {
  const [nodes, setNodes] = useState<SimNode[]>([]);
  const [links, setLinks] = useState<SimLink[]>([]);
  const [hovered, setHovered] = useState<SimNode | null>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const svgRef = useRef<SVGSVGElement>(null);
  const [size, setSize] = useState({ width: WIDTH, height: HEIGHT });

  // Live simulation handles, so drag handlers can move a node and re-heat the layout
  // without tearing down and rebuilding the simulation on every pointer event.
  const simRef = useRef<ReturnType<typeof forceSimulation<SimNode>> | null>(null);
  const simNodesRef = useRef<SimNode[]>([]);
  const dragRef = useRef<{ id: string; moved: boolean } | null>(null);
  // A drag ends with a click event on the node; that must not also navigate.
  const suppressClickRef = useRef(false);

  // Drive the viewBox from the container's real pixel size so the graph fills the
  // space it is given instead of being letterboxed inside a fixed 560x380 box. The
  // container's own height comes from CSS (never from the SVG's content), so
  // observing it cannot feed back into its own measurement.
  useEffect(() => {
    const el = containerRef.current;
    if (!el || typeof ResizeObserver === "undefined") return;
    const observer = new ResizeObserver((entries) => {
      for (const entry of entries) {
        const { width, height } = entry.contentRect;
        if (width < 1 || height < 1) continue;
        setSize((prev) =>
          // Ignore sub-pixel jitter; a re-measure restarts the simulation.
          Math.abs(prev.width - width) < 2 && Math.abs(prev.height - height) < 2
            ? prev
            : { width: Math.round(width), height: Math.round(height) },
        );
      }
    });
    observer.observe(el);
    return () => observer.disconnect();
  }, []);

  const { width: vbWidth, height: vbHeight } = size;

  // A single focus node means one entity is genuinely under investigation, so it is
  // "the target". The overview graph seeds several focus hubs at once, where nothing
  // has been selected and calling every hub a target would be meaningless.
  const focusNodes = graph.nodes.filter((node) => node.is_focus);
  const soleFocusId = focusNodes.length === 1 ? focusNodes[0].id : undefined;

  /** Who each node actually transacts with -- used to spotlight one entity's real
   * relationships on hover and fade the rest of the canvas back. */
  const neighbours = useMemo(() => {
    const map = new Map<string, Set<string>>();
    const link = (a: string, b: string) => {
      if (!map.has(a)) map.set(a, new Set());
      map.get(a)!.add(b);
    };
    for (const edge of graph.edges) {
      link(edge.source, edge.target);
      link(edge.target, edge.source);
    }
    return map;
  }, [graph]);

  const hoveredId = hovered?.id;
  function isMuted(id: string): boolean {
    if (!hoveredId) return false;
    return id !== hoveredId && !neighbours.get(hoveredId)?.has(id);
  }

  useEffect(() => {
    // Scatter initial positions on a small ring rather than stacking every node on
    // the exact same point -- zero initial distance makes forceManyBody's repulsion
    // blow up (divide-by-near-zero), flinging nodes far outside the viewBox.
    const n = graph.nodes.length;

    const radii = sizeNodes(graph);
    const simNodes: SimNode[] = graph.nodes.map((node, i) => {
      const angle = (i / Math.max(1, n)) * Math.PI * 2;
      const radius = 30 + (i % 4) * 15;
      const base: SimNode = {
        ...node,
        x: vbWidth / 2 + Math.cos(angle) * radius,
        y: vbHeight / 2 + Math.sin(angle) * radius,
        radius: radii.get(node.id) ?? MIN_RADIUS,
      };
      if (node.id === soleFocusId) {
        base.x = base.fx = vbWidth / 2;
        base.y = base.fy = vbHeight / 2;
      }
      return base;
    });
    const simLinks: SimLink[] = graph.edges.map((e) => ({ ...e }));

    // Spacing scales with the area actually available per node, so the cluster fills
    // a large investigation canvas instead of staying knotted in the middle of it.
    const spread = Math.sqrt((vbWidth * vbHeight) / Math.max(1, n));

    const sim = forceSimulation(simNodes)
      .force(
        "link",
        forceLink<SimNode, SimLink>(simLinks)
          .id((d) => d.id)
          .distance(Math.max(70, spread * 0.95))
          .strength(0.25),
      )
      .force("charge", forceManyBody().strength(-Math.max(130, spread * 1.6)))
      .force("center", forceCenter(vbWidth / 2, vbHeight / 2))
      // Gentle constant pull toward center so the cluster can never drift/explode
      // outside the visible viewBox, independent of node/edge count.
      .force("x", forceX(vbWidth / 2).strength(0.03))
      .force("y", forceY(vbHeight / 2).strength(0.03))
      .force(
        // Radius padded well past the node's own circle so the persistent ID label
        // rendered underneath it (see nodeLabel below) has room to clear its
        // neighbors too, not just the node glyphs themselves.
        "collide",
        forceCollide<SimNode>((d) => nodeRadius(d) + 26),
      )
      .alphaDecay(0.045)
      .on("tick", () => {
        // Keep every node (and the ID label beneath it) inside the viewBox rather
        // than letting the simulation push them past the edge.
        for (const node of simNodes) {
          const r = nodeRadius(node) + 6;
          const bottom = r + 18;
          node.x = Math.min(vbWidth - r, Math.max(r, node.x));
          node.y = Math.min(vbHeight - bottom, Math.max(r, node.y));
        }
        setNodes([...simNodes]);
        setLinks([...simLinks]);
      });

    simRef.current = sim;
    simNodesRef.current = simNodes;

    return () => {
      sim.stop();
      simRef.current = null;
    };
  }, [graph, vbWidth, vbHeight, soleFocusId, resetKey]);

  if (graph.nodes.length === 0) {
    return <div className="network-empty">No elevated entities to visualize right now.</div>;
  }

  function activateNode(node: NetworkNode, event: React.KeyboardEvent<SVGGElement>) {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      onSelect(node);
    }
  }

  /** Pointer position in the SVG's own coordinate space. Uses the live screen CTM
   * rather than assuming a 1:1 mapping, so the node tracks the cursor exactly even
   * mid-resize or if the viewBox is ever scaled. */
  function toGraphPoint(event: React.PointerEvent<SVGGElement>): { x: number; y: number } | null {
    const svg = svgRef.current;
    const ctm = svg?.getScreenCTM();
    if (!svg || !ctm) return null;
    const point = svg.createSVGPoint();
    point.x = event.clientX;
    point.y = event.clientY;
    const local = point.matrixTransform(ctm.inverse());
    return { x: local.x, y: local.y };
  }

  function startDrag(node: SimNode, event: React.PointerEvent<SVGGElement>) {
    if (event.button !== 0) return;
    event.currentTarget.setPointerCapture(event.pointerId);
    suppressClickRef.current = false;
    dragRef.current = { id: node.id, moved: false };
    const target = simNodesRef.current.find((n) => n.id === node.id);
    if (target) {
      // Pin to where it already is, so it does not jump on the first move.
      target.fx = target.x;
      target.fy = target.y;
    }
    // Keep the layout warm while dragging so neighbours ease out of the way.
    simRef.current?.alphaTarget(0.15).restart();
  }

  function moveDrag(event: React.PointerEvent<SVGGElement>) {
    const drag = dragRef.current;
    if (!drag) return;
    const point = toGraphPoint(event);
    const target = simNodesRef.current.find((n) => n.id === drag.id);
    if (!point || !target) return;

    const r = nodeRadius(target) + 6;
    const x = Math.min(vbWidth - r, Math.max(r, point.x));
    const y = Math.min(vbHeight - r - 18, Math.max(r, point.y));
    // Past a few pixels this is a drag, not a click -- suppress the navigation that
    // would otherwise fire when the pointer is released.
    if (!drag.moved && Math.hypot(x - (target.fx ?? target.x), y - (target.fy ?? target.y)) > 3) {
      drag.moved = true;
    }
    target.fx = x;
    target.fy = y;
  }

  function endDrag(event: React.PointerEvent<SVGGElement>) {
    const drag = dragRef.current;
    if (!drag) return;
    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId);
    }
    // fx/fy are deliberately left set: a node stays where it was dropped, which is
    // the point of rearranging. "Reset layout" clears them.
    simRef.current?.alphaTarget(0);
    suppressClickRef.current = drag.moved;
    dragRef.current = null;
  }

  function handleNodeClick(node: NetworkNode) {
    if (suppressClickRef.current) {
      suppressClickRef.current = false;
      return;
    }
    onSelect(node);
  }

  return (
    <div className="network-graph" ref={containerRef}>
      <svg ref={svgRef} viewBox={`0 0 ${vbWidth} ${vbHeight}`} role="img" aria-label="Entity risk relationship network">
        <defs>
          {/* Blueprint grid, matching the reference's analytical canvas treatment. */}
          <pattern id="network-grid" width="40" height="40" patternUnits="userSpaceOnUse">
            <path d="M40 0H0V40" fill="none" stroke="currentColor" strokeWidth="1" />
          </pattern>
          {/* Fades the grid out toward the edges so the canvas reads as depth rather
              than as a flat sheet of graph paper. */}
          <radialGradient id="network-grid-fade" cx="50%" cy="50%" r="70%">
            <stop offset="0%" stopColor="#fff" stopOpacity="0.85" />
            <stop offset="100%" stopColor="#fff" stopOpacity="0" />
          </radialGradient>
          <mask id="network-grid-mask">
            <rect width={vbWidth} height={vbHeight} fill="url(#network-grid-fade)" />
          </mask>
        </defs>
        <rect
          className="network-grid"
          width={vbWidth}
          height={vbHeight}
          fill="url(#network-grid)"
          mask="url(#network-grid-mask)"
        />

        <g className="network-edges">
          {links.map((l, i) => {
            const s = l.source as SimNode;
            const t = l.target as SimNode;
            if (typeof s !== "object" || typeof t !== "object") return null;
            const touchesHover = hoveredId !== undefined && (s.id === hoveredId || t.id === hoveredId);
            const muted = hoveredId !== undefined && !touchesHover;
            return (
              <path
                key={i}
                d={edgePath(s, t)}
                fill="none"
                stroke={linkColor(s, t)}
                className={`network-edge ${s.is_focus || t.is_focus ? "is-focus-link" : ""} ${
                  touchesHover ? "is-hot" : ""
                } ${muted ? "is-muted" : ""}`}
                strokeWidth={Math.min(2.5, 0.6 + l.weight * 0.15)}
              />
            );
          })}
        </g>
        <g className="network-nodes">
          {nodes.map((n) => {
            const r = nodeRadius(n);
            const color = graphStateColor(n);
            const isSelected = n.id === selectedId;
            return (
              <g
                key={n.id}
                transform={`translate(${n.x}, ${n.y})`}
                // Lets currentColor in the stylesheet (the focus glow) resolve to
                // this node's own risk colour rather than inherited text colour.
                style={{ color }}
                className={`network-node ${n.is_focus ? "is-focus" : ""} ${isSelected ? "is-selected" : ""} ${
                  n.risk_state === "HIGH_RISK" ? "is-high" : ""
                } ${isMuted(n.id) ? "is-muted" : ""} ${n.id === hoveredId ? "is-hovered" : ""}`}
                onClick={() => handleNodeClick(n)}
                onKeyDown={(event) => activateNode(n, event)}
                onPointerDown={(event) => startDrag(n, event)}
                onPointerMove={moveDrag}
                onPointerUp={endDrag}
                onPointerCancel={endDrag}
                onMouseEnter={() => setHovered(n)}
                onMouseLeave={() => setHovered((h) => (h?.id === n.id ? null : h))}
                tabIndex={0}
                role="button"
                aria-label={`${n.entity_type} ${n.entity_id}, ${n.risk_state ?? "unavailable"}`}
              >
                {/* High-risk marker: a soft static glow plus a slowly blinking ring
                    on the boundary, so elevated entities are findable at a glance.
                    Opacity-only -- it never changes size, which is what made the
                    earlier expanding pulse read as stray dots. */}
                {n.risk_state === "HIGH_RISK" && (
                  <>
                    <circle r={r * 1.9} fill={color} className="network-node-halo" />
                    <circle r={r + 7} stroke={color} className="network-node-alert" />
                  </>
                )}
                {isSelected && !n.is_focus && <circle r={r + 5} className="network-node-ring" />}
                {n.is_focus ? (
                  <rect x={-r} y={-r} width={r * 2} height={r * 2} rx={5} fill={color} className="network-node-fill" />
                ) : (
                  <circle r={r} fill={color} className="network-node-fill" />
                )}
                <text dy="0.32em" textAnchor="middle" className="network-node-glyph">
                  {nodeGlyph(n)}
                </text>
                {/* No "(TARGET)" caption: the page header, the entity panel and the
                    node's own square/red/centred treatment already identify it. */}
                <text dy={r + 13} textAnchor="middle" className="network-node-label">
                  {nodeLabel(n)}
                </text>
              </g>
            );
          })}
        </g>
      </svg>

      {hovered && (
        <div
          className="network-tooltip"
          style={{ left: `${(hovered.x / vbWidth) * 100}%`, top: `${(hovered.y / vbHeight) * 100}%` }}
        >
          <strong>{nodeLabel(hovered)}</strong>
          <span>{hovered.risk_state ? hovered.risk_state.replaceAll("_", " ").toLowerCase() : "unavailable"}</span>
          <span className="network-tooltip-meta">
            {neighbours.get(hovered.id)?.size ?? 0} connected · {sharedTotal(graph, hovered.id)} shared txns
          </span>
        </div>
      )}

      {legend && (
        <div className="network-legend">
          <span>
            <i style={{ background: "var(--risk-low)" }} /> normal / recovery
          </span>
          <span>
            <i style={{ background: "var(--risk-medium)" }} /> risk rising
          </span>
          <span>
            <i style={{ background: "var(--risk-high)" }} /> high risk
          </span>
          <span className="network-legend-sep">C customer · T terminal</span>
        </div>
      )}
    </div>
  );
}
