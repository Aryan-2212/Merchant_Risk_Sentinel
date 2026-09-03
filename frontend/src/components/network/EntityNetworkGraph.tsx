import { useEffect, useRef, useState } from "react";
import { forceCenter, forceCollide, forceLink, forceManyBody, forceSimulation, forceX, forceY } from "d3-force";
import type { NetworkGraph, NetworkNode } from "../../lib/types";
import { severityColor, stateColor } from "../../lib/riskColor";
import "./EntityNetworkGraph.css";

const WIDTH = 560;
const HEIGHT = 380;

interface SimNode extends NetworkNode {
  x: number;
  y: number;
  vx?: number;
  vy?: number;
}

interface SimLink {
  source: string | SimNode;
  target: string | SimNode;
  weight: number;
}

function nodeRadius(node: NetworkNode): number {
  return node.is_focus ? 16 : 9;
}

function nodeGlyph(node: NetworkNode): string {
  return node.entity_type === "terminal" ? "T" : "C";
}

function nodeLabel(node: NetworkNode): string {
  return node.entity_type === "terminal" ? `TERM_${node.entity_id}` : `CUST_${node.entity_id}`;
}

/** A link's color/weight is driven by whichever endpoint carries the higher real
 * computed severity -- the same risk_severity already used for that node's own fill,
 * never an invented "link risk tier". */
function linkColor(s: SimNode, t: SimNode): string {
  const worse = (s.risk_severity ?? -1) >= (t.risk_severity ?? -1) ? s : t;
  return severityColor(worse.risk_severity);
}

interface Props {
  graph: NetworkGraph;
  selectedId?: string;
  onSelect: (node: NetworkNode) => void;
  /** Suppress the built-in bottom legend bar -- used on Network.tsx, which renders a
   * richer floating "Network Topology" panel of its own instead. */
  legend?: boolean;
}

/**
 * The Command Center's signature Entity Risk Network -- a real, bounded neighborhood
 * graph from GET /stats/network (real customer<->terminal edges derived from actual
 * transactions, never fabricated relationships). Force-directed layout via d3-force;
 * this component only drives an SVG from its simulated positions, it does not
 * recompute risk or relationships itself.
 */
export function EntityNetworkGraph({ graph, selectedId, onSelect, legend = true }: Props) {
  const [nodes, setNodes] = useState<SimNode[]>([]);
  const [links, setLinks] = useState<SimLink[]>([]);
  const [hovered, setHovered] = useState<SimNode | null>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    // Scatter initial positions on a small ring rather than stacking every node on
    // the exact same point -- zero initial distance makes forceManyBody's repulsion
    // blow up (divide-by-near-zero), flinging nodes far outside the viewBox.
    const n = graph.nodes.length;
    const simNodes: SimNode[] = graph.nodes.map((node, i) => {
      const angle = (i / Math.max(1, n)) * Math.PI * 2;
      const radius = 30 + (i % 4) * 15;
      return { ...node, x: WIDTH / 2 + Math.cos(angle) * radius, y: HEIGHT / 2 + Math.sin(angle) * radius };
    });
    const simLinks: SimLink[] = graph.edges.map((e) => ({ ...e }));

    const sim = forceSimulation(simNodes)
      .force(
        "link",
        forceLink<SimNode, SimLink>(simLinks)
          .id((d) => d.id)
          .distance(70)
          .strength(0.25),
      )
      .force("charge", forceManyBody().strength(-130))
      .force("center", forceCenter(WIDTH / 2, HEIGHT / 2))
      // Gentle constant pull toward center so the cluster can never drift/explode
      // outside the visible viewBox, independent of node/edge count.
      .force("x", forceX(WIDTH / 2).strength(0.03))
      .force("y", forceY(HEIGHT / 2).strength(0.03))
      .force(
        // Radius padded well past the node's own circle so the persistent ID label
        // rendered underneath it (see nodeLabel below) has room to clear its
        // neighbors too, not just the node glyphs themselves.
        "collide",
        forceCollide<SimNode>((d) => nodeRadius(d) + 24),
      )
      .alphaDecay(0.045)
      .on("tick", () => {
        const margin = 16;
        for (const node of simNodes) {
          node.x = Math.min(WIDTH - margin, Math.max(margin, node.x));
          node.y = Math.min(HEIGHT - margin, Math.max(margin, node.y));
        }
        setNodes([...simNodes]);
        setLinks([...simLinks]);
      });

    return () => {
      sim.stop();
    };
  }, [graph]);

  if (graph.nodes.length === 0) {
    return <div className="network-empty">No elevated entities to visualize right now.</div>;
  }

  return (
    <div className="network-graph" ref={containerRef}>
      <svg viewBox={`0 0 ${WIDTH} ${HEIGHT}`} role="img" aria-label="Entity risk relationship network">
        <g className="network-edges">
          {links.map((l, i) => {
            const s = l.source as SimNode;
            const t = l.target as SimNode;
            if (typeof s !== "object" || typeof t !== "object") return null;
            return (
              <line
                key={i}
                x1={s.x}
                y1={s.y}
                x2={t.x}
                y2={t.y}
                stroke={linkColor(s, t)}
                className={`network-edge ${s.is_focus || t.is_focus ? "is-focus-link" : ""}`}
                strokeWidth={Math.min(2.5, 0.6 + l.weight * 0.15)}
              />
            );
          })}
        </g>
        <g className="network-nodes">
          {nodes.map((n) => {
            const r = nodeRadius(n);
            const color = stateColor(n.risk_state);
            const isSelected = n.id === selectedId;
            return (
              <g
                key={n.id}
                transform={`translate(${n.x}, ${n.y})`}
                className={`network-node ${n.is_focus ? "is-focus" : ""} ${isSelected ? "is-selected" : ""}`}
                onClick={() => onSelect(n)}
                onMouseEnter={() => setHovered(n)}
                onMouseLeave={() => setHovered((h) => (h?.id === n.id ? null : h))}
                tabIndex={0}
                role="button"
                aria-label={`${n.entity_type} ${n.entity_id}, ${n.risk_state ?? "unavailable"}`}
              >
                {isSelected && !n.is_focus && <circle r={r + 5} className="network-node-ring" />}
                {n.is_focus ? (
                  <rect x={-r} y={-r} width={r * 2} height={r * 2} rx={5} fill={color} className="network-node-fill" />
                ) : (
                  <circle r={r} fill={color} className="network-node-fill" />
                )}
                <text dy="0.32em" textAnchor="middle" className="network-node-glyph">
                  {nodeGlyph(n)}
                </text>
                <text dy={r + 13} textAnchor="middle" className="network-node-label">
                  {nodeLabel(n)}
                </text>
                {n.is_focus && (
                  <text dy={r + 25} textAnchor="middle" className="network-node-label network-node-target">
                    (TARGET)
                  </text>
                )}
              </g>
            );
          })}
        </g>
      </svg>

      {hovered && (
        <div
          className="network-tooltip"
          style={{ left: `${(hovered.x / WIDTH) * 100}%`, top: `${(hovered.y / HEIGHT) * 100}%` }}
        >
          <strong>
            {hovered.entity_type} #{hovered.entity_id}
          </strong>
          <span>{hovered.risk_state ? hovered.risk_state.replaceAll("_", " ").toLowerCase() : "unavailable"}</span>
        </div>
      )}

      {legend && (
        <div className="network-legend">
          <span>
            <i style={{ background: "var(--risk-low)" }} /> normal
          </span>
          <span>
            <i style={{ background: "var(--risk-medium)" }} /> rising / recovery
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
