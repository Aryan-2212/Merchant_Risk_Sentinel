import type { RiskLevel } from "../../lib/types";
import { formatCount, formatPercent } from "../../lib/format";
import "./RiskDistribution.css";

const ORDER: RiskLevel[] = ["LOW", "MEDIUM", "HIGH", "CRITICAL", "INSUFFICIENT_EVIDENCE"];

const LABEL: Record<RiskLevel, string> = {
  LOW: "Low",
  MEDIUM: "Medium",
  HIGH: "High",
  CRITICAL: "Critical",
  INSUFFICIENT_EVIDENCE: "Insufficient evidence",
};

const COLOR_VAR: Record<RiskLevel, string> = {
  LOW: "--risk-low",
  MEDIUM: "--risk-medium",
  HIGH: "--risk-high",
  CRITICAL: "--risk-critical",
  INSUFFICIENT_EVIDENCE: "--risk-unknown",
};

/**
 * Risk levels are an ordinal severity scale (mrs.risk.aggregate), not independent
 * categories -- bars stay in LOW..CRITICAL(..INSUFFICIENT_EVIDENCE) order rather than
 * sorted by count, so the shape of the distribution along that scale stays legible.
 * Every bar carries a visible count/percent label -- MEDIUM and HIGH's status colors
 * sit close in isolation, so identity never rests on hue alone.
 */
export function RiskDistribution({ counts }: { counts: Partial<Record<RiskLevel, number>> }) {
  const total = ORDER.reduce((sum, level) => sum + (counts[level] ?? 0), 0);
  const max = Math.max(1, ...ORDER.map((level) => counts[level] ?? 0));

  return (
    <div className="rdist" role="img" aria-label="Risk level distribution across all scored transactions">
      {ORDER.map((level) => {
        const count = counts[level] ?? 0;
        if (count === 0 && level === "INSUFFICIENT_EVIDENCE") return null;
        const fraction = total === 0 ? 0 : Math.max(0.02, count / max);
        return (
          <div className="rdist-row" key={level}>
            <span className="rdist-label">{LABEL[level]}</span>
            <div className="rdist-track">
              <div
                className="rdist-fill"
                style={{ transform: `scaleX(${fraction})`, background: `var(${COLOR_VAR[level]})` }}
              />
            </div>
            <span className="rdist-value mono">
              {formatCount(count)} <span className="rdist-pct">({formatPercent(count, total)})</span>
            </span>
          </div>
        );
      })}
    </div>
  );
}
