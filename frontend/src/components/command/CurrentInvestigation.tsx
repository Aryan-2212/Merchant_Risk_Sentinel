import { useQuery } from "@tanstack/react-query";
import { Link } from "react-router-dom";
import { api } from "../../lib/api";
import { useRiskHistory } from "../../lib/useRiskHistory";
import type { EntityType } from "../../lib/types";
import { Loading, ErrorBlock, EmptyState } from "../common/States";
import { RiskBadge } from "../risk/RiskBadge";
import { StateBadge } from "../risk/StateBadge";
import { ActionBadge } from "../risk/ActionBadge";
import "./CurrentInvestigation.css";

interface Props {
  entity: { type: EntityType; id: number } | null;
}

/**
 * The Command Center's contextual "what should I look at" panel (Dev Plan Sec 7):
 * the selected entity's current behavioral state, its most recent scored
 * transaction's evidence, and the deterministic policy decision on that transaction.
 * Reuses GET /transactions/{id} (already the authoritative detail view) rather than
 * re-deriving evidence/policy logic here.
 */
export function CurrentInvestigation({ entity }: Props) {
  const history = useRiskHistory(
    entity?.id ?? Number.NaN,
    entity?.type === "customer" ? api.getCustomerRiskHistory : api.getTerminalRiskHistory,
    `${entity?.type ?? "none"}-risk-focus`,
  );
  const latestTxId = history.items[history.items.length - 1]?.transaction_id;

  const detail = useQuery({
    queryKey: ["transaction", latestTxId],
    queryFn: () => api.getTransaction(latestTxId!),
    enabled: latestTxId !== undefined,
  });

  if (!entity) {
    return (
      <div className="cinv">
        <span className="cinv-title">Current investigation</span>
        <EmptyState>Select an entity in the network to inspect it here.</EmptyState>
      </div>
    );
  }

  if (history.isLoading || (latestTxId !== undefined && detail.isLoading)) {
    return (
      <div className="cinv">
        <span className="cinv-title">Current investigation</span>
        <Loading label="Loading entity context…" />
      </div>
    );
  }
  if (history.isError) {
    return (
      <div className="cinv">
        <span className="cinv-title">Current investigation</span>
        <ErrorBlock error={history.error} onRetry={history.refetch} />
      </div>
    );
  }
  if (history.items.length === 0) {
    return (
      <div className="cinv">
        <span className="cinv-title">Current investigation</span>
        <EmptyState>
          {entity.type} #{entity.id} has no scored transactions yet -- insufficient historical baseline.
        </EmptyState>
      </div>
    );
  }

  const risk = detail.data?.risk_score;
  const alert = detail.data?.alert;
  const entityState = entity.type === "customer" ? risk?.customer_risk_state : risk?.terminal_risk_state;
  const deterministicAction = alert?.recommended_action ?? "ALLOW";
  const detailHref = entity.type === "customer" ? `/customers/${entity.id}` : `/terminals/${entity.id}`;

  return (
    <div className="cinv">
      <span className="cinv-title">Current investigation</span>

      <div className="cinv-header">
        <div>
          <span className="cinv-entity-type">{entity.type}</span>
          <span className="cinv-entity-id mono">#{entity.id}</span>
        </div>
        {entityState !== undefined && <StateBadge state={entityState ?? null} />}
      </div>

      {risk && (
        <>
          <div className="cinv-level">
            <span className="cinv-level-label">Most recent unified risk</span>
            <RiskBadge level={risk.unified_risk_level} />
          </div>

          {risk.contributing_signals.length > 0 && (
            <div className="cinv-block">
              <span className="cinv-block-label">Why is this elevated?</span>
              <ul className="cinv-signals">
                {risk.contributing_signals.map((s) => (
                  <li key={s}>{s}</li>
                ))}
              </ul>
            </div>
          )}

          <div className="cinv-block">
            <span className="cinv-block-label">Policy recommendation</span>
            <ActionBadge action={deterministicAction} />
          </div>
        </>
      )}

      <Link to={detailHref} className="btn btn-primary cinv-link">
        Open full investigation
      </Link>
    </div>
  );
}
