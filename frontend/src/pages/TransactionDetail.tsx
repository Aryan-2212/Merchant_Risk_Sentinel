import { useQuery } from "@tanstack/react-query";
import { Link, useParams } from "react-router-dom";
import { api } from "../lib/api";
import { formatAmount, formatDateTime } from "../lib/format";
import { Loading, ErrorBlock, EmptyState } from "../components/common/States";
import { RiskBadge } from "../components/risk/RiskBadge";
import { ActionBadge } from "../components/risk/ActionBadge";
import { RiskDecomposition } from "../components/risk/RiskDecomposition";
import { EvidenceChain } from "../components/risk/EvidenceChain";
import { AnalystPanel } from "../components/analyst/AnalystPanel";
import { AuditTrail } from "../components/audit/AuditTrail";
import "./TransactionDetail.css";

export function TransactionDetail() {
  const { id } = useParams<{ id: string }>();
  const transactionId = Number(id);

  const detail = useQuery({
    queryKey: ["transaction", transactionId],
    queryFn: () => api.getTransaction(transactionId),
    enabled: Number.isFinite(transactionId),
  });

  if (!Number.isFinite(transactionId)) {
    return <EmptyState>Invalid transaction ID.</EmptyState>;
  }
  if (detail.isLoading) return <Loading label="Loading transaction…" />;
  if (detail.isError) return <ErrorBlock error={detail.error} onRetry={() => detail.refetch()} />;

  const { transaction, risk_score, alert } = detail.data!;
  // Mirrors mrs.analyst.evidence.build_evidence exactly: an alert only ever exists
  // for a non-ALLOW decision, so its absence means the deterministic policy is ALLOW.
  const deterministicAction = alert?.recommended_action ?? "ALLOW";

  return (
    <div className="page">
      <div className="page-header">
        <div>
          <h1 className="page-title">Transaction #{transaction.transaction_id}</h1>
          <p className="page-subtitle">{formatDateTime(transaction.tx_datetime)}</p>
        </div>
        {risk_score && <RiskBadge level={risk_score.unified_risk_level} />}
      </div>

      <div className="card tx-meta">
        <div>
          <span className="field-label">Amount</span>
          <p className="mono">{formatAmount(transaction.tx_amount)}</p>
        </div>
        <div>
          <span className="field-label">Customer</span>
          <p>
            <Link className="link-id mono" to={`/customers/${transaction.customer_id}`}>
              CUST_{transaction.customer_id}
            </Link>
          </p>
        </div>
        <div>
          <span className="field-label">Terminal</span>
          <p>
            <Link className="link-id mono" to={`/terminals/${transaction.terminal_id}`}>
              TERM_{transaction.terminal_id}
            </Link>
          </p>
        </div>
        <div>
          <span className="field-label">Split</span>
          <p>{transaction.split}</p>
        </div>
      </div>

      {!risk_score && (
        <EmptyState>This transaction has not been scored yet -- no risk components are available.</EmptyState>
      )}

      {risk_score && (
        <>
          <div className="grid-2">
            <RiskDecomposition risk={risk_score} />
            <EvidenceChain
              contributingSignals={risk_score.contributing_signals}
              unifiedRiskLevel={risk_score.unified_risk_level}
              deterministicAction={deterministicAction}
            />
          </div>

          <div className="section">
            <span className="section-title">Policy decision</span>
            <div className="card tx-policy">
              <ActionBadge action={deterministicAction} />
              <span className="tx-policy-reason">{alert?.reason ?? "No elevated component signals -- ALLOW."}</span>
              {detail.data!.policy_version && (
                <span className="tx-policy-version mono">{detail.data!.policy_version}</span>
              )}
            </div>
          </div>

          <div className="section">
            <span className="section-title">AI risk analyst</span>
            <AnalystPanel transactionId={transaction.transaction_id} />
          </div>
        </>
      )}

      <div className="section">
        <span className="section-title">Audit trail</span>
        <div className="card">
          <AuditTrail transactionId={transaction.transaction_id} />
        </div>
      </div>
    </div>
  );
}
