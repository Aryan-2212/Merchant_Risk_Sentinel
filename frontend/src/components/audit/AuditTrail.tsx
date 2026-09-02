import { useQuery } from "@tanstack/react-query";
import { api } from "../../lib/api";
import { formatDateTime } from "../../lib/format";
import { Loading, ErrorBlock, EmptyState } from "../common/States";
import "./AuditTrail.css";

/** GET /transactions/{id}/audit, verbatim audit_logs rows -- oldest first, matching
 * how they were written (mrs.policy.engine.apply_policy). */
export function AuditTrail({ transactionId }: { transactionId: number }) {
  const audit = useQuery({
    queryKey: ["audit", transactionId],
    queryFn: () => api.getTransactionAudit(transactionId),
  });

  if (audit.isLoading) return <Loading label="Loading audit trail…" />;
  if (audit.isError) return <ErrorBlock error={audit.error} onRetry={() => audit.refetch()} />;
  if (audit.data!.length === 0) return <EmptyState>No audit events recorded for this transaction yet.</EmptyState>;

  return (
    <ol className="audit-trail">
      {audit.data!.map((entry) => (
        <li key={entry.audit_id} className="audit-entry">
          <span className="audit-dot" aria-hidden="true" />
          <div className="audit-body">
            <div className="audit-row">
              <span className="audit-event">{entry.event_type.replaceAll("_", " ")}</span>
              <span className="audit-time mono">{formatDateTime(entry.created_at)}</span>
            </div>
            {typeof entry.payload.action === "string" && (
              <span className="audit-detail">action: {String(entry.payload.action)}</span>
            )}
            {typeof entry.payload.reason === "string" && (
              <span className="audit-detail">{String(entry.payload.reason)}</span>
            )}
            {entry.model_version && <span className="audit-detail mono">model {entry.model_version}</span>}
          </div>
        </li>
      ))}
    </ol>
  );
}
