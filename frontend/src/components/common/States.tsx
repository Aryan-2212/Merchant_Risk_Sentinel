import { ApiError } from "../../lib/api";
import "./States.css";

export function Loading({ label }: { label: string }) {
  return (
    <div className="state-block state-loading">
      <span className="state-spinner" aria-hidden="true" />
      <span>{label}</span>
    </div>
  );
}

export function EmptyState({ children }: { children: React.ReactNode }) {
  return <div className="state-block state-empty">{children}</div>;
}

export function ErrorBlock({ error, onRetry }: { error: unknown; onRetry?: () => void }) {
  const message = error instanceof ApiError ? error.detail : "Risk data could not be loaded.";
  const notFound = error instanceof ApiError && error.status === 404;
  return (
    <div className="state-block state-error">
      <span>{notFound ? message : `Risk data could not be loaded. ${message}`}</span>
      {onRetry && !notFound && (
        <button className="state-retry" onClick={onRetry}>
          Retry
        </button>
      )}
    </div>
  );
}
