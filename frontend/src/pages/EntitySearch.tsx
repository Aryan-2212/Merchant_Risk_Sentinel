import { useState } from "react";
import { useNavigate } from "react-router-dom";

/**
 * No GET /customers or GET /terminals list endpoint exists (by design decision --
 * see Phase 8 Step 7 scope notes), so this page is an ID lookup rather than a browse
 * table. Reach a specific customer/terminal here, or by clicking its ID from an
 * alert, transaction, or replay row.
 */
export function EntitySearch({ kind }: { kind: "customer" | "terminal" }) {
  const [id, setId] = useState("");
  const navigate = useNavigate();
  const label = kind === "customer" ? "Customer" : "Terminal";
  const base = kind === "customer" ? "/customers" : "/terminals";

  function onSubmit(e: React.FormEvent) {
    e.preventDefault();
    const trimmed = id.trim();
    if (trimmed && /^\d+$/.test(trimmed)) {
      navigate(`${base}/${trimmed}`);
    }
  }

  return (
    <div className="page">
      <div className="page-header">
        <div>
          <h1 className="page-title">{label}s</h1>
          <p className="page-subtitle">
            No entity list exists yet -- look up a {label.toLowerCase()} by ID, or open one from an alert or
            transaction.
          </p>
        </div>
      </div>

      <form className="card" style={{ maxWidth: 420 }} onSubmit={onSubmit}>
        <div className="section">
          <label className="field-label" htmlFor="entity-id">
            {label} ID
          </label>
          <div className="toolbar">
            <input
              id="entity-id"
              className="text-input"
              style={{ flex: 1 }}
              inputMode="numeric"
              placeholder={kind === "customer" ? "e.g. 4677" : "e.g. 8935"}
              value={id}
              onChange={(e) => setId(e.target.value)}
              autoFocus
            />
            <button className="btn btn-primary" type="submit">
              Open
            </button>
          </div>
        </div>
      </form>
    </div>
  );
}
