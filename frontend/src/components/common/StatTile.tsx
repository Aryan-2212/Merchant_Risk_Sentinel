import "./StatTile.css";

export function StatTile({
  label,
  value,
  hint,
}: {
  label: string;
  value: React.ReactNode;
  hint?: React.ReactNode;
}) {
  return (
    <div className="stat-tile">
      <span className="stat-label">{label}</span>
      <span className="stat-value mono">{value}</span>
      {hint && <span className="stat-hint">{hint}</span>}
    </div>
  );
}
