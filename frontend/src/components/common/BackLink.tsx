import { Link } from "react-router-dom";
import { Icon } from "./Icon";
import "./BackLink.css";

/** Explicit contextual navigation back to a detail page's parent list -- deliberately
 * a real link to a fixed route, not browser-history navigate(-1): a detail page can be
 * reached from several different places (sidebar, search, another entity's table), so
 * "back" here always means "to the canonical list for this entity type", predictable
 * regardless of how the operator arrived. */
export function BackLink({ to, label }: { to: string; label: string }) {
  return (
    <Link to={to} className="back-link">
      <Icon name="arrow_back" size={16} />
      {label}
    </Link>
  );
}
