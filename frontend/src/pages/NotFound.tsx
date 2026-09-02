import { Link } from "react-router-dom";

export function NotFound() {
  return (
    <div className="page">
      <div className="page-header">
        <h1 className="page-title">Not found</h1>
      </div>
      <p className="page-subtitle">
        <Link to="/" className="link-id">
          Back to overview
        </Link>
      </p>
    </div>
  );
}
