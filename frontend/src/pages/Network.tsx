import { Icon } from "../components/common/Icon";
import "./Network.css";

/**
 * Entity Network -- explicitly deferred (user instruction, 2026-08-31): the network
 * visualization module is not implemented yet, and this page must not fabricate one.
 * No placeholder graph, no simulated nodes/edges, nothing decorative. Kept as a
 * reserved, functional route inside the real application shell so the nav item isn't
 * dead, with a restrained empty state until the module is built.
 */
export function Network() {
  return (
    <div className="page">
      <div className="page-header">
        <div>
          <h1 className="page-title">Entity Network</h1>
          <p className="page-subtitle">Reserved for the entity relationship investigation module.</p>
        </div>
      </div>

      <div className="net-empty">
        <Icon name="hub" size={28} className="net-empty-icon" />
        <p className="net-empty-title">Network investigation canvas</p>
        <p className="net-empty-note">Awaiting network analysis module.</p>
      </div>
    </div>
  );
}
