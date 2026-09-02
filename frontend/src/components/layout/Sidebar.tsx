import { NavLink } from "react-router-dom";
import { Icon } from "../common/Icon";
import "./Sidebar.css";

const PRIMARY = [
  { to: "/", label: "Overview", icon: "dashboard", end: true },
  { to: "/terminals", label: "Terminals", icon: "terminal" },
  { to: "/customers", label: "Customers", icon: "group" },
  { to: "/alerts", label: "Alerts", icon: "warning" },
  { to: "/network", label: "Network", icon: "hub" },
];

//: Real, functioning screens that aren't part of the approved reference's 5-item
//: nav -- kept reachable, visually subordinate rather than removed.
const SECONDARY = [
  { to: "/replay", label: "Replay", icon: "history" },
  { to: "/transactions", label: "Transactions", icon: "receipt_long" },
  { to: "/system", label: "System health", icon: "monitor_heart" },
];

export function Sidebar() {
  return (
    <nav className="sidebar" aria-label="Primary">
      <div className="sidebar-brand">
        <div className="sidebar-mark">
          <Icon name="security" size={18} />
        </div>
        <div className="sidebar-brand-text">
          <span className="sidebar-brand-name">Sentinel</span>
          <span className="sidebar-version">V0.1.0</span>
        </div>
      </div>

      <div className="sidebar-group">
        {PRIMARY.map((item) => (
          <NavLink
            key={item.to}
            to={item.to}
            end={item.end}
            className={({ isActive }) => `sidebar-link${isActive ? " active" : ""}`}
          >
            <Icon name={item.icon} size={20} />
            <span className="sidebar-link-label">{item.label}</span>
          </NavLink>
        ))}
      </div>

      <div className="sidebar-divider" />

      <div className="sidebar-group sidebar-group-secondary">
        {SECONDARY.map((item) => (
          <NavLink
            key={item.to}
            to={item.to}
            className={({ isActive }) => `sidebar-link sidebar-link-secondary${isActive ? " active" : ""}`}
          >
            <Icon name={item.icon} size={18} />
            <span className="sidebar-link-label">{item.label}</span>
          </NavLink>
        ))}
      </div>

      <div className="sidebar-footer">
        <span className="sidebar-badge">Simulated benchmark data</span>
      </div>
    </nav>
  );
}
