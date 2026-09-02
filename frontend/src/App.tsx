import { Route, Routes } from "react-router-dom";
import { AppShell } from "./components/layout/AppShell";
import { Overview } from "./pages/Overview";
import { Alerts } from "./pages/Alerts";
import { AlertDetail } from "./pages/AlertDetail";
import { TransactionsExplorer } from "./pages/TransactionsExplorer";
import { TransactionDetail } from "./pages/TransactionDetail";
import { EntitySearch } from "./pages/EntitySearch";
import { CustomerDetail } from "./pages/CustomerDetail";
import { TerminalDetail } from "./pages/TerminalDetail";
import { Replay } from "./pages/Replay";
import { SystemHealth } from "./pages/SystemHealth";
import { Network } from "./pages/Network";
import { NotFound } from "./pages/NotFound";

export default function App() {
  return (
    <Routes>
      <Route element={<AppShell />}>
        <Route index element={<Overview />} />
        <Route path="alerts" element={<Alerts />} />
        <Route path="alerts/:id" element={<AlertDetail />} />
        <Route path="transactions" element={<TransactionsExplorer />} />
        <Route path="transactions/:id" element={<TransactionDetail />} />
        <Route path="customers" element={<EntitySearch kind="customer" />} />
        <Route path="customers/:id" element={<CustomerDetail />} />
        <Route path="terminals" element={<EntitySearch kind="terminal" />} />
        <Route path="terminals/:id" element={<TerminalDetail />} />
        <Route path="replay" element={<Replay />} />
        <Route path="system" element={<SystemHealth />} />
        <Route path="network" element={<Network />} />
        <Route path="*" element={<NotFound />} />
      </Route>
    </Routes>
  );
}
