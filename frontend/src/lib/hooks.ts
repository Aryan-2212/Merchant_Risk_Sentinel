import { useQuery } from "@tanstack/react-query";
import { api } from "./api";

/** Polled in the background for the topbar system-status dot and the System Health
 * page -- a real reachability check against the database, not a fabricated status. */
export function useHealth() {
  return useQuery({
    queryKey: ["health"],
    queryFn: api.health,
    refetchInterval: 30_000,
    staleTime: 15_000,
    retry: 0,
  });
}
