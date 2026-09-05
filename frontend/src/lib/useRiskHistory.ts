import { useEffect, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import type { PaginatedRiskHistory } from "./types";

/**
 * 100, not 50: an entity that participated in BOTH the frozen 2018 benchmark and the
 * Simulated Recent Operational Stream has its combined history in one paginated list
 * here (mrs.data.recent_stream reuses existing customer/terminal ids -- see
 * docs/RECENT_STREAM.md), and recent-stream rows are always the tail (2026 postdates
 * every 2018 row). At 50, the default "last 50" landing window for several elevated
 * entities showed only their already-settled final NORMAL state, with the actual
 * NORMAL -> RISK_RISING -> HIGH_RISK -> RECOVERY arc sitting just out of view one
 * "Older" click away. Empirically checked against the real recent-stream population
 * (every terminal/customer whose behavioral state ever reached HIGH_RISK and later
 * recovered): 50 showed the complete arc in the default window for only 4/15
 * terminals and 76/210 customers; 100 raises that to 13/15 and 172/210 with no further
 * gain from going higher (150/200 tested identically at 13/15) -- the remaining few
 * simply recovered too early in a very long history for any bounded tail window to
 * reach, which paging back still reaches truthfully. No timestamps are reordered or
 * fabricated; this only changes how much of the real, already-chronological sequence
 * is visible without an extra click.
 */
const PAGE_SIZE = 100;

/**
 * GET /customers/{id}/risk and /terminals/{id}/risk are ordered oldest-first with no
 * descending option, so offset 0 is the START of an entity's history, not its current
 * state. This hook probes the total first, then lands the default page on the most
 * recent window -- otherwise a long-lived entity's "current state" badge would show
 * whatever state it was in back at its very first scored transaction.
 */
export function useRiskHistory(
  entityId: number,
  fetchPage: (id: number, limit: number, offset: number) => Promise<PaginatedRiskHistory>,
  queryKeyPrefix: string,
) {
  const [offset, setOffset] = useState<number | null>(null);

  // A route/selection change swaps entityId without unmounting this hook's owner
  // (React Router reuses the component instance across param changes, and the
  // Command Center swaps `entity` in place) -- without this, a stale offset computed
  // for the PREVIOUS entity's total would carry over and could land past the new
  // entity's actual row count, silently returning zero rows.
  useEffect(() => {
    setOffset(null);
  }, [entityId]);

  const probe = useQuery({
    queryKey: [queryKeyPrefix, "probe", entityId],
    queryFn: () => fetchPage(entityId, 1, 0),
    enabled: Number.isFinite(entityId),
  });

  useEffect(() => {
    if (probe.data && offset === null) {
      setOffset(Math.max(0, probe.data.total - PAGE_SIZE));
    }
  }, [probe.data, offset]);

  const page = useQuery({
    queryKey: [queryKeyPrefix, "page", entityId, offset],
    queryFn: () => fetchPage(entityId, PAGE_SIZE, offset!),
    enabled: Number.isFinite(entityId) && offset !== null,
    placeholderData: (prev) => prev,
  });

  return {
    items: page.data?.items ?? [],
    total: page.data?.total ?? probe.data?.total ?? 0,
    offset: offset ?? 0,
    setOffset,
    pageSize: PAGE_SIZE,
    isLoading: probe.isLoading || (offset !== null && page.isLoading),
    isError: probe.isError || page.isError,
    error: probe.error ?? page.error,
    refetch: () => {
      probe.refetch();
      page.refetch();
    },
  };
}
