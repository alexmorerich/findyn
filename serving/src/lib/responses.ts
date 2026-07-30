import type { Context } from 'hono';
import { DISCLAIMER, STALENESS_THRESHOLD_HOURS } from '../domain';

/**
 * Every public response carries `as_of`, `model_version`, `stale` and the
 * disclaimer (FINDYN_V1_SPEC.md §13, §18).
 */
export interface Envelope<T> {
  as_of: string | null;
  model_version: string | null;
  stale: boolean;
  data: T;
  disclaimer: string;
}

export function envelope<T>(
  data: T,
  meta: { as_of?: string | null; model_version?: string | null; stale?: boolean } = {},
): Envelope<T> {
  return {
    as_of: meta.as_of ?? null,
    model_version: meta.model_version ?? null,
    stale: meta.stale ?? false,
    data,
    disclaimer: DISCLAIMER,
  };
}

export function isStale(asOf: string | null | undefined, now: Date = new Date()): boolean {
  if (!asOf) return true;
  const ts = Date.parse(asOf.length === 10 ? `${asOf}T00:00:00Z` : asOf);
  if (Number.isNaN(ts)) return true;
  return now.getTime() - ts > STALENESS_THRESHOLD_HOURS * 3_600_000;
}

/**
 * Endpoint reserved by the spec but not yet delivered by its milestone.
 * Explicit 501 beats a silent empty payload — a consumer can tell the
 * difference between "not built" and "no data".
 */
export function notImplemented(c: Context, milestone: string, section: string) {
  return c.json(
    {
      error: 'not_implemented',
      message: `Reserved by FINDYN_V1_SPEC.md ${section}; delivered in milestone ${milestone}.`,
      milestone,
    },
    501,
  );
}
