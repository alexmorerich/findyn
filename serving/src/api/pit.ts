import type { Env } from '../types';

/**
 * Point-in-time demonstration (FINDYN_V1_SPEC.md §14.1).
 *
 * Mirrors compute/findynamics/data/pit.py::pit_join in SQL and additionally returns what
 * was deliberately withheld. Showing only the surviving rows proves nothing —
 * the interesting claim is that data which exists in the database today was
 * correctly invisible at the chosen date.
 */

export interface PitAvailable {
  series_id: string;
  title: string;
  provider: string;
  unit: string;
  obs_date: string;
  release_date: string;
  value: number;
  /** Days between the observed period and the as-of date. */
  staleness_days: number;
}

export interface PitWithheld {
  series_id: string;
  title: string;
  obs_date: string;
  release_date: string;
  /** Days after the as-of date that this figure was published. */
  published_days_later: number;
  reason: string;
}

export interface PitSnapshot {
  as_of: string;
  available: PitAvailable[];
  withheld: PitWithheld[];
  total_series: number;
}

const DATE_RE = /^\d{4}-\d{2}-\d{2}$/;

export class InvalidDateError extends Error {}

function dayDiff(from: string, to: string): number {
  return Math.round((Date.parse(`${to}T00:00:00Z`) - Date.parse(`${from}T00:00:00Z`)) / 86_400_000);
}

export async function pitSnapshot(env: Env, asOf: string): Promise<PitSnapshot> {
  if (!DATE_RE.test(asOf) || Number.isNaN(Date.parse(`${asOf}T00:00:00Z`))) {
    throw new InvalidDateError(`as_of must be a valid YYYY-MM-DD date, got "${asOf}"`);
  }

  // Filtered on revision_date, not release_date: release_date says when the
  // period first became observable, but a later revision of that period was not
  // knowable until it was issued. Filtering on release_date would hand back a
  // number nobody could have seen yet.
  const availableQuery = env.DB.prepare(
    `WITH knowable AS (
       SELECT s.series_id, s.obs_date, s.release_date, s.value,
              ROW_NUMBER() OVER (
                PARTITION BY s.series_id ORDER BY s.obs_date DESC, s.revision_date DESC
              ) AS rn
         FROM macro_series s
        WHERE s.revision_date <= ?
     )
     SELECT k.series_id, m.title, m.provider, m.unit, k.obs_date, k.release_date, k.value
       FROM knowable k
       JOIN series_metadata m ON m.series_id = k.series_id
      WHERE k.rn = 1
      ORDER BY m.provider, k.series_id`,
  ).bind(asOf);

  // Periods that exist today but had not been published by as_of. Restricted to
  // the next unreleased period per series — the first thing a naive join would
  // have leaked.
  const withheldQuery = env.DB.prepare(
    `WITH future AS (
       SELECT s.series_id, s.obs_date, s.release_date,
              ROW_NUMBER() OVER (
                PARTITION BY s.series_id ORDER BY s.obs_date ASC, s.release_date ASC
              ) AS rn
         FROM macro_series s
        WHERE s.release_date > ?
     )
     SELECT f.series_id, m.title, f.obs_date, f.release_date
       FROM future f
       JOIN series_metadata m ON m.series_id = f.series_id
      WHERE f.rn = 1
      ORDER BY m.provider, f.series_id`,
  ).bind(asOf);

  const countQuery = env.DB.prepare(`SELECT COUNT(*) AS n FROM series_metadata`);

  // One round trip; D1 returns results positionally.
  type AvailableRow = Omit<PitAvailable, 'staleness_days'>;
  type WithheldRow = Omit<PitWithheld, 'published_days_later' | 'reason'>;

  const batch = await env.DB.batch([availableQuery, withheldQuery, countQuery]);

  const available = ((batch[0]?.results ?? []) as AvailableRow[]).map((r) => ({
    ...r,
    staleness_days: dayDiff(r.obs_date, asOf),
  }));

  const withheld = ((batch[1]?.results ?? []) as WithheldRow[]).map((r) => ({
    ...r,
    published_days_later: dayDiff(asOf, r.release_date),
    reason: `Published ${r.release_date}, after the ${asOf} information cutoff.`,
  }));

  const total = ((batch[2]?.results ?? []) as { n: number }[])[0]?.n ?? 0;

  return { as_of: asOf, available, withheld, total_series: total };
}
