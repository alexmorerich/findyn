import type { Env } from '../types';
import { MIN_THRESHOLD, decimate } from '../lib/decimate';

/** Above the 24,761 daily closes the full S&P record holds. */
export const MAX_OBSERVATION_ROWS = 60000;

export interface SeriesSummary {
  series_id: string;
  provider: string;
  title: string;
  frequency: string;
  unit: string;
  latest_value: number | null;
  latest_obs_date: string | null;
  latest_release_date: string | null;
  observations: number;
  /** Days between the newest observed period and now. */
  freshness_days: number | null;
}

export interface SeriesMetadata {
  series_id: string;
  provider: string;
  title: string;
  frequency: string;
  unit: string;
  seasonal_adjustment: string | null;
  first_observation: string | null;
  last_observation: string | null;
  notes: string | null;
}

export interface Observation {
  obs_date: string;
  release_date: string;
  revision_date: string | null;
  value: number;
}

function daysSince(date: string | null, now: Date): number | null {
  if (!date) return null;
  const ts = Date.parse(`${date}T00:00:00Z`);
  if (Number.isNaN(ts)) return null;
  return Math.floor((now.getTime() - ts) / 86_400_000);
}

/**
 * Catalogue of ingested series with their newest observation.
 *
 * `ROW_NUMBER` picks the newest period per series, breaking ties on
 * release_date so a revision supersedes the original print.
 */
export async function listSeries(env: Env, now = new Date()): Promise<SeriesSummary[]> {
  const { results } = await env.DB.prepare(
    `WITH ranked AS (
       SELECT series_id, obs_date, release_date, value,
              ROW_NUMBER() OVER (
                PARTITION BY series_id ORDER BY obs_date DESC, revision_date DESC
              ) AS rn,
              COUNT(*) OVER (PARTITION BY series_id) AS n
         FROM macro_series
     )
     SELECT m.series_id, m.provider, m.title, m.frequency, m.unit,
            r.obs_date     AS latest_obs_date,
            r.release_date AS latest_release_date,
            r.value        AS latest_value,
            COALESCE(r.n, 0) AS observations
       FROM series_metadata m
       LEFT JOIN ranked r ON r.series_id = m.series_id AND r.rn = 1
      ORDER BY m.provider, m.series_id`,
  ).all<Omit<SeriesSummary, 'freshness_days'>>();

  return (results ?? []).map((r) => ({
    ...r,
    freshness_days: daysSince(r.latest_obs_date, now),
  }));
}

export async function getSeriesMetadata(
  env: Env,
  seriesId: string,
): Promise<SeriesMetadata | null> {
  return env.DB.prepare(
    `SELECT series_id, provider, title, frequency, unit, seasonal_adjustment,
            first_observation, last_observation, notes
       FROM series_metadata WHERE series_id = ?`,
  )
    .bind(seriesId)
    .first<SeriesMetadata>();
}

export interface ObservationResult {
  observations: Observation[];
  available: number;
  truncated: boolean;
  decimated: { from: number; to: number; method: 'lttb' } | null;
}

/**
 * Raw observations for one series, oldest-first, optionally downsampled.
 *
 * `points` exists because the daily S&P record is 24,761 rows and the chart that
 * wants "everything since 1927" is reading it straight out of `macro_series` —
 * the engine's own chart metrics only reach as far as the *publication* series,
 * which FRED licence-caps at ten years. Decimating here is the same contract as
 * `/assets/:asset/history`: shape-preserving, and never silent about it.
 */
export async function getObservations(
  env: Env,
  seriesId: string,
  opts: { from?: string; to?: string; limit?: number; points?: number } = {},
): Promise<ObservationResult> {
  const limit = Math.min(Math.max(opts.limit ?? MAX_OBSERVATION_ROWS, 1), MAX_OBSERVATION_ROWS);

  const conditions = ['series_id = ?'];
  const bindings: (string | number)[] = [seriesId];
  if (opts.from) {
    conditions.push('obs_date >= ?');
    bindings.push(opts.from);
  }
  if (opts.to) {
    conditions.push('obs_date <= ?');
    bindings.push(opts.to);
  }
  const counted = await env.DB.prepare(
    `SELECT COUNT(*) AS n FROM macro_series WHERE ${conditions.join(' AND ')}`,
  )
    .bind(...bindings)
    .first<{ n: number }>();
  const available = counted?.n ?? 0;

  // Ascending: taking the newest N and reversing would drop the *oldest* rows
  // of a wide window, which is exactly the truncation this has to make visible.
  const { results } = await env.DB.prepare(
    `SELECT obs_date, release_date, revision_date, value
       FROM macro_series
      WHERE ${conditions.join(' AND ')}
      ORDER BY obs_date ASC, revision_date DESC
      LIMIT ?`,
  )
    .bind(...bindings, limit)
    .all<Observation>();

  const rows = results ?? [];
  const target = opts.points;
  if (target && target >= MIN_THRESHOLD && rows.length > target) {
    const finite = rows.filter((r) => Number.isFinite(r.value));
    const sampled = decimate(
      finite.map((r) => ({ ...r, x: Date.parse(`${r.obs_date}T00:00:00Z`), y: r.value })),
      target,
    );
    return {
      observations: sampled.map(({ x: _x, y: _y, ...rest }) => rest),
      available,
      truncated: available > limit,
      decimated: { from: finite.length, to: sampled.length, method: 'lttb' },
    };
  }

  return { observations: rows, available, truncated: available > limit, decimated: null };
}
