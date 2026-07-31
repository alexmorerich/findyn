import type { Env } from '../types';
import { ASSETS } from '../domain';
import { MIN_THRESHOLD, decimate } from '../lib/decimate';

/**
 * Read side of the multi-asset tables (01-target-architecture.md §7).
 *
 * The endpoints are registry-driven: `/assets` enumerates the closed vocabulary
 * rather than whatever happens to be in the database, so an engine that has
 * shipped but never run is reported as "awaiting first run" instead of vanishing.
 * That distinction is what lets the dashboard's Engines panel be built once and
 * never touched again as engines land.
 */

export interface Signal {
  name: string;
  value: number;
  direction: -1 | 0 | 1;
  note: string | null;
}

export interface AssetState {
  asset: string;
  as_of: string;
  model_version: string;
  regime: string;
  expected_return: number | null;
  risk_score: number | null;
  confidence: number | null;
  signals: Signal[];
  components: Record<string, number> | null;
}

export interface AssetSummary {
  asset: string;
  /** `live` once a run has published a state; `awaiting_first_run` before that. */
  status: 'live' | 'awaiting_first_run';
  as_of: string | null;
  model_version: string | null;
  regime: string | null;
  risk_score: number | null;
  confidence: number | null;
  stale: boolean;
  freshness_days: number | null;
}

export interface HistoryPoint {
  as_of: string;
  value: number;
  meta: unknown;
  /**
   * When the run that produced this row published it — the row's true vintage.
   *
   * Exposed because the compute plane reads these series back as another
   * engine's input (P2: FinMoney discounts long horizons off FinRates' published
   * curve). Point-in-time correctness there needs the *publication* date, not the
   * date the value describes; without it a consumer would have to synthesize a
   * lag and could read a factor at a cutoff before the run that computed it.
   */
  written_at: string | null;
}

/** An engine that has not published in this long is not reporting. */
export const ASSET_STALE_DAYS = 5;

/**
 * Whether an engine's published `as_of` counts as stale.
 *
 * Deliberately *not* the `isStale` used by the ingestion endpoints. That one
 * measures hours since data arrived, which is the right question for a feed;
 * an `AssetState.as_of` is a **market date**, and a market date is a day old the
 * moment it is published and three days old every Monday. Measuring it in hours
 * flagged every healthy run as stale — including the run that had just finished
 * — and left `/assets` and `/assets/:asset/state` giving opposite answers about
 * the same row.
 *
 * Same rule as `listAssets` now, so they cannot disagree.
 */
export function isAssetStale(asOf: string | null | undefined, now = new Date()): boolean {
  const days = daysSince(asOf ?? null, now);
  return days === null || days > ASSET_STALE_DAYS;
}

interface StateRow {
  asset: string;
  as_of: string;
  model_version: string;
  regime: string;
  expected_return: number | null;
  risk_score: number | null;
  confidence: number | null;
  signals: string;
  components: string | null;
}

export class UnknownAssetError extends Error {}

export function assertKnownAsset(asset: string): string {
  if (!(ASSETS as readonly string[]).includes(asset)) {
    throw new UnknownAssetError(`unknown asset ${asset}; expected one of ${ASSETS.join('|')}`);
  }
  return asset;
}

function daysSince(date: string | null, now: Date): number | null {
  if (!date) return null;
  const ts = Date.parse(`${date}T00:00:00Z`);
  if (Number.isNaN(ts)) return null;
  return Math.floor((now.getTime() - ts) / 86_400_000);
}

/** JSON columns are parsed defensively: a malformed blob must not 500 a read. */
function parseJson<T>(raw: string | null, fallback: T): T {
  if (!raw) return fallback;
  try {
    return JSON.parse(raw) as T;
  } catch {
    return fallback;
  }
}

function toState(row: StateRow): AssetState {
  return {
    asset: row.asset,
    as_of: row.as_of,
    model_version: row.model_version,
    regime: row.regime,
    expected_return: row.expected_return,
    risk_score: row.risk_score,
    confidence: row.confidence,
    signals: parseJson<Signal[]>(row.signals, []),
    components: parseJson<Record<string, number> | null>(row.components, null),
  };
}

/**
 * One row per engine in the vocabulary, whether or not it has ever run.
 *
 * `ROW_NUMBER` picks the newest state per asset; ties on the same date break on
 * model_version so a refit's output wins over the model it replaced.
 */
export async function listAssets(env: Env, now = new Date()): Promise<AssetSummary[]> {
  const { results } = await env.DB.prepare(
    `WITH ranked AS (
       SELECT asset, as_of, model_version, regime, risk_score, confidence,
              ROW_NUMBER() OVER (
                PARTITION BY asset ORDER BY as_of DESC, model_version DESC
              ) AS rn
         FROM asset_state
     )
     SELECT asset, as_of, model_version, regime, risk_score, confidence
       FROM ranked WHERE rn = 1`,
  ).all<{
    asset: string;
    as_of: string;
    model_version: string;
    regime: string;
    risk_score: number | null;
    confidence: number | null;
  }>();

  const latest = new Map((results ?? []).map((r) => [r.asset, r]));

  return ASSETS.map((asset) => {
    const row = latest.get(asset);
    const freshness = daysSince(row?.as_of ?? null, now);
    return {
      asset,
      status: row ? 'live' : 'awaiting_first_run',
      as_of: row?.as_of ?? null,
      model_version: row?.model_version ?? null,
      regime: row?.regime ?? null,
      risk_score: row?.risk_score ?? null,
      confidence: row?.confidence ?? null,
      stale: freshness === null || freshness > ASSET_STALE_DAYS,
      freshness_days: freshness,
    };
  });
}

/** Newest published state for one engine, or null if it has never run. */
export async function getAssetState(env: Env, asset: string): Promise<AssetState | null> {
  const row = await env.DB.prepare(
    `SELECT asset, as_of, model_version, regime, expected_return, risk_score,
            confidence, signals, components
       FROM asset_state
      WHERE asset = ?
      ORDER BY as_of DESC, model_version DESC
      LIMIT 1`,
  )
    .bind(asset)
    .first<StateRow>();

  return row ? toState(row) : null;
}

/** Distinct metric names this engine has published, for discovery. */
export async function listMetrics(env: Env, asset: string): Promise<string[]> {
  const { results } = await env.DB.prepare(
    `SELECT DISTINCT metric FROM engine_output WHERE asset = ? ORDER BY metric`,
  )
    .bind(asset)
    .all<{ metric: string }>();
  return (results ?? []).map((r) => r.metric);
}

/**
 * Hard ceiling on rows read for one series.
 *
 * Above the ~24,700 daily closes a full S&P history holds, so a Max-range
 * request is expressible rather than silently clipped. It is still a ceiling:
 * exceeding it sets `truncated`, and the caller is told rather than handed a
 * prefix that looks like a short history.
 */
export const MAX_HISTORY_ROWS = 60000;

/** Default target point count when a caller asks for decimation. */
export const DEFAULT_CHART_POINTS = 2000;

export interface HistoryResult {
  points: HistoryPoint[];
  /** Rows the window actually holds, before any decimation. */
  available: number;
  /**
   * True when the window holds more than `MAX_HISTORY_ROWS` and the response is
   * therefore a *prefix* of the answer rather than the answer.
   *
   * The one failure this whole shape exists to prevent: a clipped series and a
   * genuinely short one look identical on a chart, and the reader concludes the
   * market began in 2021.
   */
  truncated: boolean;
  /** Present when the series was downsampled for rendering. */
  decimated: { from: number; to: number; method: 'lttb' } | null;
}

export async function getHistory(
  env: Env,
  asset: string,
  metric: string,
  opts: { from?: string; to?: string; limit?: number; points?: number } = {},
): Promise<HistoryResult> {
  const limit = Math.min(Math.max(opts.limit ?? MAX_HISTORY_ROWS, 1), MAX_HISTORY_ROWS);

  const conditions = ['asset = ?', 'metric = ?'];
  const bindings: (string | number)[] = [asset, metric];
  if (opts.from) {
    conditions.push('as_of >= ?');
    bindings.push(opts.from);
  }
  if (opts.to) {
    conditions.push('as_of <= ?');
    bindings.push(opts.to);
  }

  const counted = await env.DB.prepare(
    `SELECT COUNT(*) AS n FROM engine_output WHERE ${conditions.join(' AND ')}`,
  )
    .bind(...bindings)
    .first<{ n: number }>();
  const available = counted?.n ?? 0;

  // Ascending in SQL now that the window is explicit: taking the newest N and
  // reversing would have silently dropped the *oldest* rows of a wide range,
  // which is precisely the truncation this endpoint has to make visible.
  const { results } = await env.DB.prepare(
    `SELECT as_of, value, meta, written_at FROM engine_output
      WHERE ${conditions.join(' AND ')}
      ORDER BY as_of ASC
      LIMIT ?`,
  )
    .bind(...bindings, limit)
    .all<{ as_of: string; value: number; meta: string | null; written_at: string | null }>();

  const rows: HistoryPoint[] = (results ?? []).map((r) => ({
    as_of: r.as_of,
    value: r.value,
    meta: parseJson<unknown>(r.meta, null),
    written_at: r.written_at ?? null,
  }));

  const target = opts.points;
  if (target && target >= MIN_THRESHOLD && rows.length > target) {
    const finite = rows.filter((r) => Number.isFinite(r.value));
    const sampled = decimate(
      finite.map((r) => ({ ...r, x: Date.parse(`${r.as_of}T00:00:00Z`), y: r.value })),
      target,
    );
    return {
      points: sampled.map(({ x: _x, y: _y, ...rest }) => rest),
      available,
      truncated: available > limit,
      decimated: { from: finite.length, to: sampled.length, method: 'lttb' },
    };
  }

  return { points: rows, available, truncated: available > limit, decimated: null };
}
