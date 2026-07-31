import type { Env } from '../types';
import { FORCES, REGIMES } from '../domain';
import { MIN_THRESHOLD, decimate } from '../lib/decimate';
import { MAX_HISTORY_ROWS } from './assets';

/**
 * The v1 read surface (FINDYN_V1_SPEC.md §13): `/state` and `/forces`.
 *
 * Both are spec endpoints that predate the multi-asset namespace, and both are
 * kept rather than redirected — they answer a question `/assets/:asset/state`
 * does not. `/assets/equity/state` serves the engine's `AssetState`: one regime,
 * one risk score, the portfolio layer's input. `/state` serves the *two-layer
 * state* of §2 — the kinematic layer K(t) and the force layer F(t) side by side
 * — which is what the market-navigation framing is actually about.
 *
 * They also have different lifecycles. K(t) and F(t) exist from sub-milestone A;
 * an `AssetState` needs the regime model from B. So `/state` goes live here,
 * with a `regime: null` that says plainly which half has not landed.
 */

/** Feature names that make up K(t) (§2.1), in the order a reader wants them. */
export const KINEMATIC_FEATURES = [
  'price_filtered',
  'velocity',
  'acceleration',
  'jerk_z',
  'ffd_price',
] as const;

export interface ForcePoint {
  as_of: string;
  force: string;
  score: number;
  components: Record<string, number> | null;
  model_version: string;
}

export interface KinematicState {
  as_of: string | null;
  model_version: string | null;
  /** Feature name -> value, in model units (`price_filtered` is a log level). */
  features: Record<string, number>;
}

export interface ForceSnapshot {
  as_of: string | null;
  scores: Record<string, number>;
  components: Record<string, Record<string, number> | null>;
}

export interface TwoLayerState {
  kinematics: KinematicState;
  forces: ForceSnapshot;
  /**
   * Null until the regime model lands (sub-milestone B). Present as an explicit
   * null rather than an absent key: a consumer must be able to tell "not
   * computed yet" from "the field moved".
   */
  regime: null | {
    label: string;
    confidence: number | null;
    model_version: string;
  };
}

function parseJson<T>(raw: string | null, fallback: T): T {
  if (!raw) return fallback;
  try {
    return JSON.parse(raw) as T;
  } catch {
    return fallback;
  }
}

/**
 * K(t) on the newest date the engine has published features for.
 *
 * The date is chosen first and the features read for it, rather than taking the
 * newest row of each feature independently. The features differ in how far back
 * they can be computed — `jerk_z` waits for its z-score baseline, `ffd_price`
 * for a full weight window — but on any given date they either all exist or the
 * date is not one the engine can speak about. Reading them independently would
 * assemble a snapshot from several different days and label it with one.
 */
export async function getKinematicState(env: Env, asset = 'equity'): Promise<KinematicState> {
  const newest = await env.DB.prepare(
    `SELECT date, model_version FROM derived_features
      WHERE asset = ?
      ORDER BY date DESC, model_version DESC
      LIMIT 1`,
  )
    .bind(asset)
    .first<{ date: string; model_version: string }>();

  if (!newest) {
    return { as_of: null, model_version: null, features: {} };
  }

  const { results } = await env.DB.prepare(
    `SELECT feature, value FROM derived_features
      WHERE asset = ? AND date = ? AND model_version = ?`,
  )
    .bind(asset, newest.date, newest.model_version)
    .all<{ feature: string; value: number }>();

  const features: Record<string, number> = {};
  for (const row of results ?? []) features[row.feature] = row.value;

  return { as_of: newest.date, model_version: newest.model_version, features };
}

/** F(t) on the newest date the factor pipeline has scored. */
export async function getForceSnapshot(env: Env): Promise<ForceSnapshot> {
  const newest = await env.DB.prepare(
    `SELECT date FROM force_scores ORDER BY date DESC LIMIT 1`,
  ).first<{ date: string }>();

  if (!newest) return { as_of: null, scores: {}, components: {} };

  const { results } = await env.DB.prepare(
    `SELECT force, score, components FROM force_scores
      WHERE date = ?
      ORDER BY model_version DESC`,
  )
    .bind(newest.date)
    .all<{ force: string; score: number; components: string | null }>();

  const scores: Record<string, number> = {};
  const components: Record<string, Record<string, number> | null> = {};
  for (const row of results ?? []) {
    // ORDER BY model_version DESC and first-write-wins: if two model versions
    // scored the same date, the newer one is the answer.
    if (row.force in scores) continue;
    scores[row.force] = row.score;
    components[row.force] = parseJson<Record<string, number> | null>(row.components, null);
  }
  return { as_of: newest.date, scores, components };
}

export async function getTwoLayerState(env: Env, asset = 'equity'): Promise<TwoLayerState> {
  const [kinematics, forces] = await Promise.all([
    getKinematicState(env, asset),
    getForceSnapshot(env),
  ]);
  return { kinematics, forces, regime: null };
}

export class UnknownForceError extends Error {}

/**
 * Force score history with component breakdowns (§13 `/forces`).
 *
 * Generalized by `/api/v1/factors` in the target architecture, but this is the
 * endpoint the spec names and the dashboard's drill-down already speaks; it
 * reads the same `force_scores` table.
 */
export async function getForces(
  env: Env,
  opts: { from?: string; to?: string; force?: string; limit?: number } = {},
): Promise<ForcePoint[]> {
  if (opts.force && !(FORCES as readonly string[]).includes(opts.force)) {
    throw new UnknownForceError(`unknown force ${opts.force}; expected one of ${FORCES.join('|')}`);
  }

  const limit = Math.min(Math.max(opts.limit ?? 5000, 1), 20000);
  const conditions: string[] = [];
  const bindings: (string | number)[] = [];

  if (opts.force) {
    conditions.push('force = ?');
    bindings.push(opts.force);
  }
  if (opts.from) {
    conditions.push('date >= ?');
    bindings.push(opts.from);
  }
  if (opts.to) {
    conditions.push('date <= ?');
    bindings.push(opts.to);
  }
  bindings.push(limit);

  const where = conditions.length ? `WHERE ${conditions.join(' AND ')}` : '';
  const { results } = await env.DB.prepare(
    `SELECT date, force, score, components, model_version FROM force_scores
      ${where}
      ORDER BY date DESC, force ASC
      LIMIT ?`,
  )
    .bind(...bindings)
    .all<{
      date: string;
      force: string;
      score: number;
      components: string | null;
      model_version: string;
    }>();

  return (results ?? [])
    .map((r) => ({
      as_of: r.date,
      force: r.force,
      score: r.score,
      components: parseJson<Record<string, number> | null>(r.components, null),
      model_version: r.model_version,
    }))
    .reverse();
}

// ---------------------------------------------------------------------------
// §13 /regime — regime probability history
// ---------------------------------------------------------------------------

export interface RegimePoint {
  as_of: string;
  /** Regime name -> probability. The whole posterior, not the argmax. */
  probabilities: Record<string, number>;
  /** Highest-probability regime on this date, for a ribbon or a badge. */
  regime: string;
  confidence: number;
}

export interface RegimeHistory {
  asset: string;
  count: number;
  available: number;
  truncated: boolean;
  decimated: { from: number; to: number; method: 'lttb' } | null;
  regimes: readonly string[];
  model_version: string | null;
  points: RegimePoint[];
}

/**
 * The posterior per date, pivoted into one row per date.
 *
 * Stored one row per (date, regime) because that is the shape §7 gives the
 * table; served pivoted because a stacked-area chart wants a date's five
 * probabilities together, and making the browser do the pivot over 25,000 rows
 * is the work this endpoint exists to avoid.
 *
 * Decimation applies to *dates*, never to regimes: dropping a regime from a
 * date would leave a posterior that does not sum to one.
 */
export async function getRegimeHistory(
  env: Env,
  asset: string,
  opts: { from?: string; to?: string; points?: number } = {},
): Promise<RegimeHistory> {
  const newest = await env.DB.prepare(
    `SELECT model_version FROM regime_state
      WHERE asset = ?
      ORDER BY date DESC, model_version DESC
      LIMIT 1`,
  )
    .bind(asset)
    .first<{ model_version: string }>();

  if (!newest) {
    return {
      asset,
      count: 0,
      available: 0,
      truncated: false,
      decimated: null,
      regimes: REGIMES,
      model_version: null,
      points: [],
    };
  }

  // Pinned to one model version: a refit republishes the whole window under a
  // new version, and mixing two of them would show a posterior stitched from
  // two different models on one axis.
  const conditions = ['asset = ?', 'model_version = ?'];
  const bindings: (string | number)[] = [asset, newest.model_version];
  if (opts.from) {
    conditions.push('date >= ?');
    bindings.push(opts.from);
  }
  if (opts.to) {
    conditions.push('date <= ?');
    bindings.push(opts.to);
  }

  const { results } = await env.DB.prepare(
    `SELECT date, regime, probability FROM regime_state
      WHERE ${conditions.join(' AND ')}
      ORDER BY date ASC
      LIMIT ?`,
  )
    .bind(...bindings, MAX_HISTORY_ROWS)
    .all<{ date: string; regime: string; probability: number }>();

  const byDate = new Map<string, Record<string, number>>();
  for (const row of results ?? []) {
    const entry = byDate.get(row.date) ?? {};
    entry[row.regime] = row.probability;
    byDate.set(row.date, entry);
  }

  const dates = [...byDate.keys()].sort();
  const all: RegimePoint[] = dates.map((date) => {
    const probabilities = byDate.get(date)!;
    let regime = '';
    let confidence = -1;
    for (const [name, value] of Object.entries(probabilities)) {
      if (value > confidence) {
        confidence = value;
        regime = name;
      }
    }
    return {
      as_of: date,
      probabilities,
      regime,
      confidence: Math.max(confidence, 0),
    };
  });

  const target = opts.points;
  let points = all;
  let decimated: RegimeHistory['decimated'] = null;
  if (target && target >= MIN_THRESHOLD && all.length > target) {
    // Decimated on the *severity* of the posterior, so the retained dates are
    // the ones where the regime picture actually changed — a stacked area chart
    // sampled on one arbitrary regime would keep the wrong turning points.
    const severity = all.map((p, index) => ({
      index,
      x: Date.parse(`${p.as_of}T00:00:00Z`),
      y: REGIMES.reduce((sum, name, rank) => sum + rank * (p.probabilities[name] ?? 0), 0),
    }));
    const kept = decimate(severity, target);
    points = kept.map((s) => all[s.index]!);
    decimated = { from: all.length, to: points.length, method: 'lttb' };
  }

  return {
    asset,
    count: points.length,
    available: all.length,
    truncated: (results ?? []).length >= MAX_HISTORY_ROWS,
    decimated,
    regimes: REGIMES,
    model_version: newest.model_version,
    points,
  };
}

// ---------------------------------------------------------------------------
// §13 /instability and /forecast — the M4 read surface
// ---------------------------------------------------------------------------

/** Metrics `/instability` pivots. All three factors, never the composite alone. */
export const INSTABILITY_METRICS = [
  'rii',
  'p_transition',
  'p_shock',
  'p_transmission',
  'crash_risk',
] as const;

export interface InstabilityPoint {
  as_of: string;
  rii: number | null;
  p_transition: number | null;
  p_shock: number | null;
  p_transmission: number | null;
  crash_risk: number | null;
}

export interface InstabilityHistory {
  asset: string;
  count: number;
  available: number;
  decimated: { from: number; to: number; method: string } | null;
  points: InstabilityPoint[];
}

/**
 * RII and the crash decomposition per date.
 *
 * Assembled from `engine_output` rather than the v1 `instability_index` table.
 * Both hold the same numbers and `engine_output` is where per-date engine
 * metrics already live, keyed and indexed for exactly this read; adding a
 * second write path to a parallel table would mean two places to disagree about
 * what the RII was on a given day.
 *
 * §4's rule is enforced in the *shape*: a caller receives all three factors or
 * none. There is no way to request `crash_risk` from this endpoint alone.
 */
export async function getInstability(
  env: Env,
  asset = 'equity',
  opts: { from?: string; to?: string; points?: number } = {},
): Promise<InstabilityHistory> {
  const conditions = ['asset = ?', `metric IN (${INSTABILITY_METRICS.map(() => '?').join(',')})`];
  const bindings: (string | number)[] = [asset, ...INSTABILITY_METRICS];
  if (opts.from) {
    conditions.push('as_of >= ?');
    bindings.push(opts.from);
  }
  if (opts.to) {
    conditions.push('as_of <= ?');
    bindings.push(opts.to);
  }

  const { results } = await env.DB.prepare(
    `SELECT as_of, metric, value FROM engine_output
      WHERE ${conditions.join(' AND ')}
      ORDER BY as_of ASC
      LIMIT ?`,
  )
    .bind(...bindings, MAX_HISTORY_ROWS)
    .all<{ as_of: string; metric: string; value: number }>();

  const byDate = new Map<string, Record<string, number>>();
  for (const row of results ?? []) {
    const entry = byDate.get(row.as_of) ?? {};
    entry[row.metric] = row.value;
    byDate.set(row.as_of, entry);
  }

  const all: InstabilityPoint[] = [...byDate.keys()].sort().map((as_of) => {
    const row = byDate.get(as_of)!;
    return {
      as_of,
      rii: row.rii ?? null,
      p_transition: row.p_transition ?? null,
      p_shock: row.p_shock ?? null,
      p_transmission: row.p_transmission ?? null,
      crash_risk: row.crash_risk ?? null,
    };
  });

  const target = opts.points;
  if (target && target >= MIN_THRESHOLD && all.length > target) {
    // Decimated on the RII, so the dates kept are the ones where instability
    // actually moved rather than an arbitrary stride through calm stretches.
    const sampled = decimate(
      all.map((p, index) => ({
        index,
        x: Date.parse(`${p.as_of}T00:00:00Z`),
        y: p.rii ?? 0,
      })),
      target,
    );
    return {
      asset,
      count: sampled.length,
      available: all.length,
      decimated: { from: all.length, to: sampled.length, method: 'lttb' },
      points: sampled.map((s) => all[s.index]!),
    };
  }

  return {
    asset,
    count: all.length,
    available: all.length,
    decimated: null,
    points: all,
  };
}

export interface ForecastBand {
  horizon: string;
  educational_only: boolean;
  /** Quantile -> projected log index level. */
  quantiles: Record<string, number>;
}

export interface ForecastResponse {
  asset: string;
  as_of: string | null;
  model_version: string | null;
  horizons: ForecastBand[];
}

/**
 * §13 `/forecast` — quantile bands per horizon.
 *
 * `educational_only` travels on every band. §10 excludes those horizons from
 * accuracy evaluation entirely, and a consumer that cannot tell a 50-year
 * scenario from a 6-month forecast will eventually plot them on one axis.
 */
export async function getForecast(
  env: Env,
  asset = 'equity',
  horizon?: string,
): Promise<ForecastResponse> {
  const newest = await env.DB.prepare(
    `SELECT as_of, model_version FROM forecast_distribution
      WHERE asset = ?
      ORDER BY as_of DESC, model_version DESC
      LIMIT 1`,
  )
    .bind(asset)
    .first<{ as_of: string; model_version: string }>();

  if (!newest) return { asset, as_of: null, model_version: null, horizons: [] };

  const conditions = ['asset = ?', 'as_of = ?', 'model_version = ?'];
  const bindings: string[] = [asset, newest.as_of, newest.model_version];
  if (horizon) {
    conditions.push('horizon = ?');
    bindings.push(horizon);
  }

  const { results } = await env.DB.prepare(
    `SELECT horizon, quantile, value, educational_only FROM forecast_distribution
      WHERE ${conditions.join(' AND ')}
      ORDER BY horizon, quantile`,
  )
    .bind(...bindings)
    .all<{
      horizon: string;
      quantile: number;
      value: number;
      educational_only: number;
    }>();

  const bands = new Map<string, ForecastBand>();
  for (const row of results ?? []) {
    const band = bands.get(row.horizon) ?? {
      horizon: row.horizon,
      educational_only: Boolean(row.educational_only),
      quantiles: {},
    };
    band.quantiles[String(row.quantile)] = row.value;
    bands.set(row.horizon, band);
  }

  return {
    asset,
    as_of: newest.as_of,
    model_version: newest.model_version,
    horizons: [...bands.values()],
  };
}
