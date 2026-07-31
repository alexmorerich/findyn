import type { Env } from '../types';
import { FORCES } from '../domain';

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
  regime: null | { label: string; confidence: number | null; model_version: string };
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
    throw new UnknownForceError(
      `unknown force ${opts.force}; expected one of ${FORCES.join('|')}`,
    );
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
