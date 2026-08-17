import type { Env } from '../types';
import { isKnownProfile } from '../domain';

/**
 * Read side of the portfolio layer (01-target-architecture.md §Portfolio,
 * 04-ui-plan.md §P6).
 *
 * One endpoint, `GET /api/v1/portfolio?profile=`, returning the latest weight
 * *distribution* for a profile plus its templated conditional implication and
 * the input `AssetState` references. Everything the `/portfolio` page draws —
 * the fan chart, the implication block, the "why" expander, the degraded badges
 * — comes from this one row, because the compute plane packed the whole
 * distribution into the `weights` blob rather than spreading it across columns.
 */

/** The default profile when none is asked for — the balanced 60/30/10 mix. */
export const DEFAULT_PROFILE = 'balanced';

export class UnknownProfileError extends Error {}

export function assertKnownProfile(profile: string): string {
  if (!isKnownProfile(profile)) {
    throw new UnknownProfileError(`unknown profile ${profile}; expected conservative|balanced|growth`);
  }
  return profile;
}

/** One asset's weight band, as the compute plane serialised it. */
export interface WeightBand {
  asset: string;
  mean: number;
  neutral: number;
  delta: number;
  degraded: boolean;
  reason: string;
  /** Quantile label ('q05'…'q95') -> weight. A degraded asset is a point mass. */
  quantiles: Record<string, number>;
}

/** One input engine state, for the "why" expander. */
export interface PortfolioInput {
  asset: string;
  as_of: string | null;
  regime: string;
  expected_return: number | null;
  risk_score: number;
  confidence: number;
  stale: boolean;
  reason: string;
}

export interface PortfolioState {
  profile: string;
  as_of: string;
  model_version: string;
  degraded: boolean;
  implication: string;
  method: string;
  risk_free: number | null;
  quantile_labels: string[];
  assets: WeightBand[];
  inputs: PortfolioInput[];
  factors: Record<string, number>;
  degraded_reason: string[];
}

interface PortfolioRow {
  profile: string;
  as_of: string;
  model_version: string;
  weights: string;
  implication: string;
  degraded: number;
}

/** A malformed blob must not 500 a read — parse defensively, like assets.ts. */
function parseJson<T>(raw: string | null, fallback: T): T {
  if (!raw) return fallback;
  try {
    return JSON.parse(raw) as T;
  } catch {
    return fallback;
  }
}

function toState(row: PortfolioRow): PortfolioState {
  const blob = parseJson<Record<string, unknown>>(row.weights, {});
  return {
    profile: row.profile,
    as_of: row.as_of,
    model_version: row.model_version,
    degraded: row.degraded === 1,
    implication: row.implication,
    method: typeof blob.method === 'string' ? blob.method : '',
    risk_free: typeof blob.risk_free === 'number' ? blob.risk_free : null,
    quantile_labels: Array.isArray(blob.quantile_labels)
      ? (blob.quantile_labels as string[])
      : [],
    assets: Array.isArray(blob.assets) ? (blob.assets as WeightBand[]) : [],
    inputs: Array.isArray(blob.inputs) ? (blob.inputs as PortfolioInput[]) : [],
    factors:
      typeof blob.factors === 'object' && blob.factors !== null
        ? (blob.factors as Record<string, number>)
        : {},
    degraded_reason: Array.isArray(blob.degraded_reason)
      ? (blob.degraded_reason as string[])
      : [],
  };
}

/** Newest allocation for one profile, or null if that profile has never run. */
export async function getPortfolio(env: Env, profile: string): Promise<PortfolioState | null> {
  const row = await env.DB.prepare(
    `SELECT profile, as_of, model_version, weights, implication, degraded
       FROM portfolio_state
      WHERE profile = ?
      ORDER BY as_of DESC, model_version DESC
      LIMIT 1`,
  )
    .bind(profile)
    .first<PortfolioRow>();

  return row ? toState(row) : null;
}
