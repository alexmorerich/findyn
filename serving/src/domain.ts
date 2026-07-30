/**
 * Shared domain vocabulary. Every name here is fixed by FINDYN_V1_SPEC.md —
 * changing one is a schema change, not a refactor.
 */

/** §2.2 — Market Force State F(t). */
export const FORCES = [
  'valuation',
  'earnings',
  'liquidity',
  'rates',
  'credit',
  'inflation',
  'labor',
  'risk_appetite',
  'sentiment',
] as const;
export type Force = (typeof FORCES)[number];

/** §9 L2 — the five HMM regimes. */
export const REGIMES = [
  'bull_expansion',
  'normal_expansion',
  'late_cycle',
  'bear',
  'crisis',
] as const;
export type Regime = (typeof REGIMES)[number];

/** §10 — forecast horizons. Educational horizons are excluded from accuracy evaluation. */
export const HORIZONS = [
  'tactical',
  'strategic',
  'generational',
  'educational_30y',
  'educational_50y',
] as const;
export type Horizon = (typeof HORIZONS)[number];

export const EDUCATIONAL_HORIZONS: ReadonlySet<string> = new Set([
  'educational_30y',
  'educational_50y',
]);

/** §10 — quantiles stored for every forecast distribution. */
export const QUANTILES = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95] as const;

/** §18 — must appear in every API response and on the dashboard. */
export const DISCLAIMER =
  'FinDyn is a market navigation system. It estimates market position, velocity, ' +
  'acceleration, instability, and regime-transition probability. It does not claim to ' +
  'predict the future with certainty and does not provide investment advice. Final ' +
  'allocation decisions depend on investor constraints, risk tolerance, and tax jurisdiction.';

/** Data older than this is flagged `stale: true` in API responses (§13). */
export const STALENESS_THRESHOLD_HOURS = 36;
