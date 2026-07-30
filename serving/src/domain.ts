/**
 * Shared domain vocabulary. Every name here is fixed by FINDYN_V1_SPEC.md —
 * changing one is a schema change, not a refactor.
 *
 * The compute plane splits this across three modules (core/contracts/vocab.py,
 * factors/definitions.py, engines/equity/domain.py); serving keeps one flat
 * file. compute/tests/test_domain.py is the seam and fails on any drift.
 */

/** The five asset engines (docs/redesign/01-target-architecture.md §1). */
export const ASSETS = ['money', 'rates', 'equity', 'gold', 'crypto'] as const;
export type Asset = (typeof ASSETS)[number];

/**
 * §2.2 — the shared risk factors. Named FORCES here (and `force_scores` in D1)
 * because renaming the table is not worth the churn; the compute plane calls
 * them factors.
 *
 * Scores share one axis: 100 is maximally risk-supportive. `real_rate` and
 * `usd_strength` therefore score high when real rates are low and the dollar is
 * weak — the name identifies the variable, the axis fixes what the number means.
 */
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
  'real_rate',
  'usd_strength',
] as const;
export type Force = (typeof FORCES)[number];

/** Rate-regime vocabulary, owned by findynamics/engines/rates/domain.py. */
export const RATE_REGIMES = [
  'steep_easing',
  'steep_tightening',
  'flat',
  'inverted',
  're_steepening',
] as const;
export type RateRegime = (typeof RATE_REGIMES)[number];

/**
 * Liquidity-state vocabulary, owned by findynamics/engines/money/domain.py.
 *
 * The condition of the funding market, not the level of rates. Ordered
 * least-to-most constrained, because `engine_output` publishes the state as its
 * index here (`liquidity_code`) and a chart of it should read upwards as
 * conditions worsen.
 */
export const MONEY_REGIMES = ['abundant', 'normal', 'tightening', 'stressed'] as const;
export type MoneyRegime = (typeof MONEY_REGIMES)[number];

/**
 * Standard discount horizons — the tenors `D(t, h)` is published for.
 *
 * Distinct from HORIZONS, which are *forecast* horizons. These are points on a
 * curve, and every engine that discounts a cash flow uses this grid so that two
 * of them cannot disagree about what "the 3y discount factor" means.
 */
export const DISCOUNT_HORIZONS = [
  '1m',
  '3m',
  '6m',
  '1y',
  '2y',
  '3y',
  '5y',
  '7y',
  '10y',
  '20y',
  '30y',
] as const;
export type DiscountHorizon = (typeof DISCOUNT_HORIZONS)[number];

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
