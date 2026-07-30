/**
 * The single place the dashboard talks to the FinDyn serving plane.
 *
 * The site is static and the API is live, so every read happens in the browser
 * at runtime — nothing is baked in at build time. `PUBLIC_FINDYN_API` is
 * substituted by Vite at build time into the client bundle.
 *
 * Contract: FINDYN_V1_SPEC.md §13. Every 200 response is an envelope.
 */

const PLACEHOLDER_API_BASE = 'https://findyn.<account>.workers.dev';

/**
 * Origin of the serving plane, without a trailing slash.
 *
 * The Worker serves this dashboard from its own assets binding, so in the
 * deployed case the API is same-origin and the page can simply ask the browser
 * where it is. That is deliberate: a build-time origin has to be known before
 * the thing it points at exists, and a stale one produces a site that builds
 * clean and reads nothing.
 *
 * `PUBLIC_FINDYN_API` still wins when set, for a dashboard hosted apart from
 * the Worker (Pages, a custom domain, or `npm run dev` against `wrangler dev`).
 */
function resolveApiBase(): string {
  const configured = import.meta.env.PUBLIC_FINDYN_API;
  if (configured && !configured.includes('<account>')) return configured.replace(/\/+$/, '');
  if (typeof window !== 'undefined' && window.location?.origin) return window.location.origin;
  return PLACEHOLDER_API_BASE;
}

export const API_BASE: string = resolveApiBase();

export const API_V1 = `${API_BASE}/api/v1`;

/** True when there is no API origin to read at all — worth saying out loud. */
export const API_BASE_IS_PLACEHOLDER = API_BASE.includes('<account>');

// ---------------------------------------------------------------------------
// Envelope + payload types (mirrors serving/src/lib/responses.ts and api/*.ts)
// ---------------------------------------------------------------------------

export interface Envelope<T> {
  as_of: string | null;
  model_version: string | null;
  stale: boolean;
  data: T;
  disclaimer: string;
}

export interface Meta {
  version: string;
  milestone: string;
  spec: string;
  env: string;
  info_set: string;
  vocabulary: {
    forces: string[];
    regimes: string[];
    horizons: string[];
    assets?: string[];
    rate_regimes?: string[];
  };
}

export interface SourceHealth {
  source: string;
  status: string;
  last_run_at: string;
  rows_written: number;
  error: string | null;
}

export interface Health {
  ok: boolean;
  env: string;
  info_set: string;
  last_ingestion_at: string | null;
  stale: boolean;
  sources: SourceHealth[];
  degraded_reason?: string;
}

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
  freshness_days: number | null;
}

export interface SeriesList {
  count: number;
  series: SeriesSummary[];
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

export interface SeriesDetail {
  metadata: SeriesMetadata;
  /** Newest first, as returned by the API. */
  observations: Observation[];
}

export interface PitAvailable {
  series_id: string;
  title: string;
  provider: string;
  unit: string;
  obs_date: string;
  release_date: string;
  value: number;
  staleness_days: number;
}

export interface PitWithheld {
  series_id: string;
  title: string;
  obs_date: string;
  release_date: string;
  published_days_later: number;
  reason: string;
}

export interface PitSnapshot {
  as_of: string;
  available: PitAvailable[];
  withheld: PitWithheld[];
  total_series: number;
}

// ---------------------------------------------------------------------------
// Multi-asset engine surface (P1)
// ---------------------------------------------------------------------------

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
  status: 'live' | 'awaiting_first_run';
  as_of: string | null;
  model_version: string | null;
  regime: string | null;
  risk_score: number | null;
  confidence: number | null;
  stale: boolean;
  freshness_days: number | null;
}

export interface AssetList {
  count: number;
  assets: AssetSummary[];
}

export interface HistoryPoint {
  as_of: string;
  value: number;
  meta: unknown;
}

export interface AssetHistory {
  asset: string;
  metric: string;
  count: number;
  points: HistoryPoint[];
}

/**
 * Human-facing copy for each engine, keyed by the registry name. Kept here
 * rather than on the card so the Engines panel needs no template change when an
 * engine ships — an unknown name still renders, just without the blurb.
 */
export const ENGINE_LABELS: Record<string, { title: string; blurb: string; href?: string }> = {
  money: { title: 'FinMoney', blurb: 'Time value — cash carry and discount factors' },
  rates: {
    title: 'FinRates',
    blurb: 'Interest-rate dynamics — Nelson-Siegel curve factors and rate regimes',
    href: '/rates',
  },
  equity: { title: 'FinEquity', blurb: 'Growth and risk premium — kinematics, regimes, instability' },
  gold: { title: 'FinGold', blurb: 'Trust and crisis protection — regime switching on real rates' },
  crypto: { title: 'FinCrypto', blurb: 'Network scarcity — experimental, excluded from portfolios' },
};

// ---------------------------------------------------------------------------
// Result type
// ---------------------------------------------------------------------------

/**
 * A request never throws. Callers get one of four outcomes and must render
 * all of them — a blank page is not an acceptable failure mode (§14.2).
 *
 * `not_implemented` is deliberately not an error: the spec reserves those
 * endpoints for later milestones, and the UI says so rather than showing red.
 */
export type ApiResult<T> =
  | { ok: true; envelope: Envelope<T> }
  | { ok: false; kind: 'not_implemented'; milestone: string; message: string }
  | { ok: false; kind: 'not_found'; message: string }
  | { ok: false; kind: 'unreachable'; message: string }
  | { ok: false; kind: 'error'; status: number; message: string };

export type ApiFailure<T> = Exclude<ApiResult<T>, { ok: true }>;

interface ApiErrorBody {
  error?: string;
  message?: string;
  milestone?: string;
}

function isEnvelope<T>(body: unknown): body is Envelope<T> {
  return (
    typeof body === 'object' &&
    body !== null &&
    'data' in body &&
    'stale' in body &&
    'disclaimer' in body
  );
}

/**
 * GET a path under /api/v1 and normalise everything that can go wrong.
 * `path` must start with '/'. Query strings should already be encoded.
 */
export async function apiGet<T>(path: string, init?: RequestInit): Promise<ApiResult<T>> {
  if (API_BASE_IS_PLACEHOLDER) {
    return {
      ok: false,
      kind: 'unreachable',
      message:
        'PUBLIC_FINDYN_API is not configured for this build, so there is no API to read. ' +
        'Set it to your serving-plane origin and rebuild.',
    };
  }

  let response: Response;
  try {
    response = await fetch(`${API_V1}${path}`, {
      headers: { accept: 'application/json' },
      ...init,
    });
  } catch (err) {
    return {
      ok: false,
      kind: 'unreachable',
      message: err instanceof Error ? err.message : String(err),
    };
  }

  let body: unknown = null;
  try {
    body = await response.json();
  } catch {
    body = null;
  }

  if (!response.ok) {
    const detail = (body ?? {}) as ApiErrorBody;
    if (response.status === 501) {
      return {
        ok: false,
        kind: 'not_implemented',
        milestone: detail.milestone ?? 'unknown',
        message: detail.message ?? 'Reserved by the spec; not delivered yet.',
      };
    }
    if (response.status === 404) {
      return { ok: false, kind: 'not_found', message: detail.message ?? 'Not found.' };
    }
    return {
      ok: false,
      kind: 'error',
      status: response.status,
      message: detail.message ?? `${response.status} ${response.statusText}`,
    };
  }

  if (!isEnvelope<T>(body)) {
    return {
      ok: false,
      kind: 'error',
      status: response.status,
      message: 'Response was not a FinDyn envelope.',
    };
  }

  return { ok: true, envelope: body };
}

// ---------------------------------------------------------------------------
// Endpoint helpers
// ---------------------------------------------------------------------------

export const getMeta = () => apiGet<Meta>('/meta');
export const getHealth = () => apiGet<Health>('/health');
export const getSeriesList = () => apiGet<SeriesList>('/series');

export function getSeries(
  seriesId: string,
  opts: { from?: string; to?: string; limit?: number } = {},
): Promise<ApiResult<SeriesDetail>> {
  const query = new URLSearchParams();
  if (opts.from) query.set('from', opts.from);
  if (opts.to) query.set('to', opts.to);
  if (opts.limit !== undefined) query.set('limit', String(opts.limit));
  const qs = query.toString();
  const suffix = qs === '' ? '' : `?${qs}`;
  // Series ids contain ':' (e.g. SHILLER:CAPE) and must survive the path.
  return apiGet<SeriesDetail>(`/series/${encodeURIComponent(seriesId)}${suffix}`);
}

export function getPit(asOf: string): Promise<ApiResult<PitSnapshot>> {
  return apiGet<PitSnapshot>(`/pit?as_of=${encodeURIComponent(asOf)}`);
}

/** Registered engines and their staleness — drives the home Engines panel. */
export const getAssets = () => apiGet<AssetList>('/assets');

export const getAssetState = (asset: string) =>
  apiGet<AssetState>(`/assets/${encodeURIComponent(asset)}/state`);

export function getAssetHistory(
  asset: string,
  metric: string,
  opts: { from?: string; to?: string; limit?: number } = {},
): Promise<ApiResult<AssetHistory>> {
  const query = new URLSearchParams({ metric });
  if (opts.from) query.set('from', opts.from);
  if (opts.to) query.set('to', opts.to);
  if (opts.limit !== undefined) query.set('limit', String(opts.limit));
  return apiGet<AssetHistory>(`/assets/${encodeURIComponent(asset)}/history?${query}`);
}

/**
 * Endpoints the spec reserves (§13) whose milestones have not landed.
 * The dashboard probes them so the roadmap it shows is the API's own answer
 * rather than a hard-coded list of statuses.
 */
export const RESERVED_ENDPOINTS = [
  { path: '/state', summary: 'Latest kinematic state K(t) + force state F(t)' },
  { path: '/regime', summary: 'Regime probability history' },
  { path: '/forces', summary: 'Force scores with component breakdowns' },
  { path: '/instability', summary: 'RII + crash-risk decomposition history' },
  { path: '/forecast', summary: 'Quantile forecast distributions per horizon' },
  { path: '/simulate', summary: 'Monte Carlo summary statistics' },
] as const;

/** The §18 disclaimer, rendered in the footer of every page. */
export const DISCLAIMER =
  'FinDyn is a market navigation system. It estimates market position, velocity, ' +
  'acceleration, instability, and regime-transition probability. It does not claim to ' +
  'predict the future with certainty and does not provide investment advice. Final ' +
  'allocation decisions depend on investor constraints, risk tolerance, and tax jurisdiction.';
