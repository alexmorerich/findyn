/**
 * Provider adapter contract — FINDYN_V1_SPEC.md §5.1.
 *
 * Every external source is an isolated adapter behind these interfaces, with a
 * per-provider rate limiter and circuit breaker. Yahoo Finance is a fallback
 * only and must remain deletable without touching anything outside its own file.
 */

export type ProviderId =
  | 'fred'
  | 'shiller'
  | 'bls'
  | 'bea'
  | 'treasury'
  | 'alphavantage'
  | 'stooq'
  | 'yahoo';

/** One macro observation, carrying its point-in-time release date (§14.1). */
export interface MacroObservation {
  /** Period the value describes, YYYY-MM-DD. */
  obsDate: string;
  /** First date the value was publicly known, YYYY-MM-DD. */
  releaseDate: string;
  value: number;
  /** ALFRED-style vintage identifier when the source exposes one. */
  vintage?: string;
}

export interface PriceBar {
  date: string;
  close: number;
  volume?: number;
}

export interface FetchWindow {
  /** Inclusive start, YYYY-MM-DD. */
  from: string;
  /** Inclusive end, YYYY-MM-DD. */
  to: string;
}

export interface SeriesProvider {
  readonly id: ProviderId;
  /**
   * Fetch a macro series. Implementations must populate `releaseDate` from the
   * source's own vintage data where available, otherwise from the conservative
   * publication lag configured in compute/config/series.yaml (§5.2).
   */
  fetchSeries(seriesId: string, window: FetchWindow): Promise<MacroObservation[]>;
}

export interface PriceProvider {
  readonly id: ProviderId;
  fetchPrices(symbol: string, window: FetchWindow): Promise<PriceBar[]>;
}

/** Raised by an adapter when the upstream source is unavailable or over quota. */
export class ProviderError extends Error {
  constructor(
    readonly provider: ProviderId,
    message: string,
    readonly retryable = true,
  ) {
    super(`[${provider}] ${message}`);
    this.name = 'ProviderError';
  }
}
