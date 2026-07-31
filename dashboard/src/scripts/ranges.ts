/**
 * The zoom control for `/equity`, and the rules that make deep history honest.
 *
 * Three properties matter more than the widget itself:
 *
 * 1. **A range is a server-side window, not a client-side filter.** Each range
 *    sends its own `from`/`to`, so Max fetches a century and 1Y fetches a year.
 *    Fetching everything once and slicing in the browser would move 25,000
 *    points per series over the wire to draw 250 of them.
 * 2. **The range lives in the URL.** `?range=20y` is linkable and survives a
 *    reload; a control that silently resets to its default on refresh is a
 *    control that lies about what is on screen.
 * 3. **Deep ranges are logarithmic.** The S&P goes from 17 to 7,400. On a
 *    linear axis the first seventy years are a flat line along the floor, which
 *    is visually identical to not having shipped them.
 *
 * The 1871 tier is deliberately a *different source at a different resolution*
 * rather than a wider window: Shiller is monthly, and drawing month-ends on the
 * same axis as daily closes without saying so would imply a daily record that
 * does not exist.
 */

export type RangeKey = '1y' | '5y' | '20y' | 'max' | 'monthly';

export interface RangeSpec {
  key: RangeKey;
  label: string;
  /** Years back from today; `null` means "everything the source holds". */
  years: number | null;
  /** Target point count asked of the server. */
  points: number;
  /** Log price axis — mandatory once the span outgrows a linear one. */
  log: boolean;
  /** True for the monthly Shiller tier, which is a different series entirely. */
  monthly: boolean;
  /**
   * True when the range outruns the publication series.
   *
   * `FRED:SP500` is licence-capped to a rolling ten years, so the engine's own
   * `price_close` metric cannot reach past 2016 however much history is
   * republished. Beyond that the chart reads the daily S&P record straight out
   * of `macro_series` — same index, different vendor — and the provenance block
   * says so. The filtered overlay is not drawn there, because the model only
   * runs on the publication series and pretending otherwise would draw a line
   * that was never computed.
   */
  deep: boolean;
  /** Shown in the provenance block, so the reader knows what they are seeing. */
  description: string;
}

export const RANGES: readonly RangeSpec[] = [
  {
    key: '1y',
    label: '1Y',
    years: 1,
    points: 400,
    log: false,
    monthly: false,
    deep: false,
    description: 'daily closes, last twelve months',
  },
  {
    key: '5y',
    label: '5Y',
    years: 5,
    points: 900,
    log: false,
    monthly: false,
    deep: false,
    description: 'daily closes, last five years',
  },
  {
    key: '20y',
    label: '20Y',
    years: 20,
    points: 1600,
    // Twenty years of the S&P is roughly a sixfold move. Linear still works,
    // but only just, and the 2008 drawdown reads as shallower than it was.
    log: true,
    monthly: false,
    deep: true,
    description: 'YAHOO:^GSPC daily closes, last twenty years, log scale',
  },
  {
    key: 'max',
    label: 'Max',
    years: null,
    points: 2500,
    log: true,
    monthly: false,
    deep: true,
    description: 'YAHOO:^GSPC daily closes from 1927-12-30, log scale',
  },
  {
    key: 'monthly',
    label: '1871+',
    years: null,
    points: 2200,
    log: true,
    monthly: true,
    deep: false,
    description: 'Shiller month-end composite from 1871, log scale — monthly, not daily',
  },
] as const;

export const DEFAULT_RANGE: RangeKey = '5y';

export function rangeFor(key: string | null | undefined): RangeSpec {
  return RANGES.find((r) => r.key === key) ?? RANGES.find((r) => r.key === DEFAULT_RANGE)!;
}

/** `?range=` from the address bar, falling back to the default. */
export function rangeFromUrl(search: string = window.location.search): RangeSpec {
  return rangeFor(new URLSearchParams(search).get('range'));
}

/**
 * Put the range in the address bar without adding a history entry per click.
 *
 * `replaceState`: a zoom control is a view setting, and filling the back button
 * with five of them makes leaving the page require five presses.
 */
export function writeRangeToUrl(range: RangeSpec): void {
  const url = new URL(window.location.href);
  if (range.key === DEFAULT_RANGE) url.searchParams.delete('range');
  else url.searchParams.set('range', range.key);
  window.history.replaceState({}, '', url);
}

/** ISO `from` date for a range, or `undefined` for "everything". */
export function fromDate(range: RangeSpec, now: Date = new Date()): string | undefined {
  if (range.years === null) return undefined;
  const start = new Date(now);
  start.setFullYear(start.getFullYear() - range.years);
  return start.toISOString().slice(0, 10);
}
