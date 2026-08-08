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
 *
 * There used to be a fourth rule here — a `deep` flag marking the ranges that
 * outran the publication series, on which the page abandoned the engine's own
 * metrics and read `YAHOO:^GSPC` straight out of `macro_series` with no filtered
 * overlay and a banner explaining the gap. The engine now runs its filter over
 * that same record (`engines/equity/prices.py::publication_path`), so every
 * daily range is served by the engine and the special case is gone. What it was
 * protecting against is still real and is still handled: a chart must never
 * imply a model output that was not computed.
 */

export type RangeKey = '1y' | '5y' | '20y' | '50y' | 'max' | 'monthly';

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
    description: 'daily closes and filtered level, last twelve months',
  },
  {
    key: '5y',
    label: '5Y',
    years: 5,
    points: 900,
    log: false,
    monthly: false,
    description: 'daily closes and filtered level, last five years',
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
    description: 'daily closes and filtered level, last twenty years, log scale',
  },
  {
    key: '50y',
    label: '50Y',
    years: 50,
    points: 2000,
    log: true,
    monthly: false,
    description: 'daily closes and filtered level, last fifty years, log scale',
  },
  {
    // There is no separate 100Y button because there is no separate window: the
    // daily record starts 1927-12-30, so "the last hundred years" and "all of
    // it" are the same query, and two buttons issuing it would differ only in
    // implying that one of them might return less.
    key: 'max',
    label: 'Max (~99Y)',
    years: null,
    points: 2500,
    log: true,
    monthly: false,
    description: 'the whole daily record from 1927-12-30 — about ninety-nine years, log scale',
  },
  {
    key: 'monthly',
    label: '1871+',
    years: null,
    points: 2200,
    log: true,
    monthly: true,
    description: 'Shiller month-end composite from 1871, log scale — monthly, not daily',
  },
] as const;

/**
 * The whole record, not the last five years.
 *
 * The engine filters the daily S&P back to 1927-12-30, and a page that opens on
 * a five-year window makes that invisible to anyone who does not think to click.
 * The cost is bounded by the server-side windowing above: Max asks for 2,500
 * points and receives 2,500 points, whether the record holds 24,761 rows or
 * 2,500.
 */
export const DEFAULT_RANGE: RangeKey = 'max';

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
