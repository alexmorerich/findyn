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

export type RangeKey = '1y' | '5y' | '10y' | '20y' | '50y' | 'max' | 'monthly' | 'custom';

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
  /**
   * Explicit window, set only by the custom range. `years` is a *relative*
   * window and re-resolves against today on every load; these two are absolute,
   * which is what makes a custom window linkable.
   */
  from?: string;
  to?: string;
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
    key: '10y',
    label: '10Y',
    years: 10,
    points: 1200,
    // Ten years is roughly a threefold move. Linear reads honestly at that
    // ratio, and a linear axis is easier to take a slope off by eye — which is
    // what the velocity panel below it is for.
    log: false,
    monthly: false,
    description: 'daily closes and filtered level, last ten years',
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
    // One button, not two. The daily record starts 1927-12-30, so "the last
    // hundred years" and "all of it" are the same query; shipping both would
    // differ only in implying that one of them might return less. The label
    // says 100Y because that is what a reader is looking for, and the
    // description says the real first date because that is what they get.
    key: 'max',
    label: '100Y',
    years: null,
    points: 3000,
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

const ISO_DATE = /^\d{4}-\d{2}-\d{2}$/;

/**
 * A user-chosen window, as a range like any other.
 *
 * Point budget and axis are derived from the span rather than fixed, so a
 * custom window behaves like the preset nearest its length: forty years of
 * custom gets the log axis that fifty years of preset gets, because the reason
 * for the log axis is the span, not the button.
 */
export function customRange(from: string, to: string): RangeSpec {
  const years = Math.max(
    (Date.parse(`${to}T00:00:00Z`) - Date.parse(`${from}T00:00:00Z`)) / (365.25 * 86_400_000),
    0,
  );
  return {
    key: 'custom',
    label: 'Custom',
    years: null,
    points: years > 40 ? 3000 : years > 15 ? 2000 : years > 5 ? 1200 : 600,
    log: years > 15,
    monthly: false,
    description: `custom window, ${from} to ${to}`,
    from,
    to,
  };
}

/** The custom window a page opens with before the reader has picked one. */
export function defaultCustomWindow(now: Date = new Date()): { from: string; to: string } {
  const start = new Date(now);
  start.setFullYear(start.getFullYear() - 30);
  return { from: start.toISOString().slice(0, 10), to: now.toISOString().slice(0, 10) };
}

export function rangeFor(key: string | null | undefined): RangeSpec {
  return RANGES.find((r) => r.key === key) ?? RANGES.find((r) => r.key === DEFAULT_RANGE)!;
}

/**
 * `?range=` from the address bar, falling back to the default.
 *
 * `range=custom` also reads `from`/`to`. A custom range with an unusable pair
 * degrades to the default rather than to an empty chart: a link that has lost
 * one of its dates should still show the reader something true.
 */
export function rangeFromUrl(search: string = window.location.search): RangeSpec {
  const params = new URLSearchParams(search);
  const key = params.get('range');
  if (key === 'custom') {
    const from = params.get('from');
    const to = params.get('to');
    if (from && to && ISO_DATE.test(from) && ISO_DATE.test(to) && from < to) {
      return customRange(from, to);
    }
    return rangeFor(null);
  }
  return rangeFor(key);
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
  if (range.key === 'custom' && range.from && range.to) {
    url.searchParams.set('from', range.from);
    url.searchParams.set('to', range.to);
  } else {
    url.searchParams.delete('from');
    url.searchParams.delete('to');
  }
  window.history.replaceState({}, '', url);
}

/** ISO `from` date for a range, or `undefined` for "everything". */
export function fromDate(range: RangeSpec, now: Date = new Date()): string | undefined {
  if (range.from) return range.from;
  if (range.years === null) return undefined;
  const start = new Date(now);
  start.setFullYear(start.getFullYear() - range.years);
  return start.toISOString().slice(0, 10);
}

/** ISO `to` date for a range — set only by a custom window. */
export function toDate(range: RangeSpec): string | undefined {
  return range.to;
}
