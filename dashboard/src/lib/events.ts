/**
 * The historical event layer for the Equity Dynamics Lab — a frontend-only
 * annotation, and deliberately nothing more.
 *
 * These dates are not model output and are not read from the API. They are
 * fixed reference points a reader already knows, put on the axis so the
 * derivatives can be read against something. Nothing here feeds a calculation:
 * the numbers an event shows are looked up out of the series already on screen,
 * so an event marker can never introduce a value the panels do not have.
 *
 * Each event carries a `window` as well as a `date`. The date is where the flag
 * sits; the window is what "jump to this event" frames, and the span the price
 * change is measured over. They are separate because a crash is a period and a
 * marker is a point, and collapsing the two would either put the flag in the
 * wrong place or frame a single day.
 */

export interface MarketEvent {
  id: string;
  /** Flag position — the day the event is conventionally dated to. */
  date: string;
  label: string;
  /** Short label for the flag itself, which has about six characters of room. */
  short: string;
  /** What the viewport frames, and the span `price change` is measured over. */
  window: { from: string; to: string };
  /** One line of context. Not a claim the model makes — a reminder of the date. */
  note: string;
}

/**
 * Ordered oldest first, which is the order the flags are laid out in and the
 * order the overflow logic drops them in when a viewport holds too many.
 */
export const MARKET_EVENTS: readonly MarketEvent[] = [
  {
    id: 'crash-1929',
    date: '1929-10-29',
    label: '1929 Crash',
    short: '1929',
    window: { from: '1929-09-03', to: '1932-07-08' },
    note: 'Black Tuesday, and the three-year decline that followed it to the July 1932 low.',
  },
  {
    id: 'oil-1973',
    date: '1973-10-17',
    label: '1973 Oil Crisis',
    short: '1973',
    window: { from: '1973-10-01', to: '1974-12-31' },
    note: 'The OPEC embargo, and the 1973–74 bear market that ran through it.',
  },
  {
    id: 'black-monday-1987',
    date: '1987-10-19',
    label: '1987 Black Monday',
    short: '1987',
    window: { from: '1987-08-25', to: '1988-01-29' },
    note: 'A single session down about a fifth — the sharpest one-day move in the daily record.',
  },
  {
    id: 'dotcom-2000',
    date: '2000-03-24',
    label: '2000 Dotcom Bubble',
    short: '2000',
    window: { from: '2000-03-24', to: '2002-10-09' },
    note: 'The March 2000 peak, and the two-and-a-half-year decline to the October 2002 low.',
  },
  {
    id: 'gfc-2008',
    date: '2008-09-15',
    label: '2008 Financial Crisis',
    short: '2008',
    window: { from: '2007-10-09', to: '2009-03-09' },
    note: 'Lehman as the marker; the window spans the October 2007 peak to the March 2009 low.',
  },
  {
    id: 'covid-2020',
    date: '2020-02-19',
    label: '2020 COVID Crash',
    short: '2020',
    window: { from: '2020-02-19', to: '2020-03-23' },
    note: 'Peak to trough in twenty-three sessions — the fastest drawdown of that size on record.',
  },
] as const;

/** Epoch milliseconds for a `YYYY-MM-DD` date, at UTC midnight. */
export function isoToTime(iso: string): number {
  return Date.parse(`${iso}T00:00:00Z`);
}

/** `YYYY-MM-DD` for an epoch-millisecond instant, in UTC. */
export function timeToIso(t: number): string {
  return new Date(t).toISOString().slice(0, 10);
}
