import { describe, expect, it } from 'vitest';
import { decimate, lttb } from '../src/lib/decimate';

/**
 * Downsampling has one job beyond "fewer points": keep the shape.
 *
 * A century of daily closes cannot be rendered whole, so something is dropped
 * whatever we do. What must not be dropped is a crash — those are single-day
 * events, and a chart of a hundred years of markets with Black Monday missing
 * is worse than one that honestly shows five years, because it looks complete.
 *
 * The synthetic series below stands in for that: a slow drift with three
 * one-day collapses at the positions of 1987, 2008 and 2020 in a 1927-2026
 * daily series. Every zoom level in the UI is exercised.
 */

const TOTAL = 24_761;
const START = Date.parse('1927-12-30T00:00:00Z');
const END = Date.parse('2026-07-30T00:00:00Z');
const DAY = 86_400_000;

/**
 * Trading days are not calendar days: 24,761 sessions span 98 years, so the
 * synthetic index advances about 1.44 calendar days per point. Deriving the
 * spacing rather than assuming one-per-day is what makes `indexOf` land on the
 * right observation.
 */
const SPACING = (END - START) / (TOTAL - 1);

function timeAt(index: number): number {
  return START + index * SPACING;
}

/** Index of a date within the synthetic 1927+ trading-day series. */
function indexOf(iso: string): number {
  return Math.round((Date.parse(`${iso}T00:00:00Z`) - START) / SPACING);
}

const CRASHES = [
  { name: 'Black Monday 1987', index: indexOf('1987-10-19'), drop: 0.2 },
  { name: 'post-Lehman 2008', index: indexOf('2008-09-29'), drop: 0.09 },
  { name: 'COVID 2020', index: indexOf('2020-03-16'), drop: 0.12 },
];

/** Smooth exponential drift with a one-day collapse at each crash. */
function series(): { x: number; y: number }[] {
  const points: { x: number; y: number }[] = [];
  for (let i = 0; i < TOTAL; i++) {
    // Deterministic wobble, so a failure is reproducible.
    const trend = 17 * Math.exp(i * 0.00025) * (1 + 0.02 * Math.sin(i / 90));
    const crash = CRASHES.find((c) => c.index === i);
    points.push({ x: timeAt(i), y: crash ? trend * (1 - crash.drop) : trend });
  }
  return points;
}

const DATA = series();

/** The zoom levels the page offers, in trading-day-ish counts. */
const RANGES = [
  { label: '1Y', span: 252 },
  { label: '5Y', span: 1260 },
  { label: '20Y', span: 5040 },
  { label: 'Max', span: TOTAL },
];

describe('LTTB decimation', () => {
  it('keeps the first and last point, so the span never shrinks', () => {
    const sampled = decimate(DATA, 500);
    expect(sampled[0]!.x).toBe(DATA[0]!.x);
    expect(sampled.at(-1)!.x).toBe(DATA.at(-1)!.x);
  });

  it('returns the raw window untouched when it already fits', () => {
    const small = DATA.slice(0, 400);
    expect(decimate(small, 2000)).toBe(small);
  });

  it('respects the target point count', () => {
    // decimate() may re-insert the global extremes, so the budget is +2.
    expect(decimate(DATA, 2000).length).toBeLessThanOrEqual(2002);
    expect(lttb(DATA, 2000)).toHaveLength(2000);
  });

  it('stays sorted by x', () => {
    const sampled = decimate(DATA, 1500);
    for (let i = 1; i < sampled.length; i++) {
      expect(sampled[i]!.x).toBeGreaterThan(sampled[i - 1]!.x);
    }
  });

  /**
   * The acceptance criterion. For each range that contains a crash, the
   * decimated series must still show it: the lowest value near the event has to
   * be within a whisker of the raw low over the same window.
   *
   * Stated as a local minimum rather than "the exact date is present" because
   * that is what a reader actually sees — a visible collapse at the right place.
   */
  describe.each(RANGES)('at $label', ({ span }) => {
    const window = DATA.slice(Math.max(DATA.length - span, 0));
    const sampled = decimate(window, 2000);

    it.each(CRASHES)('preserves $name when it is in range', ({ index, drop }) => {
      const position = index - (DATA.length - window.length);
      if (position < 0) return; // the crash predates this zoom level

      const at = DATA[index]!;
      const around = 30 * DAY;
      const nearby = sampled.filter((p) => Math.abs(p.x - at.x) <= around);
      expect(nearby.length).toBeGreaterThan(0);

      const lowest = Math.min(...nearby.map((p) => p.y));
      // The collapse has to survive as a collapse, not as a smoothed shoulder.
      const undisturbed = at.y / (1 - drop);
      expect(lowest).toBeLessThanOrEqual(undisturbed * (1 - drop * 0.9));
    });
  });

  it('would fail if stride sampling were used instead', () => {
    // The control. Every-Nth-point is the one-line alternative, and it drops a
    // single-day event for any N > 1 — this asserts the test can tell.
    const stride = Math.ceil(DATA.length / 2000);
    const strided = DATA.filter((_, i) => i % stride === 0);
    const missed = CRASHES.filter(({ index }) => index % stride !== 0);
    expect(missed.length).toBeGreaterThan(0);
    expect(strided.some((p) => p.x === DATA[missed[0]!.index]!.x)).toBe(false);
  });
});
