import { env } from 'cloudflare:test';
import { beforeEach, describe, expect, it } from 'vitest';
import { applyWriteBack, validatePayload } from '../src/admin/writeback';
import { MAX_HISTORY_ROWS, getHistory, maxHistoryRowsPerDate } from '../src/api/assets';
import { INSTABILITY_METRICS, getInstability, getRegimeHistory } from '../src/api/equity';
import { getObservations, getSeriesMetadata } from '../src/api/series';
import { REGIMES } from '../src/domain';

/**
 * The read surface at the length the equity engine actually publishes.
 *
 * FinEquity's filter used to run on `FRED:SP500` alone, which FRED licences as a
 * rolling ten years; the publication path is now that record spliced behind the
 * daily S&P back to 1927-12-30, so every per-date series it writes is roughly
 * 24,700 rows rather than 2,500.
 *
 * That is a read-side change as much as a compute one, and in one specific way:
 * three of these endpoints store **several rows per date** and pivot them. A
 * ceiling expressed in rows is therefore a much lower ceiling in dates, and the
 * two look identical from the outside — the response is well-formed, `available`
 * is a real number, and the series simply stops in the 1970s. These tests pin
 * the ceiling to dates.
 */

/** Sessions in the daily S&P record, 1927-12-30 to 2026-07-30. */
const RECORD_LENGTH = 24_761;

/** The ingested daily record the page falls back to when the engine is short. */
const RECORD_SERIES = 'YAHOO:^GSPC';

async function reset() {
  await env.DB.batch([
    env.DB.prepare('DELETE FROM engine_output'),
    env.DB.prepare('DELETE FROM regime_state'),
    env.DB.prepare('DELETE FROM macro_series'),
    env.DB.prepare('DELETE FROM series_metadata'),
  ]);
}

/** `n` consecutive trading-ish days ending today, oldest first. */
function days(n: number, end = '2026-07-30'): string[] {
  const last = Date.parse(`${end}T00:00:00Z`);
  return Array.from({ length: n }, (_, i) =>
    new Date(last - (n - 1 - i) * 86_400_000).toISOString().slice(0, 10),
  );
}

describe('reading a century (§13)', () => {
  beforeEach(reset);

  describe('the row ceiling is a ceiling on dates', () => {
    it('scales with the rows each date occupies', () => {
      expect(maxHistoryRowsPerDate(1)).toBe(MAX_HISTORY_ROWS);
      expect(maxHistoryRowsPerDate(REGIMES.length)).toBe(MAX_HISTORY_ROWS * REGIMES.length);
    });

    it('leaves the whole daily record expressible on every pivoting endpoint', () => {
      // The regression this exists for: at a flat 60,000 rows the posterior is
      // five to a date, so `/regime` could serve 12,000 of the 24,761 sessions
      // and would have reported the S&P ending in the 1970s.
      for (const rowsPerDate of [1, REGIMES.length, INSTABILITY_METRICS.length]) {
        const dates = maxHistoryRowsPerDate(rowsPerDate) / rowsPerDate;
        expect(dates).toBeGreaterThan(RECORD_LENGTH);
      }
      expect(MAX_HISTORY_ROWS * REGIMES.length).toBeGreaterThan(RECORD_LENGTH * REGIMES.length);
    });

    it('never returns a ceiling below one row per date', () => {
      // Defensive: a caller passing 0 or a fraction must not disable the read.
      expect(maxHistoryRowsPerDate(0)).toBe(MAX_HISTORY_ROWS);
      expect(maxHistoryRowsPerDate(0.5)).toBe(MAX_HISTORY_ROWS);
    });
  });

  describe('velocity over the full range', () => {
    it('serves the oldest dates rather than a recent prefix, and decimates on request', async () => {
      // Small enough to write quickly, shaped like the real thing: a long quiet
      // stretch, one single-day collapse, and a request for far fewer points
      // than there are rows.
      const dates = days(1200);
      const crash = 700;
      await applyWriteBack(
        env,
        validatePayload({
          engine_output: dates.map((as_of, i) => ({
            asset: 'equity',
            metric: 'velocity',
            as_of,
            value: i === crash ? -1.4 : 0.08 + (i % 7) * 0.001,
          })),
        }),
      );

      const history = await getHistory(env, 'equity', 'velocity', { points: 200 });

      expect(history.available).toBe(1200);
      expect(history.truncated).toBe(false);
      expect(history.decimated).toEqual({ from: 1200, to: history.points.length, method: 'lttb' });
      // The span is the whole window, not its tail: a chart that quietly starts
      // later than the data does is the same defect as a truncated response.
      expect(history.points[0]!.as_of).toBe(dates[0]);
      expect(history.points.at(-1)!.as_of).toBe(dates.at(-1));
      // And the one day that matters survives the downsampling.
      expect(Math.min(...history.points.map((p) => p.value))).toBe(-1.4);
    });

    it('windows on `from` so a zoom level costs what it draws', async () => {
      const dates = days(400);
      await applyWriteBack(
        env,
        validatePayload({
          engine_output: dates.map((as_of, i) => ({
            asset: 'equity',
            metric: 'velocity',
            as_of,
            value: i / 1000,
          })),
        }),
      );

      const windowed = await getHistory(env, 'equity', 'velocity', { from: dates[300] });
      expect(windowed.available).toBe(100);
      expect(windowed.points[0]!.as_of).toBe(dates[300]);
    });
  });

  describe('the coverage probe', () => {
    /**
     * `/equity` compares what the engine has published against what has been
     * ingested, so it can fall back to the observation store rather than drawing
     * ten years as though that were the whole record. It asks both endpoints
     * with `limit=1` — a few hundred bytes each — and needs two things from that
     * response: the **true total**, not the truncated one, and the **oldest**
     * row rather than the newest.
     *
     * Both are easy to break without noticing, because a probe that returns
     * `available: 1` still renders a chart; it just renders the wrong one.
     */
    it('reports the true total and the oldest row from a one-row request', async () => {
      const dates = days(500);
      await applyWriteBack(
        env,
        validatePayload({
          engine_output: dates.map((as_of, i) => ({
            asset: 'equity',
            metric: 'price_close',
            as_of,
            value: 1000 + i,
          })),
        }),
      );

      const probe = await getHistory(env, 'equity', 'price_close', { limit: 1 });

      // `available` counts the window, not the response.
      expect(probe.available).toBe(500);
      expect(probe.points).toHaveLength(1);
      // Oldest, because that is the question being asked — where does the
      // published record *start*. The newest date is on every response already.
      expect(probe.points[0]!.as_of).toBe(dates[0]);
      // And it says it was clipped, so the probe is never mistaken for a series.
      expect(probe.truncated).toBe(true);
    });

    it('costs one row whether the engine has published a decade or a century', async () => {
      await applyWriteBack(
        env,
        validatePayload({
          engine_output: days(3000).map((as_of, i) => ({
            asset: 'equity',
            metric: 'price_close',
            as_of,
            value: 1000 + i,
          })),
        }),
      );

      const probe = await getHistory(env, 'equity', 'price_close', { limit: 1 });
      expect(probe.available).toBe(3000);
      expect(probe.points).toHaveLength(1);
    });

    it('answers with zero rather than failing when the engine has published nothing', async () => {
      // The state the page has to survive: a fresh database, where the correct
      // reading is "no published record", not "the market starts today".
      const probe = await getHistory(env, 'equity', 'price_close', { limit: 1 });
      expect(probe.available).toBe(0);
      expect(probe.points).toEqual([]);
    });

    it('reads the ingested record the same way, for the other side of the comparison', async () => {
      const dates = days(400);
      await applyWriteBack(
        env,
        validatePayload({
          metadata: [
            {
              series_id: RECORD_SERIES,
              provider: 'yahoo',
              title: 'S&P 500 Index (daily close)',
              frequency: 'daily',
              unit: 'index',
            },
          ],
          observations: dates.map((obs_date, i) => ({
            series_id: RECORD_SERIES,
            obs_date,
            release_date: obs_date,
            revision_date: obs_date,
            value: 100 + i,
            source: 'yahoo',
          })),
        }),
      );

      const probe = await getObservations(env, RECORD_SERIES, { limit: 1 });
      expect(probe.available).toBe(400);
      expect(probe.observations).toHaveLength(1);
      expect(probe.observations[0]!.obs_date).toBe(dates[0]);

      // The page prefers the metadata's own claim and falls back to the row, so
      // both have to be reachable from this one request.
      const metadata = await getSeriesMetadata(env, RECORD_SERIES);
      expect(metadata).not.toBeNull();
      expect(metadata!.series_id).toBe(RECORD_SERIES);
    });
  });

  describe('the posterior over the full range', () => {
    it('pivots every date it is given and reports no truncation', async () => {
      const dates = days(300);
      await applyWriteBack(
        env,
        validatePayload({
          regime_state: dates.flatMap((as_of) =>
            REGIMES.map((regime, rank) => ({
              asset: 'equity',
              as_of,
              regime,
              probability: rank === 0 ? 0.6 : 0.1,
              model_version: 'equity-1.2.0+pub.fred_sp500_yahoo_gspc+cal.yahoo_gspc',
            })),
          ),
        }),
      );

      const history = await getRegimeHistory(env, 'equity', { points: 50 });

      expect(history.available).toBe(300);
      expect(history.truncated).toBe(false);
      expect(history.points[0]!.as_of).toBe(dates[0]);
      // Decimation drops dates, never regimes: a date missing one of the five
      // would be a posterior that does not sum to one.
      for (const point of history.points) {
        expect(Object.keys(point.probabilities).sort()).toEqual([...REGIMES].sort());
      }
    });
  });

  describe('the instability factors over the full range', () => {
    it('keeps all five metrics on every date it serves', async () => {
      const dates = days(200);
      await applyWriteBack(
        env,
        validatePayload({
          engine_output: dates.flatMap((as_of) =>
            INSTABILITY_METRICS.map((metric, i) => ({
              asset: 'equity',
              metric,
              as_of,
              value: 0.1 * (i + 1),
            })),
          ),
        }),
      );

      const history = await getInstability(env, 'equity');
      expect(history.available).toBe(200);
      expect(history.points[0]!.as_of).toBe(dates[0]);
      for (const point of history.points) {
        for (const metric of INSTABILITY_METRICS) {
          expect(point[metric]).not.toBeNull();
        }
      }
    });
  });
});
