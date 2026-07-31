import { SELF, env } from 'cloudflare:test';
import { beforeEach, describe, expect, it } from 'vitest';
import { PayloadError, applyWriteBack, validatePayload } from '../src/admin/writeback';
import { getForecast, getInstability } from '../src/api/equity';

/**
 * P3-C: the RII, the crash decomposition and the forecast distribution
 * (FINDYN_V1_SPEC.md §3.2, §4, §11, §13).
 *
 * What is actually load-bearing here is not that the numbers come back — it is
 * the two contracts the shapes enforce:
 *
 * **All three crash factors or none.** §4 forbids publishing a composite crash
 * risk without its decomposition, because "the regime may turn but the system is
 * robust" and "the system is fragile but the regime is stable" produce the same
 * product and call for opposite responses. `/instability` has no way to return
 * one factor alone.
 *
 * **Quantiles, never a target.** `forecast_distribution` has no column for a
 * point estimate, and `/forecast` carries `educational_only` on every band so a
 * 50-year illustration cannot be plotted as a 6-month forecast by accident.
 */

const VERSION = 'equity-1.0.0+cal.yahoo_gspc';

async function reset() {
  await env.DB.batch([
    env.DB.prepare('DELETE FROM engine_output'),
    env.DB.prepare('DELETE FROM forecast_distribution'),
  ]);
}

function metric(metric: string, value: number, as_of = '2026-07-31', asset = 'equity') {
  return { asset, as_of, metric, value, model_version: VERSION };
}

/** One complete published date: the RII and all three §4 factors. */
function fullDate(as_of: string, rii: number) {
  return [
    metric('rii', rii, as_of),
    metric('p_transition', 0.75, as_of),
    metric('p_shock', 0.22, as_of),
    metric('p_transmission', 0.1, as_of),
    metric('crash_risk', 1.65, as_of),
  ];
}

function band(horizon: string, quantile: number, value: number, educational_only = false) {
  return {
    asset: 'equity',
    as_of: '2026-07-31',
    horizon,
    quantile,
    value,
    educational_only,
    model_version: VERSION,
  };
}

describe('P3-C — instability (§3.2, §4)', () => {
  beforeEach(reset);

  it('pivots engine_output into one row per date with all three factors', async () => {
    await applyWriteBack(env, validatePayload({ engine_output: fullDate('2026-07-31', 76.03) }));

    const history = await getInstability(env);
    expect(history.count).toBe(1);
    const point = history.points[0]!;
    expect(point.rii).toBeCloseTo(76.03, 2);
    expect(point.p_transition).toBeCloseTo(0.75, 6);
    expect(point.p_shock).toBeCloseTo(0.22, 6);
    expect(point.p_transmission).toBeCloseTo(0.1, 6);
    expect(point.crash_risk).toBeCloseTo(1.65, 6);
  });

  it('reports a factor the run did not publish as null, not as zero', async () => {
    // Zero is a *measurement* on every one of these axes — it says "no chance of
    // a shock", "the system absorbs everything". An absent factor must never be
    // able to read as the safest possible one.
    await applyWriteBack(
      env,
      validatePayload({
        engine_output: [metric('rii', 52.0), metric('p_shock', 0.22)],
      }),
    );

    const point = (await getInstability(env)).points[0]!;
    expect(point.rii).toBeCloseTo(52.0, 6);
    expect(point.p_transition).toBeNull();
    expect(point.p_transmission).toBeNull();
  });

  it('windows by date and orders oldest first', async () => {
    await applyWriteBack(
      env,
      validatePayload({
        engine_output: [
          ...fullDate('2026-07-29', 50),
          ...fullDate('2026-07-30', 60),
          ...fullDate('2026-07-31', 70),
        ],
      }),
    );

    const all = await getInstability(env);
    expect(all.points.map((p) => p.as_of)).toEqual(['2026-07-29', '2026-07-30', '2026-07-31']);

    const windowed = await getInstability(env, 'equity', {
      from: '2026-07-30',
    });
    expect(windowed.points.map((p) => p.as_of)).toEqual(['2026-07-30', '2026-07-31']);
  });

  it('keeps assets apart', async () => {
    await applyWriteBack(
      env,
      validatePayload({
        engine_output: [metric('rii', 76.0), metric('rii', 12.0, '2026-07-31', 'gold')],
      }),
    );

    expect((await getInstability(env, 'equity')).points[0]!.rii).toBeCloseTo(76.0, 6);
    expect((await getInstability(env, 'gold')).points[0]!.rii).toBeCloseTo(12.0, 6);
  });

  it('decimates on the RII and says so, rather than sampling silently', async () => {
    const rows = Array.from({ length: 600 }, (_, i) => {
      const day = new Date(Date.UTC(2024, 0, 1 + i)).toISOString().slice(0, 10);
      // A single spike, which stride sampling would have a 1-in-3 chance of losing.
      return metric('rii', i === 300 ? 99 : 50 + (i % 5), day);
    });
    await applyWriteBack(env, validatePayload({ engine_output: rows }));

    const history = await getInstability(env, 'equity', { points: 100 });
    expect(history.available).toBe(600);
    expect(history.count).toBeLessThanOrEqual(100);
    expect(history.decimated).toEqual({
      from: 600,
      to: history.count,
      method: 'lttb',
    });
    // The extreme survives — that is the whole reason this is LTTB.
    expect(Math.max(...history.points.map((p) => p.rii ?? 0))).toBe(99);
  });

  it('serves an empty history rather than erroring on a fresh database', async () => {
    const history = await getInstability(env);
    expect(history.points).toEqual([]);
    expect(history.decimated).toBeNull();
  });

  it('answers /api/v1/instability over HTTP with all five fields', async () => {
    await applyWriteBack(env, validatePayload({ engine_output: fullDate('2026-07-31', 76.03) }));

    const res = await SELF.fetch('https://findyn.test/api/v1/instability');
    expect(res.status).toBe(200);
    const body = (await res.json()) as {
      as_of: string;
      data: { points: unknown[] };
    };
    expect(body.as_of).toBe('2026-07-31');
    expect(Object.keys(body.data.points[0] as object).sort()).toEqual([
      'as_of',
      'crash_risk',
      'p_shock',
      'p_transition',
      'p_transmission',
      'rii',
    ]);
  });
});

describe('P3-C — forecast distribution (§11, §13)', () => {
  beforeEach(reset);

  const BANDS = [
    band('tactical', 0.05, 8.4),
    band('tactical', 0.5, 8.72),
    band('tactical', 0.95, 9.02),
    band('educational_50y', 0.5, 12.1, true),
  ];

  it('validates and stores quantile bands', async () => {
    const result = await applyWriteBack(env, validatePayload({ forecast_distribution: BANDS }));
    expect(result.forecast_distribution).toBe(4);

    const forecast = await getForecast(env);
    expect(forecast.as_of).toBe('2026-07-31');
    expect(forecast.model_version).toBe(VERSION);
    expect(forecast.horizons).toHaveLength(2);
  });

  it('carries educational_only on every band', async () => {
    await applyWriteBack(env, validatePayload({ forecast_distribution: BANDS }));

    const forecast = await getForecast(env);
    const byHorizon = Object.fromEntries(forecast.horizons.map((h) => [h.horizon, h]));
    expect(byHorizon.tactical!.educational_only).toBe(false);
    // §10 excludes these from accuracy evaluation entirely. A consumer that
    // cannot tell them apart will eventually plot them on one axis.
    expect(byHorizon.educational_50y!.educational_only).toBe(true);
  });

  it('serves only the newest as_of, not every date ever published', async () => {
    await applyWriteBack(
      env,
      validatePayload({
        forecast_distribution: [
          { ...band('tactical', 0.5, 8.5), as_of: '2026-07-30' },
          band('tactical', 0.5, 8.72),
        ],
      }),
    );

    const forecast = await getForecast(env);
    expect(forecast.as_of).toBe('2026-07-31');
    expect(forecast.horizons[0]!.quantiles['0.5']).toBeCloseTo(8.72, 6);
  });

  it('filters to one horizon when asked', async () => {
    await applyWriteBack(env, validatePayload({ forecast_distribution: BANDS }));

    const forecast = await getForecast(env, 'equity', 'tactical');
    expect(forecast.horizons.map((h) => h.horizon)).toEqual(['tactical']);
  });

  it('rejects a horizon or quantile outside the published vocabulary', async () => {
    expect(() =>
      validatePayload({
        forecast_distribution: [{ ...band('tactical', 0.5, 8.7), horizon: 'someday' }],
      }),
    ).toThrow(PayloadError);
    expect(() => validatePayload({ forecast_distribution: [band('tactical', 1.5, 8.7)] })).toThrow(
      PayloadError,
    );
  });

  it('replaces a band on refit rather than accumulating duplicates', async () => {
    await applyWriteBack(
      env,
      validatePayload({ forecast_distribution: [band('tactical', 0.5, 8.72)] }),
    );
    await applyWriteBack(
      env,
      validatePayload({ forecast_distribution: [band('tactical', 0.5, 8.9)] }),
    );

    const forecast = await getForecast(env);
    expect(forecast.horizons[0]!.quantiles['0.5']).toBeCloseTo(8.9, 6);
  });

  it('keeps a second engine from overwriting equity bands', async () => {
    // The reason migration 0007 exists: `tactical` and `0.5` are not equity's
    // words, and under the old primary key gold's p50 landed on equity's row.
    await applyWriteBack(
      env,
      validatePayload({
        forecast_distribution: [
          band('tactical', 0.5, 8.72),
          { ...band('tactical', 0.5, 5.1), asset: 'gold' },
        ],
      }),
    );

    expect((await getForecast(env, 'equity')).horizons[0]!.quantiles['0.5']).toBeCloseTo(8.72, 6);
    expect((await getForecast(env, 'gold')).horizons[0]!.quantiles['0.5']).toBeCloseTo(5.1, 6);
  });

  it('rejects an unknown horizon at the HTTP edge with a 400, not a silent empty', async () => {
    const res = await SELF.fetch('https://findyn.test/api/v1/forecast?horizon=someday');
    expect(res.status).toBe(400);
    expect(((await res.json()) as { error: string }).error).toBe('bad_request');
  });
});
