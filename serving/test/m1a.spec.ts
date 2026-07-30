import { SELF, env } from 'cloudflare:test';
import { beforeEach, describe, expect, it } from 'vitest';
import { validatePayload, PayloadError, applyWriteBack } from '../src/admin/writeback';
import { listSeries } from '../src/api/series';
import { pitSnapshot, InvalidDateError } from '../src/api/pit';

/**
 * M1-A: canonical series storage, point-in-time reads, and the compute
 * write-back that fills them.
 */

const CPI = 'FRED:CPIAUCSL';
const CAPE = 'SHILLER:CAPE';

async function reset() {
  await env.DB.batch([
    env.DB.prepare('DELETE FROM macro_series'),
    env.DB.prepare('DELETE FROM series_metadata'),
    env.DB.prepare('DELETE FROM data_quality_report'),
    env.DB.prepare('DELETE FROM ingestion_log'),
  ]);
}

/**
 * A miniature point-in-time world:
 *   - Feb CPI was published 2025-03-12
 *   - Mar CPI was published 2025-04-10, then revised 2025-05-13
 *   - CAPE for Feb was published 2025-03-31
 * Standing on 2025-04-01, March CPI must be invisible.
 */
const FIXTURE = {
  metadata: [
    {
      series_id: CPI,
      provider: 'fred',
      title: 'Consumer Price Index',
      frequency: 'monthly',
      unit: 'index',
    },
    {
      series_id: CAPE,
      provider: 'shiller',
      title: 'Cyclically Adjusted P/E',
      frequency: 'monthly',
      unit: 'ratio',
    },
  ],
  observations: [
    { series_id: CPI, obs_date: '2025-02-01', release_date: '2025-03-12', revision_date: '2025-03-12', value: 319.1, source: 'fred' },
    { series_id: CPI, obs_date: '2025-03-01', release_date: '2025-04-10', revision_date: '2025-04-10', value: 319.6, source: 'fred' },
    { series_id: CPI, obs_date: '2025-03-01', release_date: '2025-04-10', revision_date: '2025-05-13', value: 319.8, source: 'fred' },
    { series_id: CAPE, obs_date: '2025-02-01', release_date: '2025-03-31', revision_date: '2025-03-31', value: 35.2, source: 'shiller' },
  ],
};

beforeEach(reset);

describe('write-back payload validation', () => {
  it('accepts a well-formed payload', () => {
    const payload = validatePayload(FIXTURE);
    expect(payload.metadata).toHaveLength(2);
    expect(payload.observations).toHaveLength(4);
  });

  it('rejects a release date that precedes the period it describes', () => {
    // Would license lookahead for every consumer downstream.
    expect(() =>
      validatePayload({
        observations: [
          { series_id: CPI, obs_date: '2025-03-01', release_date: '2025-02-01', value: 1, source: 'fred' },
        ],
      }),
    ).toThrow(PayloadError);
  });

  it.each([
    ['a non-date obs_date', { obs_date: 'March 2025' }],
    ['a missing series_id', { series_id: '' }],
    ['a non-numeric value', { value: 'high' }],
  ])('rejects %s', (_label, override) => {
    expect(() =>
      validatePayload({
        observations: [
          {
            series_id: CPI,
            obs_date: '2025-03-01',
            release_date: '2025-04-10',
            value: 1,
            source: 'fred',
            ...override,
          },
        ],
      }),
    ).toThrow(PayloadError);
  });

  it('rejects a non-object payload', () => {
    expect(() => validatePayload('nope')).toThrow(PayloadError);
  });
});

describe('write-back persistence', () => {
  it('stores metadata and observations', async () => {
    const written = await applyWriteBack(env, validatePayload(FIXTURE));
    expect(written.metadata).toBe(2);
    expect(written.observations).toBe(4);

    const series = await listSeries(env, new Date('2025-06-01T00:00:00Z'));
    expect(series.map((s) => s.series_id).sort()).toEqual([CPI, CAPE].sort());
  });

  it('is idempotent — a rerun does not duplicate rows', async () => {
    await applyWriteBack(env, validatePayload(FIXTURE));
    await applyWriteBack(env, validatePayload(FIXTURE));

    const row = await env.DB.prepare('SELECT COUNT(*) AS n FROM macro_series').first<{ n: number }>();
    expect(row?.n).toBe(4);
  });

  it('reports the newest observation and its freshness', async () => {
    await applyWriteBack(env, validatePayload(FIXTURE));
    const series = await listSeries(env, new Date('2025-06-01T00:00:00Z'));
    const cpi = series.find((s) => s.series_id === CPI);

    expect(cpi?.latest_obs_date).toBe('2025-03-01');
    // Newest vintage wins the tie on obs_date.
    expect(cpi?.latest_value).toBe(319.8);
    expect(cpi?.freshness_days).toBe(92);
    expect(cpi?.observations).toBe(3);
  });
});

describe('point-in-time snapshot', () => {
  beforeEach(async () => {
    await applyWriteBack(env, validatePayload(FIXTURE));
  });

  it('hides data that had not been published yet', async () => {
    const snapshot = await pitSnapshot(env, '2025-04-01');

    const cpi = snapshot.available.find((a) => a.series_id === CPI);
    expect(cpi?.obs_date).toBe('2025-02-01');
    expect(cpi?.value).toBe(319.1);

    // March CPI exists in the table today but was published on 2025-04-10.
    const withheld = snapshot.withheld.find((w) => w.series_id === CPI);
    expect(withheld?.obs_date).toBe('2025-03-01');
    expect(withheld?.published_days_later).toBe(9);
  });

  it('reveals the same period once its release date has passed', async () => {
    const snapshot = await pitSnapshot(env, '2025-04-15');
    const cpi = snapshot.available.find((a) => a.series_id === CPI);
    expect(cpi?.obs_date).toBe('2025-03-01');
    // The revision was not issued until May, so the first print is what stands.
    expect(cpi?.value).toBe(319.6);
  });

  it('prefers the newest vintage available at the cutoff', async () => {
    const snapshot = await pitSnapshot(env, '2025-06-01');
    const cpi = snapshot.available.find((a) => a.series_id === CPI);
    expect(cpi?.value).toBe(319.8);
  });

  it('reports staleness against the as-of date', async () => {
    const snapshot = await pitSnapshot(env, '2025-04-01');
    const cape = snapshot.available.find((a) => a.series_id === CAPE);
    expect(cape?.staleness_days).toBe(59); // 2025-02-01 -> 2025-04-01
  });

  it('returns nothing as available before any release', async () => {
    const snapshot = await pitSnapshot(env, '2025-01-01');
    expect(snapshot.available).toHaveLength(0);
    expect(snapshot.withheld.length).toBeGreaterThan(0);
  });

  it('rejects a malformed date', async () => {
    await expect(pitSnapshot(env, '01/04/2025')).rejects.toThrow(InvalidDateError);
    await expect(pitSnapshot(env, '2025-13-45')).rejects.toThrow(InvalidDateError);
  });
});

describe('public endpoints', () => {
  beforeEach(async () => {
    await applyWriteBack(env, validatePayload(FIXTURE));
  });

  it('serves the series catalogue', async () => {
    const res = await SELF.fetch('https://findyn.test/api/v1/series');
    expect(res.status).toBe(200);
    const body = (await res.json()) as { data: { count: number } };
    expect(body.data.count).toBe(2);
  });

  it('serves one series with its observations', async () => {
    const res = await SELF.fetch(`https://findyn.test/api/v1/series/${encodeURIComponent(CAPE)}`);
    expect(res.status).toBe(200);
    const body = (await res.json()) as {
      data: { metadata: { unit: string }; observations: unknown[] };
    };
    expect(body.data.metadata.unit).toBe('ratio');
    expect(body.data.observations).toHaveLength(1);
  });

  it('404s an unknown series', async () => {
    const res = await SELF.fetch('https://findyn.test/api/v1/series/NOPE%3AX');
    expect(res.status).toBe(404);
  });

  it('serves the PIT snapshot', async () => {
    const res = await SELF.fetch('https://findyn.test/api/v1/pit?as_of=2025-04-01');
    expect(res.status).toBe(200);
    const body = (await res.json()) as { data: { withheld: unknown[] } };
    expect(body.data.withheld.length).toBeGreaterThan(0);
  });

  it('requires as_of on the PIT endpoint', async () => {
    const res = await SELF.fetch('https://findyn.test/api/v1/pit');
    expect(res.status).toBe(400);
  });

  it('rejects a malformed as_of with 400, not 500', async () => {
    const res = await SELF.fetch('https://findyn.test/api/v1/pit?as_of=yesterday');
    expect(res.status).toBe(400);
  });

  it('exposes the vocabulary on /meta', async () => {
    const res = await SELF.fetch('https://findyn.test/api/v1/meta');
    const body = (await res.json()) as {
      data: { milestone: string; vocabulary: { regimes: string[] } };
    };
    expect(body.data.milestone).toBe('M1-A');
    expect(body.data.vocabulary.regimes).toContain('crisis');
  });

  it('sends CORS headers so a separately hosted dashboard can read it', async () => {
    const res = await SELF.fetch('https://findyn.test/api/v1/meta');
    expect(res.headers.get('access-control-allow-origin')).toBe('*');
  });
});
