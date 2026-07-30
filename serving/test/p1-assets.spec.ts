import { SELF, env } from 'cloudflare:test';
import { beforeEach, describe, expect, it } from 'vitest';
import { PayloadError, applyWriteBack, validatePayload } from '../src/admin/writeback';
import { getAssetState, getHistory, listAssets, listMetrics } from '../src/api/assets';
import { ASSETS } from '../src/domain';

/**
 * P1: the multi-asset output tables, their write-back door, and the read
 * endpoints the dashboard's Engines panel and /rates page are built on.
 */

async function reset() {
  await env.DB.batch([
    env.DB.prepare('DELETE FROM asset_state'),
    env.DB.prepare('DELETE FROM engine_output'),
    env.DB.prepare('DELETE FROM force_scores'),
  ]);
}

const STATE = {
  asset: 'rates',
  as_of: '2026-07-28',
  model_version: 'rates-1.0.0',
  regime: 're_steepening',
  expected_return: 0.0504,
  risk_score: 38.16,
  confidence: 0.65,
  signals: [
    { name: 'curve_inversion', value: 0.838, direction: 1 as const, note: 'fitted 10y-3m spread' },
    { name: 'term_premium_trend', value: 0.725, direction: 1 as const, note: null },
  ],
  components: { ns_level: 5.13, ns_slope: 1.22, ns_curvature: -1.02 },
};

const PAYLOAD = {
  model_version: 'rates-1.0.0',
  generated_at: '2026-07-30T00:00:00Z',
  factors: [
    { force: 'real_rate', as_of: '2026-07-28', score: 41.2, components: { 'FRED:DGS10': 38.0 } },
  ],
  asset_state: [STATE],
  engine_output: [
    { asset: 'rates', metric: 'ns_level', as_of: '2026-07-27', value: 5.11 },
    { asset: 'rates', metric: 'ns_level', as_of: '2026-07-28', value: 5.13 },
    {
      asset: 'rates',
      metric: 'regime_code',
      as_of: '2026-07-28',
      value: 4,
      meta: { regime: 're_steepening' },
    },
  ],
};

beforeEach(reset);

describe('write-back validation (03-contracts.md §6)', () => {
  it('accepts a well-formed multi-asset payload', () => {
    const payload = validatePayload(PAYLOAD);
    expect(payload.asset_state).toHaveLength(1);
    expect(payload.engine_output).toHaveLength(3);
    expect(payload.factors).toHaveLength(1);
  });

  it('rejects an asset outside the vocabulary', () => {
    // The two planes disagreeing about what exists would create a row no
    // endpoint can serve, so it is refused rather than stored.
    expect(() =>
      validatePayload({ asset_state: [{ ...STATE, asset: 'commodities' }] }),
    ).toThrow(PayloadError);
  });

  it('rejects a factor outside the vocabulary', () => {
    expect(() => validatePayload({ factors: [{ force: 'vibes', as_of: '2026-07-28', score: 5 }] })).toThrow(
      PayloadError,
    );
  });

  it('rejects out-of-range scores', () => {
    expect(() => validatePayload({ asset_state: [{ ...STATE, risk_score: 140 }] })).toThrow(
      /risk_score must be within \[0, 100\]/,
    );
    expect(() => validatePayload({ asset_state: [{ ...STATE, confidence: 1.4 }] })).toThrow(
      /confidence must be within \[0, 1\]/,
    );
  });

  it('rejects non-finite metric values', () => {
    expect(() =>
      validatePayload({
        engine_output: [{ asset: 'rates', metric: 'ns_level', as_of: '2026-07-28', value: 'NaN' }],
      }),
    ).toThrow(/must be a finite number/);
  });

  it('rejects a signal direction that is not -1, 0 or 1', () => {
    expect(() =>
      validatePayload({
        asset_state: [{ ...STATE, signals: [{ name: 'x', value: 1, direction: 2 }] }],
      }),
    ).toThrow(/direction must be -1, 0 or 1/);
  });

  it('allows a null expected_return, which is meaningful for some engines', () => {
    const payload = validatePayload({ asset_state: [{ ...STATE, expected_return: null }] });
    expect(payload.asset_state?.[0]?.expected_return).toBeNull();
  });
});

describe('write-back application', () => {
  it('writes states, outputs and factors', async () => {
    const result = await applyWriteBack(env, validatePayload(PAYLOAD));
    expect(result.asset_state).toBe(1);
    expect(result.engine_output).toBe(3);
    expect(result.factors).toBe(1);
  });

  it('is idempotent — a re-run updates in place rather than duplicating', async () => {
    await applyWriteBack(env, validatePayload(PAYLOAD));
    await applyWriteBack(env, validatePayload(PAYLOAD));

    const states = await env.DB.prepare('SELECT COUNT(*) AS n FROM asset_state').first<{
      n: number;
    }>();
    const outputs = await env.DB.prepare('SELECT COUNT(*) AS n FROM engine_output').first<{
      n: number;
    }>();
    expect(states?.n).toBe(1);
    expect(outputs?.n).toBe(3);
  });

  it('overwrites the value on re-run, so a corrected figure lands', async () => {
    await applyWriteBack(env, validatePayload(PAYLOAD));
    await applyWriteBack(
      env,
      validatePayload({
        ...PAYLOAD,
        engine_output: [{ asset: 'rates', metric: 'ns_level', as_of: '2026-07-28', value: 9.99 }],
      }),
    );
    const row = await env.DB.prepare(
      `SELECT value FROM engine_output WHERE asset='rates' AND metric='ns_level' AND as_of='2026-07-28'`,
    ).first<{ value: number }>();
    expect(row?.value).toBe(9.99);
  });

  it('keeps a different model_version as a separate row', async () => {
    // A refit must not silently rewrite the state the previous model published,
    // or every backtest of the old model becomes a backtest of the new one.
    await applyWriteBack(env, validatePayload(PAYLOAD));
    await applyWriteBack(
      env,
      validatePayload({ asset_state: [{ ...STATE, model_version: 'rates-1.1.0', regime: 'flat' }] }),
    );
    const { results } = await env.DB.prepare(
      `SELECT model_version, regime FROM asset_state ORDER BY model_version`,
    ).all<{ model_version: string; regime: string }>();
    expect(results).toHaveLength(2);
  });
});

describe('asset reads', () => {
  it('lists every engine in the vocabulary, run or not', async () => {
    await applyWriteBack(env, validatePayload(PAYLOAD));
    const assets = await listAssets(env, new Date('2026-07-29T00:00:00Z'));

    expect(assets.map((a) => a.asset)).toEqual([...ASSETS]);

    const rates = assets.find((a) => a.asset === 'rates');
    expect(rates?.status).toBe('live');
    expect(rates?.regime).toBe('re_steepening');
    expect(rates?.stale).toBe(false);

    // This is the property the whole Engines panel design rests on: an engine
    // that has shipped but never run is reported, not omitted.
    const gold = assets.find((a) => a.asset === 'gold');
    expect(gold?.status).toBe('awaiting_first_run');
    expect(gold?.regime).toBeNull();
    expect(gold?.stale).toBe(true);
  });

  it('flags an engine that has stopped publishing as stale', async () => {
    await applyWriteBack(env, validatePayload(PAYLOAD));
    const assets = await listAssets(env, new Date('2026-09-01T00:00:00Z'));
    expect(assets.find((a) => a.asset === 'rates')?.stale).toBe(true);
  });

  it('returns the newest state with its signals parsed', async () => {
    await applyWriteBack(env, validatePayload(PAYLOAD));
    const state = await getAssetState(env, 'rates');
    expect(state?.regime).toBe('re_steepening');
    expect(state?.signals.map((s) => s.name)).toEqual(['curve_inversion', 'term_premium_trend']);
    expect(state?.components?.ns_level).toBe(5.13);
  });

  it('returns history oldest-first, with meta parsed', async () => {
    await applyWriteBack(env, validatePayload(PAYLOAD));
    const points = await getHistory(env, 'rates', 'ns_level');
    expect(points.map((p) => p.as_of)).toEqual(['2026-07-27', '2026-07-28']);

    const coded = await getHistory(env, 'rates', 'regime_code');
    expect(coded[0]?.meta).toEqual({ regime: 're_steepening' });
  });

  it('lists the metrics an engine has published', async () => {
    await applyWriteBack(env, validatePayload(PAYLOAD));
    expect(await listMetrics(env, 'rates')).toEqual(['ns_level', 'regime_code']);
  });
});

describe('asset endpoints', () => {
  it('GET /assets enumerates the registry', async () => {
    await applyWriteBack(env, validatePayload(PAYLOAD));
    const res = await SELF.fetch('https://findyn.test/api/v1/assets');
    expect(res.status).toBe(200);

    const body = await res.json<{ data: { count: number; assets: { asset: string }[] } }>();
    expect(body.data.count).toBe(ASSETS.length);
    expect(body.data.assets.map((a) => a.asset)).toEqual([...ASSETS]);
  });

  it('GET /assets/rates/state returns the envelope', async () => {
    await applyWriteBack(env, validatePayload(PAYLOAD));
    const res = await SELF.fetch('https://findyn.test/api/v1/assets/rates/state');
    expect(res.status).toBe(200);

    const body = await res.json<{
      as_of: string;
      model_version: string;
      disclaimer: string;
      data: { regime: string };
    }>();
    expect(body.as_of).toBe('2026-07-28');
    expect(body.model_version).toBe('rates-1.0.0');
    expect(body.data.regime).toBe('re_steepening');
    expect(body.disclaimer).toContain('does not provide investment advice');
  });

  it('501s with a phase tag for an engine that has never run', async () => {
    // "Registered but silent" must be distinguishable from "no such engine"
    // and from "ran and found nothing".
    const res = await SELF.fetch('https://findyn.test/api/v1/assets/gold/state');
    expect(res.status).toBe(501);
    const body = await res.json<{ milestone: string; error: string }>();
    expect(body.error).toBe('not_implemented');
    expect(body.milestone).toBe('P4');
  });

  it('404s for an asset outside the vocabulary', async () => {
    const res = await SELF.fetch('https://findyn.test/api/v1/assets/tulips/state');
    expect(res.status).toBe(404);
  });

  it('GET /assets/rates/history requires a metric and suggests the options', async () => {
    await applyWriteBack(env, validatePayload(PAYLOAD));
    const res = await SELF.fetch('https://findyn.test/api/v1/assets/rates/history');
    expect(res.status).toBe(400);
    const body = await res.json<{ available: string[] }>();
    expect(body.available).toEqual(['ns_level', 'regime_code']);
  });

  it('GET /assets/rates/history returns points ascending', async () => {
    await applyWriteBack(env, validatePayload(PAYLOAD));
    const res = await SELF.fetch(
      'https://findyn.test/api/v1/assets/rates/history?metric=ns_level',
    );
    expect(res.status).toBe(200);
    const body = await res.json<{ data: { points: { as_of: string; value: number }[] } }>();
    expect(body.data.points.map((p) => p.value)).toEqual([5.11, 5.13]);
  });

  it('honours the from/to range', async () => {
    await applyWriteBack(env, validatePayload(PAYLOAD));
    const res = await SELF.fetch(
      'https://findyn.test/api/v1/assets/rates/history?metric=ns_level&from=2026-07-28',
    );
    const body = await res.json<{ data: { count: number } }>();
    expect(body.data.count).toBe(1);
  });

  it('returns an empty series rather than an error for an unknown metric', async () => {
    await applyWriteBack(env, validatePayload(PAYLOAD));
    const res = await SELF.fetch(
      'https://findyn.test/api/v1/assets/rates/history?metric=not_a_metric',
    );
    expect(res.status).toBe(200);
    const body = await res.json<{ data: { count: number }; stale: boolean }>();
    expect(body.data.count).toBe(0);
    expect(body.stale).toBe(true);
  });
});
