import { SELF, env } from 'cloudflare:test';
import { beforeEach, describe, expect, it } from 'vitest';
import { PayloadError, applyWriteBack, validatePayload } from '../src/admin/writeback';
import {
  getAssetState,
  getHistory,
  isAssetStale,
  listAssets,
  listMetrics,
} from '../src/api/assets';
import { DISCOUNT_HORIZONS, MONEY_REGIMES } from '../src/domain';

/**
 * P2: FinMoney's write-back and reads.
 *
 * Two things here are load-bearing beyond "money round-trips". First, the read
 * endpoints needed no change to serve a second engine — that was the point of
 * P1's registry-driven design, and it is asserted rather than assumed. Second,
 * `written_at` is now part of the history payload, because the compute plane
 * reads these rows back as another engine's *input* and point-in-time
 * correctness there depends on knowing when a row was published.
 */

async function reset() {
  await env.DB.batch([
    env.DB.prepare('DELETE FROM asset_state'),
    env.DB.prepare('DELETE FROM engine_output'),
    env.DB.prepare('DELETE FROM force_scores'),
  ]);
}

const MONEY_STATE = {
  asset: 'money',
  as_of: '2026-07-28',
  model_version: 'money-1.0.0',
  regime: 'tightening',
  expected_return: 0.0431,
  risk_score: 0.0,
  confidence: 0.8,
  signals: [
    {
      name: 'real_carry',
      value: 0.0447,
      direction: 1 as const,
      note: 'trailing 12m realized carry, annualized decimal',
    },
    { name: 'bill_sofr_spread', value: -0.24, direction: -1 as const, note: 'bill minus overnight' },
  ],
  components: {
    short_rate_pct: 4.31,
    wealth_index: 1.4212,
    carry_1m: 0.0433,
    carry_3m: 0.044,
    carry_12m: 0.0447,
    discount_1y: 0.9578,
    discount_3y: 0.8792,
    discount_10y: 0.6431,
    bill_sofr_spread: -0.24,
  },
};

const PAYLOAD = {
  model_version: 'money-1.0.0',
  generated_at: '2026-07-30T03:00:00Z',
  as_of: '2026-07-28',
  asset_state: [MONEY_STATE],
  engine_output: [
    {
      asset: 'money',
      metric: 'wealth_index',
      as_of: '2026-07-27',
      value: 1.4211,
      meta: { base: '1954-01-04' },
    },
    {
      asset: 'money',
      metric: 'wealth_index',
      as_of: '2026-07-28',
      value: 1.4212,
      meta: { base: '1954-01-04' },
    },
    { asset: 'money', metric: 'carry_3m', as_of: '2026-07-28', value: 0.044 },
    {
      asset: 'money',
      metric: 'discount_10y',
      as_of: '2026-07-28',
      value: 0.6431,
      meta: { curve_source: 'ns' },
    },
    {
      asset: 'money',
      metric: 'liquidity_code',
      as_of: '2026-07-28',
      value: 2,
      meta: { liquidity: 'tightening' },
    },
  ],
};

/** ISO date ``n`` days before now, for assertions about *recency*. */
function daysAgo(n: number): string {
  return new Date(Date.now() - n * 86_400_000).toISOString().slice(0, 10);
}

/**
 * ``PAYLOAD`` re-dated so its state and outputs land ``n`` days ago.
 *
 * The staleness endpoints measure against the wall clock, so a fixture with a
 * literal date tests recency only until that date is five days old. Everything
 * else in this file asserts on shape and value and is right to use the fixed
 * dates; only the staleness block needs the clock, and it uses this.
 */
function recentPayload(n: number) {
  const asOf = daysAgo(n);
  return {
    ...PAYLOAD,
    as_of: asOf,
    asset_state: PAYLOAD.asset_state.map((state) => ({ ...state, as_of: asOf })),
    engine_output: PAYLOAD.engine_output.map((row) => ({ ...row, as_of: asOf })),
  };
}

beforeEach(reset);

describe('money vocabulary parity', () => {
  it('carries the four liquidity states, ordered by tightness', () => {
    // The order is the wire order: engine_output publishes the state as its index
    // here, so reordering this array silently relabels every published row.
    expect([...MONEY_REGIMES]).toEqual(['abundant', 'normal', 'tightening', 'stressed']);
  });

  it('carries the standard discount horizons in ascending maturity', () => {
    expect([...DISCOUNT_HORIZONS]).toEqual([
      '1m',
      '3m',
      '6m',
      '1y',
      '2y',
      '3y',
      '5y',
      '7y',
      '10y',
      '20y',
      '30y',
    ]);
  });

  it('exposes the vocabulary on /meta so a consumer can discover it', async () => {
    const res = await SELF.fetch('https://findyn.test/api/v1/meta');
    const body = await res.json<{ data: { vocabulary: Record<string, string[]> } }>();
    expect(body.data.vocabulary.money_regimes).toEqual([...MONEY_REGIMES]);
    expect(body.data.vocabulary.discount_horizons).toEqual([...DISCOUNT_HORIZONS]);
  });
});

describe('money write-back', () => {
  it('accepts a money payload with no code change to the door', async () => {
    const payload = validatePayload(PAYLOAD);
    expect(payload.asset_state).toHaveLength(1);
    expect(payload.engine_output).toHaveLength(5);

    const result = await applyWriteBack(env, payload);
    expect(result.asset_state).toBe(1);
    expect(result.engine_output).toBe(5);
  });

  it('accepts a zero risk score, because cash really is riskless', () => {
    // A validator that treated 0 as missing would reject the numeraire's most
    // characteristic output.
    const payload = validatePayload({ asset_state: [{ ...MONEY_STATE, risk_score: 0 }] });
    expect(payload.asset_state?.[0]?.risk_score).toBe(0);
  });

  it('still rejects a nonsensical money state', () => {
    expect(() =>
      validatePayload({ asset_state: [{ ...MONEY_STATE, confidence: 4 }] }),
    ).toThrow(PayloadError);
    expect(() =>
      validatePayload({
        asset_state: [{ ...MONEY_STATE, signals: [{ name: 'x', value: 1, direction: 7 }] }],
      }),
    ).toThrow(PayloadError);
  });

  it('is idempotent, so a re-run does not duplicate the history', async () => {
    await applyWriteBack(env, validatePayload(PAYLOAD));
    await applyWriteBack(env, validatePayload(PAYLOAD));
    const row = await env.DB.prepare(
      `SELECT COUNT(*) AS n FROM engine_output WHERE asset='money'`,
    ).first<{ n: number }>();
    expect(row?.n).toBe(5);
  });
});

describe('money reads need no new endpoint', () => {
  it('appears in /assets alongside rates, from the same query', async () => {
    await applyWriteBack(env, validatePayload(PAYLOAD));
    const assets = await listAssets(env, new Date('2026-07-29T00:00:00Z'));

    const money = assets.find((a) => a.asset === 'money');
    expect(money?.status).toBe('live');
    expect(money?.regime).toBe('tightening');
    expect(money?.risk_score).toBe(0);
    expect(money?.stale).toBe(false);
  });

  it('serves the state with its signals and full component trace', async () => {
    await applyWriteBack(env, validatePayload(PAYLOAD));
    const state = await getAssetState(env, 'money');

    expect(state?.model_version).toBe('money-1.0.0');
    expect(state?.expected_return).toBe(0.0431);
    expect(state?.signals.map((s) => s.name)).toEqual(['real_carry', 'bill_sofr_spread']);
    expect(state?.components?.discount_10y).toBe(0.6431);
  });

  it('lists the metrics money published', async () => {
    await applyWriteBack(env, validatePayload(PAYLOAD));
    expect(await listMetrics(env, 'money')).toEqual([
      'carry_3m',
      'discount_10y',
      'liquidity_code',
      'wealth_index',
    ]);
  });

  it('serves the wealth index oldest-first with its base date', async () => {
    await applyWriteBack(env, validatePayload(PAYLOAD));
    const points = (await getHistory(env, 'money', 'wealth_index')).points;

    expect(points.map((p) => p.as_of)).toEqual(['2026-07-27', '2026-07-28']);
    // A wealth index without its base date is not interpretable.
    expect(points[0]?.meta).toEqual({ base: '1954-01-04' });
  });

  it('carries the liquidity label beside its code', async () => {
    await applyWriteBack(env, validatePayload(PAYLOAD));
    const points = (await getHistory(env, 'money', 'liquidity_code')).points;
    expect(points[0]?.meta).toEqual({ liquidity: 'tightening' });
    expect(points[0]?.value).toBe(MONEY_REGIMES.indexOf('tightening'));
  });

  it('records which curve each discount factor came from', async () => {
    await applyWriteBack(env, validatePayload(PAYLOAD));
    const points = (await getHistory(env, 'money', 'discount_10y')).points;
    expect(points[0]?.meta).toEqual({ curve_source: 'ns' });
  });

  it('GET /assets/money/state answers over HTTP', async () => {
    await applyWriteBack(env, validatePayload(PAYLOAD));
    const res = await SELF.fetch('https://findyn.test/api/v1/assets/money/state');
    expect(res.status).toBe(200);

    const body = await res.json<{ data: { regime: string }; model_version: string }>();
    expect(body.data.regime).toBe('tightening');
    expect(body.model_version).toBe('money-1.0.0');
  });

  it('GET /assets/money/state is 501 with the phase tag before the first run', async () => {
    // The header ribbon relies on telling "not run yet" from "broken".
    const res = await SELF.fetch('https://findyn.test/api/v1/assets/money/state');
    expect(res.status).toBe(501);
    const body = await res.json<{ milestone: string }>();
    expect(body.milestone).toBe('P2');
  });
});

describe('staleness is measured in market days, not ingestion hours', () => {
  /**
   * An `AssetState.as_of` is a market date: it is a day old the moment it is
   * published, and three days old every Monday. The 36-hour ingestion rule
   * therefore flagged every healthy run as stale — including the one that had
   * just finished — and made `/assets` and `/assets/:asset/state` give opposite
   * answers about the same row. The header ribbon hides itself when the engine
   * is stale, so under the old rule it would essentially never have appeared.
   */
  it('a state published yesterday is not stale', async () => {
    // Dated relative to the run, not to a literal. The endpoint compares against
    // `new Date()` inside the Worker, so a hard-coded as_of makes this assertion
    // true only for the five days after it was written — and this test did in
    // fact start failing on 2026-08-03 for that reason and nothing else. What it
    // means is "yesterday", so it says yesterday.
    await applyWriteBack(env, validatePayload(recentPayload(1)));
    const res = await SELF.fetch('https://findyn.test/api/v1/assets/money/state');
    const body = await res.json<{ stale: boolean; as_of: string }>();

    expect(body.as_of).toBe(daysAgo(1));
    expect(body.stale).toBe(false);
  });

  it('the two endpoints agree about the same row', async () => {
    await applyWriteBack(env, validatePayload(PAYLOAD));

    const listed = (await listAssets(env)).find((a) => a.asset === 'money');
    const res = await SELF.fetch('https://findyn.test/api/v1/assets/money/state');
    const state = await res.json<{ stale: boolean }>();

    expect(state.stale).toBe(listed?.stale);
  });

  it('an engine that really has stopped reporting is still flagged', () => {
    expect(isAssetStale('2026-07-28', new Date('2026-07-30T12:00:00Z'))).toBe(false);
    expect(isAssetStale('2026-07-28', new Date('2026-08-10T12:00:00Z'))).toBe(true);
    expect(isAssetStale(null)).toBe(true);
  });

  it('history is judged the same way', async () => {
    await applyWriteBack(env, validatePayload(recentPayload(1)));
    const res = await SELF.fetch(
      'https://findyn.test/api/v1/assets/money/history?metric=wealth_index',
    );
    const body = await res.json<{ stale: boolean }>();
    expect(body.stale).toBe(false);
  });
});

describe('written_at is exposed as the row vintage', () => {
  it('history points carry the publication timestamp', async () => {
    await applyWriteBack(env, validatePayload(PAYLOAD));
    const points = (await getHistory(env, 'money', 'wealth_index')).points;

    for (const point of points) {
      expect(point.written_at).toBeTruthy();
      expect(Number.isNaN(Date.parse(point.written_at!))).toBe(false);
    }
  });

  it('written_at is the run date, not the date the value describes', async () => {
    // The distinction the compute plane's read-back depends on: a run publishing
    // a five-year window stamps every row with today, and a consumer standing in
    // the past must not be able to see any of them.
    await applyWriteBack(env, validatePayload(PAYLOAD));
    const points = (await getHistory(env, 'money', 'wealth_index')).points;

    const early = points.find((p) => p.as_of === '2026-07-27');
    expect(early?.written_at! > early!.as_of).toBe(true);
  });

  it('is surfaced over HTTP too', async () => {
    await applyWriteBack(env, validatePayload(PAYLOAD));
    const res = await SELF.fetch(
      'https://findyn.test/api/v1/assets/money/history?metric=wealth_index',
    );
    const body = await res.json<{ data: { points: { written_at: string | null }[] } }>();
    expect(body.data.points.every((p) => p.written_at !== null)).toBe(true);
  });
});
