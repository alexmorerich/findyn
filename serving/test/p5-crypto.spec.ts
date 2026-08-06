import { SELF, env } from 'cloudflare:test';
import { beforeEach, describe, expect, it } from 'vitest';
import { applyWriteBack, validatePayload } from '../src/admin/writeback';
import { listAssets } from '../src/api/assets';
import {
  CRYPTO_REGIMES,
  DISCLAIMER,
  EXPERIMENTAL_ASSETS,
  EXPERIMENTAL_DISCLAIMER,
  isExperimentalAsset,
} from '../src/domain';

/**
 * P5: FinCrypto's read surface.
 *
 * The engine itself is quarantined in the compute plane and ships disabled, so
 * what the serving plane owes is narrower and sharper: **a consumer must never
 * be able to read a crypto number without being told it is experimental.**
 *
 * That is asserted here on every route that can return one, including the 501 an
 * engine that has not run yet returns — a client polling for a state has to know
 * what it is waiting for before it arrives, not after.
 *
 * The read endpoints themselves needed no per-engine code, which was the point
 * of P1's registry-driven design and is asserted rather than assumed.
 */

async function reset() {
  await env.DB.batch([
    env.DB.prepare('DELETE FROM asset_state'),
    env.DB.prepare('DELETE FROM engine_output'),
  ]);
}

const CRYPTO_STATE = {
  asset: 'crypto',
  as_of: '2026-08-04',
  model_version: 'crypto-0.1.0',
  regime: 'winter',
  // The whole point of the engine, and therefore of this fixture.
  expected_return: null,
  risk_score: 27.8114,
  confidence: 0.48,
  signals: [
    {
      name: 'speculation_index',
      value: 0.0,
      direction: 1 as const,
      note: '0-100 from 3 of 3 terms',
    },
    {
      name: 'experimental',
      value: 1.0,
      direction: 0 as const,
      note: 'Research only. No expected return by design; confidence capped at 0.5.',
    },
  ],
  components: {
    regime_code: 0,
    speculation_index: 0.0,
    liquidity_beta: 0.656414,
    expected_return_is_deliberately_absent: 1,
    confidence_ceiling: 0.5,
  },
};

const PAYLOAD = {
  model_version: 'crypto-0.1.0',
  generated_at: '2026-08-05T03:00:00Z',
  as_of: '2026-08-04',
  asset_state: [CRYPTO_STATE],
  engine_output: [
    { asset: 'crypto', metric: 'price', as_of: '2026-08-03', value: 63500.0 },
    { asset: 'crypto', metric: 'price', as_of: '2026-08-04', value: 64017.73 },
    { asset: 'crypto', metric: 'speculation_index', as_of: '2026-08-04', value: 0.0 },
    { asset: 'crypto', metric: 'liquidity_beta', as_of: '2026-08-04', value: 0.656414 },
    {
      asset: 'crypto',
      metric: 'regime_code',
      as_of: '2026-08-04',
      value: 0,
      meta: { regime: 'winter' },
    },
  ],
};

beforeEach(reset);

describe('crypto vocabulary parity', () => {
  it('carries the three states, ordered by increasing speculation', () => {
    // Wire order: engine_output publishes the state as its index here, so
    // reordering this array silently relabels every row the engine has written.
    expect([...CRYPTO_REGIMES]).toEqual(['winter', 'normal', 'frenzy']);
  });

  it('marks crypto and only crypto as experimental', () => {
    expect([...EXPERIMENTAL_ASSETS]).toEqual(['crypto']);
    expect(isExperimentalAsset('crypto')).toBe(true);
    for (const asset of ['money', 'rates', 'equity', 'gold']) {
      expect(isExperimentalAsset(asset)).toBe(false);
    }
  });
});

describe('every crypto response is flagged experimental', () => {
  it('the state endpoint carries the flag and the extra disclaimer', async () => {
    await applyWriteBack(env, validatePayload(PAYLOAD));
    const res = await SELF.fetch('https://findyn.test/api/v1/assets/crypto/state');
    const body = await res.json<{
      experimental?: boolean;
      disclaimer: string;
      data: { expected_return: number | null; confidence: number };
    }>();

    expect(res.status).toBe(200);
    expect(body.experimental).toBe(true);
    // Appended, not substituted: both claims are true at once.
    expect(body.disclaimer).toContain(DISCLAIMER);
    expect(body.disclaimer).toContain(EXPERIMENTAL_DISCLAIMER);
  });

  it('the history endpoint carries it too', async () => {
    await applyWriteBack(env, validatePayload(PAYLOAD));
    const res = await SELF.fetch(
      'https://findyn.test/api/v1/assets/crypto/history?metric=price',
    );
    const body = await res.json<{ experimental?: boolean; disclaimer: string }>();

    expect(body.experimental).toBe(true);
    expect(body.disclaimer).toContain(EXPERIMENTAL_DISCLAIMER);
  });

  it('the 501 carries it before the engine has ever run', async () => {
    // The sharp case. A client polling for a first state must learn what it is
    // waiting for from the wait, not from the arrival.
    const res = await SELF.fetch('https://findyn.test/api/v1/assets/crypto/state');
    const body = await res.json<{ error: string; milestone: string; experimental?: boolean }>();

    expect(res.status).toBe(501);
    expect(body.milestone).toBe('P5');
    expect(body.experimental).toBe(true);
  });

  it('a production engine is not flagged, and its disclaimer is unchanged', async () => {
    const res = await SELF.fetch('https://findyn.test/api/v1/assets/gold/state');
    const body = await res.json<{ experimental?: boolean }>();
    // Absent rather than false: an older consumer that does not know the field
    // must not read its absence as a promise about a different engine.
    expect(body.experimental).toBeUndefined();

    const listed = await SELF.fetch('https://findyn.test/api/v1/assets');
    const assets = await listed.json<{ disclaimer: string }>();
    expect(assets.disclaimer).toBe(DISCLAIMER);
  });
});

describe('the /assets listing tags the row, not the envelope', () => {
  it('marks the crypto row and leaves the others alone', async () => {
    await applyWriteBack(env, validatePayload(PAYLOAD));
    const rows = await listAssets(env);

    // Per-row, because this array mixes experimental and production engines:
    // an envelope-level flag would have to describe all five at once and would
    // therefore describe none of them.
    expect(rows.find((r) => r.asset === 'crypto')?.experimental).toBe(true);
    expect(rows.find((r) => r.asset === 'gold')?.experimental).toBe(false);
  });

  it('tags crypto even before it has ever published', async () => {
    const rows = await listAssets(env);
    const crypto = rows.find((r) => r.asset === 'crypto');

    expect(crypto?.status).toBe('awaiting_first_run');
    expect(crypto?.experimental).toBe(true);
  });
});

describe('the state round-trips the things that make it experimental', () => {
  it('keeps expected_return null rather than coercing it to zero', async () => {
    // A zero would be a claim. The compute engine sets None deliberately and the
    // schema was built to carry it; a write-back that quietly turned it into 0.0
    // would put a number into the field the portfolio layer optimizes against.
    await applyWriteBack(env, validatePayload(PAYLOAD));
    const res = await SELF.fetch('https://findyn.test/api/v1/assets/crypto/state');
    const body = await res.json<{ data: { expected_return: number | null } }>();

    expect(body.data.expected_return).toBeNull();
  });

  it('keeps the confidence under the engine ceiling', async () => {
    await applyWriteBack(env, validatePayload(PAYLOAD));
    const res = await SELF.fetch('https://findyn.test/api/v1/assets/crypto/state');
    const body = await res.json<{ data: { confidence: number } }>();

    expect(body.data.confidence).toBeLessThanOrEqual(0.5);
  });

  it('serves the regime history the page charts', async () => {
    await applyWriteBack(env, validatePayload(PAYLOAD));
    const res = await SELF.fetch(
      'https://findyn.test/api/v1/assets/crypto/history?metric=regime_code',
    );
    const body = await res.json<{ data: { points: { value: number; meta: unknown }[] } }>();

    expect(body.data.points).toHaveLength(1);
    expect(body.data.points[0]?.value).toBe(0);
    expect(body.data.points[0]?.meta).toEqual({ regime: 'winter' });
  });
});

describe('/meta publishes which engines are experimental', () => {
  it('so a client need not hard-code the name', async () => {
    const res = await SELF.fetch('https://findyn.test/api/v1/meta');
    const body = await res.json<{
      data: { vocabulary: { experimental_assets: string[]; crypto_regimes: string[] } };
    }>();

    expect(body.data.vocabulary.experimental_assets).toEqual(['crypto']);
    expect(body.data.vocabulary.crypto_regimes).toEqual(['winter', 'normal', 'frenzy']);
  });
});
