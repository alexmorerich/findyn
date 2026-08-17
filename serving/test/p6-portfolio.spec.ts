import { SELF, env } from 'cloudflare:test';
import { beforeEach, describe, expect, it } from 'vitest';
import { PayloadError, applyWriteBack, validatePayload } from '../src/admin/writeback';
import { getPortfolio } from '../src/api/portfolio';

/**
 * P6: the portfolio layer's write-back door and its one read endpoint, plus the
 * serving half of the chaos test — a degraded allocation still serves 200 with
 * the flag set end-to-end.
 */

async function reset() {
  await env.DB.batch([env.DB.prepare('DELETE FROM portfolio_state')]);
}

const WEIGHTS = {
  method: 'exponential tilt',
  risk_free: 0.045,
  quantile_grid: [0.05, 0.25, 0.5, 0.75, 0.95],
  quantile_labels: ['q05', 'q25', 'q50', 'q75', 'q95'],
  factors: { risk_appetite: 55 },
  degraded_reason: [],
  assets: [
    {
      asset: 'equity',
      mean: 0.628,
      neutral: 0.6,
      delta: 0.028,
      degraded: false,
      reason: '',
      quantiles: { q05: 0.6, q25: 0.615, q50: 0.628, q75: 0.641, q95: 0.655 },
    },
    {
      asset: 'rates',
      mean: 0.284,
      neutral: 0.3,
      delta: -0.016,
      degraded: false,
      reason: '',
      quantiles: { q05: 0.258, q25: 0.272, q50: 0.284, q75: 0.296, q95: 0.309 },
    },
    {
      asset: 'gold',
      mean: 0.088,
      neutral: 0.1,
      delta: -0.012,
      degraded: false,
      reason: '',
      quantiles: { q05: 0.078, q25: 0.083, q50: 0.088, q75: 0.093, q95: 0.1 },
    },
  ],
  inputs: [
    {
      asset: 'equity',
      as_of: '2026-07-28',
      regime: 'normal_expansion',
      expected_return: 0.085,
      risk_score: 55,
      confidence: 0.6,
      stale: false,
      reason: '',
    },
  ],
};

const STATE = {
  profile: 'balanced',
  as_of: '2026-07-28',
  model_version: 'portfolio-1.0.0',
  weights: WEIGHTS,
  implication: 'Under this distribution the allocation leans modestly toward equities. Not investment advice.',
  degraded: false,
};

const PAYLOAD = {
  model_version: 'portfolio-1.0.0',
  generated_at: '2026-07-30T00:00:00Z',
  portfolio_state: [STATE],
};

beforeEach(reset);

describe('portfolio write-back validation', () => {
  it('accepts a well-formed allocation', () => {
    const payload = validatePayload(PAYLOAD);
    expect(payload.portfolio_state).toHaveLength(1);
    expect(payload.portfolio_state?.[0]?.degraded).toBe(false);
  });

  it('rejects a profile outside the vocabulary', () => {
    expect(() =>
      validatePayload({ portfolio_state: [{ ...STATE, profile: 'aggressive' }] }),
    ).toThrow(PayloadError);
  });

  it('rejects a weights blob that is not an object', () => {
    expect(() =>
      validatePayload({ portfolio_state: [{ ...STATE, weights: '[]' }] }),
    ).toThrow(/must be a JSON object/);
  });

  it('requires an implication string', () => {
    expect(() =>
      validatePayload({ portfolio_state: [{ ...STATE, implication: '' }] }),
    ).toThrow(PayloadError);
  });
});

describe('portfolio write-back application', () => {
  it('writes one row per profile', async () => {
    const result = await applyWriteBack(env, validatePayload(PAYLOAD));
    expect(result.portfolio_state).toBe(1);
  });

  it('is idempotent on (profile, as_of, model_version)', async () => {
    await applyWriteBack(env, validatePayload(PAYLOAD));
    await applyWriteBack(env, validatePayload(PAYLOAD));
    const row = await env.DB.prepare('SELECT COUNT(*) AS n FROM portfolio_state').first<{
      n: number;
    }>();
    expect(row?.n).toBe(1);
  });

  it('keeps a different model_version as a separate row', async () => {
    await applyWriteBack(env, validatePayload(PAYLOAD));
    await applyWriteBack(
      env,
      validatePayload({ portfolio_state: [{ ...STATE, model_version: 'portfolio-1.1.0' }] }),
    );
    const row = await env.DB.prepare('SELECT COUNT(*) AS n FROM portfolio_state').first<{
      n: number;
    }>();
    expect(row?.n).toBe(2);
  });
});

describe('portfolio reads', () => {
  it('returns the newest allocation with its blob parsed', async () => {
    await applyWriteBack(env, validatePayload(PAYLOAD));
    const state = await getPortfolio(env, 'balanced');
    expect(state?.risk_free).toBe(0.045);
    expect(state?.assets.map((a) => a.asset)).toEqual(['equity', 'rates', 'gold']);
    expect(state?.inputs[0]?.asset).toBe('equity');
    expect(state?.degraded).toBe(false);
  });
});

describe('GET /api/v1/portfolio', () => {
  it('defaults to the balanced profile and carries the disclaimer', async () => {
    await applyWriteBack(env, validatePayload(PAYLOAD));
    const res = await SELF.fetch('https://findyn.test/api/v1/portfolio');
    expect(res.status).toBe(200);
    const body = await res.json<{
      as_of: string;
      model_version: string;
      disclaimer: string;
      data: { profile: string; assets: { asset: string }[] };
    }>();
    expect(body.data.profile).toBe('balanced');
    expect(body.as_of).toBe('2026-07-28');
    expect(body.model_version).toBe('portfolio-1.0.0');
    expect(body.disclaimer).toContain('does not provide investment advice');
    expect(body.data.assets.map((a) => a.asset)).toEqual(['equity', 'rates', 'gold']);
  });

  it('honours ?profile=', async () => {
    await applyWriteBack(
      env,
      validatePayload({ portfolio_state: [{ ...STATE, profile: 'growth' }] }),
    );
    const res = await SELF.fetch('https://findyn.test/api/v1/portfolio?profile=growth');
    expect(res.status).toBe(200);
    const body = await res.json<{ data: { profile: string } }>();
    expect(body.data.profile).toBe('growth');
  });

  it('400s for a profile outside the vocabulary', async () => {
    const res = await SELF.fetch('https://findyn.test/api/v1/portfolio?profile=aggressive');
    expect(res.status).toBe(400);
    const body = await res.json<{ profiles: string[] }>();
    expect(body.profiles).toContain('balanced');
  });

  it('501s with the phase tag for a profile that has never run', async () => {
    const res = await SELF.fetch('https://findyn.test/api/v1/portfolio?profile=conservative');
    expect(res.status).toBe(501);
    const body = await res.json<{ milestone: string; error: string }>();
    expect(body.error).toBe('not_implemented');
    expect(body.milestone).toBe('P6');
  });
});

describe('chaos: a degraded allocation still serves, flagged end-to-end', () => {
  const DEGRADED = {
    ...STATE,
    degraded: true,
    weights: {
      ...WEIGHTS,
      degraded_reason: ['equity: no state published'],
      assets: [
        // equity's engine data was killed → pinned at neutral, point mass, flagged.
        {
          asset: 'equity',
          mean: 0.6,
          neutral: 0.6,
          delta: 0,
          degraded: true,
          reason: 'no state published',
          quantiles: { q05: 0.6, q25: 0.6, q50: 0.6, q75: 0.6, q95: 0.6 },
        },
        {
          asset: 'rates',
          mean: 0.3,
          neutral: 0.3,
          delta: 0,
          degraded: false,
          reason: '',
          quantiles: { q05: 0.28, q25: 0.29, q50: 0.3, q75: 0.31, q95: 0.32 },
        },
        {
          asset: 'gold',
          mean: 0.1,
          neutral: 0.1,
          delta: 0,
          degraded: false,
          reason: '',
          quantiles: { q05: 0.09, q25: 0.095, q50: 0.1, q75: 0.105, q95: 0.11 },
        },
      ],
    },
    implication:
      'Under this distribution the allocation sits at its strategic neutral mix. One or more ' +
      'engines did not report for this run. Not investment advice.',
  };

  it('returns 200 with degraded=true and the equity band pinned at neutral', async () => {
    await applyWriteBack(env, validatePayload({ portfolio_state: [DEGRADED] }));
    const res = await SELF.fetch('https://findyn.test/api/v1/portfolio');
    expect(res.status).toBe(200);
    const body = await res.json<{
      data: {
        degraded: boolean;
        degraded_reason: string[];
        assets: { asset: string; degraded: boolean; mean: number; neutral: number }[];
      };
    }>();
    expect(body.data.degraded).toBe(true);
    expect(body.data.degraded_reason.length).toBeGreaterThan(0);
    const equity = body.data.assets.find((a) => a.asset === 'equity');
    expect(equity?.degraded).toBe(true);
    expect(equity?.mean).toBe(equity?.neutral);
  });
});
