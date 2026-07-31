import { SELF } from 'cloudflare:test';
import { describe, expect, it } from 'vitest';

describe('public API (FINDYN_V1_SPEC.md §13)', () => {
  it('serves /api/v1/health with the standard envelope', async () => {
    const res = await SELF.fetch('https://findyn.test/api/v1/health');
    expect(res.status).toBe(200);

    const body = (await res.json()) as Record<string, unknown>;
    expect(body).toHaveProperty('as_of');
    expect(body).toHaveProperty('model_version');
    expect(body).toHaveProperty('stale');
    // §18 — the disclaimer travels with every response.
    expect(body.disclaimer).toContain('does not provide investment advice');

    const data = body.data as Record<string, unknown>;
    expect(data.info_set).toBe('t-1');
    expect(Array.isArray(data.sources)).toBe(true);
  });

  it('flags an empty database as stale rather than failing', async () => {
    const res = await SELF.fetch('https://findyn.test/api/v1/health');
    const body = (await res.json()) as { stale: boolean; data: { ok: boolean } };
    expect(body.stale).toBe(true);
    expect(body.data.ok).toBe(true);
  });

  // Nothing is reserved any more. /state and /forces went live in P3-A,
  // /regime in P3-B, and /instability, /forecast and /simulate in P3-C — every
  // endpoint FINDYN_V1_SPEC.md §13 names now answers.
  it.each(['state', 'forces', 'regime', 'instability', 'forecast', 'simulate'])(
    'serves /api/v1/%s on an empty database rather than erroring',
    async (route) => {
      // The endpoints are live from P3-A, but a fresh database has no features
      // and no scores. That must read as "nothing yet", not as a 500 — the
      // dashboard renders an awaiting state off exactly this shape.
      const res = await SELF.fetch(`https://findyn.test/api/v1/${route}`);
      expect(res.status).toBe(200);
      const body = (await res.json()) as { as_of: string | null; stale: boolean };
      expect(body.as_of).toBeNull();
      expect(body.stale).toBe(true);
    },
  );

  it('returns 404 for unknown paths', async () => {
    const res = await SELF.fetch('https://findyn.test/api/v1/nope');
    expect(res.status).toBe(404);
  });

  it('rejects unsigned admin write-back', async () => {
    const res = await SELF.fetch('https://findyn.test/admin/v1/results', {
      method: 'POST',
      body: '{}',
    });
    // 401 when the secret is configured, 503 when it is not — never 200.
    expect([401, 503]).toContain(res.status);
  });
});
