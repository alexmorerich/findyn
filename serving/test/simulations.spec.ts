import { SELF, env } from 'cloudflare:test';
import { beforeEach, describe, expect, it } from 'vitest';
import { getSimulation, listSimulations, putSimulation } from '../src/admin/simulations';

/**
 * Monte Carlo path archives in R2 (FINDYN_V1_SPEC.md §11).
 *
 * Two properties, and the second is the one that would be easy to lose:
 *
 * **A run is immutable.** An archived simulation is evidence about what the
 * model said on a date. Evidence that can be edited afterwards is not evidence.
 *
 * **It never reaches the public API.** The archive exists for offline analysis;
 * `/api/v1/simulate` returns the run's parameters and points at `/forecast` for
 * the distribution. A public endpoint returning 10,000 paths per horizon is a
 * denial of service with extra steps.
 */

const ASSET = 'equity';
const AS_OF = '2026-08-01';
const VERSION = 'equity-1.1.0+cal.yahoo_gspc';

function body(overrides: Record<string, unknown> = {}): string {
  return JSON.stringify({
    asset: ASSET,
    as_of: AS_OF,
    model_version: VERSION,
    seed: 20260731,
    horizons: { tactical: { years: 0.5, paths: 2, terminal: [8.9, 9.1] } },
    ...overrides,
  });
}

async function clear() {
  const listed = await env.ARCHIVE.list({ prefix: 'simulations/' });
  await Promise.all(listed.objects.map((o) => env.ARCHIVE.delete(o.key)));
}

describe('simulation archive (§11)', () => {
  beforeEach(clear);

  it('stores and reads back one run', async () => {
    const result = await putSimulation(env, ASSET, AS_OF, VERSION, body());
    expect(result.created).toBe(true);

    const stored = JSON.parse(await getSimulation(env, ASSET, AS_OF, VERSION));
    expect(stored.horizons.tactical.terminal).toEqual([8.9, 9.1]);
  });

  it('treats an identical re-write as success', async () => {
    await putSimulation(env, ASSET, AS_OF, VERSION, body());
    expect((await putSimulation(env, ASSET, AS_OF, VERSION, body())).created).toBe(false);
  });

  it('refuses to revise a run', async () => {
    await putSimulation(env, ASSET, AS_OF, VERSION, body());
    await expect(putSimulation(env, ASSET, AS_OF, VERSION, body({ seed: 999 }))).rejects.toThrow(
      /not a document to revise/,
    );
  });

  it('keeps two models that simulated the same day', async () => {
    // A refit landing mid-day. Neither run gets to be "the" simulation for that
    // date without naming the model it came from.
    await putSimulation(env, ASSET, AS_OF, 'equity-1.0.0', body({ model_version: 'equity-1.0.0' }));
    await putSimulation(env, ASSET, AS_OF, VERSION, body());

    expect(await listSimulations(env, ASSET)).toEqual([
      { as_of: AS_OF, model_version: 'equity-1.0.0' },
      { as_of: AS_OF, model_version: VERSION },
    ]);
  });

  it('keeps assets and dates apart', async () => {
    await putSimulation(env, ASSET, '2026-07-31', VERSION, body({ as_of: '2026-07-31' }));
    await putSimulation(env, ASSET, AS_OF, VERSION, body());
    await putSimulation(env, 'gold', AS_OF, 'gold-1.0.0', body({ asset: 'gold' }));

    expect((await listSimulations(env, ASSET)).map((r) => r.as_of)).toEqual(['2026-07-31', AS_OF]);
    expect(await listSimulations(env, 'gold')).toHaveLength(1);
  });

  it('404s for a run that was never archived', async () => {
    await expect(getSimulation(env, ASSET, '1999-01-01', VERSION)).rejects.toThrow(/not found/);
  });

  it('rejects parts that would not be safe as keys', async () => {
    await expect(putSimulation(env, '../etc', AS_OF, VERSION, body())).rejects.toThrow(/invalid/);
    await expect(putSimulation(env, ASSET, '2026/08/01', VERSION, body())).rejects.toThrow(
      /invalid/,
    );
    await expect(putSimulation(env, ASSET, AS_OF, 'a/b', body())).rejects.toThrow(/invalid/);
  });
});

describe('the archive is private and the public API says where the data is', () => {
  beforeEach(clear);

  it('refuses an unsigned read', async () => {
    await putSimulation(env, ASSET, AS_OF, VERSION, body());
    const res = await SELF.fetch(
      `https://findyn.test/admin/v1/simulations/${ASSET}/${AS_OF}/${VERSION}`,
    );
    expect([401, 503]).toContain(res.status);
    expect(await res.text()).not.toContain('terminal');
  });

  it('never serves paths from /api/v1/simulate', async () => {
    await putSimulation(env, ASSET, AS_OF, VERSION, body());

    const res = await SELF.fetch('https://findyn.test/api/v1/simulate');
    expect(res.status).toBe(200);
    const payload = (await res.json()) as {
      data: { horizons: unknown[]; summary: Record<string, unknown>; note: string };
    };

    // Asserted on the shape rather than on the text: the note legitimately
    // mentions what is archived, and a substring check would pass or fail on
    // the prose. What must stay true is that nothing here grows with the number
    // of simulated paths.
    expect(payload.data.horizons.every((h) => typeof h === 'string')).toBe(true);
    expect(Object.values(payload.data.summary).every((v) => !Array.isArray(v))).toBe(true);
    expect(payload.data.note).toContain('/forecast');
  });
});
