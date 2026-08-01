import { SELF, env } from 'cloudflare:test';
import { beforeEach, describe, expect, it } from 'vitest';
import { getArtifact, listVersions, putArtifact } from '../src/admin/artifacts';

/**
 * Fitted-model storage in R2 (§6).
 *
 * The property under test is immutability, and the subject of that property is
 * a **(model_version, fit date) pair** rather than a version alone. A published
 * `model_version` is a claim about which model produced a number; if the bytes
 * behind it can change, the claim is unfalsifiable. But an expanding-window
 * refit changes the bytes every month *without* changing the specification, so
 * a version-only key made the second refit of any engine a permanent conflict —
 * which is what took the monthly job down until this layout existed.
 */

const NAME = 'equity';
const VERSION = 'equity-1.0.0+cal.yahoo_gspc';
const FIT = '2026-07-31';

function body(overrides: Record<string, unknown> = {}): string {
  return JSON.stringify({ model_version: VERSION, fit_date: FIT, hmm: { seed: 7 }, ...overrides });
}

async function clear() {
  const listed = await env.ARCHIVE.list({ prefix: 'artifacts/' });
  await Promise.all(listed.objects.map((o) => env.ARCHIVE.delete(o.key)));
}

describe('artifact storage (FINDYN_V1_SPEC.md §6)', () => {
  beforeEach(clear);

  it('stores and reads back an exact version and fit', async () => {
    await putArtifact(env, NAME, VERSION, FIT, body());
    const result = await getArtifact(env, NAME, VERSION, FIT);

    expect(result.version).toBe(VERSION);
    expect(result.fit).toBe(FIT);
    expect(JSON.parse(result.body).hmm.seed).toBe(7);
  });

  it('accepts a later refit of the same version', async () => {
    // The failure this whole layout exists to fix. An expanding-window refit
    // re-estimates every parameter monthly while the specification — which
    // features, which estimator, which vocabulary — does not change. Under a
    // version-only key the second refit conflicted with the first and every one
    // after it, forever.
    await putArtifact(env, NAME, VERSION, '2026-06-30', body({ hmm: { seed: 7 } }));
    const second = await putArtifact(env, NAME, VERSION, FIT, body({ hmm: { seed: 11 } }));

    expect(second.created).toBe(true);
    expect(JSON.parse((await getArtifact(env, NAME, VERSION, '2026-06-30')).body).hmm.seed).toBe(7);
    expect(JSON.parse((await getArtifact(env, NAME, VERSION, FIT)).body).hmm.seed).toBe(11);
  });

  it('resolves a bare version to its newest fit', async () => {
    await putArtifact(env, NAME, VERSION, '2026-06-30', body({ hmm: { seed: 7 } }));
    await putArtifact(env, NAME, VERSION, FIT, body({ hmm: { seed: 11 } }));

    const result = await getArtifact(env, NAME, VERSION);
    expect(result.fit).toBe(FIT);
    expect(JSON.parse(result.body).hmm.seed).toBe(11);
  });

  it('resolves `latest` through the pointer, version and fit', async () => {
    await putArtifact(
      env,
      NAME,
      'equity-1.0.0',
      '2026-06-30',
      body({ model_version: 'equity-1.0.0' }),
    );
    await putArtifact(env, NAME, 'equity-1.1.0', FIT, body({ model_version: 'equity-1.1.0' }));

    const result = await getArtifact(env, NAME, 'latest');
    expect(result.version).toBe('equity-1.1.0');
    expect(result.fit).toBe(FIT);
  });

  it('never moves the pointer backwards', async () => {
    // Refits can land out of order — a backfill of an older date after a current
    // one. Last-writer-wins would quietly make the stale fit current, and every
    // daily run after it would publish from a model the operator thought was
    // superseded.
    await putArtifact(env, NAME, VERSION, FIT, body());
    await putArtifact(env, NAME, VERSION, '2026-01-31', body({ hmm: { seed: 3 } }));

    expect((await getArtifact(env, NAME, 'latest')).fit).toBe(FIT);
  });

  it('rejects a different body under an existing version and fit', async () => {
    await putArtifact(env, NAME, VERSION, FIT, body());
    await expect(putArtifact(env, NAME, VERSION, FIT, body({ hmm: { seed: 99 } }))).rejects.toThrow(
      /immutable/,
    );
  });

  it('treats an identical re-write as success, not a conflict', async () => {
    // Retries after a network failure are routine and should not need a human.
    const first = await putArtifact(env, NAME, VERSION, FIT, body());
    const second = await putArtifact(env, NAME, VERSION, FIT, body());
    expect(first.created).toBe(true);
    expect(second.created).toBe(false);
  });

  it('404s for a version that was never stored', async () => {
    await expect(getArtifact(env, NAME, 'equity-9.9.9')).rejects.toThrow(/not found/);
  });

  it('404s for `latest` before anything has been stored', async () => {
    await expect(getArtifact(env, NAME, 'latest')).rejects.toThrow(/no artifact/);
  });

  it('lists every stored fit without the pointer', async () => {
    await putArtifact(
      env,
      NAME,
      'equity-1.0.0',
      '2026-06-30',
      body({ model_version: 'equity-1.0.0' }),
    );
    await putArtifact(env, NAME, VERSION, FIT, body());

    const refs = await listVersions(env, NAME);
    expect(refs).toEqual([
      { version: 'equity-1.0.0', fit: '2026-06-30' },
      { version: VERSION, fit: FIT },
    ]);
    expect(refs.map((r) => r.version)).not.toContain('latest');
  });

  it('rejects names, versions and fits that would not be safe as keys', async () => {
    await expect(putArtifact(env, '../etc', VERSION, FIT, body())).rejects.toThrow(/invalid/);
    await expect(putArtifact(env, NAME, 'a/b', FIT, body())).rejects.toThrow(/invalid/);
    await expect(putArtifact(env, NAME, VERSION, '2026/07/31', body())).rejects.toThrow(/invalid/);
    await expect(putArtifact(env, NAME, VERSION, 'latest', body())).rejects.toThrow(/invalid/);
  });
});

describe('artifacts stored before the fit date joined the key', () => {
  beforeEach(clear);

  /**
   * The back-migration, and the reason it needs no data movement.
   *
   * Production holds one flat `artifacts/<name>/<version>.json` and a pointer
   * with no `fit`. A worker carrying this change has to keep serving both
   * through the deploy, because the alternative — resolving nothing until the
   * next monthly refit — means the equity engine publishes no state at all for
   * up to a month.
   */
  async function seedLegacy(version: string, payload: string) {
    await env.ARCHIVE.put(`artifacts/${NAME}/${encodeURIComponent(version)}.json`, payload);
    await env.ARCHIVE.put(
      `artifacts/${NAME}/latest.json`,
      JSON.stringify({ name: NAME, version, updated_at: new Date().toISOString() }),
    );
  }

  it('serves a flat artifact through a pointer that predates the fit date', async () => {
    await seedLegacy(VERSION, body());

    const result = await getArtifact(env, NAME, 'latest');
    expect(result.version).toBe(VERSION);
    // Reported as null rather than guessed. The bytes genuinely do not say when
    // they were fitted, and inventing a date would make an unreplayable model
    // look replayable.
    expect(result.fit).toBeNull();
    expect(JSON.parse(result.body).hmm.seed).toBe(7);
  });

  it('prefers a nested fit once one exists, without deleting the flat one', async () => {
    await seedLegacy(VERSION, body({ hmm: { seed: 1 } }));
    await putArtifact(env, NAME, VERSION, FIT, body({ hmm: { seed: 42 } }));

    const result = await getArtifact(env, NAME, 'latest');
    expect(result.fit).toBe(FIT);
    expect(JSON.parse(result.body).hmm.seed).toBe(42);

    // The flat object is still readable by anything that asks for it directly.
    expect(
      await env.ARCHIVE.get(`artifacts/${NAME}/${encodeURIComponent(VERSION)}.json`),
    ).not.toBeNull();
  });

  it('lists a pre-dated artifact rather than hiding it', async () => {
    await seedLegacy(VERSION, body());
    expect(await listVersions(env, NAME)).toEqual([{ version: VERSION, fit: null }]);
  });
});

describe('artifact routes are behind HMAC', () => {
  beforeEach(clear);

  /**
   * The claim is "an unsigned request never receives an artifact", not a
   * specific status code. Both refusals are legitimate and which one you get
   * depends on the environment: 401 where a secret is configured and the
   * signature is missing, 503 where the admin surface is disabled because no
   * secret is set at all — which is exactly CI, and is what an earlier version
   * of this test failed on for asserting 401 outright.
   */
  function assertRefused(res: Response) {
    expect([401, 503]).toContain(res.status);
    expect(res.status).not.toBe(200);
  }

  it('refuses an unsigned read', async () => {
    assertRefused(await SELF.fetch(`https://findyn.test/admin/v1/artifacts/${NAME}/latest`));
  });

  it('refuses an unsigned write', async () => {
    assertRefused(
      await SELF.fetch(`https://findyn.test/admin/v1/artifacts/${NAME}/${VERSION}`, {
        method: 'PUT',
        body: body(),
      }),
    );
  });

  it('never leaks a stored artifact to an unsigned reader', async () => {
    // The assertion that actually matters: even with something in the bucket,
    // an unsigned caller gets a refusal rather than the model.
    await putArtifact(env, NAME, VERSION, FIT, body());
    const res = await SELF.fetch(`https://findyn.test/admin/v1/artifacts/${NAME}/${VERSION}`);
    assertRefused(res);
    expect(await res.text()).not.toContain('seed');
  });
});
