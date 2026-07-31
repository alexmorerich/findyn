import { SELF, env } from 'cloudflare:test';
import { beforeEach, describe, expect, it } from 'vitest';
import { getArtifact, listVersions, putArtifact } from '../src/admin/artifacts';

/**
 * Fitted-model storage in R2 (§6).
 *
 * The property under test is immutability. A published `model_version` is a
 * claim about which model produced a number; if the bytes behind that version
 * can change, the claim is unfalsifiable and every backtest of it silently
 * becomes a backtest of something else.
 */

const NAME = 'equity';
const VERSION = 'equity-1.0.0+cal.yahoo_gspc';

function body(overrides: Record<string, unknown> = {}): string {
  return JSON.stringify({ model_version: VERSION, hmm: { seed: 7 }, ...overrides });
}

async function clear() {
  const listed = await env.ARCHIVE.list({ prefix: 'artifacts/' });
  await Promise.all(listed.objects.map((o) => env.ARCHIVE.delete(o.key)));
}

describe('artifact storage (FINDYN_V1_SPEC.md §6)', () => {
  beforeEach(clear);

  it('stores and reads back an exact version', async () => {
    await putArtifact(env, NAME, VERSION, body());
    const result = await getArtifact(env, NAME, VERSION);

    expect(result.version).toBe(VERSION);
    expect(JSON.parse(result.body).hmm.seed).toBe(7);
  });

  it('resolves `latest` through the pointer', async () => {
    await putArtifact(env, NAME, 'equity-1.0.0', body({ model_version: 'equity-1.0.0' }));
    await putArtifact(env, NAME, 'equity-1.1.0', body({ model_version: 'equity-1.1.0' }));

    const result = await getArtifact(env, NAME, 'latest');
    expect(result.version).toBe('equity-1.1.0');
  });

  it('rejects a different body under an existing version', async () => {
    await putArtifact(env, NAME, VERSION, body());
    await expect(putArtifact(env, NAME, VERSION, body({ hmm: { seed: 99 } }))).rejects.toThrow(
      /immutable/,
    );
  });

  it('treats an identical re-write as success, not a conflict', async () => {
    // Retries after a network failure are routine and should not need a human.
    const first = await putArtifact(env, NAME, VERSION, body());
    const second = await putArtifact(env, NAME, VERSION, body());
    expect(first.created).toBe(true);
    expect(second.created).toBe(false);
  });

  it('404s for a version that was never stored', async () => {
    await expect(getArtifact(env, NAME, 'equity-9.9.9')).rejects.toThrow(/not found/);
  });

  it('404s for `latest` before anything has been stored', async () => {
    await expect(getArtifact(env, NAME, 'latest')).rejects.toThrow(/no artifact/);
  });

  it('lists stored versions without the pointer', async () => {
    await putArtifact(env, NAME, 'equity-1.0.0', body({ model_version: 'equity-1.0.0' }));
    await putArtifact(env, NAME, VERSION, body());

    const versions = await listVersions(env, NAME);
    expect(versions).toEqual(['equity-1.0.0', VERSION]);
    expect(versions).not.toContain('latest');
  });

  it('rejects names and versions that would not be safe as keys', async () => {
    await expect(putArtifact(env, '../etc', VERSION, body())).rejects.toThrow(/invalid/);
    await expect(putArtifact(env, NAME, 'a/b', body())).rejects.toThrow(/invalid/);
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
    await putArtifact(env, NAME, VERSION, body());
    const res = await SELF.fetch(`https://findyn.test/admin/v1/artifacts/${NAME}/${VERSION}`);
    assertRefused(res);
    expect(await res.text()).not.toContain('seed');
  });
});
