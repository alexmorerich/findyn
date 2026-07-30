import { describe, expect, it } from 'vitest';
import { verifyHmac } from '../src/admin/hmac';

const SECRET = 'test-secret-do-not-use';

async function sign(timestamp: string, body: string, secret = SECRET): Promise<string> {
  const key = await crypto.subtle.importKey(
    'raw',
    new TextEncoder().encode(secret),
    { name: 'HMAC', hash: 'SHA-256' },
    false,
    ['sign'],
  );
  const mac = await crypto.subtle.sign('HMAC', key, new TextEncoder().encode(`${timestamp}.${body}`));
  return [...new Uint8Array(mac)].map((b) => b.toString(16).padStart(2, '0')).join('');
}

describe('compute write-back authentication (FINDYN_V1_SPEC.md §6)', () => {
  const now = new Date('2026-07-30T12:00:00Z');
  const timestamp = String(Math.floor(now.getTime() / 1000));
  const body = JSON.stringify({ model_version: 'v1.0.0', rows: [] });

  it('accepts a correctly signed payload', async () => {
    const signature = await sign(timestamp, body);
    await expect(verifyHmac({ secret: SECRET, signature, timestamp, body, now })).resolves.toBe(true);
  });

  it('rejects a tampered body', async () => {
    const signature = await sign(timestamp, body);
    const tampered = JSON.stringify({ model_version: 'v1.0.0', rows: [{ evil: true }] });
    await expect(
      verifyHmac({ secret: SECRET, signature, timestamp, body: tampered, now }),
    ).resolves.toBe(false);
  });

  it('rejects the wrong secret', async () => {
    const signature = await sign(timestamp, body, 'other-secret');
    await expect(verifyHmac({ secret: SECRET, signature, timestamp, body, now })).resolves.toBe(false);
  });

  it('rejects a replayed timestamp outside the skew window', async () => {
    const stale = String(Math.floor(now.getTime() / 1000) - 3600);
    const signature = await sign(stale, body);
    await expect(
      verifyHmac({ secret: SECRET, signature, timestamp: stale, body, now }),
    ).resolves.toBe(false);
  });

  it('agrees with the Python signer on a fixed vector', async () => {
    // Same vector as compute/tests/test_domain.py::test_hmac_matches_the_typescript_verifier.
    // Pins the canonical string `${timestamp}.${body}` across both planes.
    const vector = {
      secret: 'findyn-parity-vector',
      timestamp: '1750000000',
      body: '{"a":1}',
      signature: '14f1496721f7ac017aa6b6f0ce9edb1bc7f68ef26ca45e434817413526145747',
    };
    expect(await sign(vector.timestamp, vector.body, vector.secret)).toBe(vector.signature);
    await expect(
      verifyHmac({ ...vector, now: new Date(1_750_000_000_000) }),
    ).resolves.toBe(true);
  });

  it('rejects malformed signatures without throwing', async () => {
    for (const signature of ['', 'zz', 'abc', 'not-hex-at-all']) {
      await expect(verifyHmac({ secret: SECRET, signature, timestamp, body, now })).resolves.toBe(
        false,
      );
    }
  });
});
