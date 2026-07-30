/** Reject signatures whose timestamp is further than this from now (replay window). */
const MAX_SKEW_SECONDS = 300;

const encoder = new TextEncoder();

function hexToBytes(hex: string): Uint8Array | null {
  if (hex.length === 0 || hex.length % 2 !== 0) return null;
  const out = new Uint8Array(hex.length / 2);
  for (let i = 0; i < out.length; i++) {
    const byte = Number.parseInt(hex.slice(i * 2, i * 2 + 2), 16);
    if (Number.isNaN(byte)) return null;
    out[i] = byte;
  }
  return out;
}

/**
 * Verify an HMAC-SHA256 signature over `${timestamp}.${body}`.
 *
 * Uses crypto.subtle.verify rather than comparing hex strings, so the
 * comparison is constant-time.
 */
export async function verifyHmac(params: {
  secret: string;
  signature: string;
  timestamp: string;
  body: string;
  now?: Date;
}): Promise<boolean> {
  const { secret, signature, timestamp, body } = params;

  const ts = Number.parseInt(timestamp, 10);
  if (!Number.isFinite(ts)) return false;
  const nowSeconds = Math.floor((params.now ?? new Date()).getTime() / 1000);
  if (Math.abs(nowSeconds - ts) > MAX_SKEW_SECONDS) return false;

  const sigBytes = hexToBytes(signature.trim().toLowerCase());
  if (!sigBytes) return false;

  const key = await crypto.subtle.importKey(
    'raw',
    encoder.encode(secret),
    { name: 'HMAC', hash: 'SHA-256' },
    false,
    ['verify'],
  );

  return crypto.subtle.verify(
    'HMAC',
    key,
    sigBytes as unknown as ArrayBuffer,
    encoder.encode(`${timestamp}.${body}`),
  );
}
