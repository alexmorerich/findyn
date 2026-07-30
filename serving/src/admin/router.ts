import { Hono } from 'hono';
import type { Env } from '../types';
import { verifyHmac } from './hmac';
import { PayloadError, applyWriteBack, validatePayload } from './writeback';

type Vars = { rawBody: string };

/**
 * Private write-back surface for the Python compute plane (FINDYN_V1_SPEC.md §6).
 * Cloudflare Workers cannot run the scientific Python stack, so all provider and
 * model output arrives here as HMAC-signed JSON rather than being computed in-Worker.
 */
export const admin = new Hono<{ Bindings: Env; Variables: Vars }>();

admin.use('*', async (c, next) => {
  const secret = c.env.ADMIN_HMAC_SECRET;
  if (!secret) {
    return c.json({ error: 'admin_disabled', message: 'ADMIN_HMAC_SECRET is not configured' }, 503);
  }

  const signature = c.req.header('x-findyn-signature');
  const timestamp = c.req.header('x-findyn-timestamp');
  if (!signature || !timestamp) {
    return c.json({ error: 'unauthorized', message: 'missing signature headers' }, 401);
  }

  // Signature covers the exact bytes, so the body is read once here and reused
  // downstream rather than re-parsed from the request.
  const body = await c.req.text();
  const ok = await verifyHmac({ secret, signature, timestamp, body });
  if (!ok) {
    return c.json({ error: 'unauthorized', message: 'invalid or expired signature' }, 401);
  }

  c.set('rawBody', body);
  await next();
});

admin.post('/results', async (c) => {
  let parsed: unknown;
  try {
    parsed = JSON.parse(c.get('rawBody'));
  } catch {
    return c.json({ error: 'bad_request', message: 'body is not valid JSON' }, 400);
  }

  try {
    const payload = validatePayload(parsed);
    const written = await applyWriteBack(c.env, payload);
    return c.json({
      ok: true,
      model_version: payload.model_version,
      generated_at: payload.generated_at,
      written,
    });
  } catch (err) {
    if (err instanceof PayloadError) {
      return c.json({ error: 'invalid_payload', message: err.message }, 422);
    }
    throw err;
  }
});
