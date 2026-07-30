import { Hono } from 'hono';
import type { Env } from '../types';
import { verifyHmac } from './hmac';

/**
 * Private write-back surface for the Python compute plane (FINDYN_V1_SPEC.md §6).
 * Cloudflare Workers cannot run the scientific Python stack, so all model output
 * arrives here as HMAC-signed JSON rather than being computed in-Worker.
 */
export const admin = new Hono<{ Bindings: Env }>();

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

  const body = await c.req.text();
  const ok = await verifyHmac({ secret, signature, timestamp, body });
  if (!ok) {
    return c.json({ error: 'unauthorized', message: 'invalid or expired signature' }, 401);
  }

  c.set('rawBody' as never, body as never);
  await next();
});

// M2 — accepts derived_features / force_scores batches
admin.post('/results', (c) =>
  c.json(
    {
      error: 'not_implemented',
      message: 'Compute write-back lands in milestone M2 (FINDYN_V1_SPEC.md §6).',
      milestone: 'M2',
    },
    501,
  ),
);
