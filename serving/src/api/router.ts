import { Hono } from 'hono';
import type { Env } from '../types';
import { envelope, notImplemented } from '../lib/responses';
import { getHealth } from './health';

/**
 * Public read-only API — FINDYN_V1_SPEC.md §13.
 * Endpoints are stubbed with 501 until their milestone lands; the route table
 * itself is fixed now so the dashboard and compute plane can code against it.
 */
export const api = new Hono<{ Bindings: Env }>();

api.get('/health', async (c) => {
  const health = await getHealth(c.env);
  return c.json(envelope(health, { as_of: health.last_ingestion_at, stale: health.stale }));
});

// M2 — kinematic state K(t) + force state F(t) snapshot
api.get('/state', (c) => notImplemented(c, 'M2', '§13 /state'));

// M3 — regime probability history
api.get('/regime', (c) => notImplemented(c, 'M3', '§13 /regime'));

// M2 — force score history with component breakdowns
api.get('/forces', (c) => notImplemented(c, 'M2', '§13 /forces'));

// M4 — RII + crash decomposition history
api.get('/instability', (c) => notImplemented(c, 'M4', '§13 /instability'));

// M4 — quantile forecast distributions per horizon
api.get('/forecast', (c) => notImplemented(c, 'M4', '§13 /forecast'));

// M4 — Monte Carlo summary statistics
api.get('/simulate', (c) => notImplemented(c, 'M4', '§13 /simulate'));
