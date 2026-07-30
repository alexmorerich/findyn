import { Hono } from 'hono';
import { cors } from 'hono/cors';
import type { Env } from '../types';
import { envelope, notImplemented } from '../lib/responses';
import { getHealth } from './health';
import { getObservations, getSeriesMetadata, listSeries } from './series';
import { InvalidDateError, pitSnapshot } from './pit';
import { FORCES, HORIZONS, REGIMES } from '../domain';

/**
 * Public read-only API — FINDYN_V1_SPEC.md §13.
 * Endpoints not yet delivered return 501 with their milestone, so a consumer can
 * tell "not built" from "no data".
 */
export const api = new Hono<{ Bindings: Env }>();

// The dashboard may be served from Pages (a different origin) or from this
// Worker's own asset binding. Reads are public and unauthenticated either way.
api.use('*', cors({ origin: '*', allowMethods: ['GET', 'OPTIONS'], maxAge: 86400 }));

api.get('/meta', (c) =>
  c.json(
    envelope({
      version: '1.0.0',
      milestone: 'M1-A',
      spec: 'FINDYN_V1_SPEC.md',
      env: c.env.FINDYN_ENV,
      info_set: c.env.INFO_SET,
      vocabulary: { forces: FORCES, regimes: REGIMES, horizons: HORIZONS },
    }),
  ),
);

api.get('/health', async (c) => {
  const health = await getHealth(c.env);
  return c.json(envelope(health, { as_of: health.last_ingestion_at, stale: health.stale }));
});

api.get('/series', async (c) => {
  const series = await listSeries(c.env);
  const newest = series.reduce<string | null>(
    (max, s) => (s.latest_obs_date && (max === null || s.latest_obs_date > max) ? s.latest_obs_date : max),
    null,
  );
  return c.json(envelope({ count: series.length, series }, { as_of: newest }));
});

api.get('/series/:id{.+}', async (c) => {
  const seriesId = decodeURIComponent(c.req.param('id'));
  const metadata = await getSeriesMetadata(c.env, seriesId);
  if (!metadata) {
    return c.json({ error: 'not_found', message: `unknown series: ${seriesId}` }, 404);
  }
  const observations = await getObservations(c.env, seriesId, {
    from: c.req.query('from'),
    to: c.req.query('to'),
    limit: c.req.query('limit') ? Number(c.req.query('limit')) : undefined,
  });
  return c.json(
    envelope({ metadata, observations }, { as_of: observations[0]?.obs_date ?? null }),
  );
});

api.get('/pit', async (c) => {
  const asOf = c.req.query('as_of');
  if (!asOf) {
    return c.json({ error: 'bad_request', message: 'as_of=YYYY-MM-DD is required' }, 400);
  }
  try {
    const snapshot = await pitSnapshot(c.env, asOf);
    return c.json(envelope(snapshot, { as_of: asOf }));
  } catch (err) {
    if (err instanceof InvalidDateError) {
      return c.json({ error: 'bad_request', message: err.message }, 400);
    }
    throw err;
  }
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
