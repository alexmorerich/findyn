import { Hono } from 'hono';
import { cors } from 'hono/cors';
import type { Env } from '../types';
// `isStale` is not imported here any more: it measures hours since ingestion,
// which is the right question for /health (where it still lives) and the wrong
// one for a market date. The asset endpoints use isAssetStale instead.
import { envelope, notImplemented } from '../lib/responses';
import { getHealth } from './health';
import { getObservations, getSeriesMetadata, listSeries } from './series';
import { InvalidDateError, pitSnapshot } from './pit';
import {
  UnknownAssetError,
  assertKnownAsset,
  getAssetState,
  getHistory,
  isAssetStale,
  listAssets,
  listMetrics,
} from './assets';
import {
  ASSETS,
  DISCOUNT_HORIZONS,
  FORCES,
  HORIZONS,
  MONEY_REGIMES,
  RATE_REGIMES,
  REGIMES,
} from '../domain';

/**
 * Which phase delivers each engine, for the 501 an unpublished engine returns.
 * Kept beside the route rather than in domain.ts: it is a roadmap fact, not
 * vocabulary the compute plane has to agree with.
 */
const ENGINE_PHASE: Record<string, string> = {
  rates: 'P1',
  money: 'P2',
  equity: 'P3',
  gold: 'P4',
  crypto: 'P5',
};

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
      vocabulary: {
        forces: FORCES,
        regimes: REGIMES,
        horizons: HORIZONS,
        assets: ASSETS,
        rate_regimes: RATE_REGIMES,
        money_regimes: MONEY_REGIMES,
        discount_horizons: DISCOUNT_HORIZONS,
      },
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

// ---------------------------------------------------------------------------
// Multi-asset engine surface (01-target-architecture.md §7)
// ---------------------------------------------------------------------------

/**
 * The registry, not the data: every engine in the vocabulary appears, with
 * `status: 'awaiting_first_run'` until one has published. The dashboard's
 * Engines panel is driven entirely by this, so a new engine shows up on the
 * home page the first time it writes back — no template change.
 */
api.get('/assets', async (c) => {
  const assets = await listAssets(c.env);
  const newest = assets.reduce<string | null>(
    (max, a) => (a.as_of && (max === null || a.as_of > max) ? a.as_of : max),
    null,
  );
  return c.json(
    envelope(
      { count: assets.length, assets },
      { as_of: newest, stale: assets.every((a) => a.stale) },
    ),
  );
});

api.get('/assets/:asset/state', async (c) => {
  const asset = c.req.param('asset');
  try {
    assertKnownAsset(asset);
  } catch (err) {
    if (err instanceof UnknownAssetError) {
      return c.json({ error: 'not_found', message: err.message }, 404);
    }
    throw err;
  }

  const state = await getAssetState(c.env, asset);
  if (!state) {
    // Registered but silent. 501 with the phase tag is the existing convention
    // for "reserved, not delivered", and it is the honest answer here too: the
    // consumer can tell this apart from an engine that ran and found nothing.
    return c.json(
      {
        error: 'not_implemented',
        message: `The ${asset} engine has not published a state yet.`,
        milestone: ENGINE_PHASE[asset] ?? 'P6',
        asset,
      },
      501,
    );
  }

  return c.json(
    envelope(state, {
      as_of: state.as_of,
      model_version: state.model_version,
      // Engine dates are market dates, not ingestion timestamps — see isAssetStale.
      stale: isAssetStale(state.as_of),
    }),
  );
});

api.get('/assets/:asset/history', async (c) => {
  const asset = c.req.param('asset');
  try {
    assertKnownAsset(asset);
  } catch (err) {
    if (err instanceof UnknownAssetError) {
      return c.json({ error: 'not_found', message: err.message }, 404);
    }
    throw err;
  }

  const metric = c.req.query('metric');
  if (!metric) {
    const available = await listMetrics(c.env, asset);
    return c.json(
      {
        error: 'bad_request',
        message: 'metric= is required',
        available,
      },
      400,
    );
  }

  const points = await getHistory(c.env, asset, metric, {
    from: c.req.query('from'),
    to: c.req.query('to'),
    limit: c.req.query('limit') ? Number(c.req.query('limit')) : undefined,
  });
  const newest = points.at(-1)?.as_of ?? null;

  return c.json(
    envelope(
      { asset, metric, count: points.length, points },
      { as_of: newest, stale: isAssetStale(newest) },
    ),
  );
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
