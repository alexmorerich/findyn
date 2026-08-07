import { Hono } from 'hono';
import { cors } from 'hono/cors';
import type { Env } from '../types';
// `isStale` is not imported here any more: it measures hours since ingestion,
// which is the right question for /health (where it still lives) and the wrong
// one for a market date. The asset endpoints use isAssetStale instead.
import { envelope } from '../lib/responses';
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
import { isExperimentalAsset } from '../domain';
import {
  UnknownForceError,
  getForces,
  getForecast,
  getInstability,
  getRegimeHistory,
  getTwoLayerState,
} from './equity';
import {
  ASSETS,
  CRYPTO_REGIMES,
  DISCOUNT_HORIZONS,
  EXPERIMENTAL_ASSETS,
  FORCES,
  GOLD_REGIMES,
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
        gold_regimes: GOLD_REGIMES,
        crypto_regimes: CRYPTO_REGIMES,
        discount_horizons: DISCOUNT_HORIZONS,
        // Which of `assets` are research-only. Published so a client can find
        // out without hard-coding the name of the one engine that is.
        experimental_assets: EXPERIMENTAL_ASSETS,
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
    (max, s) =>
      s.latest_obs_date && (max === null || s.latest_obs_date > max) ? s.latest_obs_date : max,
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
  const result = await getObservations(c.env, seriesId, {
    from: c.req.query('from'),
    to: c.req.query('to'),
    limit: c.req.query('limit') ? Number(c.req.query('limit')) : undefined,
    points: c.req.query('points') ? Number(c.req.query('points')) : undefined,
  });
  return c.json(
    envelope(
      {
        metadata,
        observations: result.observations,
        available: result.available,
        truncated: result.truncated,
        decimated: result.decimated,
      },
      { as_of: result.observations.at(-1)?.obs_date ?? null },
    ),
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
    //
    // The experimental flag rides on the 501 as well. A consumer polling for a
    // crypto state has to know what it is waiting for before it arrives, not
    // after — otherwise the first response it can act on is the first one that
    // tells it not to.
    return c.json(
      {
        error: 'not_implemented',
        message: `The ${asset} engine has not published a state yet.`,
        milestone: ENGINE_PHASE[asset] ?? 'P6',
        asset,
        ...(isExperimentalAsset(asset) ? { experimental: true } : {}),
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
      experimental: isExperimentalAsset(asset),
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

  const history = await getHistory(c.env, asset, metric, {
    from: c.req.query('from'),
    to: c.req.query('to'),
    limit: c.req.query('limit') ? Number(c.req.query('limit')) : undefined,
    // `points` asks the server to downsample for rendering. Opt-in, because a
    // consumer reading these series back as model input (P2's curve read-back)
    // wants every row and would be quietly given a sketch otherwise.
    points: c.req.query('points') ? Number(c.req.query('points')) : undefined,
  });
  const newest = history.points.at(-1)?.as_of ?? null;

  return c.json(
    envelope(
      {
        asset,
        metric,
        count: history.points.length,
        // Reported even when nothing was dropped: a client that has to infer
        // whether it received the whole window will eventually infer wrong.
        available: history.available,
        truncated: history.truncated,
        decimated: history.decimated,
        points: history.points,
      },
      {
        as_of: newest,
        stale: isAssetStale(newest),
        experimental: isExperimentalAsset(asset),
      },
    ),
  );
});

// M2 — kinematic state K(t) + force state F(t) snapshot.
//
// Live from P3-A. `regime` is an explicit null until the HMM lands in P3-B —
// the endpoint reports which half of the two-layer state exists rather than
// withholding the half that does.
api.get('/state', async (c) => {
  const state = await getTwoLayerState(c.env);
  const asOf = state.kinematics.as_of ?? state.forces.as_of;
  return c.json(
    envelope(state, {
      as_of: asOf,
      model_version: state.kinematics.model_version,
      stale: isAssetStale(asOf),
    }),
  );
});

// M3 — regime probability history. Live from P3-B.
api.get('/regime', async (c) => {
  const history = await getRegimeHistory(c.env, c.req.query('asset') ?? 'equity', {
    from: c.req.query('from'),
    to: c.req.query('to'),
    points: c.req.query('points') ? Number(c.req.query('points')) : undefined,
  });
  const newest = history.points.at(-1)?.as_of ?? null;
  return c.json(
    envelope(history, {
      as_of: newest,
      model_version: history.model_version,
      stale: isAssetStale(newest),
    }),
  );
});

// M2 — force score history with component breakdowns. Live from P3-A; the
// scores themselves have been published since P1, when Layer 0 shipped.
api.get('/forces', async (c) => {
  try {
    const points = await getForces(c.env, {
      from: c.req.query('from'),
      to: c.req.query('to'),
      force: c.req.query('force'),
      limit: c.req.query('limit') ? Number(c.req.query('limit')) : undefined,
    });
    const newest = points.at(-1)?.as_of ?? null;
    return c.json(
      envelope(
        { count: points.length, forces: FORCES, points },
        { as_of: newest, stale: isAssetStale(newest) },
      ),
    );
  } catch (err) {
    if (err instanceof UnknownForceError) {
      return c.json({ error: 'bad_request', message: err.message }, 400);
    }
    throw err;
  }
});

// M4 — RII + crash decomposition history. Live from P3-C.
api.get('/instability', async (c) => {
  const history = await getInstability(c.env, c.req.query('asset') ?? 'equity', {
    from: c.req.query('from'),
    to: c.req.query('to'),
    points: c.req.query('points') ? Number(c.req.query('points')) : undefined,
  });
  const newest = history.points.at(-1)?.as_of ?? null;
  return c.json(envelope(history, { as_of: newest, stale: isAssetStale(newest) }));
});

// M4 — quantile forecast distributions per horizon. Live from P3-C.
api.get('/forecast', async (c) => {
  const horizon = c.req.query('horizon');
  if (horizon && !(HORIZONS as readonly string[]).includes(horizon)) {
    return c.json(
      {
        error: 'bad_request',
        message: `unknown horizon ${horizon}; expected one of ${HORIZONS.join('|')}`,
      },
      400,
    );
  }
  const forecast = await getForecast(c.env, c.req.query('asset') ?? 'equity', horizon);
  return c.json(
    envelope(forecast, {
      as_of: forecast.as_of,
      model_version: forecast.model_version,
      stale: isAssetStale(forecast.as_of),
    }),
  );
});

// M4 — Monte Carlo summary statistics.
//
// The path bundles themselves are not served: §11 puts them in R2 for offline
// analysis, and a public endpoint returning 10,000 paths would be a denial of
// service with extra steps. What a caller wants — the distribution — is
// /forecast, and the run's parameters are in the AssetState's components.
api.get('/simulate', async (c) => {
  const forecast = await getForecast(c.env, c.req.query('asset') ?? 'equity');
  const state = await getAssetState(c.env, c.req.query('asset') ?? 'equity');
  const components = state?.components ?? {};
  return c.json(
    envelope(
      {
        asset: forecast.asset,
        as_of: forecast.as_of,
        model_version: forecast.model_version,
        horizons: forecast.horizons.map((band) => band.horizon),
        summary: Object.fromEntries(
          Object.entries(components).filter(([key]) => key.startsWith('mc_')),
        ),
        note:
          'Quantile bands are served by /forecast. Per-path outcomes — terminal ' +
          'level, maximum drawdown and time under water for every simulated path — ' +
          'are archived to R2 for offline analysis rather than served here: ' +
          '10,000 paths per horizon is not a public payload. The step-by-step ' +
          'paths themselves are not retained; see PathSample for why.',
      },
      {
        as_of: forecast.as_of,
        model_version: forecast.model_version,
        stale: isAssetStale(forecast.as_of),
      },
    ),
  );
});
