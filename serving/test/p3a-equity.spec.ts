import { SELF, env } from 'cloudflare:test';
import { beforeEach, describe, expect, it } from 'vitest';
import { PayloadError, applyWriteBack, validatePayload } from '../src/admin/writeback';
import { getForceSnapshot, getKinematicState, getTwoLayerState } from '../src/api/equity';

/**
 * P3-A: FinEquity's feature write-back and the two v1 endpoints it lights up.
 *
 * The load-bearing assertions here are about *versioning* and *honesty*.
 *
 * Versioning: `derived_features` is keyed by model_version, unlike
 * `engine_output`. A refit must land beside the features the previous model was
 * fitted on, not on top of them, or every backtest of that model silently
 * re-runs on inputs it never saw.
 *
 * Honesty: `/state` serves the two-layer state of FINDYN_V1_SPEC.md §2, and in
 * this sub-milestone only one of those layers exists. It says so with an
 * explicit `regime: null` rather than omitting the key, so a consumer can tell
 * "not computed yet" from "the field moved".
 */

async function reset() {
  await env.DB.batch([
    env.DB.prepare('DELETE FROM derived_features'),
    env.DB.prepare('DELETE FROM force_scores'),
    env.DB.prepare('DELETE FROM asset_state'),
    env.DB.prepare('DELETE FROM engine_output'),
  ]);
}

const VERSION = 'equity-1.0.0+cal.fred_nasdaq100';

function feature(feature: string, value: number, as_of = '2026-07-29', model_version = VERSION) {
  return { asset: 'equity', feature, as_of, value, model_version };
}

const FEATURES = [
  feature('price_filtered', 8.8998),
  feature('velocity', 0.1349),
  feature('acceleration', -1.0294),
  feature('jerk_z', -0.0812),
  feature('ffd_price', 0.9194),
  feature('momentum_12m', 235.1027),
];

describe('P3-A — equity features (FINDYN_V1_SPEC.md §7, §13)', () => {
  beforeEach(reset);

  it('validates and stores a derived_features batch', async () => {
    const payload = validatePayload({ model_version: VERSION, derived_features: FEATURES });
    const result = await applyWriteBack(env, payload);
    expect(result.derived_features).toBe(FEATURES.length);

    const state = await getKinematicState(env);
    expect(state.as_of).toBe('2026-07-29');
    expect(state.model_version).toBe(VERSION);
    expect(state.features.velocity).toBeCloseTo(0.1349, 6);
  });

  it('is idempotent, so a re-run repairs a partial write instead of doubling it', async () => {
    const payload = validatePayload({ derived_features: FEATURES });
    await applyWriteBack(env, payload);
    await applyWriteBack(env, payload);

    const row = await env.DB.prepare(
      `SELECT COUNT(*) AS n FROM derived_features`,
    ).first<{ n: number }>();
    expect(row?.n).toBe(FEATURES.length);
  });

  it('keeps two model versions of the same feature side by side', async () => {
    // The refit case. Overwriting would mean the old model's backtest silently
    // starts running on features it was never fitted on.
    await applyWriteBack(env, validatePayload({ derived_features: [feature('velocity', 0.11)] }));
    await applyWriteBack(
      env,
      validatePayload({
        derived_features: [feature('velocity', 0.22, '2026-07-29', 'equity-1.1.0+cal.stooq_spx')],
      }),
    );

    const { results } = await env.DB.prepare(
      `SELECT model_version, value FROM derived_features
        WHERE feature = 'velocity' ORDER BY model_version`,
    ).all<{ model_version: string; value: number }>();

    expect(results).toHaveLength(2);
    expect(results.map((r) => r.value)).toEqual([0.11, 0.22]);
  });

  it('carries model_version per row, not from the envelope', async () => {
    // A run publishing two engines has two versions; the envelope's joined
    // string is a version no single model ever had.
    const payload = validatePayload({
      model_version: 'equity-1.0.0,rates-1.0.0',
      derived_features: [feature('velocity', 0.11)],
    });
    await applyWriteBack(env, payload);

    const row = await env.DB.prepare(
      `SELECT model_version FROM derived_features LIMIT 1`,
    ).first<{ model_version: string }>();
    expect(row?.model_version).toBe(VERSION);
  });

  it('rejects an unknown asset', () => {
    expect(() =>
      validatePayload({ derived_features: [{ ...feature('velocity', 1), asset: 'commodities' }] }),
    ).toThrow(PayloadError);
  });

  it('rejects a non-finite value and a missing model_version', () => {
    expect(() =>
      validatePayload({ derived_features: [{ ...feature('velocity', 1), value: 'NaN' }] }),
    ).toThrow(PayloadError);
    expect(() =>
      validatePayload({ derived_features: [{ ...feature('velocity', 1), model_version: '' }] }),
    ).toThrow(PayloadError);
  });

  it('accepts a feature name the Worker has never heard of', async () => {
    // Momentum windows are configured in compute/config/engines/equity.yaml, so
    // their names are decided at runtime. A closed vocabulary here would mean a
    // yaml edit could only ship together with a Worker deploy.
    const payload = validatePayload({ derived_features: [feature('momentum_6m', 1.5)] });
    expect((await applyWriteBack(env, payload)).derived_features).toBe(1);
  });

  it('assembles the snapshot from one date rather than per-feature newest', async () => {
    // The features have different start dates — jerk_z waits for its z-score
    // baseline. Taking each column's newest row independently would assemble a
    // snapshot out of several different days and label it with one.
    await applyWriteBack(
      env,
      validatePayload({
        derived_features: [
          feature('velocity', 0.1, '2026-07-29'),
          feature('jerk_z', 9.9, '2026-07-20'),
        ],
      }),
    );

    const state = await getKinematicState(env);
    expect(state.as_of).toBe('2026-07-29');
    expect(state.features.jerk_z).toBeUndefined();
  });
});

describe('P3-A — /state and /forces (§13)', () => {
  beforeEach(reset);

  it('serves both layers of the two-layer state', async () => {
    await applyWriteBack(env, validatePayload({ derived_features: FEATURES }));
    await env.DB.prepare(
      `INSERT INTO force_scores (date, force, score, components, model_version)
       VALUES ('2026-07-29', 'valuation', 22.5, '{"SHILLER:CAPE": 18.0}', 'factors-1.0.0')`,
    ).run();

    const state = await getTwoLayerState(env);
    expect(state.kinematics.features.velocity).toBeCloseTo(0.1349, 6);
    expect(state.forces.scores.valuation).toBe(22.5);
    expect(state.forces.components.valuation).toEqual({ 'SHILLER:CAPE': 18.0 });
  });

  it('reports the regime layer as an explicit null until P3-B', async () => {
    await applyWriteBack(env, validatePayload({ derived_features: FEATURES }));
    const res = await SELF.fetch('https://findyn.test/api/v1/state');
    expect(res.status).toBe(200);

    const body = (await res.json()) as { data: { regime: unknown }; model_version: string | null };
    expect(body.data).toHaveProperty('regime');
    expect(body.data.regime).toBeNull();
    expect(body.model_version).toBe(VERSION);
  });

  it('serves force history with its component breakdowns', async () => {
    await env.DB.batch([
      env.DB.prepare(
        `INSERT INTO force_scores (date, force, score, components, model_version)
         VALUES ('2026-07-28', 'liquidity', 61.0, '{"FRED:M2SL": 55.0}', 'f-1')`,
      ),
      env.DB.prepare(
        `INSERT INTO force_scores (date, force, score, components, model_version)
         VALUES ('2026-07-29', 'liquidity', 62.5, '{"FRED:M2SL": 56.0}', 'f-1')`,
      ),
    ]);

    const res = await SELF.fetch('https://findyn.test/api/v1/forces?force=liquidity');
    const body = (await res.json()) as {
      data: { count: number; points: { as_of: string; score: number; components: unknown }[] };
    };

    expect(body.data.count).toBe(2);
    // Ascending on the way out: a chart wants time to move left to right.
    expect(body.data.points.map((p) => p.as_of)).toEqual(['2026-07-28', '2026-07-29']);
    expect(body.data.points[1]?.components).toEqual({ 'FRED:M2SL': 56.0 });
  });

  it('rejects an unknown force with 400 rather than serving an empty series', async () => {
    const res = await SELF.fetch('https://findyn.test/api/v1/forces?force=vibes');
    expect(res.status).toBe(400);
  });

  it('prefers the newest model version when one date was scored twice', async () => {
    await env.DB.batch([
      env.DB.prepare(
        `INSERT INTO force_scores (date, force, score, components, model_version)
         VALUES ('2026-07-29', 'credit', 10.0, NULL, 'factors-1.0.0')`,
      ),
      env.DB.prepare(
        `INSERT INTO force_scores (date, force, score, components, model_version)
         VALUES ('2026-07-29', 'credit', 40.0, NULL, 'factors-2.0.0')`,
      ),
    ]);

    expect((await getForceSnapshot(env)).scores.credit).toBe(40.0);
  });

  it('leaves /assets/equity/state at 501 while the engine publishes no state', async () => {
    // The engine is enabled and publishing features; it declines to publish an
    // AssetState until the regime model lands. Those are different answers and
    // the two endpoints have to give different ones.
    await applyWriteBack(env, validatePayload({ derived_features: FEATURES }));

    const features = await SELF.fetch('https://findyn.test/api/v1/state');
    const state = await SELF.fetch('https://findyn.test/api/v1/assets/equity/state');

    expect(features.status).toBe(200);
    expect(state.status).toBe(501);
    expect(((await state.json()) as { milestone: string }).milestone).toBe('P3');
  });
});
