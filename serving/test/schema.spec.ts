import { env } from 'cloudflare:test';
import { beforeEach, describe, expect, it } from 'vitest';
import { applyWriteBack, validatePayload } from '../src/admin/writeback';

const EXPECTED_TABLES = [
  'asset_state',
  'data_quality_report',
  'derived_features',
  'engine_output',
  'force_scores',
  'forecast_distribution',
  'ingestion_log',
  'instability_index',
  'macro_series',
  'market_price',
  'regime_state',
  'series_metadata',
  'tradable_proxy_mapping',
];

describe('D1 schema (FINDYN_V1_SPEC.md §7)', () => {
  it('creates every table the spec defines', async () => {
    const { results } = await env.DB.prepare(
      `SELECT name FROM sqlite_master
        WHERE type = 'table' AND name NOT LIKE 'sqlite_%' AND name NOT LIKE '_cf_%'
          AND name != 'd1_migrations'
        ORDER BY name`,
    ).all<{ name: string }>();

    expect(results.map((r) => r.name)).toEqual(EXPECTED_TABLES);
  });

  it('keys macro_series on the vintage so revisions survive (§14.1)', async () => {
    // Revisions of one period share a release_date, so keying on release_date
    // would upsert each revision over the previous one and lose the history a
    // point-in-time query needs.
    const { results } = await env.DB.prepare(
      `SELECT name, pk FROM pragma_table_info('macro_series') WHERE pk > 0 ORDER BY pk`,
    ).all<{ name: string; pk: number }>();

    expect(results.map((r) => r.name)).toEqual(['series_id', 'obs_date', 'revision_date']);
  });

  it('still carries release_date, which is what "first knowable" means', async () => {
    const { results } = await env.DB.prepare(
      `SELECT name, "notnull" FROM pragma_table_info('macro_series')
        WHERE name IN ('release_date', 'revision_date') ORDER BY name`,
    ).all<{ name: string; notnull: number }>();

    expect(results).toEqual([
      { name: 'release_date', notnull: 1 },
      { name: 'revision_date', notnull: 1 },
    ]);
  });

  it('keys asset_state on the model version too (0004)', async () => {
    // Without model_version in the key, a refit overwrites the state the
    // previous model published and every backtest of it changes retroactively.
    const { results } = await env.DB.prepare(
      `SELECT name, pk FROM pragma_table_info('asset_state') WHERE pk > 0 ORDER BY pk`,
    ).all<{ name: string; pk: number }>();

    expect(results.map((r) => r.name)).toEqual(['asset', 'as_of', 'model_version']);
  });

  it('keys engine_output on (asset, metric, as_of) so re-runs upsert (0004)', async () => {
    const { results } = await env.DB.prepare(
      `SELECT name, pk FROM pragma_table_info('engine_output') WHERE pk > 0 ORDER BY pk`,
    ).all<{ name: string; pk: number }>();

    expect(results.map((r) => r.name)).toEqual(['asset', 'metric', 'as_of']);
  });

  it('adds an asset column to the v1 output tables, defaulting to SPX (0004)', async () => {
    // Additive only: existing rows are S&P500 by definition, so the default
    // makes the migration safe for queries that never mention the column.
    // derived_features (0005), regime_state (0006) and forecast_distribution
    // (0007) are excluded — each was rebuilt with asset in the primary key and a
    // required value; see below.
    for (const table of ['force_scores', 'instability_index']) {
      const row = await env.DB.prepare(
        `SELECT dflt_value FROM pragma_table_info('${table}') WHERE name = 'asset'`,
      ).first<{ dflt_value: string }>();
      expect(row?.dflt_value, `${table}.asset`).toBe("'SPX'");
    }
  });

  it('keys derived_features by asset as well as date and model (0005)', async () => {
    // 0004 could add the column but not extend the key, and `velocity` is
    // exactly the kind of feature name two engines would both want: without
    // asset in the key they would upsert over each other and the loser would
    // look like missing data.
    const { results } = await env.DB.prepare(
      `SELECT name FROM pragma_table_info('derived_features') WHERE pk > 0 ORDER BY pk`,
    ).all<{ name: string }>();

    expect(results.map((r) => r.name)).toEqual(['asset', 'date', 'feature', 'model_version']);
  });

  it('keys regime_state by asset as well (0006)', async () => {
    // `bear` and `crisis` are names a second engine would also publish, and a
    // shared row would silently overwrite equity's posterior.
    const { results } = await env.DB.prepare(
      `SELECT name FROM pragma_table_info('regime_state') WHERE pk > 0 ORDER BY pk`,
    ).all<{ name: string }>();

    expect(results.map((r) => r.name)).toEqual(['asset', 'date', 'regime', 'model_version']);
  });

  it('keys forecast_distribution by asset as well (0007)', async () => {
    // `tactical` and `0.5` are the least engine-specific words in the schema.
    // Under the old key a second engine's p50 landed on equity's row and won by
    // whichever job ran second — with no error, no count change, and no way to
    // tell from the data which asset the number belonged to.
    const { results } = await env.DB.prepare(
      `SELECT name FROM pragma_table_info('forecast_distribution') WHERE pk > 0 ORDER BY pk`,
    ).all<{ name: string }>();

    expect(results.map((r) => r.name)).toEqual([
      'asset',
      'as_of',
      'horizon',
      'quantile',
      'model_version',
    ]);
  });

  it('has no column a point forecast could be written to (§0 non-goal 1)', async () => {
    // Structural, not conventional. "No deterministic price target" survives a
    // reviewer who thinks one would be convenient only if there is nowhere to
    // put it.
    const { results } = await env.DB.prepare(
      `SELECT name FROM pragma_table_info('forecast_distribution')`,
    ).all<{ name: string }>();

    const names = results.map((r) => r.name);
    expect(names).toContain('quantile');
    expect(names).not.toContain('target');
    expect(names).not.toContain('point_estimate');
  });

  it('accepts two engines publishing the same feature name on one date', async () => {
    const insert = `INSERT INTO derived_features
        (asset, date, feature, value, model_version, computed_at)
      VALUES (?, '2026-07-29', 'velocity', ?, 'v1', '2026-07-30T00:00:00Z')`;
    await env.DB.prepare(insert).bind('equity', 0.11).run();
    await env.DB.prepare(insert).bind('rates', 0.22).run();

    const { results } = await env.DB.prepare(
      `SELECT asset, value FROM derived_features WHERE feature = 'velocity' ORDER BY asset`,
    ).all<{ asset: string; value: number }>();

    expect(results).toEqual([
      { asset: 'equity', value: 0.11 },
      { asset: 'rates', value: 0.22 },
    ]);
  });

  it('seeds tradable proxies for every jurisdiction (§7, §12)', async () => {
    const { results } = await env.DB.prepare(
      `SELECT DISTINCT jurisdiction FROM tradable_proxy_mapping ORDER BY jurisdiction`,
    ).all<{ jurisdiction: string }>();

    expect(results.map((r) => r.jurisdiction)).toEqual(['EU_UCITS', 'LATAM_DEFAULT', 'US']);
  });
});

describe('observation-level quality flags (0008)', () => {
  beforeEach(async () => {
    await env.DB.prepare('DELETE FROM macro_series').run();
  });

  /**
   * The distinction this column exists to draw: a defect that belongs to the
   * SERIES blocks ingestion upstream, because it leaves no trustworthy subset.
   * A single extreme observation leaves the rest of the series perfectly usable,
   * and in macro data it is usually not an error at all — March 2020 and
   * September 2008 are the history this system is for.
   */
  it('stores a flag on the row and serves it back', async () => {
    await applyWriteBack(
      env,
      validatePayload({
        observations: [
          {
            series_id: 'FRED:UNRATE',
            obs_date: '2020-04-01',
            release_date: '2020-05-08',
            value: 14.7,
            source: 'fred',
            quality_flag: 'abnormal_jump',
          },
          {
            series_id: 'FRED:UNRATE',
            obs_date: '2020-03-01',
            release_date: '2020-04-03',
            value: 4.4,
            source: 'fred',
          },
        ],
      }),
    );

    const { results } = await env.DB.prepare(
      `SELECT obs_date, value, quality_flag FROM macro_series
        WHERE series_id = 'FRED:UNRATE' ORDER BY obs_date`,
    ).all<{ obs_date: string; value: number; quality_flag: string | null }>();

    // Both rows are present. The extreme one is marked, not missing.
    expect(results).toEqual([
      { obs_date: '2020-03-01', value: 4.4, quality_flag: null },
      { obs_date: '2020-04-01', value: 14.7, quality_flag: 'abnormal_jump' },
    ]);
  });

  it('rejects a flag outside the vocabulary', () => {
    // Bounded so a consumer can branch on it. Free text would become a place to
    // put messages nobody parses.
    expect(() =>
      validatePayload({
        observations: [
          {
            series_id: 'FRED:UNRATE',
            obs_date: '2020-04-01',
            release_date: '2020-05-08',
            value: 14.7,
            source: 'fred',
            quality_flag: 'looks_weird_to_me',
          },
        ],
      }),
    ).toThrow(/abnormal_jump/);
  });

  it('treats an absent flag as "nothing to say", not as a value', async () => {
    await applyWriteBack(
      env,
      validatePayload({
        observations: [
          {
            series_id: 'FRED:DGS10',
            obs_date: '2026-01-02',
            release_date: '2026-01-03',
            value: 4.2,
            source: 'fred',
          },
        ],
      }),
    );

    const row = await env.DB.prepare(
      `SELECT quality_flag FROM macro_series WHERE series_id = 'FRED:DGS10'`,
    ).first<{ quality_flag: string | null }>();

    expect(row?.quality_flag).toBeNull();
  });
});
