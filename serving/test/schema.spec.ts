import { env } from 'cloudflare:test';
import { describe, expect, it } from 'vitest';

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
    // derived_features is excluded — 0005 rebuilt it with asset in the primary
    // key and a required value; see the test below.
    for (const table of [
      'force_scores',
      'regime_state',
      'instability_index',
      'forecast_distribution',
    ]) {
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
