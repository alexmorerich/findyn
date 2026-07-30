import { env } from 'cloudflare:test';
import { describe, expect, it } from 'vitest';

const EXPECTED_TABLES = [
  'data_quality_report',
  'derived_features',
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

  it('seeds tradable proxies for every jurisdiction (§7, §12)', async () => {
    const { results } = await env.DB.prepare(
      `SELECT DISTINCT jurisdiction FROM tradable_proxy_mapping ORDER BY jurisdiction`,
    ).all<{ jurisdiction: string }>();

    expect(results.map((r) => r.jurisdiction)).toEqual(['EU_UCITS', 'LATAM_DEFAULT', 'US']);
  });
});
