import type { Env } from '../types';
import { isStale } from '../lib/responses';

export interface SourceHealth {
  source: string;
  status: string;
  last_run_at: string;
  rows_written: number;
  error: string | null;
}

export interface HealthReport {
  ok: boolean;
  env: string;
  info_set: string;
  last_ingestion_at: string | null;
  stale: boolean;
  sources: SourceHealth[];
  degraded_reason?: string;
}

interface HealthRow {
  source: string;
  status: string;
  last_run_at: string;
  rows_written: number | null;
  error: string | null;
}

/**
 * Per-source ingestion status (FINDYN_V1_SPEC.md §13 /health).
 *
 * Degradation contract (§14.2): a database failure must not produce a 5xx.
 * The endpoint reports the failure as data so monitoring can see it.
 */
export async function getHealth(env: Env): Promise<HealthReport> {
  const base: HealthReport = {
    ok: true,
    env: env.FINDYN_ENV,
    info_set: env.INFO_SET,
    last_ingestion_at: null,
    stale: true,
    sources: [],
  };

  try {
    const { results } = await env.DB.prepare(
      `SELECT source,
              status,
              MAX(run_at)         AS last_run_at,
              SUM(rows_written)   AS rows_written,
              MAX(error)          AS error
         FROM ingestion_log
        GROUP BY source, status
        ORDER BY last_run_at DESC`,
    ).all<HealthRow>();

    const sources: SourceHealth[] = (results ?? []).map((r) => ({
      source: r.source,
      status: r.status,
      last_run_at: r.last_run_at,
      rows_written: r.rows_written ?? 0,
      error: r.error,
    }));

    const lastIngestionAt =
      sources.reduce<string | null>(
        (max, s) => (max === null || s.last_run_at > max ? s.last_run_at : max),
        null,
      ) ?? null;

    return {
      ...base,
      ok: sources.every((s) => s.status !== 'failed'),
      last_ingestion_at: lastIngestionAt,
      stale: isStale(lastIngestionAt),
      sources,
    };
  } catch (err) {
    return {
      ...base,
      ok: false,
      degraded_reason: err instanceof Error ? err.message : String(err),
    };
  }
}
