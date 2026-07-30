import type { Env } from '../types';

/**
 * Cron dispatch — FINDYN_V1_SPEC.md §6 ("Ingestion strategy").
 * Cron expressions are declared in wrangler.jsonc; this maps each to a job.
 */
export type JobName = 'realtime_cache' | 'daily_macro' | 'daily_market_close' | 'weekly_slow';

export const CRON_JOBS: Readonly<Record<string, JobName>> = {
  '*/15 13-21 * * 1-5': 'realtime_cache',
  '0 13 * * *': 'daily_macro',
  '30 22 * * 1-5': 'daily_market_close',
  '0 6 * * 1': 'weekly_slow',
};

export function resolveJob(cron: string): JobName | null {
  return CRON_JOBS[cron] ?? null;
}

/**
 * A scheduled run must never throw: an unhandled error in one job would abort
 * the whole invocation and leave no trace of what failed. Every outcome is
 * written to `ingestion_log` instead (§14.2).
 */
export async function runScheduled(event: ScheduledController, env: Env): Promise<void> {
  const job = resolveJob(event.cron);
  const runAt = new Date(event.scheduledTime).toISOString();

  if (!job) {
    await logIngestion(env, { runAt, source: 'cron', status: 'failed', error: `unmapped cron: ${event.cron}` });
    return;
  }

  try {
    // M1 wires the provider adapters behind each job. Until then the dispatch
    // path itself is exercised and recorded, so cron wiring is verifiable.
    await logIngestion(env, {
      runAt,
      source: `cron:${job}`,
      status: 'degraded',
      error: 'ingestion adapters land in M1',
    });
  } catch (err) {
    await logIngestion(env, {
      runAt,
      source: `cron:${job}`,
      status: 'failed',
      error: err instanceof Error ? err.message : String(err),
    });
  }
}

export async function logIngestion(
  env: Env,
  entry: {
    runAt: string;
    source: string;
    seriesId?: string | null;
    status: 'ok' | 'degraded' | 'failed';
    rowsWritten?: number;
    error?: string | null;
  },
): Promise<void> {
  try {
    await env.DB.prepare(
      `INSERT INTO ingestion_log (run_at, source, series_id, status, rows_written, error)
       VALUES (?, ?, ?, ?, ?, ?)`,
    )
      .bind(
        entry.runAt,
        entry.source,
        entry.seriesId ?? null,
        entry.status,
        entry.rowsWritten ?? 0,
        entry.error ?? null,
      )
      .run();
  } catch (err) {
    // Logging must not become the failure. Surface to Workers Logs and move on.
    console.error('ingestion_log write failed', err);
  }
}
