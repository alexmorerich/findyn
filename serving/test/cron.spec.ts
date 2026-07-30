import { describe, expect, it } from 'vitest';
import { CRON_JOBS, resolveJob } from '../src/ingest/scheduled';

// The cross-check that every cron in wrangler.jsonc has a job (and vice versa)
// runs in scripts/check-crons.mjs — wrangler.jsonc cannot be imported into the
// workerd test runtime, which has no filesystem and no JSONC parser.

describe('cron dispatch (FINDYN_V1_SPEC.md §6)', () => {
  it('covers all four ingestion cadences', () => {
    expect(new Set(Object.values(CRON_JOBS))).toEqual(
      new Set(['realtime_cache', 'daily_macro', 'daily_market_close', 'weekly_slow']),
    );
  });

  it('resolves each declared cron to its job', () => {
    expect(resolveJob('30 22 * * 1-5')).toBe('daily_market_close');
    expect(resolveJob('0 6 * * 1')).toBe('weekly_slow');
  });

  it('returns null for an unknown cron rather than guessing', () => {
    expect(resolveJob('0 0 1 1 *')).toBeNull();
  });
});
