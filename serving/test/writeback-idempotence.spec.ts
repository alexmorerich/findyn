import { env } from 'cloudflare:test';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { applyWriteBack, validatePayload } from '../src/admin/writeback';

/**
 * The nightly republish must not be billed as a write.
 *
 * Each engine offers a multi-year window of every metric it charts on every
 * run (`history_days`), so a normal night sends ~125k rows of which a handful
 * are new. An unguarded `DO UPDATE` rewrites all of them — D1 counts that as
 * 125k rows written and the free tier allows 100k/day, which is how the
 * database came to reject every write mid-month.
 *
 * `written_at` is the probe: the guard leaves it alone precisely when it
 * skipped the row, so an unchanged timestamp is the observable proof that no
 * write happened, and a moved one proves a real change still lands.
 */

const AS_OF = '2026-07-28';

function payload(value: number, meta: Record<string, unknown> | null = null) {
  return validatePayload({
    model_version: 'rates-1.0.0',
    generated_at: '2026-07-30T00:00:00Z',
    engine_output: [{ asset: 'rates', metric: 'ns_level', as_of: AS_OF, value, meta }],
    derived_features: [
      { asset: 'rates', as_of: AS_OF, feature: 'ns_slope', value, model_version: 'rates-1.0.0' },
    ],
    regime_state: [
      {
        asset: 'rates',
        as_of: AS_OF,
        regime: 're_steepening',
        probability: 0.5,
        model_version: 'rates-1.0.0',
      },
    ],
  });
}

async function stamps() {
  const output = await env.DB.prepare(
    'SELECT value, written_at FROM engine_output WHERE asset = ? AND metric = ? AND as_of = ?',
  )
    .bind('rates', 'ns_level', AS_OF)
    .first<{ value: number; written_at: string }>();
  const feature = await env.DB.prepare(
    'SELECT value, computed_at FROM derived_features WHERE asset = ? AND feature = ? AND date = ?',
  )
    .bind('rates', 'ns_slope', AS_OF)
    .first<{ value: number; computed_at: string }>();
  return { output, feature };
}

describe('write-back skips unchanged rows (D1 row-write budget)', () => {
  beforeEach(async () => {
    await env.DB.batch([
      env.DB.prepare('DELETE FROM engine_output'),
      env.DB.prepare('DELETE FROM derived_features'),
      env.DB.prepare('DELETE FROM regime_state'),
    ]);
    vi.useRealTimers();
  });

  it('leaves the row — and its timestamp — untouched when every value is identical', async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date('2026-07-30T03:00:00Z'));
    await applyWriteBack(env, payload(2.167026));
    const first = await stamps();
    expect(first.output?.written_at).toBe('2026-07-30T03:00:00.000Z');

    // The next night offers the same history again.
    vi.setSystemTime(new Date('2026-07-31T03:00:00Z'));
    await applyWriteBack(env, payload(2.167026));
    const second = await stamps();

    expect(second.output?.written_at).toBe(first.output?.written_at);
    expect(second.feature?.computed_at).toBe(first.feature?.computed_at);
  });

  it('still writes a revised value, and a changed meta blob', async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date('2026-07-30T03:00:00Z'));
    await applyWriteBack(env, payload(2.167026));

    vi.setSystemTime(new Date('2026-07-31T03:00:00Z'));
    await applyWriteBack(env, payload(2.9));
    const revised = await stamps();
    expect(revised.output?.value).toBeCloseTo(2.9);
    expect(revised.output?.written_at).toBe('2026-07-31T03:00:00.000Z');
    expect(revised.feature?.value).toBeCloseTo(2.9);
    expect(revised.feature?.computed_at).toBe('2026-07-31T03:00:00.000Z');

    // Same value, different meta — still a change.
    vi.setSystemTime(new Date('2026-08-01T03:00:00Z'));
    await applyWriteBack(env, payload(2.9, { regime: 'steep_tightening' }));
    const remeta = await stamps();
    expect(remeta.output?.written_at).toBe('2026-08-01T03:00:00.000Z');
  });

  it('inserts a genuinely new date', async () => {
    await applyWriteBack(env, payload(2.167026));
    const next = validatePayload({
      model_version: 'rates-1.0.0',
      generated_at: '2026-07-31T00:00:00Z',
      engine_output: [
        { asset: 'rates', metric: 'ns_level', as_of: AS_OF, value: 2.167026, meta: null },
        { asset: 'rates', metric: 'ns_level', as_of: '2026-07-29', value: 2.2, meta: null },
      ],
    });
    await applyWriteBack(env, next);

    const row = await env.DB.prepare(
      'SELECT COUNT(*) AS n FROM engine_output WHERE asset = ? AND metric = ?',
    )
      .bind('rates', 'ns_level')
      .first<{ n: number }>();
    expect(row?.n).toBe(2);
  });
});
