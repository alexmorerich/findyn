/**
 * Display formatting. Nothing here invents a value: every function either
 * formats something the API returned or returns an explicit placeholder.
 */

const PLACEHOLDER = '—';

/** Numbers keep their magnitude readable without implying false precision. */
export function formatValue(value: number | null | undefined): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return PLACEHOLDER;
  const abs = Math.abs(value);
  const digits = abs >= 1000 ? 1 : abs >= 10 ? 2 : abs >= 1 ? 3 : 4;
  return value.toLocaleString('en-US', {
    minimumFractionDigits: 0,
    maximumFractionDigits: digits,
  });
}

/** `YYYY-MM-DD` passes through; ISO timestamps are shown to the minute, UTC. */
export function formatDate(value: string | null | undefined): string {
  if (!value) return PLACEHOLDER;
  if (/^\d{4}-\d{2}-\d{2}$/.test(value)) return value;
  const ts = Date.parse(value);
  if (Number.isNaN(ts)) return value;
  return `${new Date(ts).toISOString().slice(0, 16).replace('T', ' ')} UTC`;
}

export function formatDays(days: number | null | undefined): string {
  if (days === null || days === undefined || !Number.isFinite(days)) return PLACEHOLDER;
  const n = Math.round(days);
  return `${n} ${Math.abs(n) === 1 ? 'day' : 'days'}`;
}

export function formatCount(n: number): string {
  return n.toLocaleString('en-US');
}

export type Tone = 'ok' | 'warn' | 'bad' | 'idle' | 'info';

export interface Freshness {
  label: string;
  tone: Tone;
  /** Spelled out so the badge is auditable rather than a mystery colour. */
  explanation: string;
}

/**
 * Freshness thresholds are frequency-relative: a quarterly series 60 days
 * behind its last period is on schedule, a daily series 60 days behind is not.
 * `freshness_days` is the age of the newest *observed period*, which for macro
 * data is always older than the release that carried it.
 */
const THRESHOLDS: Record<string, { fresh: number; aging: number }> = {
  daily: { fresh: 7, aging: 30 },
  weekly: { fresh: 14, aging: 45 },
  biweekly: { fresh: 30, aging: 75 },
  monthly: { fresh: 60, aging: 120 },
  quarterly: { fresh: 150, aging: 270 },
  annual: { fresh: 450, aging: 730 },
};

const FREQUENCY_ALIASES: Record<string, string> = {
  d: 'daily',
  day: 'daily',
  daily: 'daily',
  w: 'weekly',
  week: 'weekly',
  weekly: 'weekly',
  bw: 'biweekly',
  biweekly: 'biweekly',
  m: 'monthly',
  month: 'monthly',
  monthly: 'monthly',
  q: 'quarterly',
  quarter: 'quarterly',
  quarterly: 'quarterly',
  a: 'annual',
  y: 'annual',
  year: 'annual',
  yearly: 'annual',
  annual: 'annual',
};

export function normaliseFrequency(frequency: string | null | undefined): string | null {
  if (!frequency) return null;
  return FREQUENCY_ALIASES[frequency.trim().toLowerCase()] ?? null;
}

export function freshness(
  freshnessDays: number | null | undefined,
  frequency: string | null | undefined,
): Freshness {
  if (freshnessDays === null || freshnessDays === undefined || !Number.isFinite(freshnessDays)) {
    return {
      label: 'no data',
      tone: 'idle',
      explanation: 'No observation has been ingested for this series yet.',
    };
  }
  const key = normaliseFrequency(frequency) ?? 'monthly';
  const limits = THRESHOLDS[key] ?? THRESHOLDS['monthly']!;
  const days = Math.round(freshnessDays);
  const basis = `Newest observed period is ${formatDays(days)} old; ${key} series are counted fresh under ${limits.fresh} days and aging under ${limits.aging}.`;
  if (days <= limits.fresh) return { label: 'fresh', tone: 'ok', explanation: basis };
  if (days <= limits.aging) return { label: 'aging', tone: 'warn', explanation: basis };
  return { label: 'stale', tone: 'bad', explanation: basis };
}

/** Ingestion-log status strings mapped onto badge tones. */
export function statusTone(status: string): Tone {
  const s = status.toLowerCase();
  if (s === 'ok' || s === 'success') return 'ok';
  if (s === 'degraded' || s === 'partial') return 'warn';
  if (s === 'failed' || s === 'error') return 'bad';
  return 'idle';
}
