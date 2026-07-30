import type { Env } from '../types';

/**
 * Compute-plane write-back (FINDYN_V1_SPEC.md §6).
 *
 * The Python plane owns provider access and model computation; this is the only
 * door its results come through. Everything is upserted so a re-run of the same
 * job is idempotent — reruns after a partial failure are routine, and duplicate
 * rows would quietly corrupt point-in-time reads.
 */

/** D1 free tier caps a batch at 1000 statements. */
const MAX_BATCH = 900;

export interface SeriesMetadataRow {
  series_id: string;
  provider: string;
  title: string;
  frequency: string;
  unit: string;
  seasonal_adjustment?: string | null;
  first_observation?: string | null;
  last_observation?: string | null;
  notes?: string | null;
}

export interface ObservationRow {
  series_id: string;
  obs_date: string;
  release_date: string;
  revision_date?: string | null;
  value: number;
  vintage?: string | null;
  source: string;
}

export interface PriceRow {
  date: string;
  symbol: string;
  close: number;
  volume?: number | null;
  source: string;
}

export interface QualityRow {
  series_id: string;
  provider: string;
  status: 'ok' | 'warning' | 'error';
  observations: number;
  warnings?: unknown[];
  errors?: unknown[];
  checked_range?: string | null;
}

export interface IngestionRow {
  source: string;
  series_id?: string | null;
  status: 'ok' | 'degraded' | 'failed';
  rows_written?: number;
  error?: string | null;
}

export interface WriteBackPayload {
  model_version?: string;
  generated_at?: string;
  metadata?: SeriesMetadataRow[];
  observations?: ObservationRow[];
  prices?: PriceRow[];
  quality?: QualityRow[];
  ingestion?: IngestionRow[];
}

export interface WriteBackResult {
  metadata: number;
  observations: number;
  prices: number;
  quality: number;
  ingestion: number;
}

export class PayloadError extends Error {}

function requireString(value: unknown, field: string): string {
  if (typeof value !== 'string' || value.length === 0) {
    throw new PayloadError(`${field} must be a non-empty string`);
  }
  return value;
}

function requireDate(value: unknown, field: string): string {
  const s = requireString(value, field);
  if (!/^\d{4}-\d{2}-\d{2}$/.test(s)) {
    throw new PayloadError(`${field} must be YYYY-MM-DD, got ${s}`);
  }
  return s;
}

function requireNumber(value: unknown, field: string): number {
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    throw new PayloadError(`${field} must be a finite number`);
  }
  return value;
}

function optionalDate(value: unknown, field: string): string | null {
  if (value === undefined || value === null) return null;
  return requireDate(value, field);
}

/**
 * Validate before touching the database. A half-applied batch is worse than a
 * rejected one, because the next run's upsert would hide the gap.
 */
export function validatePayload(raw: unknown): WriteBackPayload {
  if (typeof raw !== 'object' || raw === null) {
    throw new PayloadError('payload must be a JSON object');
  }
  const p = raw as Record<string, unknown>;

  const metadata = (p.metadata as SeriesMetadataRow[] | undefined)?.map((m, i) => ({
    series_id: requireString(m?.series_id, `metadata[${i}].series_id`),
    provider: requireString(m?.provider, `metadata[${i}].provider`),
    title: requireString(m?.title, `metadata[${i}].title`),
    frequency: requireString(m?.frequency, `metadata[${i}].frequency`),
    unit: requireString(m?.unit, `metadata[${i}].unit`),
    seasonal_adjustment: m?.seasonal_adjustment ?? null,
    first_observation: optionalDate(m?.first_observation, `metadata[${i}].first_observation`),
    last_observation: optionalDate(m?.last_observation, `metadata[${i}].last_observation`),
    notes: m?.notes ?? null,
  }));

  const observations = (p.observations as ObservationRow[] | undefined)?.map((o, i) => {
    const obsDate = requireDate(o?.obs_date, `observations[${i}].obs_date`);
    const releaseDate = requireDate(o?.release_date, `observations[${i}].release_date`);
    // The point-in-time invariant, enforced at the boundary rather than trusted:
    // a value published before the period it describes would license lookahead.
    if (releaseDate < obsDate) {
      throw new PayloadError(
        `observations[${i}]: release_date ${releaseDate} precedes obs_date ${obsDate}`,
      );
    }
    // A source with no vintage information issues each period exactly once,
    // so its revision date is its release date.
    const revisionDate =
      optionalDate(o?.revision_date, `observations[${i}].revision_date`) ?? releaseDate;
    if (revisionDate < releaseDate) {
      throw new PayloadError(
        `observations[${i}]: revision_date ${revisionDate} precedes release_date ${releaseDate}`,
      );
    }
    return {
      series_id: requireString(o?.series_id, `observations[${i}].series_id`),
      obs_date: obsDate,
      release_date: releaseDate,
      revision_date: revisionDate,
      value: requireNumber(o?.value, `observations[${i}].value`),
      vintage: o?.vintage ?? null,
      source: requireString(o?.source, `observations[${i}].source`),
    };
  });

  const prices = (p.prices as PriceRow[] | undefined)?.map((r, i) => ({
    date: requireDate(r?.date, `prices[${i}].date`),
    symbol: requireString(r?.symbol, `prices[${i}].symbol`),
    close: requireNumber(r?.close, `prices[${i}].close`),
    volume: r?.volume ?? null,
    source: requireString(r?.source, `prices[${i}].source`),
  }));

  const quality = (p.quality as QualityRow[] | undefined)?.map((q, i) => {
    const status = requireString(q?.status, `quality[${i}].status`);
    if (!['ok', 'warning', 'error'].includes(status)) {
      throw new PayloadError(`quality[${i}].status must be ok|warning|error`);
    }
    return {
      series_id: requireString(q?.series_id, `quality[${i}].series_id`),
      provider: requireString(q?.provider, `quality[${i}].provider`),
      status: status as QualityRow['status'],
      observations: q?.observations ?? 0,
      warnings: q?.warnings ?? [],
      errors: q?.errors ?? [],
      checked_range: q?.checked_range ?? null,
    };
  });

  const ingestion = (p.ingestion as IngestionRow[] | undefined)?.map((g, i) => ({
    source: requireString(g?.source, `ingestion[${i}].source`),
    series_id: g?.series_id ?? null,
    status: requireString(g?.status, `ingestion[${i}].status`) as IngestionRow['status'],
    rows_written: g?.rows_written ?? 0,
    error: g?.error ?? null,
  }));

  return {
    model_version: typeof p.model_version === 'string' ? p.model_version : 'unversioned',
    generated_at: typeof p.generated_at === 'string' ? p.generated_at : new Date().toISOString(),
    metadata,
    observations,
    prices,
    quality,
    ingestion,
  };
}

async function runBatched(db: D1Database, statements: D1PreparedStatement[]): Promise<number> {
  for (let i = 0; i < statements.length; i += MAX_BATCH) {
    await db.batch(statements.slice(i, i + MAX_BATCH));
  }
  return statements.length;
}

export async function applyWriteBack(env: Env, payload: WriteBackPayload): Promise<WriteBackResult> {
  const now = new Date().toISOString();
  const db = env.DB;

  const metaStmt = db.prepare(
    `INSERT INTO series_metadata
       (series_id, provider, title, frequency, unit, seasonal_adjustment,
        first_observation, last_observation, notes, updated_at)
     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
     ON CONFLICT(series_id) DO UPDATE SET
       provider = excluded.provider, title = excluded.title,
       frequency = excluded.frequency, unit = excluded.unit,
       seasonal_adjustment = excluded.seasonal_adjustment,
       first_observation = excluded.first_observation,
       last_observation = excluded.last_observation,
       notes = excluded.notes, updated_at = excluded.updated_at`,
  );

  // Keyed on the vintage, so re-ingesting a series updates each figure in place
  // instead of collapsing every revision of a period onto one row.
  const obsStmt = db.prepare(
    `INSERT INTO macro_series
       (series_id, obs_date, release_date, revision_date, value, vintage, source, ingested_at)
     VALUES (?, ?, ?, ?, ?, ?, ?, ?)
     ON CONFLICT(series_id, obs_date, revision_date) DO UPDATE SET
       release_date = excluded.release_date, value = excluded.value,
       vintage = excluded.vintage, ingested_at = excluded.ingested_at`,
  );

  const priceStmt = db.prepare(
    `INSERT INTO market_price (date, symbol, close, volume, source, ingested_at)
     VALUES (?, ?, ?, ?, ?, ?)
     ON CONFLICT(date, symbol, source) DO UPDATE SET
       close = excluded.close, volume = excluded.volume, ingested_at = excluded.ingested_at`,
  );

  const dqStmt = db.prepare(
    `INSERT INTO data_quality_report
       (run_at, series_id, provider, status, observations, warnings, errors, checked_range)
     VALUES (?, ?, ?, ?, ?, ?, ?, ?)`,
  );

  const logStmt = db.prepare(
    `INSERT INTO ingestion_log (run_at, source, series_id, status, rows_written, error)
     VALUES (?, ?, ?, ?, ?, ?)`,
  );

  const result: WriteBackResult = {
    metadata: 0,
    observations: 0,
    prices: 0,
    quality: 0,
    ingestion: 0,
  };

  if (payload.metadata?.length) {
    result.metadata = await runBatched(
      db,
      payload.metadata.map((m) =>
        metaStmt.bind(
          m.series_id,
          m.provider,
          m.title,
          m.frequency,
          m.unit,
          m.seasonal_adjustment ?? null,
          m.first_observation ?? null,
          m.last_observation ?? null,
          m.notes ?? null,
          now,
        ),
      ),
    );
  }

  if (payload.observations?.length) {
    result.observations = await runBatched(
      db,
      payload.observations.map((o) =>
        obsStmt.bind(
          o.series_id,
          o.obs_date,
          o.release_date,
          o.revision_date ?? o.release_date,
          o.value,
          o.vintage ?? null,
          o.source,
          now,
        ),
      ),
    );
  }

  if (payload.prices?.length) {
    result.prices = await runBatched(
      db,
      payload.prices.map((r) =>
        priceStmt.bind(r.date, r.symbol, r.close, r.volume ?? null, r.source, now),
      ),
    );
  }

  if (payload.quality?.length) {
    result.quality = await runBatched(
      db,
      payload.quality.map((q) =>
        dqStmt.bind(
          now,
          q.series_id,
          q.provider,
          q.status,
          q.observations,
          JSON.stringify(q.warnings ?? []),
          JSON.stringify(q.errors ?? []),
          q.checked_range ?? null,
        ),
      ),
    );
  }

  if (payload.ingestion?.length) {
    result.ingestion = await runBatched(
      db,
      payload.ingestion.map((g) =>
        logStmt.bind(now, g.source, g.series_id ?? null, g.status, g.rows_written ?? 0, g.error ?? null),
      ),
    );
  }

  return result;
}
