import type { Env } from '../types';
import { ASSETS, FORCES } from '../domain';

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

/**
 * P1 batch types (docs/redesign/03-contracts.md §6). The design sketch showed
 * these as separate `{kind, rows}` envelopes; they are carried as named arrays
 * on the existing envelope instead, so one signed request can hold a whole run
 * and there is exactly one payload shape rather than two.
 */
export interface SignalRow {
  name: string;
  value: number;
  direction: -1 | 0 | 1;
  note?: string | null;
}

export interface AssetStateRow {
  asset: string;
  as_of: string;
  model_version: string;
  regime: string;
  expected_return: number | null;
  risk_score: number;
  confidence: number;
  signals: SignalRow[];
  components?: Record<string, number> | null;
}

export interface EngineOutputRow {
  asset: string;
  metric: string;
  as_of: string;
  value: number;
  meta?: unknown;
}

export interface FactorRow {
  force: string;
  as_of: string;
  score: number;
  components?: Record<string, number> | null;
}

/**
 * P3-A: the model inputs an engine was fitted on.
 *
 * Near-identical to EngineOutputRow, and the difference is the point:
 * `model_version` is part of the primary key here. `engine_output` is what the
 * dashboard draws, so the newest run owns each date; a feature is what a model
 * *saw*, so a refit lands beside the old features rather than on top of them.
 *
 * Feature names are deliberately not checked against a closed vocabulary. The
 * fixed five come from FINDYN_V1_SPEC.md §2.1, but the momentum windows are
 * configurable in `config/engines/equity.yaml`, so their names are decided by
 * the compute plane at runtime. Rejecting an unknown one here would mean a yaml
 * edit could only ship together with a Worker deploy.
 */
export interface DerivedFeatureRow {
  asset: string;
  feature: string;
  as_of: string;
  value: number;
  model_version: string;
}

export interface WriteBackPayload {
  model_version?: string;
  generated_at?: string;
  metadata?: SeriesMetadataRow[];
  observations?: ObservationRow[];
  prices?: PriceRow[];
  quality?: QualityRow[];
  ingestion?: IngestionRow[];
  factors?: FactorRow[];
  asset_state?: AssetStateRow[];
  engine_output?: EngineOutputRow[];
  derived_features?: DerivedFeatureRow[];
}

export interface WriteBackResult {
  metadata: number;
  observations: number;
  prices: number;
  quality: number;
  ingestion: number;
  factors: number;
  asset_state: number;
  engine_output: number;
  derived_features: number;
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

function requireRange(value: unknown, lo: number, hi: number, field: string): number {
  const n = requireNumber(value, field);
  if (n < lo || n > hi) {
    throw new PayloadError(`${field} must be within [${lo}, ${hi}], got ${n}`);
  }
  return n;
}

function requireMember(value: unknown, allowed: readonly string[], field: string): string {
  const s = requireString(value, field);
  if (!allowed.includes(s)) {
    throw new PayloadError(`${field} must be one of ${allowed.join('|')}, got ${s}`);
  }
  return s;
}

/**
 * Explanation traces are free-form maps, but every value must still be a finite
 * number: a NaN in `components` renders as an empty cell on the dashboard and
 * looks like missing data rather than a bug.
 */
function optionalComponents(value: unknown, field: string): Record<string, number> | null {
  if (value === undefined || value === null) return null;
  if (typeof value !== 'object' || Array.isArray(value)) {
    throw new PayloadError(`${field} must be an object`);
  }
  const out: Record<string, number> = {};
  for (const [key, raw] of Object.entries(value as Record<string, unknown>)) {
    out[key] = requireNumber(raw, `${field}.${key}`);
  }
  return out;
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

  const modelVersion = typeof p.model_version === 'string' ? p.model_version : 'unversioned';

  const factors = (p.factors as FactorRow[] | undefined)?.map((f, i) => ({
    force: requireMember(f?.force, FORCES, `factors[${i}].force`),
    as_of: requireDate(f?.as_of, `factors[${i}].as_of`),
    score: requireRange(f?.score, 0, 100, `factors[${i}].score`),
    components: optionalComponents(f?.components, `factors[${i}].components`),
  }));

  // The engine vocabulary is closed. An unknown asset means the compute plane
  // and this plane disagree about what exists, and writing it would create a
  // row no endpoint can ever serve.
  const assetState = (p.asset_state as AssetStateRow[] | undefined)?.map((s, i) => ({
    asset: requireMember(s?.asset, ASSETS, `asset_state[${i}].asset`),
    as_of: requireDate(s?.as_of, `asset_state[${i}].as_of`),
    model_version: requireString(s?.model_version, `asset_state[${i}].model_version`),
    regime: requireString(s?.regime, `asset_state[${i}].regime`),
    expected_return:
      s?.expected_return === undefined || s?.expected_return === null
        ? null
        : requireNumber(s.expected_return, `asset_state[${i}].expected_return`),
    risk_score: requireRange(s?.risk_score, 0, 100, `asset_state[${i}].risk_score`),
    confidence: requireRange(s?.confidence, 0, 1, `asset_state[${i}].confidence`),
    signals: (s?.signals ?? []).map((sig, j) => {
      const direction = requireNumber(sig?.direction, `asset_state[${i}].signals[${j}].direction`);
      if (direction !== -1 && direction !== 0 && direction !== 1) {
        throw new PayloadError(`asset_state[${i}].signals[${j}].direction must be -1, 0 or 1`);
      }
      return {
        name: requireString(sig?.name, `asset_state[${i}].signals[${j}].name`),
        value: requireNumber(sig?.value, `asset_state[${i}].signals[${j}].value`),
        direction: direction as -1 | 0 | 1,
        note: sig?.note ?? null,
      };
    }),
    components: optionalComponents(s?.components, `asset_state[${i}].components`),
  }));

  const engineOutput = (p.engine_output as EngineOutputRow[] | undefined)?.map((o, i) => ({
    asset: requireMember(o?.asset, ASSETS, `engine_output[${i}].asset`),
    metric: requireString(o?.metric, `engine_output[${i}].metric`),
    as_of: requireDate(o?.as_of, `engine_output[${i}].as_of`),
    value: requireNumber(o?.value, `engine_output[${i}].value`),
    meta: o?.meta ?? null,
  }));

  const derivedFeatures = (p.derived_features as DerivedFeatureRow[] | undefined)?.map((f, i) => ({
    asset: requireMember(f?.asset, ASSETS, `derived_features[${i}].asset`),
    feature: requireString(f?.feature, `derived_features[${i}].feature`),
    as_of: requireDate(f?.as_of, `derived_features[${i}].as_of`),
    value: requireNumber(f?.value, `derived_features[${i}].value`),
    // Per row, not from the envelope: a run publishing two engines carries two
    // versions, and this column is part of the key.
    model_version: requireString(f?.model_version, `derived_features[${i}].model_version`),
  }));

  return {
    model_version: modelVersion,
    generated_at: typeof p.generated_at === 'string' ? p.generated_at : new Date().toISOString(),
    metadata,
    observations,
    prices,
    quality,
    ingestion,
    factors,
    asset_state: assetState,
    engine_output: engineOutput,
    derived_features: derivedFeatures,
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

  const factorStmt = db.prepare(
    `INSERT INTO force_scores (date, force, score, components, model_version)
     VALUES (?, ?, ?, ?, ?)
     ON CONFLICT(date, force, model_version) DO UPDATE SET
       score = excluded.score, components = excluded.components`,
  );

  // Keyed on (asset, as_of, model_version), so re-running a day replaces that
  // day's state for that model and leaves other models' history alone.
  const stateStmt = db.prepare(
    `INSERT INTO asset_state
       (asset, as_of, model_version, regime, expected_return, risk_score,
        confidence, signals, components, written_at)
     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
     ON CONFLICT(asset, as_of, model_version) DO UPDATE SET
       regime = excluded.regime, expected_return = excluded.expected_return,
       risk_score = excluded.risk_score, confidence = excluded.confidence,
       signals = excluded.signals, components = excluded.components,
       written_at = excluded.written_at`,
  );

  const outputStmt = db.prepare(
    `INSERT INTO engine_output (asset, metric, as_of, value, meta, written_at)
     VALUES (?, ?, ?, ?, ?, ?)
     ON CONFLICT(asset, metric, as_of) DO UPDATE SET
       value = excluded.value, meta = excluded.meta, written_at = excluded.written_at`,
  );

  const featureStmt = db.prepare(
    `INSERT INTO derived_features (asset, date, feature, value, model_version, computed_at)
     VALUES (?, ?, ?, ?, ?, ?)
     ON CONFLICT(asset, date, feature, model_version) DO UPDATE SET
       value = excluded.value, computed_at = excluded.computed_at`,
  );

  const result: WriteBackResult = {
    metadata: 0,
    observations: 0,
    prices: 0,
    quality: 0,
    ingestion: 0,
    factors: 0,
    asset_state: 0,
    engine_output: 0,
    derived_features: 0,
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

  if (payload.factors?.length) {
    result.factors = await runBatched(
      db,
      payload.factors.map((f) =>
        factorStmt.bind(
          f.as_of,
          f.force,
          f.score,
          f.components ? JSON.stringify(f.components) : null,
          payload.model_version ?? 'unversioned',
        ),
      ),
    );
  }

  if (payload.asset_state?.length) {
    result.asset_state = await runBatched(
      db,
      payload.asset_state.map((s) =>
        stateStmt.bind(
          s.asset,
          s.as_of,
          s.model_version,
          s.regime,
          s.expected_return,
          s.risk_score,
          s.confidence,
          JSON.stringify(s.signals ?? []),
          s.components ? JSON.stringify(s.components) : null,
          now,
        ),
      ),
    );
  }

  if (payload.engine_output?.length) {
    result.engine_output = await runBatched(
      db,
      payload.engine_output.map((o) =>
        outputStmt.bind(
          o.asset,
          o.metric,
          o.as_of,
          o.value,
          o.meta === null || o.meta === undefined ? null : JSON.stringify(o.meta),
          now,
        ),
      ),
    );
  }

  if (payload.derived_features?.length) {
    result.derived_features = await runBatched(
      db,
      payload.derived_features.map((f) =>
        featureStmt.bind(f.asset, f.as_of, f.feature, f.value, f.model_version, now),
      ),
    );
  }

  return result;
}
