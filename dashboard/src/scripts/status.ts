/**
 * System Status page.
 *
 * Reads /meta and /health and reports the three planes of the architecture.
 * Anything the API does not tell us is reported as unknown — the page never
 * invents a status, and never renders blank.
 */
import {
  apiGet,
  getHealth,
  getMeta,
  RESERVED_ENDPOINTS,
  type ApiResult,
  type Health,
  type Meta,
  type SourceHealth,
} from '../lib/api';
import {
  badge,
  defRow,
  el,
  emptyBlock,
  failureBlock,
  loadingBlock,
  replace,
  stateBlock,
  table,
  text,
} from '../lib/dom';
import { formatCount, formatDate, statusTone, type Tone } from '../lib/format';

const tilesHost = document.querySelector('#status-tiles');
const engineHost = document.querySelector('#engine-meta');
const sourcesHost = document.querySelector('#sources');
const roadmapHost = document.querySelector('#roadmap');

function tile(opts: {
  label: string;
  value: string;
  tone: Tone;
  detail?: string;
  badgeText?: string;
}): HTMLElement {
  return el(
    'div',
    { class: `tile tile--${opts.tone}` },
    el('div', { class: 'tile__label' }, opts.label),
    el(
      'div',
      { class: 'tile__value' },
      opts.value,
      opts.badgeText ? ' ' : null,
      opts.badgeText ? badge(opts.badgeText, opts.tone) : null,
    ),
    opts.detail ? el('div', { class: 'tile__detail' }, opts.detail) : null,
  );
}

/** Newest `last_run_at` among sources that actually succeeded. */
function lastSuccessfulRun(sources: SourceHealth[]): SourceHealth | null {
  return sources
    .filter((s) => statusTone(s.status) === 'ok')
    .reduce<SourceHealth | null>(
      (best, s) => (best === null || s.last_run_at > best.last_run_at ? s : best),
      null,
    );
}

function renderTiles(metaRes: ApiResult<Meta>, healthRes: ApiResult<Health>): void {
  if (!tilesHost) return;

  const reachable = metaRes.ok || healthRes.ok;
  const health = healthRes.ok ? healthRes.envelope.data : null;
  const sources = health?.sources ?? [];
  const lastOk = lastSuccessfulRun(sources);
  const failing = sources.filter((s) => statusTone(s.status) === 'bad');

  // Serving plane — can the browser reach the Workers API at all?
  const serving = reachable
    ? tile({
        label: 'Serving plane',
        value: 'reachable',
        tone: 'ok',
        badgeText: 'up',
        detail: metaRes.ok
          ? `Workers API responding · env ${text(metaRes.envelope.data.env)}`
          : 'Workers API responding',
      })
    : tile({
        label: 'Serving plane',
        value: 'unreachable',
        tone: 'bad',
        badgeText: 'down',
        detail: !metaRes.ok ? metaRes.message : 'No response from the Workers API.',
      });

  // Compute plane — inferred from ingestion history, not directly observable.
  let compute: HTMLElement;
  if (!healthRes.ok) {
    compute = tile({
      label: 'Compute plane',
      value: 'unknown',
      tone: 'idle',
      detail: 'Cannot infer ingestion state without /health.',
    });
  } else if (lastOk) {
    compute = tile({
      label: 'Compute plane',
      value: formatDate(lastOk.last_run_at),
      tone: health?.stale ? 'warn' : 'ok',
      badgeText: health?.stale ? 'stale' : 'current',
      detail: `Last successful ingestion, source ${lastOk.source}${
        failing.length > 0 ? ` · ${failing.length} source(s) failing` : ''
      }`,
    });
  } else {
    compute = tile({
      label: 'Compute plane',
      value: 'no successful run',
      tone: sources.length > 0 ? 'warn' : 'idle',
      detail:
        sources.length > 0
          ? 'The ingestion log has entries but none reported success.'
          : 'The ingestion log is empty — no pipeline run has been recorded yet.',
    });
  }

  // D1 — the API reports its own database health rather than 5xx-ing (§14.2).
  let database: HTMLElement;
  if (!healthRes.ok) {
    database = tile({
      label: 'D1 database',
      value: 'unknown',
      tone: 'idle',
      detail: healthRes.message,
    });
  } else if (health?.ok) {
    database = tile({
      label: 'D1 database',
      value: 'ok',
      tone: 'ok',
      badgeText: 'ok',
      detail: `${formatCount(sources.length)} source(s) in the ingestion log`,
    });
  } else {
    database = tile({
      label: 'D1 database',
      value: 'degraded',
      tone: 'bad',
      badgeText: 'degraded',
      detail: health?.degraded_reason ?? 'One or more ingestion sources reported failure.',
    });
  }

  const pipeline = tile({
    label: 'Last successful pipeline run',
    value: lastOk ? formatDate(lastOk.last_run_at) : '—',
    tone: lastOk ? (health?.stale ? 'warn' : 'ok') : 'idle',
    detail: lastOk
      ? `${formatCount(lastOk.rows_written)} row(s) written by ${lastOk.source}`
      : 'Recorded here once an ingestion run completes successfully.',
  });

  replace(tilesHost, serving, compute, database, pipeline);
}

function renderEngine(metaRes: ApiResult<Meta>, healthRes: ApiResult<Health>): void {
  if (!engineHost) return;
  if (!metaRes.ok) {
    replace(engineHost, failureBlock(metaRes, 'Engine metadata'));
    return;
  }
  const m = metaRes.envelope.data;
  const health = healthRes.ok ? healthRes.envelope.data : null;

  replace(
    engineHost,
    el(
      'dl',
      { class: 'deflist' },
      defRow('FinDyn version', el('span', { class: 'mono' }, text(m.version))),
      defRow('Milestone', badge(text(m.milestone), 'info')),
      defRow('Specification', el('span', { class: 'mono' }, text(m.spec))),
      defRow('Environment', el('span', { class: 'mono' }, text(m.env))),
      defRow(
        'Information set',
        el('span', { class: 'mono' }, text(m.info_set)),
      ),
      defRow('Model version', el('span', { class: 'mono' }, text(metaRes.envelope.model_version))),
      defRow(
        'Data freshness flag',
        health
          ? badge(health.stale ? 'stale' : 'current', health.stale ? 'warn' : 'ok')
          : badge('unknown', 'idle'),
      ),
      defRow('Forces in vocabulary', text(m.vocabulary.forces.join(', '))),
      defRow('Regimes in vocabulary', text(m.vocabulary.regimes.join(', '))),
      defRow('Forecast horizons', text(m.vocabulary.horizons.join(', '))),
    ),
  );
}

function renderSources(healthRes: ApiResult<Health>): void {
  if (!sourcesHost) return;
  if (!healthRes.ok) {
    replace(sourcesHost, failureBlock(healthRes, 'Ingestion health'));
    return;
  }
  const health = healthRes.envelope.data;

  const degraded = health.degraded_reason
    ? stateBlock({
        tone: 'bad',
        title: 'The API reported a degraded database read',
        detail: health.degraded_reason,
      })
    : null;

  if (health.sources.length === 0) {
    replace(
      sourcesHost,
      degraded,
      emptyBlock(
        'ingestion runs',
        'The ingestion log is empty. Per-source rows appear here after the first cron-triggered run writes to D1.',
      ),
    );
    return;
  }

  const rows = health.sources.map((s) =>
    el(
      'tr',
      {},
      el('td', { class: 'mono' }, s.source),
      el('td', {}, badge(s.status, statusTone(s.status))),
      el('td', { class: 'mono nowrap' }, formatDate(s.last_run_at)),
      el('td', { class: 'num' }, formatCount(s.rows_written)),
      el('td', {}, text(s.error, '—')),
    ),
  );

  replace(
    sourcesHost,
    degraded,
    table(['Source', 'Status', 'Last run', 'Rows written', 'Error'], rows),
  );
}

async function renderRoadmap(apiReachable: boolean): Promise<void> {
  if (!roadmapHost) return;

  if (!apiReachable) {
    replace(
      roadmapHost,
      stateBlock({
        tone: 'warn',
        title: 'Milestone status unavailable — API unreachable',
        detail:
          'These endpoints are reserved by FINDYN_V1_SPEC.md §13. Their delivery milestone is read from the API, which did not respond.',
      }),
      el(
        'ul',
        { class: 'prose' },
        ...RESERVED_ENDPOINTS.map((e) =>
          el('li', {}, el('code', {}, `/api/v1${e.path}`), ` — ${e.summary}`),
        ),
      ),
    );
    return;
  }

  const results = await Promise.all(
    RESERVED_ENDPOINTS.map(async (endpoint) => ({
      endpoint,
      result: await apiGet<unknown>(endpoint.path),
    })),
  );

  const rows = results.map(({ endpoint, result }) => {
    let status: HTMLElement;
    let note: string;
    if (result.ok) {
      status = badge('live', 'ok');
      note = 'Delivered — this endpoint now returns data.';
    } else if (result.kind === 'not_implemented') {
      status = badge(`planned · ${result.milestone}`, 'info');
      note = result.message;
    } else {
      status = badge('unknown', 'idle');
      note = result.message;
    }
    return el(
      'tr',
      {},
      el('td', { class: 'mono nowrap' }, `/api/v1${endpoint.path}`),
      el('td', {}, endpoint.summary),
      el('td', {}, status),
      el('td', {}, note),
    );
  });

  replace(roadmapHost, table(['Endpoint', 'Returns', 'Status', 'Detail'], rows));
}

async function main(): Promise<void> {
  if (tilesHost) replace(tilesHost, loadingBlock('system status'));
  if (engineHost) replace(engineHost, loadingBlock('engine metadata'));
  if (sourcesHost) replace(sourcesHost, loadingBlock('ingestion health'));
  if (roadmapHost) replace(roadmapHost, loadingBlock('milestone status'));

  const [metaRes, healthRes] = await Promise.all([getMeta(), getHealth()]);

  renderTiles(metaRes, healthRes);
  renderEngine(metaRes, healthRes);
  renderSources(healthRes);
  await renderRoadmap(metaRes.ok || healthRes.ok);
}

void main();
