/**
 * Macro Data Explorer.
 *
 * Lists everything in the feature store's series catalogue, and on selection
 * loads that series' observations — plotted, and tabulated with the three
 * dates the point-in-time model depends on (period, release, revision).
 */
import {
  getSeries,
  getSeriesList,
  type Observation,
  type SeriesDetail,
  type SeriesSummary,
} from '../lib/api';
import {
  badge,
  defRow,
  el,
  emptyBlock,
  failureBlock,
  loadingBlock,
  replace,
  svgEl,
  table,
  text,
} from '../lib/dom';
import { formatCount, formatDate, formatDays, formatValue, freshness } from '../lib/format';

const listHost = document.querySelector('#series-list');
const summaryHost = document.querySelector('#series-count');
const detailHost = document.querySelector('#series-detail');

/** Observations pulled per series for the chart. The API caps at 5000. */
const OBSERVATION_LIMIT = 1000;
/** Rows shown in the recent-observations table. */
const RECENT_ROWS = 24;

let selectedId: string | null = null;

// ---------------------------------------------------------------- chart

const W = 900;
const H = 300;
const PAD = { top: 16, right: 18, bottom: 30, left: 62 };

/**
 * Inline SVG line chart. Deliberately plain: it draws exactly the observed
 * values, with no smoothing, interpolation or extrapolation of any kind.
 */
function chart(observations: Observation[], unit: string): SVGSVGElement {
  // API returns newest-first; plot oldest-first.
  const points = observations
    .filter((o) => Number.isFinite(o.value))
    .slice()
    .reverse();

  const svg = svgEl('svg', {
    class: 'chart',
    viewBox: `0 0 ${W} ${H}`,
    role: 'img',
    preserveAspectRatio: 'xMidYMid meet',
    'aria-label':
      points.length > 0
        ? `Line chart of ${points.length} observations from ${points[0]!.obs_date} to ${
            points[points.length - 1]!.obs_date
          }`
        : 'No observations to plot',
  });

  if (points.length === 0) {
    svg.appendChild(
      svgEl('text', { x: W / 2, y: H / 2, 'text-anchor': 'middle' }, 'no observations to plot'),
    );
    return svg;
  }

  const values = points.map((p) => p.value);
  const rawMin = Math.min(...values);
  const rawMax = Math.max(...values);
  const span = rawMax - rawMin;
  const pad = span === 0 ? Math.max(Math.abs(rawMax) * 0.05, 1) : span * 0.08;
  const yMin = rawMin - pad;
  const yMax = rawMax + pad;

  const plotW = W - PAD.left - PAD.right;
  const plotH = H - PAD.top - PAD.bottom;
  const x = (i: number) =>
    PAD.left + (points.length === 1 ? plotW / 2 : (i / (points.length - 1)) * plotW);
  const y = (v: number) => PAD.top + plotH - ((v - yMin) / (yMax - yMin)) * plotH;

  // Horizontal gridlines + value labels.
  const TICKS = 4;
  for (let t = 0; t <= TICKS; t++) {
    const value = yMin + ((yMax - yMin) * t) / TICKS;
    const gy = y(value);
    svg.appendChild(
      svgEl('line', { class: 'grid', x1: PAD.left, x2: W - PAD.right, y1: gy, y2: gy }),
    );
    svg.appendChild(
      svgEl(
        'text',
        { x: PAD.left - 8, y: gy + 3, 'text-anchor': 'end' },
        formatValue(value),
      ),
    );
  }

  svg.appendChild(
    svgEl('line', {
      class: 'axis',
      x1: PAD.left,
      x2: PAD.left,
      y1: PAD.top,
      y2: PAD.top + plotH,
    }),
  );

  const path = points.map((p, i) => `${i === 0 ? 'M' : 'L'}${x(i).toFixed(2)},${y(p.value).toFixed(2)}`);
  svg.appendChild(svgEl('path', { class: 'line', d: path.join(' ') }));
  if (points.length === 1) {
    svg.appendChild(svgEl('circle', { class: 'line', cx: x(0), cy: y(points[0]!.value), r: 3 }));
  }

  // X labels: first, middle, last observed period.
  const labelIdx = points.length > 2 ? [0, Math.floor((points.length - 1) / 2), points.length - 1] : [0, points.length - 1];
  for (const i of new Set(labelIdx)) {
    const anchor = i === 0 ? 'start' : i === points.length - 1 ? 'end' : 'middle';
    svg.appendChild(
      svgEl(
        'text',
        { x: x(i), y: H - 10, 'text-anchor': anchor },
        points[i]!.obs_date,
      ),
    );
  }

  if (unit) {
    svg.appendChild(svgEl('text', { x: PAD.left, y: 11 }, unit));
  }

  return svg;
}

// ------------------------------------------------------------ detail view

function renderDetail(detail: SeriesDetail): HTMLElement[] {
  const { metadata: m, observations } = detail;

  const meta = el(
    'dl',
    { class: 'deflist' },
    defRow('Series id', el('span', { class: 'mono' }, m.series_id)),
    defRow('Provider', text(m.provider)),
    defRow('Frequency', text(m.frequency)),
    defRow('Unit', text(m.unit)),
    defRow('Seasonal adjustment', text(m.seasonal_adjustment)),
    defRow('First observation', formatDate(m.first_observation)),
    defRow('Last observation', formatDate(m.last_observation)),
    defRow('Observations loaded', formatCount(observations.length)),
    defRow('Notes', text(m.notes)),
  );

  if (observations.length === 0) {
    return [
      el('h3', {}, text(m.title, m.series_id)),
      meta,
      emptyBlock(
        'observations',
        `The catalogue knows ${m.series_id}, but no observations have been ingested for it yet.`,
      ),
    ];
  }

  const rows = observations.slice(0, RECENT_ROWS).map((o) => {
    const lagDays = Math.round(
      (Date.parse(`${o.release_date}T00:00:00Z`) - Date.parse(`${o.obs_date}T00:00:00Z`)) / 86_400_000,
    );
    return el(
      'tr',
      {},
      el('td', { class: 'mono nowrap' }, o.obs_date),
      el('td', { class: 'mono nowrap' }, o.release_date),
      el('td', { class: 'num' }, Number.isFinite(lagDays) ? formatDays(lagDays) : '—'),
      el('td', { class: 'mono nowrap' }, text(o.revision_date)),
      el('td', { class: 'num' }, formatValue(o.value)),
    );
  });

  return [
    el('h3', {}, text(m.title, m.series_id)),
    meta,
    el('div', { style: 'margin-top:0.9rem' }, chart(observations, m.unit)),
    el(
      'p',
      { class: 'prose', style: 'margin-top:0.9rem' },
      `Most recent ${Math.min(RECENT_ROWS, observations.length)} of ${formatCount(
        observations.length,
      )} loaded observations. `,
      el('strong', {}, 'Period'),
      ' is what the figure describes; ',
      el('strong', {}, 'released'),
      ' is the first date it was publicly knowable; ',
      el('strong', {}, 'revised'),
      ' is when it was restated, if it was.',
    ),
    table(['Period (obs_date)', 'Released', 'Publication lag', 'Revised', 'Value'], rows),
  ];
}

async function loadDetail(seriesId: string): Promise<void> {
  if (!detailHost) return;
  selectedId = seriesId;
  for (const row of document.querySelectorAll<HTMLTableRowElement>('tr.rowbutton')) {
    row.dataset['selected'] = String(row.dataset['seriesId'] === seriesId);
  }
  replace(detailHost, loadingBlock(seriesId));

  const result = await getSeries(seriesId, { limit: OBSERVATION_LIMIT });
  // A newer click won already — drop this response.
  if (selectedId !== seriesId) return;

  if (!result.ok) {
    replace(detailHost, failureBlock(result, seriesId));
    return;
  }
  replace(detailHost, ...renderDetail(result.envelope.data));
  detailHost.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
}

// -------------------------------------------------------------- catalogue

function seriesRow(s: SeriesSummary): HTMLTableRowElement {
  const fresh = freshness(s.freshness_days, s.frequency);

  const trigger = el(
    'button',
    { type: 'button', class: 'linklike' },
    text(s.title, s.series_id),
  );
  trigger.addEventListener('click', (event) => {
    event.stopPropagation();
    void loadDetail(s.series_id);
  });

  const row = el(
    'tr',
    { class: 'rowbutton', 'data-series-id': s.series_id, 'data-selected': 'false' },
    el('td', {}, trigger),
    el('td', { class: 'mono nowrap' }, s.series_id),
    el('td', {}, text(s.provider)),
    el('td', {}, text(s.frequency)),
    el('td', {}, text(s.unit)),
    el('td', { class: 'num' }, formatValue(s.latest_value)),
    el('td', { class: 'mono nowrap' }, formatDate(s.latest_obs_date)),
    el('td', { class: 'mono nowrap' }, formatDate(s.latest_release_date)),
    el('td', { class: 'num' }, formatCount(s.observations)),
    el('td', { title: fresh.explanation }, badge(fresh.label, fresh.tone)),
  );
  row.addEventListener('click', () => void loadDetail(s.series_id));
  return row;
}

async function main(): Promise<void> {
  if (!listHost) return;
  replace(listHost, loadingBlock('series catalogue'));
  if (detailHost) {
    replace(
      detailHost,
      emptyBlock('series selected', 'Choose a series above to load its observations.'),
    );
  }

  const result = await getSeriesList();

  if (!result.ok) {
    replace(listHost, failureBlock(result, 'Series catalogue'));
    if (summaryHost) replace(summaryHost, 'unavailable');
    return;
  }

  const { count, series } = result.envelope.data;
  if (summaryHost) {
    replace(summaryHost, `${formatCount(count)} series · as of ${formatDate(result.envelope.as_of)}`);
  }

  if (series.length === 0) {
    replace(
      listHost,
      emptyBlock(
        'series',
        'The catalogue is empty. Series appear here once the ingestion workers have written to D1.',
      ),
    );
    return;
  }

  replace(
    listHost,
    table(
      [
        'Title',
        'Series id',
        'Provider',
        'Frequency',
        'Unit',
        'Latest value',
        'Period',
        'Released',
        'Observations',
        'Freshness',
      ],
      series.map(seriesRow),
    ),
  );
}

void main();
