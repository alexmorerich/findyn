/**
 * FinRates page (04-ui-plan.md §P1).
 *
 * Four views over the same engine: the fitted curve today (with ghosts from a
 * month and a year ago), the three Nelson-Siegel factors as history, the regime
 * as a coloured timeline, and the signals as a table.
 *
 * Everything is drawn from the live API in the browser. The curve chart plots
 * the *fitted* curve, reconstructed in the browser from the published NS
 * factors — the engine publishes level, slope, curvature and lambda, and the
 * Nelson-Siegel form is a closed expression, so no extra endpoint is needed.
 * Ghost curves are the same expression evaluated on older factor values.
 *
 * Empty, stale and 501 states are rendered explicitly everywhere: an engine
 * that has not run must say so, not show an empty box.
 */
import {
  getAssetHistory,
  getAssetState,
  type ApiResult,
  type AssetHistory,
  type AssetState,
  type HistoryPoint,
  type Signal,
} from '../lib/api';
import {
  badge,
  el,
  emptyBlock,
  failureBlock,
  loadingBlock,
  replace,
  stateBlock,
  svgEl,
  table,
  text,
} from '../lib/dom';
import { formatDate, formatValue, type Tone } from '../lib/format';
import { regimeTone } from './engines';

const ASSET = 'rates';

/** Enough history for a year of context on the sparklines without huge payloads. */
const HISTORY_LIMIT = 2600;

const stateHost = document.querySelector('#rates-state');
const curveHost = document.querySelector('#rates-curve');
const factorsHost = document.querySelector('#rates-factors');
const regimeHost = document.querySelector('#rates-regime');
const signalsHost = document.querySelector('#rates-signals');

// --------------------------------------------------------------- NS maths

/**
 * The Nelson-Siegel curve, evaluated in the browser exactly as the engine fits
 * it. Kept identical to compute/findynamics/engines/rates/nelson_siegel.py —
 * if these two ever disagree, the chart stops describing the model.
 *
 *   y(t) = b0 + b1*(1-exp(-Lt))/(Lt) + b2*[(1-exp(-Lt))/(Lt) - exp(-Lt)]
 *
 * `slope` as published is -b1, so b1 is negated on the way in.
 */
export function nsYield(
  maturity: number,
  factors: { level: number; slope: number; curvature: number; lambda: number },
): number {
  const lt = factors.lambda * maturity;
  const decay = lt > 1e-12 ? (1 - Math.exp(-lt)) / lt : 1;
  return (
    factors.level + -factors.slope * decay + factors.curvature * (decay - Math.exp(-lt))
  );
}

/** Maturities the curve is drawn over — dense at the short end, where it bends. */
const CURVE_GRID: number[] = [];
for (let m = 0.08; m <= 30.001; m += m < 2 ? 0.08 : m < 10 ? 0.25 : 1) CURVE_GRID.push(m);

// ------------------------------------------------------------------ chart

const W = 900;
const H = 320;
const PAD = { top: 18, right: 18, bottom: 32, left: 58 };

interface CurveSeries {
  label: string;
  ghost: boolean;
  factors: { level: number; slope: number; curvature: number; lambda: number };
}

function axes(
  svg: SVGSVGElement,
  yMin: number,
  yMax: number,
  xTicks: { at: number; label: string }[],
  xScale: (v: number) => number,
  yScale: (v: number) => number,
  unit: string,
): void {
  const TICKS = 4;
  for (let t = 0; t <= TICKS; t++) {
    const value = yMin + ((yMax - yMin) * t) / TICKS;
    const gy = yScale(value);
    svg.appendChild(svgEl('line', { class: 'grid', x1: PAD.left, x2: W - PAD.right, y1: gy, y2: gy }));
    svg.appendChild(
      svgEl('text', { x: PAD.left - 8, y: gy + 3, 'text-anchor': 'end' }, formatValue(value)),
    );
  }
  svg.appendChild(
    svgEl('line', {
      class: 'axis',
      x1: PAD.left,
      x2: PAD.left,
      y1: PAD.top,
      y2: H - PAD.bottom,
    }),
  );
  for (const tick of xTicks) {
    svg.appendChild(
      svgEl('text', { x: xScale(tick.at), y: H - 12, 'text-anchor': 'middle' }, tick.label),
    );
  }
  if (unit) svg.appendChild(svgEl('text', { x: PAD.left, y: 12 }, unit));
}

/**
 * Yield-curve snapshot: the fitted curve, the raw tenor points it was fitted
 * to, and ghost curves for a month and a year ago.
 *
 * The x axis is sqrt(maturity), not maturity: on a linear axis the first two
 * years — where every interesting bend lives — occupy 7% of the width.
 */
function curveChart(series: CurveSeries[], observed: { maturity: number; value: number }[]): SVGSVGElement {
  const svg = svgEl('svg', {
    class: 'chart',
    viewBox: `0 0 ${W} ${H}`,
    role: 'img',
    preserveAspectRatio: 'xMidYMid meet',
    'aria-label': `Fitted Nelson-Siegel yield curve with ${series.length - 1} historical comparison curve(s)`,
  });

  const all = [
    ...series.flatMap((s) => CURVE_GRID.map((m) => nsYield(m, s.factors))),
    ...observed.map((o) => o.value),
  ].filter(Number.isFinite);

  if (all.length === 0) {
    svg.appendChild(
      svgEl('text', { x: W / 2, y: H / 2, 'text-anchor': 'middle' }, 'no curve to plot'),
    );
    return svg;
  }

  const lo = Math.min(...all);
  const hi = Math.max(...all);
  const pad = hi - lo < 1e-6 ? 0.5 : (hi - lo) * 0.12;
  const yMin = lo - pad;
  const yMax = hi + pad;

  const plotW = W - PAD.left - PAD.right;
  const plotH = H - PAD.top - PAD.bottom;
  const root = Math.sqrt(30);
  const x = (m: number) => PAD.left + (Math.sqrt(m) / root) * plotW;
  const y = (v: number) => PAD.top + plotH - ((v - yMin) / (yMax - yMin)) * plotH;

  axes(
    svg,
    yMin,
    yMax,
    [0.25, 1, 2, 5, 10, 20, 30].map((m) => ({ at: m, label: m < 1 ? `${m * 12}m` : `${m}y` })),
    x,
    y,
    'percent',
  );

  // Zero line, so an inverted or negative curve reads at a glance.
  if (yMin < 0 && yMax > 0) {
    svg.appendChild(
      svgEl('line', { class: 'axis', x1: PAD.left, x2: W - PAD.right, y1: y(0), y2: y(0) }),
    );
  }

  // Ghosts first so today's curve draws on top of them.
  for (const s of [...series].sort((a, b) => Number(b.ghost) - Number(a.ghost))) {
    const d = CURVE_GRID.map(
      (m, i) => `${i === 0 ? 'M' : 'L'}${x(m).toFixed(2)},${y(nsYield(m, s.factors)).toFixed(2)}`,
    ).join(' ');
    svg.appendChild(svgEl('path', { class: s.ghost ? 'line line--ghost' : 'line', d }));
  }

  for (const point of observed) {
    svg.appendChild(
      svgEl('circle', { class: 'point', cx: x(point.maturity), cy: y(point.value), r: 3 }),
    );
  }

  return svg;
}

/** Compact line chart for one metric's history. */
function sparkline(points: HistoryPoint[], label: string): SVGSVGElement {
  const SW = 900;
  const SH = 120;
  const SP = { top: 14, right: 14, bottom: 20, left: 58 };

  const svg = svgEl('svg', {
    class: 'chart chart--spark',
    viewBox: `0 0 ${SW} ${SH}`,
    role: 'img',
    preserveAspectRatio: 'none',
    'aria-label': `${label} over ${points.length} observations`,
  });

  const usable = points.filter((p) => Number.isFinite(p.value));
  if (usable.length === 0) {
    svg.appendChild(
      svgEl('text', { x: SW / 2, y: SH / 2, 'text-anchor': 'middle' }, `no ${label} history`),
    );
    return svg;
  }

  const values = usable.map((p) => p.value);
  const lo = Math.min(...values);
  const hi = Math.max(...values);
  const pad = hi - lo < 1e-9 ? Math.max(Math.abs(hi) * 0.05, 0.1) : (hi - lo) * 0.1;
  const yMin = lo - pad;
  const yMax = hi + pad;

  const plotW = SW - SP.left - SP.right;
  const plotH = SH - SP.top - SP.bottom;
  const x = (i: number) =>
    SP.left + (usable.length === 1 ? plotW / 2 : (i / (usable.length - 1)) * plotW);
  const yy = (v: number) => SP.top + plotH - ((v - yMin) / (yMax - yMin)) * plotH;

  for (const value of [yMin, (yMin + yMax) / 2, yMax]) {
    const gy = yy(value);
    svg.appendChild(svgEl('line', { class: 'grid', x1: SP.left, x2: SW - SP.right, y1: gy, y2: gy }));
    svg.appendChild(
      svgEl('text', { x: SP.left - 8, y: gy + 3, 'text-anchor': 'end' }, formatValue(value)),
    );
  }
  if (yMin < 0 && yMax > 0) {
    svg.appendChild(
      svgEl('line', { class: 'axis', x1: SP.left, x2: SW - SP.right, y1: yy(0), y2: yy(0) }),
    );
  }

  svg.appendChild(
    svgEl('path', {
      class: 'line',
      d: usable
        .map((p, i) => `${i === 0 ? 'M' : 'L'}${x(i).toFixed(2)},${yy(p.value).toFixed(2)}`)
        .join(' '),
    }),
  );

  const first = usable[0];
  const last = usable[usable.length - 1];
  if (first) svg.appendChild(svgEl('text', { x: SP.left, y: SH - 6 }, first.as_of));
  if (last) {
    svg.appendChild(
      svgEl('text', { x: SW - SP.right, y: SH - 6, 'text-anchor': 'end' }, last.as_of),
    );
  }
  return svg;
}

/**
 * Regime timeline: one coloured band per contiguous run of the same regime.
 *
 * Runs are drawn proportionally to how many observations they cover, not to
 * elapsed time — the underlying series is trading days, and spacing by calendar
 * date would silently widen every holiday.
 */
function regimeStrip(points: HistoryPoint[]): HTMLElement {
  const labelled = points
    .map((p) => ({
      as_of: p.as_of,
      regime: (p.meta as { regime?: string } | null)?.regime ?? null,
    }))
    .filter((p): p is { as_of: string; regime: string } => Boolean(p.regime));

  if (labelled.length === 0) {
    return emptyBlock('regime history', 'The engine has not published a regime series yet.');
  }

  const runs: { regime: string; from: string; to: string; count: number }[] = [];
  for (const point of labelled) {
    const current = runs[runs.length - 1];
    if (current && current.regime === point.regime) {
      current.to = point.as_of;
      current.count += 1;
    } else {
      runs.push({ regime: point.regime, from: point.as_of, to: point.as_of, count: 1 });
    }
  }

  const total = labelled.length;
  const strip = el(
    'div',
    { class: 'regimestrip', role: 'img', 'aria-label': `Rate regime over ${total} trading days` },
    ...runs.map((run) =>
      el('span', {
        class: `regimestrip__run regimestrip__run--${regimeTone(run.regime)}`,
        style: `flex-grow:${run.count}`,
        title: `${run.regime}: ${run.from} to ${run.to} (${run.count} days)`,
      }),
    ),
  );

  const present = [...new Set(runs.map((r) => r.regime))];
  const legend = el(
    'div',
    { class: 'legend' },
    ...present.map((regime) =>
      el(
        'span',
        { class: 'legend__item' },
        el('span', { class: `legend__swatch legend__swatch--${regimeTone(regime)}` }),
        regime,
      ),
    ),
  );

  const first = labelled[0];
  const last = labelled[labelled.length - 1];
  return el(
    'div',
    {},
    strip,
    el(
      'p',
      { class: 'enginecard__detail' },
      `${first?.as_of ?? '—'} to ${last?.as_of ?? '—'} · ${total} trading days · ${runs.length} regime spells`,
    ),
    legend,
  );
}

// --------------------------------------------------------------- rendering

const DIRECTION_ARROW: Record<number, string> = { [-1]: '▼', 0: '▬', 1: '▲' };
const DIRECTION_TONE: Record<number, Tone> = { [-1]: 'bad', 0: 'idle', 1: 'ok' };
const DIRECTION_WORD: Record<number, string> = {
  [-1]: 'adverse',
  0: 'neutral',
  1: 'supportive',
};

function renderState(result: ApiResult<AssetState>): AssetState | null {
  if (!stateHost) return null;
  if (!result.ok) {
    replace(stateHost, failureBlock(result, 'FinRates state'));
    return null;
  }

  const state = result.envelope.data;
  const stale = result.envelope.stale;

  const tiles = el(
    'div',
    { class: 'tiles' },
    tile('Regime', state.regime, regimeTone(state.regime), `As of ${state.as_of}`),
    tile(
      'Expected return',
      state.expected_return === null ? '—' : `${(state.expected_return * 100).toFixed(2)}%`,
      'info',
      'Carry + rolldown on a 10y constant-maturity position if the curve does not move. Not a forecast.',
    ),
    tile(
      'Risk score',
      formatValue(state.risk_score),
      state.risk_score !== null && state.risk_score >= 70
        ? 'bad'
        : state.risk_score !== null && state.risk_score >= 40
          ? 'warn'
          : 'ok',
      'Duration times rate volatility, scaled 0-100.',
    ),
    tile(
      'Confidence',
      formatValue(state.confidence),
      'idle',
      'Fit quality times how much of the curve quoted that day.',
    ),
  );

  const staleness = stale
    ? stateBlock({
        tone: 'warn',
        title: 'This state is stale',
        detail: `Last published ${formatDate(state.as_of)}. The daily run has not reported since.`,
      })
    : null;

  // Every score on this page must be explainable (04-ui-plan.md global rules).
  const components = state.components ?? {};
  const keys = Object.keys(components).sort();
  const explain = keys.length
    ? el(
        'details',
        { class: 'explain' },
        el('summary', {}, 'How these numbers were produced'),
        el(
          'dl',
          { class: 'deflist' },
          ...keys.map((key) =>
            el(
              'div',
              { class: 'defrow' },
              el('dt', { class: 'mono' }, key),
              el('dd', { class: 'mono' }, formatValue(components[key])),
            ),
          ),
        ),
      )
    : null;

  replace(
    stateHost,
    staleness,
    tiles,
    el(
      'p',
      { class: 'enginecard__detail' },
      `Model ${state.model_version} · information set ${state.as_of}`,
    ),
    explain,
  );
  return state;
}

function tile(label: string, value: string, tone: Tone, detail: string): HTMLElement {
  return el(
    'div',
    { class: `tile tile--${tone}` },
    el('div', { class: 'tile__label' }, label),
    el('div', { class: 'tile__value' }, value),
    el('div', { class: 'tile__detail' }, detail),
  );
}

/** Value of a metric on the observation closest to (but not after) a target date. */
function valueAsOf(points: HistoryPoint[], target: string): HistoryPoint | null {
  let best: HistoryPoint | null = null;
  for (const point of points) {
    if (point.as_of <= target && (best === null || point.as_of > best.as_of)) best = point;
  }
  return best;
}

function shiftDays(isoDate: string, days: number): string {
  const ts = Date.parse(`${isoDate}T00:00:00Z`);
  if (Number.isNaN(ts)) return isoDate;
  return new Date(ts + days * 86_400_000).toISOString().slice(0, 10);
}

function renderCurve(
  state: AssetState | null,
  histories: Record<string, HistoryPoint[]>,
): void {
  if (!curveHost) return;

  if (!state) {
    replace(
      curveHost,
      emptyBlock('curve', 'The engine has not published a state, so there is no curve to fit.'),
    );
    return;
  }

  const c = state.components ?? {};
  const lambda = c.ns_lambda;
  if (
    !Number.isFinite(c.ns_level) ||
    !Number.isFinite(c.ns_slope) ||
    !Number.isFinite(c.ns_curvature) ||
    !Number.isFinite(lambda)
  ) {
    replace(
      curveHost,
      emptyBlock(
        'curve',
        'The published state carries no Nelson-Siegel factors, so the fitted curve cannot be drawn.',
      ),
    );
    return;
  }

  const today = {
    level: c.ns_level as number,
    slope: c.ns_slope as number,
    curvature: c.ns_curvature as number,
    lambda: lambda as number,
  };

  const series: CurveSeries[] = [{ label: state.as_of, ghost: false, factors: today }];

  // Ghosts are only drawn when all three factors exist at that date. A curve
  // built from two of three factors and today's third would be a fabrication.
  for (const [label, days] of [
    ['1 month ago', -30],
    ['1 year ago', -365],
  ] as const) {
    const target = shiftDays(state.as_of, days);
    const level = valueAsOf(histories.ns_level ?? [], target);
    const slope = valueAsOf(histories.ns_slope ?? [], target);
    const curvature = valueAsOf(histories.ns_curvature ?? [], target);
    if (!level || !slope || !curvature) continue;
    series.push({
      label: `${label} (${level.as_of})`,
      ghost: true,
      factors: {
        level: level.value,
        slope: slope.value,
        curvature: curvature.value,
        lambda: today.lambda,
      },
    });
  }

  const legend = el(
    'div',
    { class: 'legend' },
    ...series.map((s) =>
      el(
        'span',
        { class: 'legend__item' },
        el('span', { class: s.ghost ? 'legend__swatch legend__swatch--ghost' : 'legend__swatch legend__swatch--accent' }),
        s.ghost ? s.label : `today (${s.label})`,
      ),
    ),
  );

  const missingGhosts =
    series.length < 3
      ? el(
          'p',
          { class: 'enginecard__detail' },
          'Some comparison curves are omitted: the factor history does not reach back that far yet.',
        )
      : null;

  replace(curveHost, curveChart(series, []), legend, missingGhosts);
}

function renderFactors(histories: Record<string, ApiResult<AssetHistory>>): void {
  if (!factorsHost) return;

  const rows: (HTMLElement | null)[] = [];
  for (const [metric, label, description] of [
    ['ns_level', 'Level (β₀)', 'Where the curve settles at the long end.'],
    ['ns_slope', 'Slope (−β₁)', 'Long minus short. Negative means an inverted curve.'],
    ['ns_curvature', 'Curvature (β₂)', 'The hump in the belly of the curve.'],
  ] as const) {
    const result = histories[metric];
    if (!result || !result.ok) {
      rows.push(
        el(
          'div',
          { class: 'factorrow' },
          el('h3', { class: 'factorrow__title' }, label),
          failureBlock(
            result ?? { kind: 'error', message: 'not requested', status: 0 },
            label,
          ),
        ),
      );
      continue;
    }
    const points = result.envelope.data.points ?? [];
    rows.push(
      el(
        'div',
        { class: 'factorrow' },
        el(
          'div',
          { class: 'factorrow__head' },
          el('h3', { class: 'factorrow__title' }, label),
          el('span', { class: 'panel__note' }, description),
        ),
        points.length === 0
          ? emptyBlock(label, 'No history published for this factor yet.')
          : sparkline(points, label),
      ),
    );
  }

  replace(factorsHost, ...rows);
}

function renderRegime(result: ApiResult<AssetHistory> | undefined): void {
  if (!regimeHost) return;
  if (!result) return;
  if (!result.ok) {
    replace(regimeHost, failureBlock(result, 'Regime history'));
    return;
  }
  replace(regimeHost, regimeStrip(result.envelope.data.points ?? []));
}

function renderSignals(state: AssetState | null): void {
  if (!signalsHost) return;
  if (!state) {
    replace(signalsHost, emptyBlock('signals', 'The engine has not published a state yet.'));
    return;
  }
  const signals: Signal[] = state.signals ?? [];
  if (signals.length === 0) {
    replace(signalsHost, emptyBlock('signals', 'This run produced no signals.'));
    return;
  }

  const rows = signals.map((signal) =>
    el(
      'tr',
      {},
      // `nowrap`: td.mono breaks anywhere so long series ids fit elsewhere, which
      // here splits `term_premium_trend` across two lines and reads as a typo.
      el('td', { class: 'mono nowrap' }, signal.name),
      el('td', { class: 'num mono nowrap' }, formatValue(signal.value)),
      el(
        'td',
        {},
        badge(
          `${DIRECTION_ARROW[signal.direction] ?? '?'} ${DIRECTION_WORD[signal.direction] ?? 'unknown'}`,
          DIRECTION_TONE[signal.direction] ?? 'idle',
        ),
      ),
      el('td', {}, text(signal.note, '—')),
    ),
  );

  replace(signalsHost, table(['Signal', 'Value', 'Direction', 'What it measures'], rows));
}

// ------------------------------------------------------------------- main

const METRICS = ['ns_level', 'ns_slope', 'ns_curvature'] as const;

async function main(): Promise<void> {
  for (const host of [stateHost, curveHost, factorsHost, regimeHost, signalsHost]) {
    if (host) replace(host, loadingBlock('FinRates'));
  }

  const [stateResult, ...historyResults] = await Promise.all([
    getAssetState(ASSET),
    ...METRICS.map((m) => getAssetHistory(ASSET, m, { limit: HISTORY_LIMIT })),
    getAssetHistory(ASSET, 'regime_code', { limit: HISTORY_LIMIT }),
  ]);

  const byMetric: Record<string, ApiResult<AssetHistory>> = {};
  METRICS.forEach((metric, i) => {
    const result = historyResults[i];
    if (result) byMetric[metric] = result;
  });
  const regimeResult = historyResults[METRICS.length];

  const state = renderState(stateResult);

  const points: Record<string, HistoryPoint[]> = {};
  for (const [metric, result] of Object.entries(byMetric)) {
    points[metric] = result.ok ? (result.envelope.data.points ?? []) : [];
  }

  renderCurve(state, points);
  renderFactors(byMetric);
  renderRegime(regimeResult);
  renderSignals(state);
}

void main();
