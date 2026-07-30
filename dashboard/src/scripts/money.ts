/**
 * FinMoney page (04-ui-plan.md §P2).
 *
 * Five views over one engine: the state tiles, the liquidity badge with the
 * spreads that produced it, the wealth index as a chart, the trailing carry as a
 * table, and the discount-factor curve.
 *
 * The discount curve is drawn from `AssetState.components` rather than from a
 * history call: the engine publishes the whole horizon grid in the state, so the
 * current curve needs no extra request. The wealth index is the only series that
 * genuinely needs history.
 *
 * Empty, stale and 501 states are rendered explicitly everywhere. Unlike the
 * header ribbon, this page must say when the engine has not run — it is the page
 * about the engine.
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

const ASSET = 'money';

/** Five years of daily observations, matching the engine's publish window. */
const HISTORY_LIMIT = 2600;

const stateHost = document.querySelector('#money-state');
const liquidityHost = document.querySelector('#money-liquidity');
const wealthHost = document.querySelector('#money-wealth');
const carryHost = document.querySelector('#money-carry');
const discountHost = document.querySelector('#money-discount');
const signalsHost = document.querySelector('#money-signals');

/** The standard grid, mirroring core/contracts/vocab.py::DISCOUNT_HORIZONS. */
const DISCOUNT_HORIZONS = [
  '1m',
  '3m',
  '6m',
  '1y',
  '2y',
  '3y',
  '5y',
  '7y',
  '10y',
  '20y',
  '30y',
] as const;

const HORIZON_YEARS: Record<string, number> = {
  '1m': 1 / 12,
  '3m': 0.25,
  '6m': 0.5,
  '1y': 1,
  '2y': 2,
  '3y': 3,
  '5y': 5,
  '7y': 7,
  '10y': 10,
  '20y': 20,
  '30y': 30,
};

const CARRY_WINDOWS = [
  ['carry_1m', '1 month', 30],
  ['carry_3m', '3 months', 91],
  ['carry_12m', '12 months', 365],
] as const;

// ------------------------------------------------------------------ charts

const W = 900;
const H = 300;
const PAD = { top: 18, right: 18, bottom: 34, left: 66 };

/**
 * The wealth index over time. A single ascending line, so the y axis is not
 * forced to include zero — a dollar that grew from 1.00 to 1.42 would otherwise
 * be a flat line in the top third of an empty box.
 */
function wealthChart(points: HistoryPoint[], base: string | null): SVGSVGElement {
  const svg = svgEl('svg', {
    class: 'chart',
    viewBox: `0 0 ${W} ${H}`,
    role: 'img',
    preserveAspectRatio: 'xMidYMid meet',
    'aria-label': `Value of one dollar compounded at the short rate over ${points.length} observations`,
  });

  const usable = points.filter((p) => Number.isFinite(p.value));
  if (usable.length < 2) {
    svg.appendChild(
      svgEl('text', { x: W / 2, y: H / 2, 'text-anchor': 'middle' }, 'not enough history to plot'),
    );
    return svg;
  }

  const values = usable.map((p) => p.value);
  const lo = Math.min(...values);
  const hi = Math.max(...values);
  const pad = hi - lo < 1e-9 ? Math.max(Math.abs(hi) * 0.02, 0.01) : (hi - lo) * 0.08;
  const yMin = lo - pad;
  const yMax = hi + pad;

  const plotW = W - PAD.left - PAD.right;
  const plotH = H - PAD.top - PAD.bottom;
  const x = (i: number) => PAD.left + (i / (usable.length - 1)) * plotW;
  const y = (v: number) => PAD.top + plotH - ((v - yMin) / (yMax - yMin)) * plotH;

  const TICKS = 4;
  for (let t = 0; t <= TICKS; t++) {
    const value = yMin + ((yMax - yMin) * t) / TICKS;
    const gy = y(value);
    svg.appendChild(
      svgEl('line', { class: 'grid', x1: PAD.left, x2: W - PAD.right, y1: gy, y2: gy }),
    );
    svg.appendChild(
      svgEl('text', { x: PAD.left - 8, y: gy + 3, 'text-anchor': 'end' }, value.toFixed(4)),
    );
  }
  svg.appendChild(
    svgEl('line', { class: 'axis', x1: PAD.left, x2: PAD.left, y1: PAD.top, y2: H - PAD.bottom }),
  );

  svg.appendChild(
    svgEl('path', {
      class: 'line',
      d: usable
        .map((p, i) => `${i === 0 ? 'M' : 'L'}${x(i).toFixed(2)},${y(p.value).toFixed(2)}`)
        .join(' '),
    }),
  );

  const first = usable[0];
  const last = usable[usable.length - 1];
  if (first) svg.appendChild(svgEl('text', { x: PAD.left, y: H - 12 }, first.as_of));
  if (last) {
    svg.appendChild(
      svgEl('text', { x: W - PAD.right, y: H - 12, 'text-anchor': 'end' }, last.as_of),
    );
  }
  svg.appendChild(
    svgEl(
      'text',
      { x: PAD.left, y: 12 },
      base ? `$1 invested ${base}` : 'value of $1 compounded',
    ),
  );
  return svg;
}

/**
 * The discount-factor curve across horizons.
 *
 * x is sqrt(years), for the same reason the rates page uses it: on a linear axis
 * everything inside two years — where the two curve sources are stitched —
 * occupies a fifteenth of the width. Points sourced from the short rate and from
 * the fitted curve are drawn differently, because they are different claims.
 */
/** One plotted horizon, and whether its value came from a fitted curve. */
interface DiscountPoint {
  horizon: string;
  years: number;
  value: number;
  fitted: boolean;
}

function discountChart(factors: DiscountPoint[]): SVGSVGElement {
  const svg = svgEl('svg', {
    class: 'chart',
    viewBox: `0 0 ${W} ${H}`,
    role: 'img',
    preserveAspectRatio: 'xMidYMid meet',
    'aria-label': `Discount factors across ${factors.length} standard horizons`,
  });

  if (factors.length < 2) {
    svg.appendChild(
      svgEl('text', { x: W / 2, y: H / 2, 'text-anchor': 'middle' }, 'no discount curve published'),
    );
    return svg;
  }

  const values = factors.map((f) => f.value);
  const lo = Math.min(...values, 0);
  const hi = Math.max(...values, 1);

  const plotW = W - PAD.left - PAD.right;
  const plotH = H - PAD.top - PAD.bottom;
  const root = Math.sqrt(30);
  const x = (years: number) => PAD.left + (Math.sqrt(years) / root) * plotW;
  const y = (v: number) => PAD.top + plotH - ((v - lo) / (hi - lo || 1)) * plotH;

  const TICKS = 4;
  for (let t = 0; t <= TICKS; t++) {
    const value = lo + ((hi - lo) * t) / TICKS;
    const gy = y(value);
    svg.appendChild(
      svgEl('line', { class: 'grid', x1: PAD.left, x2: W - PAD.right, y1: gy, y2: gy }),
    );
    svg.appendChild(
      svgEl('text', { x: PAD.left - 8, y: gy + 3, 'text-anchor': 'end' }, value.toFixed(3)),
    );
  }
  svg.appendChild(
    svgEl('line', { class: 'axis', x1: PAD.left, x2: PAD.left, y1: PAD.top, y2: H - PAD.bottom }),
  );
  for (const f of factors) {
    svg.appendChild(
      svgEl('text', { x: x(f.years), y: H - 12, 'text-anchor': 'middle' }, f.horizon),
    );
  }

  svg.appendChild(
    svgEl('path', {
      class: 'line',
      d: factors
        .map((f, i) => `${i === 0 ? 'M' : 'L'}${x(f.years).toFixed(2)},${y(f.value).toFixed(2)}`)
        .join(' '),
    }),
  );

  for (const f of factors) {
    svg.appendChild(
      svgEl('circle', {
        class: f.fitted ? 'point' : 'point point--assumed',
        cx: x(f.years),
        cy: y(f.value),
        r: 3.5,
      }),
    );
  }
  svg.appendChild(svgEl('text', { x: PAD.left, y: 12 }, 'D(t, h) — present value of $1'));
  return svg;
}

// --------------------------------------------------------------- rendering

const DIRECTION_ARROW: Record<number, string> = { [-1]: '▼', 0: '▬', 1: '▲' };
const DIRECTION_TONE: Record<number, Tone> = { [-1]: 'bad', 0: 'idle', 1: 'ok' };
const DIRECTION_WORD: Record<number, string> = {
  [-1]: 'adverse',
  0: 'neutral',
  1: 'supportive',
};

function tile(label: string, value: string, tone: Tone, detail: string): HTMLElement {
  return el(
    'div',
    { class: `tile tile--${tone}` },
    el('div', { class: 'tile__label' }, label),
    el('div', { class: 'tile__value' }, value),
    el('div', { class: 'tile__detail' }, detail),
  );
}

function percent(value: number | null | undefined, digits = 2): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return '—';
  return `${(value * 100).toFixed(digits)}%`;
}

function renderState(result: ApiResult<AssetState>): AssetState | null {
  if (!stateHost) return null;
  if (!result.ok) {
    replace(stateHost, failureBlock(result, 'FinMoney state'));
    return null;
  }

  const state = result.envelope.data;
  const components = state.components ?? {};

  const tiles = el(
    'div',
    { class: 'tiles' },
    tile(
      'Short rate',
      percent(state.expected_return),
      'info',
      'Annualized, as observed. What cash is earning right now — not a forecast.',
    ),
    tile('Liquidity', state.regime, regimeTone(state.regime), `As of ${state.as_of}`),
    tile(
      'Risk score',
      formatValue(state.risk_score),
      state.risk_score !== null && state.risk_score >= 5 ? 'warn' : 'ok',
      'Zero by construction: cash has no duration and no credit. Non-zero only when its own funding market is dislocated, capped at 10 on the shared 0-100 axis.',
    ),
    tile(
      'Confidence',
      formatValue(state.confidence),
      'idle',
      'Deducted for a spliced short rate, a missing bill-SOFR spread, and a missing fitted curve.',
    ),
  );

  const staleness = result.envelope.stale
    ? stateBlock({
        tone: 'warn',
        title: 'This state is stale',
        detail: `Last published ${formatDate(state.as_of)}. The daily run has not reported since.`,
      })
    : null;

  // Every score shown must be explainable (04-ui-plan.md global rules).
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
      `Model ${state.model_version} · information set ${state.as_of} · ` +
        `${formatValue(components.rate_path_days)} days of rate path`,
    ),
    explain,
  );
  return state;
}

function renderLiquidity(state: AssetState | null): void {
  if (!liquidityHost) return;
  if (!state) {
    replace(
      liquidityHost,
      emptyBlock('liquidity state', 'The engine has not published a state yet.'),
    );
    return;
  }

  const c = state.components ?? {};
  const rows: HTMLTableRowElement[] = [];
  for (const [key, label, note] of [
    ['bill_sofr_spread', 'Bill − SOFR spread', '3m constant-maturity bill minus the overnight rate, pp'],
    ['bill_sofr_spread_mean', 'Spread, sustained', 'Mean over the last 3 sessions — the gate for stress'],
    ['bill_change', 'Bill, 10-session change', 'Sharply negative is a dash for cash'],
    ['overnight_change', 'Overnight, 10-session change', 'Sharply positive is a funding squeeze'],
    ['rrp_level_bn', 'Reverse repo, $bn', '5-session average of take-up at the Fed'],
    ['rrp_ratio', 'Reverse repo trend', 'Fast average over slow; below 1.0 the buffer is draining'],
    ['stressed_spread_threshold_pp', 'Stress threshold, pp', 'From config/engines/money.yaml'],
  ] as const) {
    const value = c[key];
    if (value === undefined) continue;
    rows.push(
      el(
        'tr',
        {},
        el('td', {}, label),
        el('td', { class: 'num mono' }, formatValue(value)),
        el('td', {}, note),
      ) as HTMLTableRowElement,
    );
  }

  replace(
    liquidityHost,
    el(
      'div',
      { class: 'badgerow' },
      badge(state.regime, regimeTone(state.regime)),
      el('span', { class: 'enginecard__detail' }, `funding-market state on ${state.as_of}`),
    ),
    rows.length
      ? table(['Component', 'Value', 'What it measures'], rows)
      : emptyBlock('liquidity components', 'This run published no component spreads.'),
  );
}

function renderWealth(result: ApiResult<AssetHistory>): void {
  if (!wealthHost) return;
  if (!result.ok) {
    replace(wealthHost, failureBlock(result, 'Wealth index'));
    return;
  }

  const points = result.envelope.data.points ?? [];
  if (points.length === 0) {
    replace(
      wealthHost,
      emptyBlock('wealth index', 'The engine has not published a wealth index yet.'),
    );
    return;
  }

  // The base date travels in each row's meta; without it the chart is unlabelled.
  const base = (points[0]?.meta as { base?: string } | null)?.base ?? null;
  const last = points[points.length - 1];

  replace(
    wealthHost,
    wealthChart(points, base),
    el(
      'p',
      { class: 'enginecard__detail' },
      base
        ? `$1 invested ${base} is worth ${formatValue(last?.value)} on ${last?.as_of} · ` +
            `${points.length} observations`
        : `${points.length} observations through ${last?.as_of}`,
    ),
  );
}

function renderCarry(state: AssetState | null): void {
  if (!carryHost) return;
  if (!state) {
    replace(carryHost, emptyBlock('carry', 'The engine has not published a state yet.'));
    return;
  }

  const c = state.components ?? {};
  const rows = CARRY_WINDOWS.map(([key, label, days]) => {
    const value = c[key];
    return el(
      'tr',
      {},
      el('td', {}, label),
      el('td', { class: 'num mono' }, percent(value, 3)),
      el('td', { class: 'num mono' }, `${days}`),
      el(
        'td',
        {},
        value === undefined
          ? badge('not enough history', 'idle')
          : badge('realized', 'ok'),
      ),
    ) as HTMLTableRowElement;
  });

  const missing = CARRY_WINDOWS.filter(([key]) => c[key] === undefined);

  replace(
    carryHost,
    table(['Window', 'Annualized carry', 'Calendar days', 'Status'], rows),
    missing.length
      ? el(
          'p',
          { class: 'enginecard__detail' },
          `${missing.length} window(s) are absent: the point-in-time rate path does not reach back that far, and a shorter window reported as a longer one would be worse than nothing.`,
        )
      : null,
  );
}

function renderDiscount(state: AssetState | null): void {
  if (!discountHost) return;
  if (!state) {
    replace(
      discountHost,
      emptyBlock('discount curve', 'The engine has not published a state yet.'),
    );
    return;
  }

  const c = state.components ?? {};
  const fitted = c.ns_lambda !== undefined;

  const factors: DiscountPoint[] = [];
  for (const horizon of DISCOUNT_HORIZONS) {
    const value = c[`discount_${horizon}`];
    const years = HORIZON_YEARS[horizon];
    if (value === undefined || !Number.isFinite(value) || years === undefined) continue;
    factors.push({
      horizon,
      years,
      value,
      // Beyond a year the point is fitted only when a curve was actually published.
      fitted: fitted && years > 1,
    });
  }

  if (factors.length === 0) {
    replace(
      discountHost,
      emptyBlock('discount curve', 'This run published no discount factors.'),
    );
    return;
  }

  const rows = factors.map((f) =>
    el(
      'tr',
      {},
      el('td', { class: 'mono' }, f.horizon),
      el('td', { class: 'num mono' }, f.value.toFixed(6)),
      // The implied continuously-compounded zero rate, which is the more
      // readable of the two numbers and is exactly -ln(D)/h.
      el('td', { class: 'num mono' }, percent(-Math.log(f.value) / f.years, 2)),
      el('td', {}, f.fitted ? badge('fitted NS curve', 'ok') : badge('flat short rate', 'idle')),
    ) as HTMLTableRowElement,
  );

  const legend = el(
    'div',
    { class: 'legend' },
    el(
      'span',
      { class: 'legend__item' },
      el('span', { class: 'legend__swatch legend__swatch--accent' }),
      'fitted Nelson-Siegel curve (h > 1y)',
    ),
    el(
      'span',
      { class: 'legend__item' },
      el('span', { class: 'legend__swatch legend__swatch--idle' }),
      'flat short rate (h ≤ 1y, or no curve published)',
    ),
  );

  replace(
    discountHost,
    discountChart(factors),
    legend,
    table(['Horizon', 'D(t, h)', 'Implied zero rate', 'Source'], rows),
    fitted
      ? null
      : el(
          'p',
          { class: 'enginecard__detail' },
          'No fitted Nelson-Siegel curve was in this run’s information set, so every horizon is the short rate extrapolated flat. The long end is a much weaker statement than usual.',
        ),
  );
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
      // here splits `bill_sofr_spread` across two lines and reads as a typo.
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
    ) as HTMLTableRowElement,
  );

  replace(signalsHost, table(['Signal', 'Value', 'Direction', 'What it measures'], rows));
}

// ------------------------------------------------------------------- main

async function main(): Promise<void> {
  for (const host of [
    stateHost,
    liquidityHost,
    wealthHost,
    carryHost,
    discountHost,
    signalsHost,
  ]) {
    if (host) replace(host, loadingBlock('FinMoney'));
  }

  const [stateResult, wealthResult] = await Promise.all([
    getAssetState(ASSET),
    getAssetHistory(ASSET, 'wealth_index', { limit: HISTORY_LIMIT }),
  ]);

  const state = renderState(stateResult);
  renderLiquidity(state);
  renderWealth(wealthResult);
  renderCarry(state);
  renderDiscount(state);
  renderSignals(state);
}

void main();
