/**
 * FinGold page (04-ui-plan.md §P4).
 *
 * Four views over one engine: the regime badge with its full posterior, the
 * hedge-score history, the drivers panel as labelled tiles, and the signals.
 *
 * The posterior is drawn as a stacked area rather than a line of winners,
 * because the winner is the least interesting part: a 0.51/0.49 split between a
 * hedge bid and a rate headwind is a completely different statement from a 0.95
 * hedge bid, and a badge alone cannot tell them apart. Same reasoning as the
 * equity page's regime chart and §4's three-bar crash decomposition.
 *
 * Empty, stale and 501 states are rendered explicitly everywhere. This is the
 * page about the engine, so it must say when the engine has not run.
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

const ASSET = 'gold';

/** Ten years of daily observations, matching the engine's nightly window. */
const HISTORY_LIMIT = 2600;

/** Mirrors findynamics/engines/gold/domain.py::GOLD_REGIMES, in wire order. */
const GOLD_REGIMES = ['hedge_bid', 'carry_headwind', 'crisis_bid'] as const;
type GoldRegime = (typeof GOLD_REGIMES)[number];

const REGIME_COPY: Record<GoldRegime, string> = {
  hedge_bid:
    'The ordinary state. No crisis is being priced and real rates are not moving against gold; it trades on the dollar and on flows.',
  carry_headwind:
    'Real rates are rising. The one genuine cost of holding a non-yielding asset, and the only one that is arithmetic rather than sentimental.',
  crisis_bid:
    'Stress is being paid for. A statement about demand, not direction — the first days of a crisis usually see gold sold for liquidity before it is bought for safety.',
};

const stateHost = document.querySelector('#gold-state');
const regimeHost = document.querySelector('#gold-regime');
const hedgeHost = document.querySelector('#gold-hedge');
const driversHost = document.querySelector('#gold-drivers');
const signalsHost = document.querySelector('#gold-signals');

// ------------------------------------------------------------------ charts

const W = 900;
const H = 300;
const PAD = { top: 18, right: 18, bottom: 34, left: 56 };

function noData(svg: SVGSVGElement, message: string): SVGSVGElement {
  svg.appendChild(svgEl('text', { x: W / 2, y: H / 2, 'text-anchor': 'middle' }, message));
  return svg;
}

/**
 * The hedge score over time, on a fixed 0-100 axis.
 *
 * Fixed, not auto-scaled: the score is a position on a defined scale, and
 * letting the axis follow the data would make a flat decade look like drama and
 * hide how far from either end the number actually sits. The 50 line is drawn
 * because it is the meaningful reference — above it gold is diversifying an
 * equity drawdown, below it gold is moving with one.
 */
function hedgeChart(points: HistoryPoint[]): SVGSVGElement {
  const svg = svgEl('svg', {
    class: 'chart',
    viewBox: `0 0 ${W} ${H}`,
    role: 'img',
    preserveAspectRatio: 'xMidYMid meet',
    'aria-label': `Gold hedge score over ${points.length} observations, 0 to 100`,
  });

  const usable = points.filter((p) => Number.isFinite(p.value));
  if (usable.length < 2) return noData(svg, 'not enough history to plot');

  const plotW = W - PAD.left - PAD.right;
  const plotH = H - PAD.top - PAD.bottom;
  const x = (i: number) => PAD.left + (i / (usable.length - 1)) * plotW;
  const y = (v: number) => PAD.top + plotH - (v / 100) * plotH;

  for (let t = 0; t <= 4; t++) {
    const value = t * 25;
    const gy = y(value);
    svg.appendChild(
      svgEl('line', { class: 'grid', x1: PAD.left, x2: W - PAD.right, y1: gy, y2: gy }),
    );
    svg.appendChild(
      svgEl('text', { x: PAD.left - 8, y: gy + 3, 'text-anchor': 'end' }, String(value)),
    );
  }
  svg.appendChild(
    svgEl('line', { class: 'axis', x1: PAD.left, x2: PAD.left, y1: PAD.top, y2: H - PAD.bottom }),
  );
  svg.appendChild(
    svgEl('line', {
      class: 'axis',
      x1: PAD.left,
      x2: W - PAD.right,
      y1: y(50),
      y2: y(50),
      'stroke-dasharray': '4 4',
    }),
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
  svg.appendChild(svgEl('text', { x: PAD.left, y: 12 }, 'hedge score — 50 is neutral'));
  return svg;
}

/**
 * The regime posterior as a stacked area, always summing to 1.
 *
 * Three series drawn as bands rather than three separate lines: the quantity is
 * a distribution, and a stack is the shape that cannot accidentally show one
 * regime rising without another falling.
 */
function posteriorChart(series: Record<GoldRegime, HistoryPoint[]>): SVGSVGElement {
  const svg = svgEl('svg', {
    class: 'chart',
    viewBox: `0 0 ${W} ${H}`,
    role: 'img',
    preserveAspectRatio: 'xMidYMid meet',
    'aria-label': 'Gold regime posterior over time, stacked to 1',
  });

  // Only dates every regime published are stackable; a partial column would
  // draw a stack that does not sum to one and read as a gap in the model.
  const byDate = new Map<string, Partial<Record<GoldRegime, number>>>();
  for (const regime of GOLD_REGIMES) {
    for (const point of series[regime] ?? []) {
      if (!Number.isFinite(point.value)) continue;
      const row = byDate.get(point.as_of) ?? {};
      row[regime] = point.value;
      byDate.set(point.as_of, row);
    }
  }
  const dates = [...byDate.keys()]
    .filter((d) => GOLD_REGIMES.every((r) => byDate.get(d)?.[r] !== undefined))
    .sort();

  if (dates.length < 2) return noData(svg, 'no regime posterior published yet');

  const plotW = W - PAD.left - PAD.right;
  const plotH = H - PAD.top - PAD.bottom;
  const x = (i: number) => PAD.left + (i / (dates.length - 1)) * plotW;
  const y = (v: number) => PAD.top + plotH - v * plotH;

  for (let t = 0; t <= 4; t++) {
    const gy = y(t / 4);
    svg.appendChild(
      svgEl('line', { class: 'grid', x1: PAD.left, x2: W - PAD.right, y1: gy, y2: gy }),
    );
    svg.appendChild(
      svgEl('text', { x: PAD.left - 8, y: gy + 3, 'text-anchor': 'end' }, (t / 4).toFixed(2)),
    );
  }

  let floor = new Array(dates.length).fill(0) as number[];
  for (const regime of GOLD_REGIMES) {
    const tops = dates.map((date, i) => (floor[i] ?? 0) + (byDate.get(date)?.[regime] ?? 0));
    const forward = tops.map(
      (v, i) => `${i === 0 ? 'M' : 'L'}${x(i).toFixed(2)},${y(v).toFixed(2)}`,
    );
    const back = floor
      .map((v, i) => ({ v, i }))
      .reverse()
      .map(({ v, i }) => `L${x(i).toFixed(2)},${y(v).toFixed(2)}`);
    svg.appendChild(
      svgEl('path', {
        // Keyed by the regime name, matching the equity chart. The engine
        // vocabularies are disjoint by construction (03-contracts.md §1), so one
        // stylesheet can carry both without a prefix.
        class: `regimeband regimeband--${regime}`,
        d: `${forward.join(' ')} ${back.join(' ')} Z`,
      }),
    );
    floor = tops;
  }

  svg.appendChild(
    svgEl('line', { class: 'axis', x1: PAD.left, x2: PAD.left, y1: PAD.top, y2: H - PAD.bottom }),
  );
  const first = dates[0];
  const last = dates[dates.length - 1];
  if (first) svg.appendChild(svgEl('text', { x: PAD.left, y: H - 12 }, first));
  if (last) {
    svg.appendChild(svgEl('text', { x: W - PAD.right, y: H - 12, 'text-anchor': 'end' }, last));
  }
  return svg;
}

// --------------------------------------------------------------- rendering

const DIRECTION_ARROW: Record<number, string> = { [-1]: '▼', 0: '▬', 1: '▲' };
const DIRECTION_TONE: Record<number, Tone> = { [-1]: 'bad', 0: 'idle', 1: 'ok' };
const DIRECTION_WORD: Record<number, string> = { [-1]: 'adverse', 0: 'neutral', 1: 'supportive' };

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

function signed(value: number | undefined, digits = 2, unit = ''): string {
  if (value === undefined || !Number.isFinite(value)) return '—';
  return `${value >= 0 ? '+' : ''}${value.toFixed(digits)}${unit}`;
}

function renderState(result: ApiResult<AssetState>): AssetState | null {
  if (!stateHost) return null;
  if (!result.ok) {
    replace(stateHost, failureBlock(result, 'FinGold state'));
    return null;
  }

  const state = result.envelope.data;

  const tiles = el(
    'div',
    { class: 'tiles' },
    tile('Regime', state.regime, regimeTone(state.regime), `As of ${state.as_of}`),
    tile(
      'Hedge score',
      formatValue(state.components?.hedge_score),
      hedgeTone(state.components?.hedge_score),
      '0-100. Gold’s diversification of an equity drawdown, blended with the regime posterior.',
    ),
    tile(
      'Risk score',
      formatValue(state.risk_score),
      state.risk_score !== null && state.risk_score >= 60 ? 'warn' : 'ok',
      'Realized volatility and jump intensity. Calibrated on gold, where 100 is roughly 1980.',
    ),
    tile(
      'Confidence',
      formatValue(state.confidence),
      'idle',
      'Capped at 0.7 by construction: the expected return is a historical mean and the regime is half a fitted chain, half two configured gates.',
    ),
  );

  const expected =
    state.expected_return === null
      ? null
      : el(
          'p',
          { class: 'prose' },
          el('strong', {}, 'Not a forecast. '),
          `The ${percent(state.expected_return)} figure carried as expected_return is the mean annualized return gold has delivered in this regime across the fitted sample. It describes the past tense of a 600-month history and nothing about the next twelve months.`,
        );

  const staleness = result.envelope.stale
    ? stateBlock({
        tone: 'warn',
        title: 'This state is stale',
        detail: `Last published ${formatDate(state.as_of)}. The daily run has not reported since.`,
      })
    : null;

  replace(
    stateHost,
    tiles,
    expected,
    staleness,
    el(
      'p',
      { class: 'enginecard__detail' },
      `Model ${state.model_version}. There is no valuation here and never will be: gold has no cash flow, so this engine models drivers.`,
    ),
  );
  return state;
}

function hedgeTone(score: number | undefined): Tone {
  if (score === undefined || !Number.isFinite(score)) return 'idle';
  if (score >= 60) return 'ok';
  if (score < 40) return 'bad';
  return 'warn';
}

function renderRegime(
  state: AssetState | null,
  histories: Record<GoldRegime, ApiResult<AssetHistory>>,
): void {
  if (!regimeHost) return;
  if (!state) {
    replace(regimeHost, emptyBlock('regime', 'The engine has not published a state yet.'));
    return;
  }

  const components = state.components ?? {};
  const rows = GOLD_REGIMES.map((regime) => {
    const probability = components[`regime_posterior_${regime}`];
    return el(
      'tr',
      {},
      el('td', { class: 'mono nowrap' }, regime),
      el('td', { class: 'num mono' }, formatValue(probability)),
      el('td', {}, regime === state.regime ? badge('current', regimeTone(regime)) : ''),
      el('td', {}, REGIME_COPY[regime]),
    ) as HTMLTableRowElement;
  });

  const series = {} as Record<GoldRegime, HistoryPoint[]>;
  const failed: string[] = [];
  for (const regime of GOLD_REGIMES) {
    const result = histories[regime];
    if (result?.ok) series[regime] = result.envelope.data.points ?? [];
    else {
      series[regime] = [];
      failed.push(regime);
    }
  }

  const legend = el(
    'div',
    { class: 'legend' },
    ...GOLD_REGIMES.map((regime) =>
      el(
        'span',
        { class: 'legend__item' },
        el('span', { class: `legend__swatch legend__swatch--${regime}` }),
        regime,
      ),
    ),
  );

  replace(
    regimeHost,
    posteriorChart(series),
    legend,
    failed.length
      ? stateBlock({
          tone: 'warn',
          title: 'Part of the posterior history could not be read',
          detail: `Missing: ${failed.join(', ')}. The chart shows only dates on which all three were published.`,
        })
      : null,
    table(['Regime', 'Posterior', '', 'What it means'], rows),
  );
}

function renderHedge(result: ApiResult<AssetHistory>): void {
  if (!hedgeHost) return;
  if (!result.ok) {
    replace(hedgeHost, failureBlock(result, 'Hedge-score history'));
    return;
  }
  const points = result.envelope.data.points ?? [];
  if (points.length === 0) {
    replace(
      hedgeHost,
      emptyBlock('hedge-score history', 'The engine has not published this metric yet.'),
    );
    return;
  }
  replace(
    hedgeHost,
    hedgeChart(points),
    result.envelope.data.truncated
      ? stateBlock({
          tone: 'warn',
          title: 'This series is clipped',
          detail: 'The requested window exceeded the API’s row ceiling, so this is a prefix rather than the whole history.',
        })
      : null,
  );
}

/** One driver tile: label, formatted value, and what it is telling you. */
interface Driver {
  label: string;
  key: string;
  format: (value: number | undefined) => string;
  detail: string;
  tone?: (value: number | undefined) => Tone;
}

const DRIVERS: Driver[] = [
  {
    label: 'Real 10y rate',
    key: 'real_rate',
    format: (v) => signed(v, 2, '%'),
    detail:
      'Nominal 10y minus the breakeven, or minus trailing CPI before TIPS existed. The opportunity cost of holding an asset that yields nothing.',
  },
  {
    label: 'Real rate, 12m change',
    key: 'real_rate_change_12m',
    format: (v) => signed(v, 2, 'pp'),
    detail: 'Rising is the headwind. This is the driver that separates 2013 and 2022 from 2011.',
    tone: (v) => (v === undefined ? 'idle' : v > 0.25 ? 'bad' : v < -0.25 ? 'ok' : 'idle'),
  },
  {
    label: 'USD trend',
    key: 'usd_trend',
    format: (v) => (v === undefined ? '—' : signed(v * 100, 1, '%')),
    detail:
      '12-month log change of the broad dollar. Gold is quoted in dollars, so a stronger dollar is a mechanical headwind before any story about it.',
    tone: (v) => (v === undefined ? 'idle' : v > 0.02 ? 'warn' : 'idle'),
  },
  {
    label: 'Liquidity stress',
    key: 'z_stress',
    format: (v) => signed(v, 2, 'σ'),
    detail:
      'NFCI, standardized on an expanding window. Positive is tight. This is the crisis channel.',
    tone: (v) => (v === undefined ? 'idle' : v > 0.5 ? 'bad' : v < -0.5 ? 'ok' : 'idle'),
  },
  {
    label: 'Jump intensity',
    key: 'jump_intensity',
    format: (v) => (v === undefined ? '—' : `${v.toFixed(1)}/yr`),
    detail:
      'Lee-Mykland detections in the trailing year, annualized. Bipower local volatility, so the day of a crash cannot raise its own threshold and hide.',
    tone: (v) => (v === undefined ? 'idle' : v >= 4 ? 'bad' : v >= 2 ? 'warn' : 'ok'),
  },
  {
    label: 'Crisis premium',
    key: 'crisis_premium',
    format: (v) => (v === undefined ? '—' : v.toFixed(2)),
    detail:
      '0-1: jump intensity lifted by financial stress. Not a probability of a crisis — how much of one is already being paid for.',
    tone: (v) => (v === undefined ? 'idle' : v >= 0.4 ? 'bad' : 'idle'),
  },
  {
    label: 'Equity instability',
    key: 'z_equity_rii',
    format: (v) => signed(v, 2, 'σ'),
    detail:
      'FinEquity’s RII, read back as published data rather than by importing that engine. Absent until FinEquity has published.',
  },
  {
    label: 'Realized volatility',
    key: 'realized_vol_annual_pct',
    format: (v) => (v === undefined ? '—' : `${v.toFixed(1)}%`),
    detail: 'Annualized standard deviation of daily log returns over the trailing year.',
    tone: (v) => (v === undefined ? 'idle' : v >= 25 ? 'warn' : 'idle'),
  },
];

function renderDrivers(state: AssetState | null): void {
  if (!driversHost) return;
  if (!state) {
    replace(driversHost, emptyBlock('drivers', 'The engine has not published a state yet.'));
    return;
  }
  const components = state.components ?? {};

  const tiles = DRIVERS.map((driver) => {
    const value = components[driver.key];
    return tile(
      driver.label,
      driver.format(value),
      driver.tone ? driver.tone(value) : 'info',
      driver.detail,
    );
  });

  const absent = DRIVERS.filter((d) => components[d.key] === undefined).map((d) => d.label);

  // Layer 0's reading of the same forces, on its own axis. Shown beside the
  // engine's drivers rather than mixed into them: these are 0-100 percentiles
  // where 100 is most risk-supportive, and the tiles above are percentage
  // points and standard deviations. The two agreeing is reassurance; the two
  // disagreeing is the interesting case, and neither is visible if they are
  // averaged together.
  const SHARED = [
    ['factor_real_rate', 'real_rate'],
    ['factor_usd_strength', 'usd_strength'],
    ['factor_liquidity', 'liquidity'],
  ] as const;
  const shared = SHARED.filter(([key]) => components[key] !== undefined);

  replace(
    driversHost,
    el('div', { class: 'tiles' }, ...tiles),
    shared.length
      ? el(
          'p',
          { class: 'enginecard__detail' },
          `Shared factors (Layer 0, 0-100 where 100 is most risk-supportive): ${shared
            .map(([key, label]) => `${label} ${formatValue(components[key])}`)
            .join(' · ')}. Computed by an independent pipeline from overlapping series — a cross-check on the drivers above, not an input to them.`,
        )
      : null,
    absent.length
      ? el(
          'p',
          { class: 'enginecard__detail' },
          `${absent.length} driver(s) are not in this run’s information set: ${absent.join(', ')}. They are shown as absent rather than as zero, because those are different statements.`,
        )
      : null,
    el(
      'div',
      { class: 'legend' },
      el(
        'span',
        { class: 'legend__item' },
        `Markov violent-state probability ${formatValue(components.markov_violent_probability)} · stress gate ${formatValue(components.stress_gate)} · carry gate ${formatValue(components.carry_gate)}`,
      ),
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
  for (const host of [stateHost, regimeHost, hedgeHost, driversHost, signalsHost]) {
    if (host) replace(host, loadingBlock('FinGold'));
  }

  const [stateResult, hedgeResult, ...posteriors] = await Promise.all([
    getAssetState(ASSET),
    getAssetHistory(ASSET, 'hedge_score', { limit: HISTORY_LIMIT }),
    ...GOLD_REGIMES.map((regime) =>
      getAssetHistory(ASSET, `regime_posterior_${regime}`, { limit: HISTORY_LIMIT }),
    ),
  ]);

  const histories = {} as Record<GoldRegime, ApiResult<AssetHistory>>;
  GOLD_REGIMES.forEach((regime, i) => {
    const result = posteriors[i];
    if (result) histories[regime] = result;
  });

  const state = renderState(stateResult);
  renderRegime(state, histories);
  renderHedge(hedgeResult);
  renderDrivers(state);
  renderSignals(state);
}

void main();
