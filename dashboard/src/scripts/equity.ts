/**
 * FinEquity page (04-ui-plan.md §P3, sub-milestones A and B).
 *
 * Kinematics, regime posterior, calibrated transition probabilities, and the
 * force layer — over a range the reader chooses, from one year to the whole
 * daily record back to 1927.
 *
 * Three things here are load-bearing beyond "draw the data":
 *
 * **Ranges are server-side windows.** Every range refetches with its own
 * `from`/`points`. Fetching the century once and slicing in the browser would
 * ship 25,000 points per series to draw a few hundred, and is explicitly not
 * what `from`/`to` are for.
 *
 * **Truncation is never silent.** The API reports `available`, `truncated` and
 * `decimated`; all three are surfaced. A clipped series and a genuinely short
 * one look identical on a chart, and the reader would conclude the market began
 * in 2021.
 *
 * **The regime panel says what it is not.** Transition probabilities are shown
 * against their base rate rather than against 50%, because an 89% chance of a
 * regime change beside a 97% bull call reads as a contradiction until you know
 * the unconditional rate is 28%.
 */
import {
  getAssetHistory,
  getAssetState,
  getForces,
  getRegimeHistory,
  getSeries,
  getTwoLayerState,
  type ApiResult,
  type AssetHistory,
  type AssetState,
  type ForceHistory,
  type HistoryPoint,
  type RegimeHistory,
  type SeriesDetail,
  type TwoLayerState,
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
} from '../lib/dom';
import { formatCount, formatDate, formatValue, type Tone } from '../lib/format';
import { DEFAULT_RANGE, RANGES, fromDate, rangeFromUrl, writeRangeToUrl, type RangeSpec } from './ranges';

const ASSET = 'equity';
const DEEP_SERIES = 'SHILLER:NOMINAL_PRICE';
/** The daily S&P record, 1927+. Same index as the publication series. */
const DAILY_DEEP_SERIES = 'YAHOO:^GSPC';

/** §3.1 thresholds, mirroring features/kinematics.py. */
const JERK_ELEVATED = 2.0;
const JERK_EXTREME = 3.0;

/** §9 L2 vocabulary, most to least risk-on — the stacking order. */
const REGIMES = ['bull_expansion', 'normal_expansion', 'late_cycle', 'bear', 'crisis'] as const;

const REGIME_TONE: Record<string, Tone> = {
  bull_expansion: 'ok',
  normal_expansion: 'ok',
  late_cycle: 'warn',
  bear: 'bad',
  crisis: 'bad',
};

const host = (id: string) => document.querySelector(`#equity-${id}`);

const hosts = {
  range: host('range'),
  state: host('state'),
  regime: host('regime'),
  transitions: host('transitions'),
  price: host('price'),
  kinematics: host('kinematics'),
  jerk: host('jerk'),
  forces: host('forces'),
  provenance: host('provenance'),
};

// ------------------------------------------------------------------ charts

const W = 900;
const H = 300;
const PAD = { top: 18, right: 18, bottom: 34, left: 66 };

interface Line {
  points: HistoryPoint[];
  className: string;
}

function finite(points: HistoryPoint[]): HistoryPoint[] {
  return points.filter((p) => Number.isFinite(p.value));
}

interface ChartOptions {
  label: string;
  caption: string;
  zeroLine?: boolean;
  bands?: number[];
  /** Log price axis — required once a range spans decades. */
  log?: boolean;
}

/**
 * Several series against a shared date axis.
 *
 * x is proportional to *calendar time*, not to position in the array. With
 * server-side decimation the points are no longer evenly spaced — LTTB keeps
 * more of them around a crash — so a positional axis would stretch the busy
 * stretches and compress the quiet ones.
 */
function lineChart(lines: Line[], opts: ChartOptions): SVGSVGElement {
  const svg = svgEl('svg', {
    class: 'chart',
    viewBox: `0 0 ${W} ${H}`,
    role: 'img',
    preserveAspectRatio: 'xMidYMid meet',
    'aria-label': opts.label,
  });

  const drawn = lines
    .map((l) => ({ ...l, points: finite(l.points) }))
    .filter((l) => l.points.length > 1);
  if (drawn.length === 0) {
    svg.appendChild(
      svgEl('text', { x: W / 2, y: H / 2, 'text-anchor': 'middle' }, 'not enough history to plot'),
    );
    return svg;
  }

  const time = (iso: string) => Date.parse(`${iso}T00:00:00Z`);
  const times = drawn.flatMap((l) => l.points.map((p) => time(p.as_of)));
  const tMin = Math.min(...times);
  const tMax = Math.max(...times);

  // A log axis cannot show a non-positive value; those only occur on the signed
  // panels, which never ask for one.
  const useLog = Boolean(opts.log) && drawn.every((l) => l.points.every((p) => p.value > 0));
  const project = (v: number) => (useLog ? Math.log(v) : v);

  const values = drawn.flatMap((l) => l.points.map((p) => project(p.value)));
  let lo = Math.min(...values, ...(opts.zeroLine && !useLog ? [0] : []));
  let hi = Math.max(...values, ...(opts.zeroLine && !useLog ? [0] : []));
  if (opts.bands) {
    const widest = Math.max(...opts.bands);
    lo = Math.min(lo, -widest);
    hi = Math.max(hi, widest);
  }
  const pad = hi - lo < 1e-9 ? Math.max(Math.abs(hi) * 0.02, 0.01) : (hi - lo) * 0.08;
  const yMin = lo - pad;
  const yMax = hi + pad;

  const plotW = W - PAD.left - PAD.right;
  const plotH = H - PAD.top - PAD.bottom;
  const x = (iso: string) =>
    PAD.left + (tMax === tMin ? 0.5 : (time(iso) - tMin) / (tMax - tMin)) * plotW;
  const y = (v: number) => PAD.top + plotH - ((project(v) - yMin) / (yMax - yMin)) * plotH;

  const TICKS = 4;
  for (let t = 0; t <= TICKS; t++) {
    const projected = yMin + ((yMax - yMin) * t) / TICKS;
    const value = useLog ? Math.exp(projected) : projected;
    const gy = PAD.top + plotH - ((projected - yMin) / (yMax - yMin)) * plotH;
    svg.appendChild(
      svgEl('line', { class: 'grid', x1: PAD.left, x2: W - PAD.right, y1: gy, y2: gy }),
    );
    svg.appendChild(
      svgEl('text', { x: PAD.left - 8, y: gy + 3, 'text-anchor': 'end' }, formatValue(value)),
    );
  }
  svg.appendChild(
    svgEl('line', { class: 'axis', x1: PAD.left, x2: PAD.left, y1: PAD.top, y2: H - PAD.bottom }),
  );

  if (opts.zeroLine && !useLog && yMin < 0 && yMax > 0) {
    svg.appendChild(
      svgEl('line', { class: 'axis', x1: PAD.left, x2: W - PAD.right, y1: y(0), y2: y(0) }),
    );
  }
  for (const band of opts.bands ?? []) {
    for (const level of [band, -band]) {
      if (level < yMin || level > yMax) continue;
      svg.appendChild(
        svgEl('line', {
          class: 'grid',
          'stroke-dasharray': '4 4',
          x1: PAD.left,
          x2: W - PAD.right,
          y1: y(level),
          y2: y(level),
        }),
      );
    }
  }

  for (const line of drawn) {
    svg.appendChild(
      svgEl('path', {
        class: line.className,
        d: line.points
          .map((p, i) => `${i === 0 ? 'M' : 'L'}${x(p.as_of).toFixed(2)},${y(p.value).toFixed(2)}`)
          .join(' '),
      }),
    );
  }

  const first = drawn[0]!.points[0]!.as_of;
  const last = drawn[0]!.points.at(-1)!.as_of;
  svg.appendChild(svgEl('text', { x: PAD.left, y: H - 12 }, first));
  svg.appendChild(svgEl('text', { x: W - PAD.right, y: H - 12, 'text-anchor': 'end' }, last));
  svg.appendChild(
    svgEl('text', { x: PAD.left, y: 12 }, opts.caption + (useLog ? ' · log scale' : '')),
  );
  return svg;
}

/**
 * The regime posterior as a stacked area, ordered most to least risk-on.
 *
 * Stacked because the five probabilities sum to one: the reader should see a
 * full band whose *composition* shifts, not five lines that happen to move
 * together. Ordering matters — crisis on top means the risk-off share grows
 * downward from the roof and is legible at a glance.
 */
function regimeChart(points: RegimeHistory['points']): SVGSVGElement {
  const svg = svgEl('svg', {
    class: 'chart',
    viewBox: `0 0 ${W} ${H}`,
    role: 'img',
    preserveAspectRatio: 'xMidYMid meet',
    'aria-label': `Regime posterior over ${points.length} dates`,
  });

  if (points.length < 2) {
    svg.appendChild(
      svgEl('text', { x: W / 2, y: H / 2, 'text-anchor': 'middle' }, 'no regime history yet'),
    );
    return svg;
  }

  const time = (iso: string) => Date.parse(`${iso}T00:00:00Z`);
  const tMin = time(points[0]!.as_of);
  const tMax = time(points.at(-1)!.as_of);
  const plotW = W - PAD.left - PAD.right;
  const plotH = H - PAD.top - PAD.bottom;
  const x = (iso: string) =>
    PAD.left + (tMax === tMin ? 0.5 : (time(iso) - tMin) / (tMax - tMin)) * plotW;
  const y = (v: number) => PAD.top + plotH - v * plotH;

  // Cumulative bands: each regime's ribbon sits on the sum of the ones above it.
  let below = points.map(() => 0);
  for (const regime of REGIMES) {
    const top = points.map((p, i) => below[i]! + (p.probabilities[regime] ?? 0));
    const forward = points.map((p, i) => `${x(p.as_of).toFixed(2)},${y(top[i]!).toFixed(2)}`);
    const back = points
      .map((p, i) => `${x(p.as_of).toFixed(2)},${y(below[i]!).toFixed(2)}`)
      .reverse();
    svg.appendChild(
      svgEl('path', {
        class: `regimeband regimeband--${regime}`,
        d: `M${forward.join(' L')} L${back.join(' L')} Z`,
      }),
    );
    below = top;
  }

  svg.appendChild(svgEl('text', { x: PAD.left, y: H - 12 }, points[0]!.as_of));
  svg.appendChild(
    svgEl('text', { x: W - PAD.right, y: H - 12, 'text-anchor': 'end' }, points.at(-1)!.as_of),
  );
  svg.appendChild(svgEl('text', { x: PAD.left, y: 12 }, 'P(regime) — stacked, sums to 1'));
  return svg;
}

/** A transition probability as a dial, read against its base rate. */
function dial(months: number, probability: number, base: number | undefined): HTMLElement {
  const pct = Math.round(probability * 100);
  const elevated = base !== undefined && probability > base * 1.15;
  const size = 120;
  const radius = 46;
  const circumference = 2 * Math.PI * radius;

  const svg = svgEl('svg', {
    class: 'dial',
    viewBox: `0 0 ${size} ${size}`,
    role: 'img',
    'aria-label': `${pct} percent probability of entering bear or crisis within ${months} months`,
  });
  svg.appendChild(
    svgEl('circle', { class: 'dial__track', cx: size / 2, cy: size / 2, r: radius }),
  );
  svg.appendChild(
    svgEl('circle', {
      class: `dial__value dial__value--${elevated ? 'warn' : 'ok'}`,
      cx: size / 2,
      cy: size / 2,
      r: radius,
      'stroke-dasharray': `${(circumference * probability).toFixed(2)} ${circumference.toFixed(2)}`,
      transform: `rotate(-90 ${size / 2} ${size / 2})`,
    }),
  );
  if (base !== undefined) {
    // The base rate as a tick: without it the number has no scale, and 89%
    // against a 28% base reads the same as 89% against an 85% one.
    const angle = -90 + 360 * base;
    svg.appendChild(
      svgEl('line', {
        class: 'dial__base',
        x1: size / 2,
        y1: size / 2 - radius + 6,
        x2: size / 2,
        y2: size / 2 - radius - 6,
        transform: `rotate(${angle} ${size / 2} ${size / 2})`,
      }),
    );
  }
  svg.appendChild(
    svgEl('text', { class: 'dial__label', x: size / 2, y: size / 2 + 6, 'text-anchor': 'middle' }, `${pct}%`),
  );

  return el(
    'div',
    { class: 'dialcard' },
    svg,
    el('div', { class: 'dialcard__title' }, `${months} months`),
    el(
      'div',
      { class: 'dialcard__detail' },
      base === undefined
        ? 'P(enters bear or crisis)'
        : `vs ${Math.round(base * 100)}% unconditional`,
    ),
  );
}

// --------------------------------------------------------------- rendering

function tile(label: string, value: string, tone: Tone, detail: string): HTMLElement {
  return el(
    'div',
    { class: `tile tile--${tone}` },
    el('div', { class: 'tile__label' }, label),
    el('div', { class: 'tile__value' }, value),
    el('div', { class: 'tile__detail' }, detail),
  );
}

function jerkLamp(z: number | undefined): { label: string; tone: Tone; detail: string } {
  if (z === undefined || !Number.isFinite(z)) {
    return { label: 'unknown', tone: 'idle', detail: 'The expanding baseline has not filled yet.' };
  }
  const magnitude = Math.abs(z);
  if (magnitude >= JERK_EXTREME) {
    return {
      label: 'extreme',
      tone: 'bad',
      detail: `|z| = ${formatValue(magnitude)}, beyond three standard deviations of this series' own history.`,
    };
  }
  if (magnitude >= JERK_ELEVATED) {
    return {
      label: 'elevated',
      tone: 'warn',
      detail: `|z| = ${formatValue(magnitude)}, beyond two standard deviations. The trend is changing shape.`,
    };
  }
  return {
    label: 'calm',
    tone: 'ok',
    detail: `|z| = ${formatValue(magnitude)}, inside two standard deviations.`,
  };
}

function percent(value: number | undefined, digits = 2): string {
  if (value === undefined || !Number.isFinite(value)) return '—';
  return `${(value * 100).toFixed(digits)}%`;
}

/** Truncation and decimation, stated rather than left to be inferred. */
function coverageNote(history: {
  count: number;
  available: number;
  truncated: boolean;
  decimated: { from: number; to: number; method: string } | null;
}): HTMLElement | null {
  if (history.truncated) {
    return stateBlock({
      tone: 'warn',
      title: 'This series was truncated',
      detail:
        `The window holds ${formatCount(history.available)} observations and the API ` +
        `returned ${formatCount(history.count)}. What is drawn is a prefix of the range, ` +
        'not the whole of it — narrow the range rather than reading the start date off the chart.',
    });
  }
  if (history.decimated) {
    return el(
      'p',
      { class: 'enginecard__detail' },
      `${formatCount(history.decimated.from)} observations, drawn as ` +
        `${formatCount(history.decimated.to)} points (${history.decimated.method.toUpperCase()}). ` +
        'Shape-preserving: single-day crashes survive downsampling.',
    );
  }
  return el(
    'p',
    { class: 'enginecard__detail' },
    `${formatCount(history.count)} observations, drawn in full.`,
  );
}

function renderRangeControl(current: RangeSpec, onSelect: (range: RangeSpec) => void): void {
  if (!hosts.range) return;
  replace(
    hosts.range,
    el(
      'div',
      { class: 'rangebar', role: 'group', 'aria-label': 'Chart range' },
      ...RANGES.map((range) => {
        const button = el(
          'button',
          {
            type: 'button',
            class: `rangebar__button${range.key === current.key ? ' rangebar__button--on' : ''}`,
            'aria-pressed': range.key === current.key ? 'true' : 'false',
          },
          range.label,
        );
        button.addEventListener('click', () => onSelect(range));
        return button;
      }),
    ),
    el('p', { class: 'enginecard__detail' }, current.description),
  );
}

function renderState(result: ApiResult<TwoLayerState>): TwoLayerState | null {
  if (!hosts.state) return null;
  if (!result.ok) {
    replace(hosts.state, failureBlock(result, 'Kinematic state'));
    return null;
  }

  const state = result.envelope.data;
  const k = state.kinematics;
  if (k.as_of === null) {
    replace(
      hosts.state,
      emptyBlock(
        'kinematic state',
        'The equity engine has not published features yet. It publishes on every daily run once its price backbone is backfilled.',
      ),
    );
    return state;
  }

  const f = k.features;
  const lamp = jerkLamp(f.jerk_z);
  const level = f.price_filtered === undefined ? undefined : Math.exp(f.price_filtered);

  replace(
    hosts.state,
    result.envelope.stale
      ? stateBlock({
          tone: 'warn',
          title: 'This state is stale',
          detail: `Last published ${formatDate(k.as_of)}. The daily run has not reported since.`,
        })
      : null,
    el(
      'div',
      { class: 'tiles' },
      tile(
        'Filtered level',
        formatValue(level),
        'info',
        'The local-linear-trend estimate of where the index is, in points — noise removed causally, not by looking ahead.',
      ),
      tile(
        'Velocity',
        percent(f.velocity, 1),
        f.velocity === undefined ? 'idle' : f.velocity >= 0 ? 'ok' : 'bad',
        'Annualized log drift: the trend the filter currently believes in.',
      ),
      tile(
        'Acceleration',
        formatValue(f.acceleration),
        f.acceleration === undefined ? 'idle' : f.acceleration >= 0 ? 'ok' : 'warn',
        'Change in velocity, per year squared. Negative with positive velocity is a rally losing its slope.',
      ),
      tile('Jerk', lamp.label, lamp.tone, lamp.detail),
    ),
    el(
      'p',
      { class: 'enginecard__detail' },
      `Model ${k.model_version ?? 'unknown'} · information set ${k.as_of}`,
    ),
    el(
      'details',
      { class: 'explain' },
      el('summary', {}, 'Every feature, in model units'),
      el(
        'dl',
        { class: 'deflist' },
        ...Object.keys(f)
          .sort()
          .map((key) =>
            el(
              'div',
              { class: 'defrow' },
              el('dt', { class: 'mono' }, key),
              el('dd', { class: 'mono' }, formatValue(f[key])),
            ),
          ),
      ),
    ),
  );
  return state;
}

function renderRegime(
  history: ApiResult<RegimeHistory>,
  state: ApiResult<AssetState>,
): void {
  if (!hosts.regime) return;

  if (!history.ok) {
    replace(hosts.regime, failureBlock(history, 'Regime history'));
    return;
  }
  const data = history.envelope.data;
  if (data.points.length === 0) {
    replace(
      hosts.regime,
      stateBlock({
        tone: 'info',
        title: 'No regime history yet',
        detail:
          'The regime model is fitted by the monthly refit job. Until it has run once, the engine publishes its features and declines to publish a state.',
      }),
    );
    return;
  }

  const latest = data.points.at(-1)!;
  const badgeRow = el(
    'div',
    { class: 'badgerow' },
    badge(latest.regime, REGIME_TONE[latest.regime] ?? 'idle'),
    el(
      'span',
      { class: 'enginecard__detail' },
      `${percent(latest.confidence, 1)} posterior on ${latest.as_of}` +
        (data.model_version ? ` · ${data.model_version}` : ''),
    ),
  );

  const legend = el(
    'div',
    { class: 'legend' },
    ...REGIMES.map((regime) =>
      el(
        'span',
        { class: 'legend__item' },
        el('span', { class: `legend__swatch legend__swatch--${regime}` }),
        regime,
      ),
    ),
  );

  replace(hosts.regime, badgeRow, regimeChart(data.points), legend, coverageNote(data));

  // The current posterior as a table, since a stacked area is hard to read
  // precisely at the right-hand edge — which is the part people care about.
  if (state.ok) {
    const components = state.envelope.data.components ?? {};
    const rows = REGIMES.map((regime) =>
      el(
        'tr',
        {},
        el('td', { class: 'mono nowrap' }, regime),
        el('td', { class: 'num mono' }, percent(latest.probabilities[regime] ?? 0, 1)),
        el(
          'td',
          {},
          components[`regime_mean_return`] !== undefined && regime === latest.regime
            ? badge('current', REGIME_TONE[regime] ?? 'idle')
            : '',
        ),
      ) as HTMLTableRowElement,
    );
    hosts.regime.appendChild(table(['Regime', 'P(regime)', ''], rows));
  }
}

function renderTransitions(state: ApiResult<AssetState>): void {
  if (!hosts.transitions) return;
  if (!state.ok) {
    replace(
      hosts.transitions,
      state.kind === 'not_implemented'
        ? stateBlock({
            tone: 'info',
            title: 'No state published yet',
            detail:
              'The engine publishes its features on every run and declines to publish a state until the monthly refit has fitted the regime model.',
          })
        : failureBlock(state, 'Transition probabilities'),
    );
    return;
  }

  const data = state.envelope.data;
  const components = data.components ?? {};
  const dials = data.signals
    .filter((s) => s.name.startsWith('p_transition_'))
    .map((s) => {
      const months = Number(s.name.replace('p_transition_', '').replace('m', ''));
      return dial(months, s.value, components[`base_rate_${months}m`]);
    });

  const shap = Object.entries(components)
    .filter(([key]) => key.startsWith('shap_'))
    .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]))
    .slice(0, 8);

  replace(
    hosts.transitions,
    dials.length
      ? el('div', { class: 'dialrow' }, ...dials)
      : emptyBlock('transition probabilities', 'This run published none.'),
    stateBlock({
      tone: 'info',
      title: 'Read these against the base rate, not against 50%',
      detail:
        'An "entry" is the regime path moving into bear or crisis from anywhere else. The engine changes regime a few times a year, so entries are frequent and mostly minor — a high probability here is not a forecast of a crash. The tick on each dial marks the unconditional rate.',
    }),
    shap.length
      ? el(
          'details',
          { class: 'explain' },
          el('summary', {}, 'What drove these probabilities (SHAP, log-odds)'),
          el(
            'dl',
            { class: 'deflist' },
            ...shap.map(([key, value]) =>
              el(
                'div',
                { class: 'defrow' },
                el('dt', { class: 'mono' }, key.replace('shap_', '')),
                el('dd', { class: 'mono' }, formatValue(value)),
              ),
            ),
          ),
        )
      : null,
  );
}

function renderPrice(
  close: ApiResult<AssetHistory>,
  filtered: ApiResult<AssetHistory>,
  deep: ApiResult<SeriesDetail> | null,
  range: RangeSpec,
): void {
  if (!hosts.price) return;

  // Deep daily and monthly both read `macro_series` directly: the engine's own
  // price metric is the publication series, which FRED caps at ten years.
  if (range.deep || range.monthly) {
    if (!deep || !deep.ok) {
      const failure = deep && !deep.ok ? deep : { kind: 'unreachable', message: 'not requested' };
      replace(hosts.price, failureBlock(failure, 'Shiller monthly history'));
      return;
    }
    const body = deep.envelope.data;
    const points: HistoryPoint[] = body.observations.map((o) => ({
      as_of: o.obs_date,
      value: o.value,
      meta: null,
      written_at: null,
    }));

    replace(
      hosts.price,
      range.monthly
        ? stateBlock({
            tone: 'info',
            title: 'Monthly resolution — a different series, not a wider window',
            detail: `${DEEP_SERIES}: month-end closes of the S&P composite from 1871. The daily record begins in 1927; everything before that exists only at this resolution.`,
          })
        : stateBlock({
            tone: 'info',
            title: 'Read straight from the observation store',
            detail: `${DAILY_DEEP_SERIES}: daily closes of the same index the engine publishes against. The engine's own price metric comes from FRED:SP500, which is licence-capped to a rolling ten years — so past that, this is the record. The filtered overlay is not drawn here, because the model runs on the publication series and a line that was never computed should not appear.`,
          }),
      lineChart([{ points, className: 'line' }], {
        label: range.monthly
          ? 'Shiller S&P composite, monthly, from 1871'
          : `S&P 500 daily closes, ${range.label}`,
        caption: range.monthly ? 'index points, month-end' : 'index points, daily close',
        log: range.log,
      }),
      coverageNote({
        count: points.length,
        available: body.available,
        truncated: body.truncated,
        decimated: body.decimated,
      }),
    );
    return;
  }

  if (!close.ok) {
    replace(hosts.price, failureBlock(close, 'Price history'));
    return;
  }

  const raw = close.envelope.data;
  if (raw.points.length === 0) {
    replace(
      hosts.price,
      emptyBlock('price history', 'The engine has not published a price path for this range.'),
    );
    return;
  }

  replace(
    hosts.price,
    lineChart(
      [
        { points: raw.points, className: 'line line--muted' },
        { points: filtered.ok ? filtered.envelope.data.points : [], className: 'line' },
      ],
      {
        label: `S&P 500 close and Kalman-filtered level, ${range.label}`,
        caption: 'index points',
        log: range.log,
      },
    ),
    el(
      'div',
      { class: 'legend' },
      el(
        'span',
        { class: 'legend__item' },
        el('span', { class: 'legend__swatch legend__swatch--accent' }),
        'filtered level (Kalman, filtered estimate)',
      ),
      el(
        'span',
        { class: 'legend__item' },
        el('span', { class: 'legend__swatch legend__swatch--idle' }),
        'raw close',
      ),
    ),
    coverageNote(raw),
  );
}

function renderKinematics(
  velocity: ApiResult<AssetHistory>,
  acceleration: ApiResult<AssetHistory>,
  range: RangeSpec,
): void {
  if (!hosts.kinematics) return;
  if (range.monthly) {
    replace(
      hosts.kinematics,
      stateBlock({
        tone: 'info',
        title: 'Not computed at monthly resolution',
        detail:
          'The kinematic features are built on the daily publication series. The 1871 tier is a different series at a different resolution, so its velocity would not be the same quantity.',
      }),
    );
    return;
  }
  if (!velocity.ok) {
    replace(hosts.kinematics, failureBlock(velocity, 'Velocity history'));
    return;
  }
  const v = velocity.envelope.data;
  if (v.points.length === 0) {
    replace(hosts.kinematics, emptyBlock('kinematics', 'Nothing published for this range.'));
    return;
  }

  replace(
    hosts.kinematics,
    lineChart([{ points: v.points, className: 'line' }], {
      label: 'Annualized velocity of the filtered trend',
      caption: 'velocity — annualized log drift',
      zeroLine: true,
    }),
    lineChart(
      [{ points: acceleration.ok ? acceleration.envelope.data.points : [], className: 'line line--alt' }],
      {
        label: 'Acceleration of the filtered trend',
        caption: 'acceleration — per year squared',
        zeroLine: true,
      },
    ),
    coverageNote(v),
  );
}

function renderJerk(result: ApiResult<AssetHistory>, range: RangeSpec): void {
  if (!hosts.jerk) return;
  if (range.monthly) {
    replace(
      hosts.jerk,
      stateBlock({
        tone: 'info',
        title: 'Not computed at monthly resolution',
        detail: 'The jerk indicator is built on the daily publication series.',
      }),
    );
    return;
  }
  if (!result.ok) {
    replace(hosts.jerk, failureBlock(result, 'Jerk indicator'));
    return;
  }

  const data = result.envelope.data;
  if (data.points.length === 0) {
    replace(hosts.jerk, emptyBlock('jerk indicator', 'Nothing published for this range.'));
    return;
  }

  const last = data.points.at(-1);
  const lamp = jerkLamp(last?.value);
  const elevated = data.points.filter((p) => Math.abs(p.value) >= JERK_ELEVATED).length;

  replace(
    hosts.jerk,
    el(
      'div',
      { class: 'badgerow' },
      badge(lamp.label, lamp.tone),
      el('span', { class: 'enginecard__detail' }, `${lamp.detail} As of ${last?.as_of}.`),
    ),
    lineChart([{ points: data.points, className: 'line' }], {
      label: 'Jerk z-score against its expanding baseline',
      caption: 'jerk z-score — dashed lines at ±2 and ±3',
      zeroLine: true,
      bands: [JERK_ELEVATED, JERK_EXTREME],
    }),
    el(
      'p',
      { class: 'enginecard__detail' },
      `${elevated} of ${formatCount(data.points.length)} drawn points were elevated or beyond.`,
    ),
  );
}

function renderForces(result: ApiResult<ForceHistory>, snapshot: TwoLayerState | null): void {
  if (!hosts.forces) return;
  if (!result.ok) {
    replace(hosts.forces, failureBlock(result, 'Force scores'));
    return;
  }

  const scores = snapshot?.forces.scores ?? {};
  const components = snapshot?.forces.components ?? {};
  const names = Object.keys(scores).sort();
  if (names.length === 0) {
    replace(hosts.forces, emptyBlock('force scores', 'The factor pipeline has not published scores yet.'));
    return;
  }

  const rows = names.map((name) => {
    const score = scores[name] ?? 0;
    const breakdown = components[name];
    const tone: Tone = score >= 66 ? 'ok' : score >= 33 ? 'warn' : 'bad';
    return el(
      'tr',
      {},
      el('td', { class: 'mono nowrap' }, name),
      el('td', { class: 'num mono' }, formatValue(score)),
      el('td', {}, badge(score >= 66 ? 'supportive' : score >= 33 ? 'neutral' : 'hostile', tone)),
      el(
        'td',
        {},
        breakdown && Object.keys(breakdown).length
          ? el(
              'details',
              { class: 'explain' },
              el('summary', {}, `${Object.keys(breakdown).length} input(s)`),
              el(
                'dl',
                { class: 'deflist' },
                ...Object.keys(breakdown)
                  .sort()
                  .map((key) =>
                    el(
                      'div',
                      { class: 'defrow' },
                      el('dt', { class: 'mono' }, key),
                      el('dd', { class: 'mono' }, formatValue(breakdown[key])),
                    ),
                  ),
              ),
            )
          : '—',
      ),
    ) as HTMLTableRowElement;
  });

  replace(
    hosts.forces,
    table(['Force', 'Score (0–100)', 'Reading', 'Components'], rows),
    el(
      'p',
      { class: 'enginecard__detail' },
      `Scored on ${snapshot?.forces.as_of ?? 'an unknown date'}.`,
    ),
  );
}

function renderProvenance(state: TwoLayerState | null, range: RangeSpec): void {
  if (!hosts.provenance) return;

  const version = state?.kinematics.model_version ?? null;
  const calibration = version?.split('+cal.')[1] ?? null;
  const isProxy = calibration !== null && !calibration.includes('gspc') && !calibration.includes('spx');

  replace(
    hosts.provenance,
    el(
      'p',
      { class: 'prose' },
      'Three price series feed this engine and they are not interchangeable. The ',
      el('strong', {}, 'publication'),
      ' series is what the kinematics describe. A separate ',
      el('strong', {}, 'calibration'),
      ' series — longer, reaching back through the crises the publication series does not — is what the regime model is fitted on. A monthly ',
      el('strong', {}, 'deep-history'),
      ' series back to 1871 is the only basis for the tail estimates.',
    ),
    el(
      'p',
      { class: 'prose' },
      el('strong', {}, 'On screen now: '),
      range.monthly
        ? `${DEEP_SERIES} — month-end closes from 1871. Monthly, not daily.`
        : range.deep
          ? `${DAILY_DEEP_SERIES} daily closes for the price chart (${range.description}); the kinematics below remain FRED:SP500, which is where the model runs.`
          : `FRED:SP500 for both the price chart and the kinematics, over ${range.description}.`,
    ),
    el(
      'p',
      { class: 'prose' },
      'Which series played which part is recorded in the model version, so a number can never be traced to the wrong index: ',
      version ? el('code', {}, version) : el('em', {}, 'no version published yet'),
      calibration ? el('span', {}, ' — the suffix names the calibration series.') : '',
    ),
    isProxy
      ? stateBlock({
          tone: 'warn',
          title: 'The calibration series is a proxy, not the S&P 500',
          detail:
            'A longer, more volatile index is standing in for the fitted parameters. Every fitted number on this page carries that caveat.',
        })
      : calibration
        ? stateBlock({
            tone: 'info',
            title: 'The regime model is fitted on the S&P itself',
            detail:
              'The daily record back to 1927 fills the backfill role, so 1929, 1987, 2000, 2008 and 2020 are all in sample on the same index this page publishes against.',
          })
        : null,
  );
}

// ------------------------------------------------------------------- main

async function load(range: RangeSpec): Promise<void> {
  for (const target of Object.values(hosts)) {
    if (target && target !== hosts.range) replace(target, loadingBlock(`FinEquity (${range.label})`));
  }

  const from = fromDate(range);
  const history = (metric: string) =>
    getAssetHistory(ASSET, metric, { from, points: range.points });

  const [state, twoLayer, regime, close, filtered, velocity, acceleration, jerk, forces, deep] =
    await Promise.all([
      getAssetState(ASSET),
      getTwoLayerState(),
      getRegimeHistory({ asset: ASSET, from, points: Math.min(range.points, 1200) }),
      range.deep || range.monthly ? Promise.resolve(null) : history('price_close'),
      range.deep || range.monthly ? Promise.resolve(null) : history('price_filtered'),
      range.monthly ? Promise.resolve(null) : history('velocity'),
      range.monthly ? Promise.resolve(null) : history('acceleration'),
      range.monthly ? Promise.resolve(null) : history('jerk_z'),
      getForces({ limit: 200 }),
      range.monthly
        ? getSeries(DEEP_SERIES, { points: range.points })
        : range.deep
          ? getSeries(DAILY_DEEP_SERIES, { from, points: range.points })
          : Promise.resolve(null),
    ]);

  const snapshot = renderState(twoLayer);
  renderRegime(regime, state);
  renderTransitions(state);
  renderPrice(close ?? regimePlaceholder(), filtered ?? regimePlaceholder(), deep, range);
  renderKinematics(velocity ?? regimePlaceholder(), acceleration ?? regimePlaceholder(), range);
  renderJerk(jerk ?? regimePlaceholder(), range);
  renderForces(forces, snapshot);
  renderProvenance(snapshot, range);
}

/** Stand-in for the daily series when the monthly tier is selected. */
function regimePlaceholder(): ApiResult<AssetHistory> {
  return { ok: false, kind: 'not_found', message: 'not requested for this range' };
}

async function main(): Promise<void> {
  let current = rangeFromUrl();

  const select = (range: RangeSpec) => {
    if (range.key === current.key) return;
    current = range;
    writeRangeToUrl(range);
    renderRangeControl(current, select);
    void load(current);
  };

  renderRangeControl(current, select);
  await load(current);
}

void main();

export { DEFAULT_RANGE };
