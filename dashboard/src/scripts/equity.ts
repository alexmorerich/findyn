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
  getForecast,
  getInstability,
  getRegimeHistory,
  getSeries,
  getTwoLayerState,
  type ApiResult,
  type AssetHistory,
  type AssetState,
  type ForceHistory,
  type ForecastResponse,
  type HistoryPoint,
  type InstabilityHistory,
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
/**
 * The daily S&P record as ingested — the same index the engine publishes against.
 *
 * The engine filters this record spliced behind `FRED:SP500`, so once a
 * full-history run has happened its own metrics cover the same span and this is
 * redundant. Until then it is the only place the century exists, and the page
 * falls back to it rather than drawing ten years as though that were all there
 * is. See {@link readCoverage}.
 */
const DAILY_RECORD_SERIES = 'YAHOO:^GSPC';

/**
 * How far the engine's published record may fall short of the ingested one
 * before the page stops treating it as the whole history.
 *
 * Not zero: the engine legitimately starts a little later than the raw record
 * because the Kalman filter's diffuse start-up is discarded and the trailing
 * windows have to fill. Six weeks absorbs that without absorbing a decade.
 */
const COVERAGE_TOLERANCE_DAYS = 45;

/** §3.1 thresholds, mirroring features/kinematics.py. */
const JERK_ELEVATED = 2.0;
const JERK_EXTREME = 3.0;

/** §9 L2 vocabulary, most to least risk-on — the stacking order. */
const REGIMES = ['bull_expansion', 'normal_expansion', 'late_cycle', 'bear', 'crisis'] as const;

/** §3.2 reading thresholds for the RII. Percentiles, so these are percentiles. */
const RII_ELEVATED = 60;
const RII_HIGH = 80;
/** The sparkline window, fixed independently of the page range — see below. */
const RII_SPARKLINE_DAYS = 90;

/** §11 horizons, mirroring core/contracts/vocab.py. */
const HORIZON_YEARS: Record<string, number> = {
  tactical: 0.5,
  strategic: 2,
  generational: 12,
  educational_30y: 30,
  educational_50y: 50,
};

const HORIZON_LABELS: Record<string, string> = {
  tactical: '6 months',
  strategic: '2 years',
  generational: '12 years',
  educational_30y: '30 years',
  educational_50y: '50 years',
};

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
  instability: host('instability'),
  crash: host('crash'),
  implication: host('implication'),
  forecast: host('forecast'),
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

// --------------------------------------------------------------- coverage

/**
 * What the engine has actually published, against what has actually been
 * ingested.
 *
 * These are two different things and the page used to assume they were one. The
 * engine's kinematics run over the daily S&P record back to 1927, but they only
 * *reach* D1 when a `--full-history` run publishes them; a nightly run
 * republishes five years. Between a deploy and that backfill, asking the engine
 * for "everything" returns whatever the last run happened to write — which is a
 * well-formed response, with a real `available` count, that silently describes a
 * decade as though it were the whole record.
 *
 * So the page measures the gap instead of assuming there is none. Nothing here
 * is a workaround for a slow backfill: `engineFrom` and `recordFrom` are facts a
 * chart of a century has to know regardless, because they are what distinguishes
 * "the market began in 2016" from "we have not published that far back yet".
 */
interface Coverage {
  /** Oldest date the engine has published `price_close` for, over all time. */
  engineFrom: string | null;
  /** Rows of `price_close` the engine has published, over all time. */
  engineRows: number;
  /** Oldest date the observation store holds for the same index. */
  recordFrom: string | null;
  recordRows: number;
  /** The engine's published record is materially shorter than the ingested one. */
  short: boolean;
}

function daysBetween(from: string, to: string): number {
  return Math.round((Date.parse(`${to}T00:00:00Z`) - Date.parse(`${from}T00:00:00Z`)) / 86_400_000);
}

/**
 * Two single-row probes: what the engine published, and what was ingested.
 *
 * `limit=1` on both, so each response is one observation plus the counts —
 * a few hundred bytes to answer a question that decides which source the page
 * draws from. Asking for the full record up front and discarding it would cost
 * the reader a megabyte on every load once the backfill has run.
 */
async function readCoverage(): Promise<Coverage> {
  const [engine, record] = await Promise.all([
    getAssetHistory(ASSET, 'price_close', { limit: 1 }),
    getSeries(DAILY_RECORD_SERIES, { limit: 1 }),
  ]);

  const engineFrom = engine.ok ? (engine.envelope.data.points[0]?.as_of ?? null) : null;
  const engineRows = engine.ok ? engine.envelope.data.available : 0;

  const detail = record.ok ? record.envelope.data : null;
  // `first_observation` is the metadata's own claim; the oldest returned row is
  // the same fact read from the data. Prefer the metadata, fall back to the row,
  // because an ingest that has not refreshed metadata still has real rows.
  const recordFrom =
    detail?.metadata.first_observation ?? detail?.observations[0]?.obs_date ?? null;
  const recordRows = detail?.available ?? 0;

  const short =
    recordFrom !== null &&
    (engineFrom === null || daysBetween(recordFrom, engineFrom) > COVERAGE_TOLERANCE_DAYS);

  return { engineFrom, engineRows, recordFrom, recordRows, short };
}

/** True when the requested window reaches back past what the engine published. */
function reachesPastEngine(coverage: Coverage, range: RangeSpec): boolean {
  if (!coverage.short) return false;
  if (coverage.engineFrom === null) return true;
  const from = fromDate(range);
  // `undefined` is the Max range: it asks for everything, so it always reaches.
  return from === undefined || from < coverage.engineFrom;
}

/**
 * The notice a panel shows when it is drawing less than the record holds.
 *
 * Stated in both directions — what exists, and what is drawn — because the whole
 * failure being prevented is a reader inferring the span from the axis labels.
 */
function shortfallNotice(coverage: Coverage, what: string): HTMLElement {
  const detail =
    coverage.engineFrom === null
      ? `The engine has published no ${what} at all, while the observation store holds ${formatCount(coverage.recordRows)} daily closes from ${formatDate(coverage.recordFrom ?? '')}. Run the monthly refit, then \`jobs.daily --full-history\`.`
      : `The engine has published ${what} from ${formatDate(coverage.engineFrom)} — ${formatCount(coverage.engineRows)} dates — while the observation store holds ${formatCount(coverage.recordRows)} daily closes from ${formatDate(coverage.recordFrom ?? '')}. This is a backfill that has not run, not the start of the market: the nightly job republishes a five-year window, and the full record reaches D1 only from \`jobs.daily --full-history\`.`;
  return stateBlock({ tone: 'warn', title: `Historical publication incomplete`, detail });
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
  coverage: Coverage,
  range: RangeSpec,
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

  replace(
    hosts.regime,
    reachesPastEngine(coverage, range) ? shortfallNotice(coverage, 'a regime posterior') : null,
    badgeRow,
    regimeChart(data.points),
    legend,
    coverageNote(data),
  );

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

/**
 * §3.2 — the RII, as a gauge with its own recent history.
 *
 * The sparkline is fixed at 90 days regardless of the page range. The index is a
 * percentile against expanding history, so "68" only means something next to
 * what it was last quarter; stretching it across a century turns the reading
 * that matters — the last few weeks — into two pixels.
 */
function renderInstability(
  instability: ApiResult<InstabilityHistory>,
  state: ApiResult<AssetState>,
): void {
  if (!hosts.instability) return;
  if (!instability.ok) {
    replace(hosts.instability, failureBlock(instability, 'Regime instability'));
    return;
  }

  const points = instability.envelope.data.points;
  const withRii = points.filter((p) => p.rii !== null);
  const latest = withRii.at(-1);

  if (!latest || latest.rii === null) {
    replace(
      hosts.instability,
      emptyBlock(
        'an instability index',
        'The engine publishes one once the regime model has been fitted and the slowest RII component has filled its window.',
      ),
    );
    return;
  }

  const recent = withRii.slice(-RII_SPARKLINE_DAYS);
  const value = latest.rii;
  const tone: Tone = value >= RII_HIGH ? 'bad' : value >= RII_ELEVATED ? 'warn' : 'ok';
  const previous = recent.at(0)?.rii ?? value;
  const move = value - previous;

  const components = state.ok ? (state.envelope.data.components ?? {}) : {};
  const parts = Object.entries(components)
    .filter(([key]) => key.startsWith('rii_') && !key.endsWith('_weight'))
    .sort((a, b) => b[1] - a[1]);

  replace(
    hosts.instability,
    el(
      'div',
      { class: 'dialrow' },
      gauge(value, tone),
      tile(
        'Over 90 days',
        `${move >= 0 ? '+' : ''}${formatValue(move)}`,
        move > 0 ? 'warn' : 'ok',
        `${formatValue(previous)} on ${formatDate(recent.at(0)?.as_of ?? latest.as_of)}, ${formatValue(value)} now.`,
      ),
      tile(
        'Components',
        String(parts.length),
        parts.length >= 5 ? 'ok' : 'warn',
        parts.length >= 5
          ? 'Missing components are dropped and the weights renormalized, never scored zero.'
          : 'Fewer than the seven §3.2 names — an absent input is reported, not treated as calm.',
      ),
    ),
    recent.length > 2
      ? lineChart(
          [
            {
              points: recent.map((p) => ({ as_of: p.as_of, value: p.rii as number, meta: null })),
              className: 'line line--primary',
            },
          ],
          {
            label: 'Regime instability index over the last 90 published days',
            caption: `RII, last ${recent.length} published days · 0–100`,
            bands: [RII_ELEVATED, RII_HIGH],
          },
        )
      : null,
    parts.length
      ? el(
          'details',
          { class: 'explain' },
          el('summary', {}, 'What the index is made of (each 0–100 on its own axis)'),
          el(
            'dl',
            { class: 'deflist' },
            ...parts.map(([key, componentValue]) =>
              el(
                'div',
                { class: 'defrow' },
                el('dt', { class: 'mono' }, key.replace('rii_', '')),
                el('dd', { class: 'mono' }, formatValue(componentValue)),
              ),
            ),
          ),
        )
      : null,
    stateBlock({
      tone: 'info',
      title: 'This replaces the fourth derivative, deliberately',
      detail:
        'Snap — d⁴price/dt⁴ — is not published. Each differentiation amplifies noise, and a fourth derivative of a daily index is essentially all microstructure. The question "how close is this to a state transition" is answered instead by seven things that can actually be measured, of which the model\u2019s own uncertainty is two.',
    }),
  );
}

/** The RII as a half-circle gauge. */
function gauge(value: number, tone: Tone): HTMLElement {
  const size = 120;
  const radius = 46;
  const circumference = 2 * Math.PI * radius;
  const fraction = Math.min(Math.max(value / 100, 0), 1);

  const svg = svgEl('svg', {
    class: 'dial',
    viewBox: `0 0 ${size} ${size}`,
    role: 'img',
    'aria-label': `Regime instability index ${Math.round(value)} of 100`,
  });
  svg.appendChild(svgEl('circle', { class: 'dial__track', cx: size / 2, cy: size / 2, r: radius }));
  svg.appendChild(
    svgEl('circle', {
      class: `dial__value dial__value--${tone}`,
      cx: size / 2,
      cy: size / 2,
      r: radius,
      'stroke-dasharray': `${(circumference * fraction).toFixed(2)} ${circumference.toFixed(2)}`,
      transform: `rotate(-90 ${size / 2} ${size / 2})`,
    }),
  );
  svg.appendChild(
    svgEl(
      'text',
      { class: 'dial__label', x: size / 2, y: size / 2 + 6, 'text-anchor': 'middle' },
      String(Math.round(value)),
    ),
  );

  return el(
    'div',
    { class: 'dialcard' },
    svg,
    el('div', { class: 'dialcard__title' }, 'RII'),
    el('div', { class: 'dialcard__detail' }, 'percentile against expanding history'),
  );
}

/**
 * §4 — crash risk as **three bars**, never one number.
 *
 * The composite is a product, so a single figure cannot distinguish "the regime
 * is about to turn but the system is robust" from "the system is fragile but
 * the regime is stable". Those call for opposite responses. The product is
 * shown, small, underneath — it is a summary of the three, not a substitute.
 */
function renderCrash(state: ApiResult<AssetState>): void {
  if (!hosts.crash) return;
  if (!state.ok) {
    replace(hosts.crash, failureBlock(state, 'Crash decomposition'));
    return;
  }

  const data = state.envelope.data;
  const components = data.components ?? {};
  const signal = (name: string) => data.signals.find((s) => s.name === name)?.value;

  const transitionKey = Object.keys(components).find((k) => /^p_transition_\d+m$/.test(k));
  const transition = signal('p_transition') ?? (transitionKey ? components[transitionKey] : undefined);
  const shock = signal('p_shock') ?? components.p_shock;
  const transmission = signal('p_transmission') ?? components.p_transmission;
  const composite = signal('crash_risk') ?? components.crash_risk;

  const factors: Array<[string, number | undefined, string]> = [
    [
      'P(transition)',
      transition,
      'Probability the regime *enters* bear or crisis over the horizon, propagated from today’s posterior with the adverse states made absorbing. Derived from the posterior and the transition matrix, not from the classifier.',
    ],
    [
      'P(shock)',
      shock,
      'Probability a drawdown past the severity threshold begins, from a generalized-Pareto fit to declustered historical drawdown episodes — one observation per episode, not per month underwater.',
    ],
    [
      'P(transmission)',
      transmission,
      'Whether a shock would propagate or be absorbed: credit spread level and velocity, financial conditions, and the curve. Floored, because a benign reading means a shock transmits less, not that it fails to transmit.',
    ],
  ];

  const available = factors.filter(([, v]) => v !== undefined && Number.isFinite(v));

  replace(
    hosts.crash,
    available.length === 3
      ? el(
          'div',
          { class: 'barlist' },
          ...available.map(([label, v, detail]) => factorBar(label, v as number, detail)),
        )
      : emptyBlock(
          'a crash decomposition',
          'All three factors are published together or not at all — a composite without its parts is exactly the number §4 forbids.',
        ),
    composite !== undefined && available.length === 3
      ? el(
          'p',
          { class: 'enginecard__detail' },
          `Composite: ${formatValue(composite)} / 100 — the product of the three above, published beside them and never instead of them.`,
        )
      : null,
    stateBlock({
      tone: 'info',
      title: 'Three factors, because they call for different responses',
      detail:
        'A high P(transition) with low fragility is a market that may turn without breaking. Low P(transition) with high fragility is a market that is fine until it is not. Multiplying them into one number erases the distinction that matters.',
    }),
  );
}

function factorBar(label: string, value: number, detail: string): HTMLElement {
  const pct = Math.min(Math.max(value, 0), 1) * 100;
  const tone: Tone = pct >= 60 ? 'bad' : pct >= 30 ? 'warn' : 'ok';
  return el(
    'div',
    { class: 'barrow' },
    el(
      'div',
      { class: 'barrow__head' },
      el('span', { class: 'barrow__label' }, label),
      el('span', { class: 'barrow__value mono' }, `${pct.toFixed(1)}%`),
    ),
    el(
      'div',
      { class: 'bar', role: 'img', 'aria-label': `${label} ${pct.toFixed(1)} percent` },
      el('div', { class: `bar__fill bar__fill--${tone}`, style: `width:${pct.toFixed(2)}%` }),
    ),
    el('p', { class: 'barrow__detail' }, detail),
  );
}

/**
 * §12 — the conditional implication, templated from the published numbers.
 *
 * §0's non-goal 2 is "no trading commands", and this is the sentence that has to
 * respect it while still being useful. Three rules hold it in place:
 *
 * - It is **conditional**, not imperative. "Investors with a short horizon may
 *   consider" is a statement about a class of constraint; "reduce equity" is an
 *   instruction to the reader, whose constraints this system does not know.
 * - It is **assembled from the published numbers**, in the browser, from the
 *   same fields shown above it. There is no hidden model behind the sentence and
 *   nothing it says cannot be checked against the panels.
 * - It **says what it does not know**. Tax jurisdiction, horizon, liabilities and
 *   risk tolerance are exactly what turns a distribution into a decision, and
 *   none of them are inputs here.
 */
function renderImplication(state: ApiResult<AssetState>, instability: ApiResult<InstabilityHistory>): void {
  if (!hosts.implication) return;
  if (!state.ok) {
    replace(hosts.implication, failureBlock(state, 'Conditional implication'));
    return;
  }

  const data = state.envelope.data;
  const components = data.components ?? {};
  const signal = (name: string) => data.signals.find((s) => s.name === name)?.value;
  const latest = instability.ok
    ? instability.envelope.data.points.filter((p) => p.rii !== null).at(-1)
    : undefined;

  const regime = data.regime?.replace(/_/g, ' ') ?? 'an unpublished regime';
  const rii = latest?.rii ?? null;
  const transmission = signal('p_transmission') ?? components.p_transmission;

  const clauses: string[] = [
    `The model reads the current state as ${regime}` +
      (data.confidence == null
        ? '.'
        : ` at ${Math.round(data.confidence * 100)}% posterior probability.`),
  ];

  if (rii !== null) {
    clauses.push(
      rii >= RII_HIGH
        ? `Regime instability is in the top of its historical range (${formatValue(rii)}/100), which is a statement about how close the system is to a state transition — in either direction.`
        : rii >= RII_ELEVATED
          ? `Regime instability is elevated (${formatValue(rii)}/100) but not extreme.`
          : `Regime instability is unremarkable against this series' own history (${formatValue(rii)}/100).`,
    );
  }

  if (transmission !== undefined && Number.isFinite(transmission)) {
    clauses.push(
      transmission >= 0.5
        ? 'Credit and liquidity conditions are such that a shock would propagate rather than be absorbed.'
        : 'Credit and liquidity conditions currently look absorptive rather than transmissive, which lowers the composite without lowering the probability of a shock arriving.',
    );
  }

  const implication =
    rii !== null && rii >= RII_ELEVATED
      ? 'Under this distribution, investors whose horizon is short enough that a multi-year drawdown could not be waited out may consider whether their equity concentration and liquidity match that constraint.'
      : 'Under this distribution, nothing in the published state distinguishes now from an ordinary period for an investor whose horizon spans the forecast horizons above.';

  replace(
    hosts.implication,
    el('p', { class: 'prose' }, clauses.join(' ')),
    el('p', { class: 'prose prose--lead' }, implication),
    stateBlock({
      tone: 'info',
      title: 'What this sentence cannot know',
      detail:
        'Horizon, tax jurisdiction, liabilities and risk tolerance are what turn a distribution into a decision, and none of them are inputs to this system. That is why the sentence is conditional on a constraint rather than addressed to you, and why no endpoint anywhere in this API returns an action.',
    }),
  );
}

/**
 * §11 — the Monte Carlo distribution as a fan, with the educational horizons
 * held apart.
 *
 * Thirty- and fifty-year projections are in the spec as *illustrations of
 * compounding*, and §10 excludes them from accuracy evaluation entirely. Drawn
 * on the same axis as the six-month band they would read as forecasts of the
 * same kind, so they get their own strip below the rule, greyed, and labelled.
 */
function renderForecast(forecast: ApiResult<ForecastResponse>): void {
  if (!hosts.forecast) return;
  if (!forecast.ok) {
    replace(hosts.forecast, failureBlock(forecast, 'Forecast distribution'));
    return;
  }

  const bands = forecast.envelope.data.horizons;
  if (!bands.length) {
    replace(
      hosts.forecast,
      emptyBlock(
        'a forecast distribution',
        'Published once the regime model has been fitted — the simulation is conditioned on the posterior.',
      ),
    );
    return;
  }

  const real = bands.filter((b) => !b.educational_only);
  const educational = bands.filter((b) => b.educational_only);
  const order = (b: ForecastResponse['horizons'][number]) => HORIZON_YEARS[b.horizon] ?? 0;
  real.sort((a, b) => order(a) - order(b));
  educational.sort((a, b) => order(a) - order(b));

  replace(
    hosts.forecast,
    real.length ? fanChart(real, 'Forecast horizons') : null,
    ...real.map(bandRow),
    educational.length
      ? el(
          'div',
          { class: 'educational' },
          el('h3', { class: 'educational__title' }, 'Educational horizons — not forecasts'),
          el(
            'p',
            { class: 'prose' },
            'Thirty and fifty years out, the distribution is an illustration of what compounding and its dispersion look like, not a claim about 2076. These are excluded from accuracy evaluation by §10 and are drawn apart from the horizons above for that reason.',
          ),
          ...educational.map(bandRow),
        )
      : null,
    stateBlock({
      tone: 'info',
      title: 'Quantiles, never a target',
      detail:
        'There is no “S&P at X” anywhere in this system, by construction: the table these come from has no column for a point estimate. Each band is the distribution of terminal levels across at least ten thousand simulated paths, conditioned on today’s regime posterior and carrying the same tail model the crash decomposition uses.',
    }),
  );
}

function bandRow(band: ForecastResponse['horizons'][number]): HTMLElement {
  const level = (q: string) => {
    const value = band.quantiles[q];
    return value === undefined ? '—' : formatCount(Math.round(Math.exp(value)));
  };
  return el(
    'div',
    { class: `bandrow${band.educational_only ? ' bandrow--educational' : ''}` },
    el(
      'div',
      { class: 'bandrow__head' },
      el('span', { class: 'barrow__label' }, HORIZON_LABELS[band.horizon] ?? band.horizon),
      band.educational_only ? badge('educational', 'idle') : null,
    ),
    el(
      'dl',
      { class: 'deflist deflist--inline' },
      ...['0.05', '0.25', '0.5', '0.75', '0.95'].map((q) =>
        el(
          'div',
          { class: 'defrow' },
          el('dt', { class: 'mono' }, `p${Math.round(Number(q) * 100)}`),
          el('dd', { class: 'mono' }, level(q)),
        ),
      ),
    ),
  );
}

/** Quantile bands against horizon, as nested ribbons around the median. */
function fanChart(bands: ForecastResponse['horizons'], label: string): SVGSVGElement {
  const svg = svgEl('svg', {
    class: 'chart',
    viewBox: `0 0 ${W} ${H}`,
    role: 'img',
    preserveAspectRatio: 'xMidYMid meet',
    'aria-label': label,
  });

  const years = bands.map((b) => HORIZON_YEARS[b.horizon] ?? 0);
  const all = bands.flatMap((b) => Object.values(b.quantiles));
  const minY = Math.min(...all);
  const maxY = Math.max(...all);
  const maxX = Math.max(...years, 1);

  const x = (year: number) => PAD.left + (year / maxX) * (W - PAD.left - PAD.right);
  const y = (value: number) =>
    H - PAD.bottom - ((value - minY) / (maxY - minY || 1)) * (H - PAD.top - PAD.bottom);

  // Widest band first, so the narrower ones layer over it.
  for (const [lo, hi, opacity] of [
    ['0.05', '0.95', 0.14],
    ['0.25', '0.75', 0.24],
  ] as const) {
    const upper = bands.map((b, i) => `${x(years[i]!)},${y(b.quantiles[hi] ?? 0)}`);
    const lower = bands
      .map((b, i) => `${x(years[i]!)},${y(b.quantiles[lo] ?? 0)}`)
      .reverse();
    svg.appendChild(
      svgEl('polygon', {
        class: 'fan__band',
        points: [...upper, ...lower].join(' '),
        'fill-opacity': String(opacity),
      }),
    );
  }

  svg.appendChild(
    svgEl('polyline', {
      class: 'line line--primary',
      points: bands.map((b, i) => `${x(years[i]!)},${y(b.quantiles['0.5'] ?? 0)}`).join(' '),
    }),
  );

  bands.forEach((band, i) => {
    const px = x(years[i]!);
    svg.appendChild(
      svgEl('circle', { class: 'fan__point', cx: px, cy: y(band.quantiles['0.5'] ?? 0), r: 3 }),
    );
    svg.appendChild(
      svgEl(
        'text',
        { class: 'chart__tick', x: px, y: H - 12, 'text-anchor': 'middle' },
        HORIZON_LABELS[band.horizon] ?? band.horizon,
      ),
    );
  });

  svg.appendChild(
    svgEl('text', { x: PAD.left, y: 12 }, 'log index level — median, 50% and 90% bands'),
  );
  return svg;
}

/** `macro_series` observations in the shape the charts take. */
function asPoints(observations: SeriesDetail['observations']): HistoryPoint[] {
  return observations.map((o) => ({
    as_of: o.obs_date,
    value: o.value,
    meta: null,
    written_at: null,
  }));
}

function renderPrice(
  close: ApiResult<AssetHistory>,
  filtered: ApiResult<AssetHistory>,
  shiller: ApiResult<SeriesDetail> | null,
  record: ApiResult<SeriesDetail> | null,
  coverage: Coverage,
  range: RangeSpec,
): void {
  if (!hosts.price) return;

  // The 1871 tier reads `macro_series` because it is a *monthly* series — a
  // different quantity, not a wider window — and it would do so however much
  // history the engine had published.
  if (range.monthly) {
    if (!shiller || !shiller.ok) {
      const failure = shiller && !shiller.ok
        ? shiller
        : { kind: 'unreachable', message: 'not requested' };
      replace(hosts.price, failureBlock(failure, 'Shiller monthly history'));
      return;
    }
    const body = shiller.envelope.data;
    const points = asPoints(body.observations);

    replace(
      hosts.price,
      stateBlock({
        tone: 'info',
        title: 'Monthly resolution — a different series, not a wider window',
        detail: `${DEEP_SERIES}: month-end closes of the S&P composite from 1871. The daily record begins in 1927; everything before that exists only at this resolution. No filtered overlay is drawn, because the model runs on the daily path and a line that was never computed should not appear.`,
      }),
      lineChart([{ points, className: 'line' }], {
        label: 'Shiller S&P composite, monthly, from 1871',
        caption: 'index points, month-end',
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

  // The window reaches back past what the engine has published, and the same
  // index is sitting in the observation store in full. Draw the record rather
  // than a truncated version of the engine's view of it — and say which one this
  // is, because they are not the same series of numbers.
  //
  // No filtered overlay here: the model never ran on these dates, and drawing a
  // line that was never computed is the one thing worse than a short chart.
  if (reachesPastEngine(coverage, range)) {
    if (!record || !record.ok) {
      const failure =
        record && !record.ok ? record : { kind: 'unreachable', message: 'not requested' };
      replace(
        hosts.price,
        shortfallNotice(coverage, 'a price path'),
        failureBlock(failure, `${DAILY_RECORD_SERIES} daily closes`),
      );
      return;
    }

    const body = record.envelope.data;
    replace(
      hosts.price,
      shortfallNotice(coverage, 'a filtered price path'),
      stateBlock({
        tone: 'info',
        title: 'Drawn from the observation store, not from the model',
        detail: `${DAILY_RECORD_SERIES}: daily closes of the same index the engine publishes against, read straight out of \`macro_series\`. The filtered overlay is absent because the model has not published over this span — once the backfill runs, this panel returns to the engine's own path and both lines are drawn.`,
      }),
      lineChart([{ points: asPoints(body.observations), className: 'line' }], {
        label: `S&P 500 daily closes, ${range.label}`,
        caption: 'index points, daily close',
        log: range.log,
      }),
      coverageNote({
        count: body.observations.length,
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
  coverage: Coverage,
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

  // No fallback exists here and none should be invented: velocity is a *state
  // estimate*, and there is no second source for one. Where the price panel can
  // fall back to the observation store, this panel can only say what it has.
  replace(
    hosts.kinematics,
    reachesPastEngine(coverage, range) ? shortfallNotice(coverage, 'trend states') : null,
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

function renderJerk(
  result: ApiResult<AssetHistory>,
  coverage: Coverage,
  range: RangeSpec,
): void {
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
    reachesPastEngine(coverage, range) ? shortfallNotice(coverage, 'a jerk z-score') : null,
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

function renderProvenance(
  state: TwoLayerState | null,
  coverage: Coverage,
  range: RangeSpec,
): void {
  if (!hosts.provenance) return;

  const version = state?.kinematics.model_version ?? null;
  const calibration = version?.split('+cal.')[1] ?? null;
  const isProxy = calibration !== null && !calibration.includes('gspc') && !calibration.includes('spx');
  const split = reachesPastEngine(coverage, range);

  replace(
    hosts.provenance,
    el(
      'p',
      { class: 'prose' },
      'Several price series feed this engine and they are not interchangeable. The ',
      el('strong', {}, 'publication'),
      ' path is what the kinematics describe: FRED:SP500 for every date it covers, spliced behind it the same index from YAHOO:^GSPC for the ninety years FRED does not licence. A separate ',
      el('strong', {}, 'calibration'),
      ' series is what the regime model is fitted on. A monthly ',
      el('strong', {}, 'deep-history'),
      ' series back to 1871 is the only basis for the tail estimates.',
    ),
    el(
      'p',
      { class: 'prose' },
      el('strong', {}, 'On screen now: '),
      range.monthly
        ? `${DEEP_SERIES} — month-end closes from 1871. Monthly, not daily.`
        : split
          ? `${DAILY_RECORD_SERIES} daily closes for the price chart over ${range.description}, read from the observation store because the engine has not published that far back yet. The panels below show only what the engine has published, and say where that starts — they are not the same span as the chart above.`
          : `the engine's own filtered path over ${range.description}. The chart and the velocity below are the same computation on the same dates, so the two cannot disagree about where the trend was.`,
    ),
    el(
      'p',
      { class: 'prose' },
      'The two vendors are checked against each other before they are joined: they overlap by the whole ten years FRED covers, and the median relative gap across those dates is 2e-8. A pair that did not agree would not be spliced, and the model version would say so.',
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

  // The coverage probe runs *beside* the panel requests rather than in front of
  // them: it only decides which source the price panel draws from, so blocking
  // the whole page on it would cost every reader a round trip to answer a
  // question that is usually "the engine has everything".
  const [
    coverage,
    state,
    twoLayer,
    regime,
    instability,
    forecast,
    close,
    filtered,
    velocity,
    acceleration,
    jerk,
    forces,
    shiller,
  ] = await Promise.all([
      readCoverage(),
      getAssetState(ASSET),
      getTwoLayerState(),
      getRegimeHistory({ asset: ASSET, from, points: Math.min(range.points, 1200) }),
      // Fixed window, not the page range: the RII sparkline is a 90-day read and
      // the crash factors are a snapshot. Neither gets more meaningful with a
      // century of context, and the request would cost the reader a second or two.
      getInstability({ asset: ASSET, points: 400 }),
      getForecast({ asset: ASSET }),
      range.monthly ? Promise.resolve(null) : history('price_close'),
      range.monthly ? Promise.resolve(null) : history('price_filtered'),
      range.monthly ? Promise.resolve(null) : history('velocity'),
      range.monthly ? Promise.resolve(null) : history('acceleration'),
      range.monthly ? Promise.resolve(null) : history('jerk_z'),
      getForces({ limit: 200 }),
      range.monthly ? getSeries(DEEP_SERIES, { points: range.points }) : Promise.resolve(null),
    ]);

  // Only fetched when the engine has not published this far back — one extra
  // request in the degraded case, none in the healthy one.
  const record =
    !range.monthly && reachesPastEngine(coverage, range)
      ? await getSeries(DAILY_RECORD_SERIES, { from, points: range.points })
      : null;

  const snapshot = renderState(twoLayer);
  renderRegime(regime, state, coverage, range);
  renderTransitions(state);
  renderInstability(instability, state);
  renderCrash(state);
  renderForecast(forecast);
  renderImplication(state, instability);
  renderPrice(
    close ?? regimePlaceholder(),
    filtered ?? regimePlaceholder(),
    shiller,
    record,
    coverage,
    range,
  );
  renderKinematics(
    velocity ?? regimePlaceholder(),
    acceleration ?? regimePlaceholder(),
    coverage,
    range,
  );
  renderJerk(jerk ?? regimePlaceholder(), coverage, range);
  renderForces(forces, snapshot);
  renderProvenance(snapshot, coverage, range);
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
