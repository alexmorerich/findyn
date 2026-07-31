/**
 * FinEquity page, sub-milestone A (04-ui-plan.md §P3).
 *
 * Six views over the causal feature path: the K(t) tiles, an explicit
 * placeholder where the regime will land, the filtered level against the close,
 * velocity and acceleration, the jerk lamp, and F(t) with its breakdowns.
 *
 * Two API surfaces, deliberately. `/state` carries the *snapshot* of both
 * layers, which is one request instead of eleven; `/assets/equity/history`
 * carries the series the charts draw. The snapshot is in model units — logs,
 * annualized rates, z-scores — and the histories are in the units a reader
 * expects, which is why the price chart never has to exponentiate anything.
 *
 * The regime panel says "not computed yet" rather than being omitted. The engine
 * is live and publishing features while deliberately declining to publish an
 * AssetState, and a page that quietly showed only kinematics would leave a
 * reader to conclude the market has no regime.
 */
import {
  getAssetHistory,
  getForces,
  getTwoLayerState,
  type ApiResult,
  type AssetHistory,
  type ForceHistory,
  type HistoryPoint,
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
import { formatDate, formatValue, type Tone } from '../lib/format';

const ASSET = 'equity';

/** Five years of daily observations, matching the engine's publish window. */
const HISTORY_LIMIT = 1400;

const stateHost = document.querySelector('#equity-state');
const regimeHost = document.querySelector('#equity-regime');
const priceHost = document.querySelector('#equity-price');
const kinematicsHost = document.querySelector('#equity-kinematics');
const jerkHost = document.querySelector('#equity-jerk');
const forcesHost = document.querySelector('#equity-forces');
const provenanceHost = document.querySelector('#equity-provenance');

/** §3.1 thresholds, mirroring features/kinematics.py. */
const JERK_ELEVATED = 2.0;
const JERK_EXTREME = 3.0;

// ------------------------------------------------------------------ charts

const W = 900;
const H = 300;
const PAD = { top: 18, right: 18, bottom: 34, left: 66 };

interface Line {
  points: HistoryPoint[];
  className: string;
  label: string;
}

function finite(points: HistoryPoint[]): HistoryPoint[] {
  return points.filter((p) => Number.isFinite(p.value));
}

/**
 * Several series against a shared date axis and a shared y scale.
 *
 * x is positional over the union of dates rather than proportional to calendar
 * time, because trading days are not evenly spaced and a time-proportional axis
 * puts visible gaps at every weekend on a five-year daily chart.
 */
function lineChart(
  lines: Line[],
  opts: { label: string; caption: string; zeroLine?: boolean; bands?: number[] } = {
    label: '',
    caption: '',
  },
): SVGSVGElement {
  const svg = svgEl('svg', {
    class: 'chart',
    viewBox: `0 0 ${W} ${H}`,
    role: 'img',
    preserveAspectRatio: 'xMidYMid meet',
    'aria-label': opts.label,
  });

  const drawn = lines.map((l) => ({ ...l, points: finite(l.points) })).filter((l) => l.points.length > 1);
  if (drawn.length === 0) {
    svg.appendChild(
      svgEl('text', { x: W / 2, y: H / 2, 'text-anchor': 'middle' }, 'not enough history to plot'),
    );
    return svg;
  }

  // One index per date, shared across series so they line up even where one
  // starts later than another — jerk_z waits for its z-score baseline.
  const dates = [...new Set(drawn.flatMap((l) => l.points.map((p) => p.as_of)))].sort();
  const index = new Map(dates.map((d, i) => [d, i]));

  const values = drawn.flatMap((l) => l.points.map((p) => p.value));
  let lo = Math.min(...values, ...(opts.zeroLine ? [0] : []));
  let hi = Math.max(...values, ...(opts.zeroLine ? [0] : []));
  if (opts.bands) {
    lo = Math.min(lo, -Math.max(...opts.bands));
    hi = Math.max(hi, Math.max(...opts.bands));
  }
  const pad = hi - lo < 1e-9 ? Math.max(Math.abs(hi) * 0.02, 0.01) : (hi - lo) * 0.08;
  const yMin = lo - pad;
  const yMax = hi + pad;

  const plotW = W - PAD.left - PAD.right;
  const plotH = H - PAD.top - PAD.bottom;
  const x = (date: string) => PAD.left + ((index.get(date) ?? 0) / Math.max(dates.length - 1, 1)) * plotW;
  const y = (v: number) => PAD.top + plotH - ((v - yMin) / (yMax - yMin)) * plotH;

  const TICKS = 4;
  for (let t = 0; t <= TICKS; t++) {
    const value = yMin + ((yMax - yMin) * t) / TICKS;
    const gy = y(value);
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

  if (opts.zeroLine && yMin < 0 && yMax > 0) {
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

  const first = dates[0];
  const last = dates[dates.length - 1];
  if (first) svg.appendChild(svgEl('text', { x: PAD.left, y: H - 12 }, first));
  if (last) {
    svg.appendChild(svgEl('text', { x: W - PAD.right, y: H - 12, 'text-anchor': 'end' }, last));
  }
  if (opts.caption) svg.appendChild(svgEl('text', { x: PAD.left, y: 12 }, opts.caption));
  return svg;
}

function legend(items: { label: string; swatch: string }[]): HTMLElement {
  return el(
    'div',
    { class: 'legend' },
    ...items.map((item) =>
      el(
        'span',
        { class: 'legend__item' },
        el('span', { class: `legend__swatch ${item.swatch}` }),
        item.label,
      ),
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
    return {
      label: 'unknown',
      tone: 'idle',
      detail: 'No z-score yet — the expanding baseline has not filled.',
    };
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

function renderState(result: ApiResult<TwoLayerState>): TwoLayerState | null {
  if (!stateHost) return null;
  if (!result.ok) {
    replace(stateHost, failureBlock(result, 'Kinematic state'));
    return null;
  }

  const state = result.envelope.data;
  const k = state.kinematics;
  if (k.as_of === null) {
    replace(
      stateHost,
      emptyBlock(
        'kinematic state',
        'The equity engine has not published features yet. It publishes on every daily run once its price backbone is backfilled.',
      ),
    );
    return state;
  }

  const f = k.features;
  const lamp = jerkLamp(f.jerk_z);

  // price_filtered is a log level in the feature store; index points is what a
  // reader wants on a tile, and exp() is the whole conversion.
  const level = f.price_filtered === undefined ? undefined : Math.exp(f.price_filtered);

  const staleness = result.envelope.stale
    ? stateBlock({
        tone: 'warn',
        title: 'This state is stale',
        detail: `Last published ${formatDate(k.as_of)}. The daily run has not reported since.`,
      })
    : null;

  replace(
    stateHost,
    staleness,
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

function renderRegime(state: TwoLayerState | null): void {
  if (!regimeHost) return;
  if (state?.regime) {
    replace(
      regimeHost,
      el(
        'div',
        { class: 'badgerow' },
        badge(state.regime.label, 'info'),
        el('span', { class: 'enginecard__detail' }, `confidence ${formatValue(state.regime.confidence)}`),
      ),
    );
    return;
  }

  replace(
    regimeHost,
    stateBlock({
      tone: 'info',
      title: 'Not computed yet — sub-milestone P3-B',
      detail:
        'The five-state HMM has not landed, so the engine publishes no regime and no AssetState. It declines rather than showing a placeholder: a regime badge is read as a market view, and there is no honest source for one until the model is fitted.',
      hint: 'The model will be fitted on a daily series long enough to contain 2000, 2008 and 2020, then applied to the S&P feature path — because ten years of history containing one drawdown cannot define a crisis state.',
    }),
  );
}

function renderPrice(
  close: ApiResult<AssetHistory>,
  filtered: ApiResult<AssetHistory>,
): void {
  if (!priceHost) return;
  if (!close.ok) {
    replace(priceHost, failureBlock(close, 'Price history'));
    return;
  }

  const rawPoints = close.envelope.data.points ?? [];
  const filteredPoints = filtered.ok ? (filtered.envelope.data.points ?? []) : [];
  if (rawPoints.length === 0) {
    replace(priceHost, emptyBlock('price history', 'The engine has not published a price path yet.'));
    return;
  }

  replace(
    priceHost,
    lineChart(
      [
        { points: rawPoints, className: 'line line--muted', label: 'close' },
        { points: filteredPoints, className: 'line', label: 'filtered' },
      ],
      {
        label: `S&P 500 close and Kalman-filtered level over ${rawPoints.length} sessions`,
        caption: 'index points',
      },
    ),
    legend([
      { label: 'filtered level (Kalman, filtered estimate)', swatch: 'legend__swatch--accent' },
      { label: 'raw close', swatch: 'legend__swatch--idle' },
    ]),
    el(
      'p',
      { class: 'enginecard__detail' },
      `${rawPoints.length} sessions through ${rawPoints[rawPoints.length - 1]?.as_of}`,
    ),
  );
}

function renderKinematics(
  velocity: ApiResult<AssetHistory>,
  acceleration: ApiResult<AssetHistory>,
): void {
  if (!kinematicsHost) return;
  if (!velocity.ok) {
    replace(kinematicsHost, failureBlock(velocity, 'Velocity history'));
    return;
  }

  const v = velocity.envelope.data.points ?? [];
  const a = acceleration.ok ? (acceleration.envelope.data.points ?? []) : [];
  if (v.length === 0) {
    replace(
      kinematicsHost,
      emptyBlock('kinematics', 'The engine has not published a velocity path yet.'),
    );
    return;
  }

  // Two different units on one axis would be meaningless, so they get two
  // charts sharing a date range rather than a shared, dishonest scale.
  replace(
    kinematicsHost,
    lineChart([{ points: v, className: 'line', label: 'velocity' }], {
      label: 'Annualized velocity of the filtered trend',
      caption: 'velocity — annualized log drift',
      zeroLine: true,
    }),
    lineChart([{ points: a, className: 'line line--alt', label: 'acceleration' }], {
      label: 'Acceleration of the filtered trend',
      caption: 'acceleration — per year squared',
      zeroLine: true,
    }),
  );
}

function renderJerk(result: ApiResult<AssetHistory>): void {
  if (!jerkHost) return;
  if (!result.ok) {
    replace(jerkHost, failureBlock(result, 'Jerk indicator'));
    return;
  }

  const points = result.envelope.data.points ?? [];
  if (points.length === 0) {
    replace(
      jerkHost,
      emptyBlock(
        'jerk indicator',
        'No z-score has been published yet — the expanding baseline has not filled.',
      ),
    );
    return;
  }

  const last = points[points.length - 1];
  const lamp = jerkLamp(last?.value);
  const elevated = points.filter((p) => Math.abs(p.value) >= JERK_ELEVATED).length;

  replace(
    jerkHost,
    el(
      'div',
      { class: 'badgerow' },
      badge(lamp.label, lamp.tone),
      el('span', { class: 'enginecard__detail' }, `${lamp.detail} As of ${last?.as_of}.`),
    ),
    lineChart([{ points, className: 'line', label: 'jerk z-score' }], {
      label: 'Jerk z-score against its expanding baseline',
      caption: 'jerk z-score — dashed lines at ±2 and ±3',
      zeroLine: true,
      bands: [JERK_ELEVATED, JERK_EXTREME],
    }),
    el(
      'p',
      { class: 'enginecard__detail' },
      `${elevated} of ${points.length} published sessions were elevated or beyond.`,
    ),
  );
}

function renderForces(result: ApiResult<ForceHistory>, snapshot: TwoLayerState | null): void {
  if (!forcesHost) return;
  if (!result.ok) {
    replace(forcesHost, failureBlock(result, 'Force scores'));
    return;
  }

  const scores = snapshot?.forces.scores ?? {};
  const components = snapshot?.forces.components ?? {};
  const names = Object.keys(scores).sort();

  if (names.length === 0) {
    replace(
      forcesHost,
      emptyBlock('force scores', 'The factor pipeline has not published scores yet.'),
    );
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
    forcesHost,
    table(['Force', 'Score (0–100)', 'Reading', 'Components'], rows),
    el(
      'p',
      { class: 'enginecard__detail' },
      `Scored on ${snapshot?.forces.as_of ?? 'an unknown date'} · ${result.envelope.data.count} historical points available from /forces`,
    ),
  );
}

function renderProvenance(state: TwoLayerState | null): void {
  if (!provenanceHost) return;

  const version = state?.kinematics.model_version ?? null;
  // The version carries its calibration tag: equity-1.0.0+cal.<series slug>.
  const calibration = version?.split('+cal.')[1] ?? null;

  replace(
    provenanceHost,
    el(
      'p',
      { class: 'prose' },
      'Three price series feed this engine and they are not interchangeable. The ',
      el('strong', {}, 'publication'),
      ' series is what everything on this page describes. A separate ',
      el('strong', {}, 'calibration'),
      ' series — longer, and reaching back through the crises the publication series does not — is what the regime model will be fitted on in the next sub-milestone. A monthly ',
      el('strong', {}, 'deep-history'),
      ' series back to 1871 is the only basis for the tail estimates after that.',
    ),
    el(
      'p',
      { class: 'prose' },
      'Which series played which part is recorded in the model version itself, so a number can never be traced to the wrong index: ',
      version ? el('code', {}, version) : el('em', {}, 'no version published yet'),
      calibration
        ? el(
            'span',
            {},
            ' — the suffix names the calibration series (',
            el('code', {}, calibration),
            ').',
          )
        : '',
    ),
    calibration && !calibration.includes('spx')
      ? stateBlock({
          tone: 'warn',
          title: 'The calibration series is a proxy, not the S&P 500',
          detail:
            'Daily S&P history before 2016 is not currently reachable, so a longer, more volatile index stands in for the fitted parameters. Nothing on this page depends on it yet — the kinematics above are computed on the S&P itself — but every fitted parameter from the next sub-milestone onwards will carry this caveat.',
        })
      : null,
  );
}

// ------------------------------------------------------------------- main

async function main(): Promise<void> {
  for (const host of [
    stateHost,
    regimeHost,
    priceHost,
    kinematicsHost,
    jerkHost,
    forcesHost,
    provenanceHost,
  ]) {
    if (host) replace(host, loadingBlock('FinEquity'));
  }

  const history = (metric: string) => getAssetHistory(ASSET, metric, { limit: HISTORY_LIMIT });

  const [state, close, filtered, velocity, acceleration, jerk, forces] = await Promise.all([
    getTwoLayerState(),
    history('price_close'),
    history('price_filtered'),
    history('velocity'),
    history('acceleration'),
    history('jerk_z'),
    getForces({ limit: 200 }),
  ]);

  const snapshot = renderState(state);
  renderRegime(snapshot);
  renderPrice(close, filtered);
  renderKinematics(velocity, acceleration);
  renderJerk(jerk);
  renderForces(forces, snapshot);
  renderProvenance(snapshot);
}

void main();
