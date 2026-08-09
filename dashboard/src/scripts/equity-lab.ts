/**
 * The Equity Dynamics Lab — the research surface at the top of `/equity`.
 *
 * Everything here reads the same endpoints the rest of the page already reads.
 * No engine, contract or schema is touched: the Lab is a different *view* of
 * `/api/v1/assets/equity/history`, not a different source of numbers, and the
 * request the page issues is byte-for-byte the one it issued before.
 *
 * What is new is the reading. Four derivatives of the same series answer four
 * different questions, and they can only be compared if a feature at one instant
 * is at one x-coordinate in all four panels at once. That is the whole design:
 *
 * - **Stacked, never overlaid.** Price runs to four figures, velocity is a rate
 *   near zero, jerk is a z-score. One y-axis would flatten three of them into
 *   the baseline; four panels sharing an x-axis keeps every one legible.
 * - **One fetch per range, and zoom is free.** {@link DynamicsChart} holds the
 *   decimated arrays and re-slices them, so a reader can go from a century to a
 *   month without a request.
 * - **Absent data is named.** `jerk_z` begins in 1939, twelve years after the
 *   price record, because its baseline has to fill. The Lab says so rather than
 *   drawing a shorter line and letting the axis imply the rest.
 */

import type { ApiResult, AssetHistory, AssetState, HistoryPoint, RegimeHistory } from '../lib/api';
import { badge, el, replace, stateBlock } from '../lib/dom';
import { DynamicsChart, type LabSeries, type PanelSpec, type Sample, type SeriesDiagnostics, nearestSample } from '../lib/dynamics';
import { MARKET_EVENTS, type MarketEvent, isoToTime, timeToIso } from '../lib/events';
import { formatCount, formatValue, type Tone } from '../lib/format';
import type { RangeSpec } from './ranges';

/** §3.1 thresholds, mirroring features/kinematics.py. Read, never recomputed. */
const JERK_ELEVATED = 2.0;
const JERK_EXTREME = 3.0;

/**
 * §3.2 reading thresholds for the RII, matching the instability panel below.
 *
 * `AssetState.risk_score` *is* the RII — `engine.py::_risk_score` returns the
 * composite directly. So it has to be read against the same two numbers the
 * instability panel uses: one page showing one quantity under two different
 * colour rules is a contradiction the reader has no way to resolve.
 */
const RII_ELEVATED = 60;
const RII_HIGH = 80;

/**
 * The §9 regime vocabulary, mapped to a structural reading and a risk state.
 *
 * This is a **display mapping over the engine's own labels**, not a second
 * regime model: the label on the left is what `/api/v1/regime` published, and
 * nothing here recomputes, re-thresholds or re-orders it. The mapping is stated
 * on screen for the same reason it is stated here — a coloured word whose rule
 * is not visible is a claim the reader cannot check.
 */
const REGIME_READING: Record<string, { reading: string; risk: 'Low' | 'Medium' | 'High'; tone: Tone }> = {
  bull_expansion: { reading: 'Healthy Trend', risk: 'Low', tone: 'ok' },
  normal_expansion: { reading: 'Healthy Trend', risk: 'Low', tone: 'ok' },
  late_cycle: { reading: 'Warning', risk: 'Medium', tone: 'warn' },
  bear: { reading: 'Crash / Structural Break', risk: 'High', tone: 'bad' },
  crisis: { reading: 'Crash / Structural Break', risk: 'High', tone: 'bad' },
};

const REGIME_ORDER = ['bull_expansion', 'normal_expansion', 'late_cycle', 'bear', 'crisis'] as const;

// ------------------------------------------------------------- formatting

const fmtPrice = (v: number) => formatValue(v);
const fmtRate = (v: number) => `${(v * 100).toFixed(1)}%`;
/** Precision follows magnitude: three decimals on 0.004, one on -20. */
const decimalsFor = (magnitude: number) => (magnitude >= 10 ? 1 : magnitude >= 1 ? 2 : 3);
const fmtAccel = (v: number) => v.toFixed(decimalsFor(Math.abs(v)));
/** On an axis the magnitude is the *axis's*, so every label agrees on precision. */
const axisAccel = (v: number, maxAbs: number) => v.toFixed(decimalsFor(maxAbs));
const fmtZ = (v: number) => `${v.toFixed(2)}σ`;

// ------------------------------------------------------------ series prep

/**
 * API points to plot samples: parse each date exactly once, drop what cannot be
 * drawn, and *count* what was dropped rather than quietly shrinking the series.
 */
function toSamples(points: HistoryPoint[]): { samples: Sample[]; missing: number } {
  const samples: Sample[] = [];
  let missing = 0;
  for (const point of points) {
    const t = isoToTime(point.as_of);
    if (!Number.isFinite(t) || !Number.isFinite(point.value)) {
      missing++;
      continue;
    }
    samples.push({ t, v: point.value });
  }
  // The API returns oldest-first; a sort here is a cheap guard against a source
  // that stops doing so, and a no-op when it does not.
  samples.sort((a, b) => a.t - b.t);
  return { samples, missing };
}

function buildSeries(
  key: string,
  label: string,
  className: string,
  result: ApiResult<AssetHistory>,
  format: (v: number) => string,
): LabSeries {
  if (!result.ok) {
    return {
      key,
      label,
      className,
      samples: [],
      format,
      diagnostics: {
        available: 0,
        returned: 0,
        missing: 0,
        truncated: false,
        decimated: null,
        first: null,
        last: null,
      },
    };
  }
  const data = result.envelope.data;
  const { samples, missing } = toSamples(data.points);
  const diagnostics: SeriesDiagnostics = {
    available: data.available,
    returned: data.count,
    missing,
    truncated: data.truncated,
    decimated: data.decimated,
    first: samples[0] ? timeToIso(samples[0].t) : null,
    last: samples.at(-1) ? timeToIso(samples.at(-1)!.t) : null,
  };
  return { key, label, className, samples, format, diagnostics };
}

/**
 * A series standardized over the window that was loaded.
 *
 * Used only by the overlay mode, and only to put three different units on one
 * axis. The baseline is **this window**, not the expanding history the engine
 * uses for `jerk_z` — a different quantity, so the panel says so in its own
 * subtitle rather than letting a reader assume they match.
 */
function zScored(series: LabSeries, label: string): LabSeries {
  const values = series.samples;
  if (values.length < 2) return { ...series, label, format: fmtZ };
  let sum = 0;
  for (const s of values) sum += s.v;
  const mean = sum / values.length;
  let variance = 0;
  for (const s of values) variance += (s.v - mean) ** 2;
  const sd = Math.sqrt(variance / (values.length - 1));
  if (!(sd > 0)) return { ...series, label, format: fmtZ };
  return {
    ...series,
    label,
    format: fmtZ,
    samples: values.map((s) => ({ t: s.t, v: (s.v - mean) / sd })),
  };
}

// ------------------------------------------------------------------ state

interface LabData {
  range: RangeSpec;
  price: LabSeries;
  filtered: LabSeries;
  velocity: LabSeries;
  acceleration: LabSeries;
  jerk: LabSeries;
  regime: ApiResult<RegimeHistory>;
  state: ApiResult<AssetState>;
}

interface LabHosts {
  chart: Element | null;
  regime: Element | null;
  events: Element | null;
  debug: Element | null;
  toolbar: Element | null;
}

let chart: DynamicsChart | null = null;
let overlayMode = false;
let debugMode = false;
let current: LabData | null = null;
let hosts: LabHosts = { chart: null, regime: null, events: null, debug: null, toolbar: null };
let selectedEvent: MarketEvent | null = null;
let viewport: readonly [number, number] | null = null;

// ----------------------------------------------------------------- panels

function panelsFor(data: LabData): PanelSpec[] {
  const price: PanelSpec = {
    key: 'price',
    title: 'Price',
    note: data.range.log
      ? 'historical equity curve — index points, log scale'
      : 'historical equity curve — index points',
    series: [data.price, data.filtered],
    height: 250,
    log: data.range.log,
  };

  if (overlayMode) {
    return [
      price,
      {
        key: 'overlay',
        title: 'Derivatives — overlay',
        note: 'velocity, acceleration and jerk, each standardized over the loaded window',
        series: [
          zScored(data.velocity, 'Velocity (z)'),
          zScored(data.acceleration, 'Acceleration (z)'),
          zScored(data.jerk, 'Jerk (z of z)'),
        ],
        height: 220,
        zeroLine: true,
      },
    ];
  }

  return [
    price,
    {
      key: 'velocity',
      title: 'Velocity',
      note: 'first derivative — annualized log drift of the filtered trend',
      series: [data.velocity],
      height: 150,
      zeroLine: true,
    },
    {
      key: 'acceleration',
      title: 'Acceleration',
      note: 'second derivative — change in velocity, per year squared',
      series: [data.acceleration],
      height: 150,
      zeroLine: true,
      axisFormat: axisAccel,
    },
    {
      key: 'jerk',
      title: 'Jerk',
      note: 'third derivative — z-score against its own expanding baseline, bands at ±2 and ±3',
      series: [data.jerk],
      height: 150,
      zeroLine: true,
      bands: [JERK_ELEVATED, JERK_EXTREME],
    },
  ];
}

/**
 * The union of every loaded series' extent.
 *
 * Not the price series alone: `velocity` starts a year later and `jerk_z`
 * twelve, and a domain taken from the shortest of them would hide a decade of
 * price the API did return.
 */
function domainOf(data: LabData): readonly [number, number] {
  const all = [data.price, data.filtered, data.velocity, data.acceleration, data.jerk];
  let lo = Infinity;
  let hi = -Infinity;
  for (const series of all) {
    const first = series.samples[0];
    const last = series.samples.at(-1);
    if (first) lo = Math.min(lo, first.t);
    if (last) hi = Math.max(hi, last.t);
  }
  if (!Number.isFinite(lo) || !Number.isFinite(hi) || lo >= hi) {
    const now = Date.now();
    return [now - 365 * 86_400_000, now];
  }
  return [lo, hi];
}

// -------------------------------------------------------------- rendering

export function setLabHosts(next: LabHosts): void {
  hosts = next;
}

/** The toolbar: overlay toggle, debug toggle, reset. Rendered once. */
function renderToolbar(): void {
  if (!hosts.toolbar) return;

  const toggle = (label: string, on: boolean, onChange: (next: boolean) => void): HTMLElement => {
    const input = el('input', { type: 'checkbox', class: 'labtoggle__input' }) as HTMLInputElement;
    input.checked = on;
    input.addEventListener('change', () => onChange(input.checked));
    return el('label', { class: 'labtoggle' }, input, el('span', {}, label));
  };

  const reset = el('button', { type: 'button', class: 'labbutton' }, 'Reset zoom');
  reset.addEventListener('click', () => chart?.resetViewport());

  replace(
    hosts.toolbar,
    el(
      'div',
      { class: 'labtoolbar' },
      toggle('Overlay mode (z-scored derivatives)', overlayMode, (next) => {
        overlayMode = next;
        if (current && chart) chart.setPanels(panelsFor(current));
      }),
      toggle('Enable Dynamics Debug', debugMode, (next) => {
        debugMode = next;
        renderDebug();
      }),
      reset,
      el(
        'span',
        { class: 'labtoolbar__hint' },
        'scroll to zoom · drag to pan · double-click to reset · hover for the crosshair',
      ),
    ),
  );
}

/** Mount or re-mount the chart for a freshly loaded range. */
export function renderLab(data: LabData): void {
  current = data;
  selectedEvent = null;
  renderToolbar();

  if (!hosts.chart) return;

  if (data.price.samples.length < 2) {
    chart?.destroy();
    chart = null;
    replace(
      hosts.chart,
      stateBlock({
        tone: 'warn',
        title: 'Not enough price history to plot',
        detail:
          'The API returned fewer than two drawable points for this window. The panels below read the same endpoint and will say the same thing.',
      }),
    );
    renderRegime(data);
    renderEvents(data);
    renderDebug();
    return;
  }

  const domain = domainOf(data);
  chart?.destroy();
  chart = new DynamicsChart(hosts.chart as HTMLElement, {
    panels: panelsFor(data),
    domain,
    events: MARKET_EVENTS,
    onViewport: (next) => {
      viewport = next;
      renderDebug();
    },
    onEvent: (event) => {
      selectedEvent = event;
      renderEvents(data);
    },
  });
  viewport = domain;

  renderCoverageNotes(data);
  renderRegime(data);
  renderEvents(data);
  renderDebug();
}

/**
 * What the window holds against what is drawn — and where each series starts.
 *
 * The burn-in gaps are the reason this exists. `velocity` has no value for
 * 1927–28 and `jerk_z` none before 1939, and on a shared axis three lines that
 * begin at three different dates look like missing data unless the page says
 * otherwise.
 */
function renderCoverageNotes(data: LabData): void {
  if (!hosts.chart) return;
  const gaps = [data.velocity, data.acceleration, data.jerk]
    .filter((s) => s.diagnostics.first && data.price.diagnostics.first && s.diagnostics.first > data.price.diagnostics.first)
    .map((s) => `${s.label} from ${s.diagnostics.first}`);

  const notes: Node[] = [];
  if (gaps.length) {
    notes.push(
      el(
        'p',
        { class: 'labnote' },
        `Price from ${data.price.diagnostics.first ?? '—'}; ${gaps.join(', ')}. ` +
          'A derivative cannot start where its input does — the filter discards a diffuse start-up and the jerk z-score waits for its expanding baseline to fill. The shorter lines are the engine declining to publish, not gaps in the market.',
      ),
    );
  }
  const decimated = data.price.diagnostics.decimated;
  notes.push(
    el(
      'p',
      { class: 'labnote' },
      decimated
        ? `${formatCount(decimated.from)} observations in this window, served as ${formatCount(decimated.to)} points (${decimated.method.toUpperCase()}) — shape-preserving, so single-day crashes survive. Zooming re-reads these points; it does not re-request.`
        : `${formatCount(data.price.diagnostics.returned)} observations, drawn in full.`,
    ),
  );
  for (const note of notes) hosts.chart.appendChild(note);
}

// ---------------------------------------------------------------- regime

/**
 * §9's posterior, as a state readout and a ribbon.
 *
 * The engine publishes the label, the posterior and a risk score; this panel
 * arranges them. The one derived thing is the Low/Medium/High word, which is a
 * lookup on the published label — {@link REGIME_READING} — and the lookup is
 * printed underneath so it can be checked.
 */
function renderRegime(data: LabData): void {
  if (!hosts.regime) return;

  if (!data.regime.ok) {
    replace(
      hosts.regime,
      stateBlock({
        tone: 'info',
        title: 'No regime posterior for this window',
        detail: data.regime.message,
      }),
    );
    return;
  }

  const points = data.regime.envelope.data.points;
  if (points.length === 0) {
    replace(
      hosts.regime,
      stateBlock({
        tone: 'info',
        title: 'No regime posterior yet',
        detail: 'The regime model is fitted by the monthly refit job; until it has run the engine declines to publish a state.',
      }),
    );
    return;
  }

  const latest = points.at(-1)!;
  const reading = REGIME_READING[latest.regime] ?? {
    reading: 'Unmapped',
    risk: 'Medium' as const,
    tone: 'idle' as Tone,
  };
  const riskScore = data.state.ok ? data.state.envelope.data.risk_score : null;

  replace(
    hosts.regime,
    el(
      'div',
      { class: 'labtiles' },
      labTile('Current regime', latest.regime.replace(/_/g, ' '), reading.tone, `${(latest.confidence * 100).toFixed(1)}% posterior on ${latest.as_of}`),
      labTile('Risk state', reading.risk, reading.tone, 'Mapped from the published regime label — see the rule below.'),
      labTile('Reading', reading.reading, reading.tone, 'Healthy Trend · Warning · Crash / Structural Break'),
      labTile(
        'Risk score (RII)',
        riskScore === null ? '—' : formatValue(riskScore),
        riskScore === null ? 'idle' : riskScore >= RII_HIGH ? 'bad' : riskScore >= RII_ELEVATED ? 'warn' : 'ok',
        riskScore === null
          ? 'Not published on the current state.'
          : 'The §3.2 instability index, published on /assets/equity/state — the same number, and the same thresholds, as the instability panel below.',
      ),
    ),
    regimeRibbon(points),
    el(
      'div',
      { class: 'labkey' },
      ...REGIME_ORDER.map((regime) =>
        el(
          'span',
          { class: 'labkey__item' },
          el('span', { class: `labkey__swatch labkey__swatch--${regime}` }),
          `${regime.replace(/_/g, ' ')} → ${REGIME_READING[regime]?.risk ?? '—'}`,
        ),
      ),
    ),
    el(
      'p',
      { class: 'labnote' },
      'The ribbon is the highest-posterior regime per published date over the loaded window, in the engine’s own five-state vocabulary. No threshold, smoothing or relabelling is applied here: this panel draws /api/v1/regime and the risk word is a fixed lookup on the label it returns.',
    ),
    el(
      'p',
      { class: 'labnote' },
      'Over a century the ribbon reads as banding rather than as blocks, and that is the data rather than the drawing: the engine changes its most-likely state a few times a year, so a hundred years holds a few hundred switches. Narrow the range to read it, and see the stacked posterior further down the page for the distribution the label is an argmax of — a 51/49 split and a 99/1 split produce the same colour here.',
    ),
  );
}

function labTile(label: string, value: string, tone: Tone, detail: string): HTMLElement {
  return el(
    'div',
    { class: `labtile labtile--${tone}` },
    el('div', { class: 'labtile__label' }, label),
    el('div', { class: 'labtile__value' }, value),
    el('div', { class: 'labtile__detail' }, detail),
  );
}

/** The modal regime per date, as one horizontal band across the loaded window. */
function regimeRibbon(points: RegimeHistory['points']): SVGSVGElement {
  const W = 1000;
  const H = 34;
  const node = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
  node.setAttribute('class', 'labribbon');
  node.setAttribute('viewBox', `0 0 ${W} ${H}`);
  node.setAttribute('preserveAspectRatio', 'none');
  node.setAttribute('role', 'img');
  node.setAttribute('aria-label', `Regime over ${points.length} published dates`);

  const t0 = isoToTime(points[0]!.as_of);
  const t1 = isoToTime(points.at(-1)!.as_of);
  const span = t1 - t0 || 1;
  const x = (t: number) => ((t - t0) / span) * W;

  // Merge consecutive dates that share a label into one rect: a century of
  // daily rows is 24,000 rects otherwise, for a band 34 units tall.
  let runStart = t0;
  let runLabel = points[0]!.regime;
  const push = (from: number, to: number, label: string) => {
    const rect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
    rect.setAttribute('class', `labribbon__seg labribbon__seg--${label}`);
    rect.setAttribute('x', x(from).toFixed(2));
    rect.setAttribute('y', '0');
    rect.setAttribute('width', Math.max(x(to) - x(from), 0.5).toFixed(2));
    rect.setAttribute('height', String(H));
    const title = document.createElementNS('http://www.w3.org/2000/svg', 'title');
    title.textContent = `${label.replace(/_/g, ' ')} · ${timeToIso(from)} → ${timeToIso(to)}`;
    rect.appendChild(title);
    node.appendChild(rect);
  };

  for (const point of points) {
    const t = isoToTime(point.as_of);
    if (point.regime !== runLabel) {
      push(runStart, t, runLabel);
      runStart = t;
      runLabel = point.regime;
    }
  }
  push(runStart, t1, runLabel);
  return node;
}

// ----------------------------------------------------------------- events

/**
 * The event layer's readout.
 *
 * Every number in it is looked up out of the series already loaded — the
 * nearest published sample to the event's own date. Nothing is interpolated and
 * no indicator is derived: if `jerk_z` has no value in 1929, this panel says so
 * instead of producing one.
 */
function renderEvents(data: LabData): void {
  if (!hosts.events) return;

  const chips = MARKET_EVENTS.map((event) => {
    const button = el(
      'button',
      {
        type: 'button',
        class: `labchip${selectedEvent?.id === event.id ? ' labchip--on' : ''}`,
        'aria-pressed': selectedEvent?.id === event.id ? 'true' : 'false',
      },
      event.label,
    );
    button.addEventListener('click', () => {
      selectedEvent = event;
      chart?.jumpTo(event);
      renderEvents(data);
    });
    return button;
  });

  replace(
    hosts.events,
    el('div', { class: 'labchips' }, ...chips),
    selectedEvent ? eventCard(selectedEvent, data) : null,
    el(
      'p',
      { class: 'labnote' },
      'Markers are a frontend annotation, not model output — the dates are fixed references, and the values beside them are read from the series already on screen at the nearest published date. Selecting one moves every panel to its window.',
    ),
  );
}

function readAt(series: LabSeries, iso: string): { value: number; at: string } | null {
  const sample = nearestSample(series.samples, isoToTime(iso));
  if (!sample) return null;
  return { value: sample.v, at: timeToIso(sample.t) };
}

function eventCard(event: MarketEvent, data: LabData): HTMLElement {
  const start = readAt(data.price, event.window.from);
  const end = readAt(data.price, event.window.to);
  const change =
    start && end && start.value > 0 ? ((end.value - start.value) / start.value) * 100 : null;

  const metric = (label: string, series: LabSeries, format: (v: number) => string): HTMLElement => {
    const read = readAt(series, event.date);
    const covered =
      read !== null &&
      series.diagnostics.first !== null &&
      event.date >= series.diagnostics.first;
    return el(
      'div',
      { class: 'labmetric' },
      el('div', { class: 'labmetric__label' }, label),
      el(
        'div',
        { class: 'labmetric__value' },
        covered && read ? format(read.value) : '—',
      ),
      el(
        'div',
        { class: 'labmetric__detail' },
        covered && read
          ? `at ${read.at}`
          : series.diagnostics.first
            ? `not published before ${series.diagnostics.first}`
            : 'not published for this window',
      ),
    );
  };

  return el(
    'div',
    { class: 'labevent' },
    el(
      'div',
      { class: 'labevent__head' },
      el('h4', { class: 'labevent__title' }, event.label),
      badge(event.date, 'info'),
    ),
    el('p', { class: 'labevent__note' }, event.note),
    el(
      'div',
      { class: 'labmetrics' },
      el(
        'div',
        { class: 'labmetric' },
        el('div', { class: 'labmetric__label' }, 'Price change'),
        el(
          'div',
          { class: `labmetric__value labmetric__value--${change === null ? 'idle' : change < 0 ? 'bad' : 'ok'}` },
          change === null ? '—' : `${change >= 0 ? '+' : ''}${change.toFixed(1)}%`,
        ),
        el(
          'div',
          { class: 'labmetric__detail' },
          start && end ? `${start.at} → ${end.at}` : 'outside the loaded window',
        ),
      ),
      metric('Price', data.price, fmtPrice),
      metric('Velocity', data.velocity, fmtRate),
      metric('Acceleration', data.acceleration, fmtAccel),
      metric('Jerk', data.jerk, fmtZ),
    ),
  );
}

// ------------------------------------------------------------------ debug

/**
 * §7's developer surface — diagnostics the frontend can actually establish.
 *
 * The distinction it keeps is between *observed* and *exposed*. The date
 * `jerk_z` starts on is observed: it is in the response. The burn-in period
 * that produced that date is a model parameter, and no endpoint returns it. The
 * first is reported as a fact; the second is reported as unavailable, and
 * neither is inferred from the other.
 */
function renderDebug(): void {
  if (!hosts.debug) return;
  if (!debugMode || !current) {
    replace(hosts.debug, el('p', { class: 'labnote' }, 'Debug mode is off.'));
    return;
  }

  const data = current;
  const series = [data.price, data.filtered, data.velocity, data.acceleration, data.jerk];

  const rows = series.map((s) => {
    const d = s.diagnostics;
    return el(
      'tr',
      {},
      el('td', { class: 'mono nowrap' }, s.key),
      el('td', { class: 'num mono' }, formatCount(d.returned)),
      el('td', { class: 'num mono' }, formatCount(d.available)),
      el('td', { class: 'num mono' }, formatCount(d.missing)),
      el('td', { class: 'mono nowrap' }, d.first ?? '—'),
      el('td', { class: 'mono nowrap' }, d.last ?? '—'),
      el('td', { class: 'mono' }, d.decimated ? `${d.decimated.method} ${d.decimated.from}→${d.decimated.to}` : 'none'),
      el('td', { class: 'mono' }, d.truncated ? 'yes' : 'no'),
    ) as HTMLTableRowElement;
  });

  const view = viewport;
  const zoomLine = view
    ? `${timeToIso(view[0])} → ${timeToIso(view[1])} (${((view[1] - view[0]) / (365.25 * 86_400_000)).toFixed(2)} years)`
    : 'not mounted';

  replace(
    hosts.debug,
    el(
      'div',
      { class: 'tablewrap' },
      el(
        'table',
        {},
        el(
          'thead',
          {},
          el(
            'tr',
            {},
            ...['Series', 'Drawn', 'Available', 'Dropped', 'First', 'Last', 'Server decimation', 'Truncated'].map(
              (h) => el('th', { scope: 'col' }, h),
            ),
          ),
        ),
        el('tbody', {}, ...rows),
      ),
    ),
    el(
      'dl',
      { class: 'deflist' },
      defRow('range key', data.range.key),
      defRow('range window', `${data.range.from ?? (data.range.years === null ? 'all' : `${data.range.years}y back`)} → ${data.range.to ?? 'today'}`),
      defRow('points requested', String(data.range.points)),
      defRow('price axis', data.range.log ? 'log' : 'linear'),
      defRow('current zoom range', zoomLine),
      defRow('panel mode', overlayMode ? 'overlay (z-scored derivatives)' : 'four synchronized panels'),
      defRow('regime model_version', data.regime.ok ? (data.regime.envelope.data.model_version ?? '—') : '—'),
      defRow('state model_version', data.state.ok ? data.state.envelope.data.model_version : '—'),
    ),
    stateBlock({
      tone: 'info',
      title: 'Not available from API',
      detail:
        'Burn-in period, splice boundary and smoothing parameters are not exposed by any /api/v1 endpoint, so they are not shown. The first published date per series above is an observation, not an inference about any of them — it is where the response starts, nothing more.',
    }),
  );
}

function defRow(label: string, value: string): HTMLElement {
  return el('div', { class: 'defrow' }, el('dt', { class: 'mono' }, label), el('dd', { class: 'mono' }, value));
}

// ------------------------------------------------------------------ entry

/** Build the Lab's view model from the results `equity.ts` already fetched. */
export function labDataFrom(input: {
  range: RangeSpec;
  close: ApiResult<AssetHistory>;
  filtered: ApiResult<AssetHistory>;
  velocity: ApiResult<AssetHistory>;
  acceleration: ApiResult<AssetHistory>;
  jerk: ApiResult<AssetHistory>;
  regime: ApiResult<RegimeHistory>;
  state: ApiResult<AssetState>;
}): LabData {
  return {
    range: input.range,
    price: buildSeries('price_close', 'Price', 'lab-line lab-line--raw', input.close, fmtPrice),
    filtered: buildSeries('price_filtered', 'Filtered', 'lab-line lab-line--price', input.filtered, fmtPrice),
    velocity: buildSeries('velocity', 'Velocity', 'lab-line lab-line--velocity', input.velocity, fmtRate),
    acceleration: buildSeries('acceleration', 'Acceleration', 'lab-line lab-line--acceleration', input.acceleration, fmtAccel),
    jerk: buildSeries('jerk_z', 'Jerk', 'lab-line lab-line--jerk', input.jerk, fmtZ),
    regime: input.regime,
    state: input.state,
  };
}

/** Tear the chart down — used when a range switches to the monthly tier. */
export function clearLab(message: string): void {
  chart?.destroy();
  chart = null;
  current = null;
  if (hosts.chart) {
    replace(
      hosts.chart,
      stateBlock({ tone: 'info', title: 'The Lab is a daily-resolution surface', detail: message }),
    );
  }
  if (hosts.regime) replace(hosts.regime);
  if (hosts.events) replace(hosts.events);
  if (hosts.debug) replace(hosts.debug);
  if (hosts.toolbar) replace(hosts.toolbar);
}

export { JERK_ELEVATED, JERK_EXTREME };
