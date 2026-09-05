/**
 * The synchronized multi-panel chart behind the Equity Dynamics Lab.
 *
 * There is no chart library in this project and this file does not add one. The
 * dashboard draws SVG by hand (see `lib/dom.ts`), the payloads are already
 * decimated server-side, and the one thing a library would buy here — shared
 * interaction across stacked panels — is the part every library makes hardest.
 *
 * Four rules shape the implementation:
 *
 * **One x-scale, many y-scales.** Price, velocity, acceleration and jerk differ
 * by orders of magnitude and by unit. They share a *viewport* — the same
 * `[t0, t1]` window, the same pixel x for the same instant, so a feature at
 * 2008-09 lines up vertically across all four — and nothing else. Each panel
 * scales its own y over what is visible, which is the whole point of stacking
 * them rather than overlaying them.
 *
 * **Zoom never refetches.** A range is fetched once, server-decimated, and held.
 * Zooming re-slices the cached arrays with a binary search and rewrites the `d`
 * attribute of paths that already exist. No request, no re-parse, no node
 * churn — see {@link DynamicsChart.draw}.
 *
 * **The cursor layer is separate from the series layer.** Moving the pointer
 * across a century of data touches four `<line>`s, four `<circle>`s and a
 * tooltip. It never rebuilds a path and never recomputes a scale, because
 * crosshair latency is what makes a chart feel like a terminal or like a
 * website.
 *
 * **Dates are parsed once.** `Date.parse` on every point on every frame is the
 * single most expensive thing a chart like this can do. ISO strings become epoch
 * milliseconds at load and stay that way.
 */

import { MARKET_EVENTS, type MarketEvent, isoToTime, timeToIso } from './events';

// --------------------------------------------------------------- geometry

/** viewBox width. The SVG scales to its container; the coordinate space is fixed. */
const VIEW_W = 1000;
const PAD = { left: 68, right: 16, top: 14, bottom: 14 };
/**
 * Extra bottom room on the last panel, which carries the shared date axis.
 *
 * Two rows, not one: the date labels sit directly under the plot and the event
 * flags below them. Sharing a row makes the flags overlap the decade labels at
 * exactly the dates a reader is most likely to be looking at.
 */
const AXIS_H = 38;
/** Baseline of the date labels, below the plot floor. */
const AXIS_LABEL_DY = 14;
/** Top of the event flags, clear of the labels above them. */
const FLAG_DY = 21;
const PLOT_W = VIEW_W - PAD.left - PAD.right;

/**
 * Most points any one panel will draw, whatever the viewport holds.
 *
 * The plot is 1000 units wide and rendered around 900 CSS pixels. Past roughly
 * two points per pixel the extra vertices are invisible and cost real time in
 * path serialization, so a min/max reduction stands in for them — min/max rather
 * than stride sampling because a stride drops the single-day low of 1987 and a
 * min/max bucket cannot.
 */
const DRAW_BUDGET = 1800;

/** Tightest zoom, in milliseconds. Roughly ten sessions. */
const MIN_SPAN_MS = 14 * 86_400_000;

// ------------------------------------------------------------------ types

export interface Sample {
  /** Epoch milliseconds, UTC midnight. Parsed once at load. */
  t: number;
  v: number;
}

export interface LabSeries {
  key: string;
  label: string;
  /** CSS class on the path — the palette lives in the stylesheet. */
  className: string;
  /** Ascending by `t`, non-finite values already dropped. */
  samples: Sample[];
  format: (v: number) => string;
  /** Diagnostics, surfaced by the debug panel rather than thrown away. */
  diagnostics: SeriesDiagnostics;
  /**
   * Milliseconds between consecutive samples beyond which the line **breaks**
   * instead of bridging.
   *
   * Undefined by default, and the four kinematic panels leave it that way: a
   * price series has no interior holes, so a break would only ever appear at a
   * market closure and would be noise.
   *
   * The research panels set it, because for them a hole is the message. The
   * KK-Ω coupling is undefined wherever |ΔΩ| fell below its denominator floor
   * — the estimator declining to publish, not a gap in the market — and a
   * straight segment drawn across three months of that would assert a value
   * nobody computed. Same distinction the Lab already draws for `jerk_z`.
   */
  maxGap?: number;
}

export interface SeriesDiagnostics {
  /** Rows the API said the window holds, before server-side decimation. */
  available: number;
  /** Rows the API actually returned. */
  returned: number;
  /** Rows dropped here because their value was null, NaN or infinite. */
  missing: number;
  truncated: boolean;
  decimated: { from: number; to: number; method: string } | null;
  first: string | null;
  last: string | null;
}

export interface PanelSpec {
  key: string;
  title: string;
  /** Unit line under the title. Says what the numbers are, always. */
  note: string;
  series: LabSeries[];
  height: number;
  log?: boolean;
  zeroLine?: boolean;
  /** Symmetric reference lines, e.g. ±2 and ±3 on a z-score. */
  bands?: number[];
  /**
   * Axis-label formatter, given the value and the largest tick magnitude on the
   * axis. The magnitude is the reason this exists separately from the series
   * formatter: a per-value rule prints `-20.0` beside `0.000` on the same axis,
   * and an axis whose labels disagree about precision reads as a bug.
   */
  axisFormat?: (value: number, maxAbs: number) => string;
}

export type Viewport = readonly [number, number];

export interface ChartOptions {
  panels: PanelSpec[];
  /** Full extent of the loaded data. Zoom is clamped to this. */
  domain: Viewport;
  events?: readonly MarketEvent[];
  /** Fired after every viewport change, for the debug panel and the URL. */
  onViewport?: (viewport: Viewport) => void;
  /** Fired when an event flag is clicked, before the viewport jumps. */
  onEvent?: (event: MarketEvent) => void;
}

// ------------------------------------------------------------------ ticks

interface TimeTick {
  t: number;
  label: string;
}

/**
 * Tick steps, smallest first. Deliberately **not** a uniform ladder.
 *
 * The gaps are the design. There is no 2-year or 4-year step, so twenty years
 * lands on five-year labels rather than on two-year ones, and a century lands on
 * decades rather than on twenty-year blocks. Those two readings are the ones
 * this chart is for, and a mathematically tidy ladder gets both of them wrong.
 */
const DAY = 86_400_000;
const DAY_STEPS = [1, 2, 7, 14];
const MONTH_STEPS = [1, 2, 3, 6, 12, 60, 120, 300, 600];
const MAX_TICKS = 10;

function utc(year: number, month: number, day = 1): number {
  return Date.UTC(year, month, day);
}

/**
 * Dates for the shared x-axis, at a density the span can carry.
 *
 * Ticks are aligned to calendar boundaries, not to the viewport edges: a decade
 * label reading 1934 because that is where the window happens to start is worse
 * than no label at all.
 */
export function timeTicks(t0: number, t1: number): TimeTick[] {
  const span = Math.max(t1 - t0, 1);

  for (const days of DAY_STEPS) {
    if (span / (days * DAY) > MAX_TICKS) continue;
    const step = days * DAY;
    const ticks: TimeTick[] = [];
    // Align to the epoch day grid, which is midnight UTC by construction.
    for (let t = Math.ceil(t0 / step) * step; t <= t1; t += step) {
      ticks.push({ t, label: timeToIso(t) });
    }
    if (ticks.length >= 2) return ticks;
  }

  const startYear = new Date(t0).getUTCFullYear();
  for (const months of MONTH_STEPS) {
    if (span / (months * 30.44 * DAY) > MAX_TICKS) continue;
    const ticks: TimeTick[] = [];
    // Anchor on the calendar, not on the window edge: a decade step has to land
    // on 1930 and 1940, and a step counted from wherever the data happens to
    // begin lands on 1929 and 1939 — which reads as an off-by-one in the data.
    let year: number;
    let month = 0;
    if (months >= 12) {
      const stepYears = months / 12;
      year = Math.floor(startYear / stepYears) * stepYears - stepYears;
    } else {
      year = startYear - 1;
    }
    for (let guard = 0; guard < 4000; guard++) {
      const t = utc(year, month);
      if (t > t1) break;
      if (t >= t0) {
        ticks.push({
          t,
          label: months >= 12 ? String(year) : `${year}-${String(month + 1).padStart(2, '0')}`,
        });
      }
      month += months;
      year += Math.floor(month / 12);
      month %= 12;
    }
    if (ticks.length >= 2) return ticks;
  }

  return [
    { t: t0, label: timeToIso(t0) },
    { t: t1, label: timeToIso(t1) },
  ];
}

/** Round linear gridlines onto 1/2/5×10ⁿ, so the labels read as numbers a person chose. */
function linearTicks(lo: number, hi: number, target = 5): number[] {
  if (!(hi > lo)) return [lo];
  const raw = (hi - lo) / target;
  const magnitude = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 5, 10].map((m) => m * magnitude).find((s) => (hi - lo) / s <= target) ?? magnitude * 10;
  const ticks: number[] = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + step * 1e-9; v += step) ticks.push(v);
  return ticks.length >= 2 ? ticks : [lo, hi];
}

/**
 * Log gridlines on 1/2/5×10ⁿ, thinned to decades when that is too many.
 *
 * A century of the S&P spans 17 to 7,400 — nearly three decades of index points.
 * Evenly spacing gridlines in log space and labelling them produces values like
 * 43.7, which nobody reads; decade anchors produce 100, 1,000 and a reader who
 * knows where they are.
 */
function logTicks(lo: number, hi: number): number[] {
  if (!(hi > lo) || lo <= 0) return [lo, hi];
  const build = (mantissas: number[]): number[] => {
    const out: number[] = [];
    for (let e = Math.floor(Math.log10(lo)); e <= Math.ceil(Math.log10(hi)); e++) {
      for (const m of mantissas) {
        const v = m * 10 ** e;
        if (v >= lo && v <= hi) out.push(v);
      }
    }
    return out;
  };
  const fine = build([1, 2, 5]);
  if (fine.length > 9) return build([1]);
  if (fine.length >= 3) return fine;
  const coarse = build([1, 1.5, 2, 3, 5, 7]);
  return coarse.length >= 2 ? coarse : [lo, hi];
}

// ------------------------------------------------------------- data access

/** First index whose `t` is >= target. Binary search; the arrays are sorted. */
function lowerBound(samples: Sample[], target: number): number {
  let lo = 0;
  let hi = samples.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (samples[mid]!.t < target) lo = mid + 1;
    else hi = mid;
  }
  return lo;
}

/**
 * Index range covering `[t0, t1]`, widened by one on each side.
 *
 * The extra point matters: without it a line whose neighbouring samples sit
 * outside the viewport stops short of the edge, and the reader sees a gap in
 * the data where there is only a gap in the window.
 */
function visibleRange(samples: Sample[], t0: number, t1: number): [number, number] {
  if (samples.length === 0) return [0, -1];
  const lo = Math.max(0, lowerBound(samples, t0) - 1);
  const hi = Math.min(samples.length - 1, lowerBound(samples, t1));
  return [lo, hi];
}

/**
 * Reduce a slice to at most `budget` points, keeping every local extreme.
 *
 * Buckets the slice by index and emits each bucket's minimum and maximum in
 * time order. Stride sampling would be one line shorter and would delete
 * 1987-10-19, which is the single most informative point in the series.
 */
function reduce(samples: Sample[], lo: number, hi: number, budget = DRAW_BUDGET): Sample[] {
  const count = hi - lo + 1;
  if (count <= 0) return [];
  if (count <= budget) return samples.slice(lo, hi + 1);

  const buckets = Math.max(1, Math.floor(budget / 2));
  const size = count / buckets;
  const out: Sample[] = [];
  for (let b = 0; b < buckets; b++) {
    const start = lo + Math.floor(b * size);
    const end = Math.min(hi, lo + Math.floor((b + 1) * size) - 1);
    if (end < start) continue;
    let min = samples[start]!;
    let max = samples[start]!;
    for (let i = start + 1; i <= end; i++) {
      const s = samples[i]!;
      if (s.v < min.v) min = s;
      if (s.v > max.v) max = s;
    }
    // Time order within the bucket, so the path never doubles back on itself.
    if (min.t <= max.t) {
      out.push(min);
      if (max !== min) out.push(max);
    } else {
      out.push(max, min);
    }
  }
  return out;
}

/** The sample nearest `t`, or null when the series has none in range. */
export function nearestSample(samples: Sample[], t: number): Sample | null {
  if (samples.length === 0) return null;
  const i = lowerBound(samples, t);
  const before = samples[i - 1];
  const after = samples[i];
  if (!before) return after ?? null;
  if (!after) return before;
  return t - before.t <= after.t - t ? before : after;
}

// ------------------------------------------------------------------- svg

const SVG_NS = 'http://www.w3.org/2000/svg';

function svg<K extends keyof SVGElementTagNameMap>(
  tag: K,
  attrs: Record<string, string | number> = {},
): SVGElementTagNameMap[K] {
  const node = document.createElementNS(SVG_NS, tag);
  for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, String(value));
  return node;
}

function clear(node: Element): void {
  node.replaceChildren();
}

// ----------------------------------------------------------------- panels

interface PanelRuntime {
  spec: PanelSpec;
  root: HTMLElement;
  svg: SVGSVGElement;
  gridG: SVGGElement;
  seriesG: SVGGElement;
  eventsG: SVGGElement;
  flagsG: SVGGElement;
  axisG: SVGGElement;
  cursorG: SVGGElement;
  cursorLine: SVGLineElement;
  cursorDots: SVGCircleElement[];
  /** One per series, created once and mutated thereafter. */
  paths: SVGPathElement[];
  readout: HTMLElement;
  plotH: number;
  height: number;
  /** Current y scale, kept so the cursor layer can place dots without a rebuild. */
  y: (v: number) => number;
  /** Current visible slice per series, reused by the cursor layer. */
  slices: Array<[number, number]>;
}

// ------------------------------------------------------------------ chart

export class DynamicsChart {
  private readonly host: HTMLElement;
  private readonly stack: HTMLElement;
  private readonly tip: HTMLElement;
  private readonly uid = `lab${Math.random().toString(36).slice(2, 8)}`;

  private panels: PanelRuntime[] = [];
  private events: readonly MarketEvent[];
  private domain: Viewport;
  private viewport: Viewport;

  private frame = 0;
  private cursorFrame = 0;
  private cursorT: number | null = null;
  private lastDrawn: Viewport | null = null;

  private readonly onViewport?: (viewport: Viewport) => void;
  private readonly onEvent?: (event: MarketEvent) => void;

  private drag: { pointerId: number; startX: number; view: Viewport } | null = null;
  private readonly disposers: Array<() => void> = [];

  constructor(host: HTMLElement, opts: ChartOptions) {
    this.host = host;
    this.domain = opts.domain;
    this.viewport = opts.domain;
    this.events = opts.events ?? MARKET_EVENTS;
    this.onViewport = opts.onViewport;
    this.onEvent = opts.onEvent;

    host.replaceChildren();
    host.classList.add('labchart');

    this.stack = document.createElement('div');
    this.stack.className = 'labchart__stack';
    host.appendChild(this.stack);

    this.tip = document.createElement('div');
    this.tip.className = 'labtip';
    this.tip.hidden = true;
    host.appendChild(this.tip);

    this.setPanels(opts.panels);
  }

  // ------------------------------------------------------------ lifecycle

  /**
   * Swap the panel set, keeping the viewport.
   *
   * The overlay toggle replaces three panels with one. Rebuilding the chart
   * would reset the zoom, which is exactly the state the reader was using the
   * toggle to compare.
   */
  setPanels(specs: PanelSpec[]): void {
    clear(this.stack);
    this.panels = specs.map((spec, index) => this.buildPanel(spec, index === specs.length - 1));
    this.lastDrawn = null;
    this.schedule();
  }

  setDomain(domain: Viewport, resetViewport = true): void {
    this.domain = domain;
    if (resetViewport) this.viewport = domain;
    else this.viewport = this.clamp(this.viewport);
    this.lastDrawn = null;
    this.schedule();
  }

  getViewport(): Viewport {
    return this.viewport;
  }

  setViewport(next: Viewport): void {
    const clamped = this.clamp(next);
    if (clamped[0] === this.viewport[0] && clamped[1] === this.viewport[1]) return;
    this.viewport = clamped;
    this.onViewport?.(this.viewport);
    this.schedule();
  }

  resetViewport(): void {
    this.setViewport(this.domain);
  }

  /** Frame an event's own window, with a margin so it is not flush to the edge. */
  jumpTo(event: MarketEvent): void {
    const from = isoToTime(event.window.from);
    const to = isoToTime(event.window.to);
    const margin = Math.max((to - from) * 0.35, 120 * DAY);
    this.setViewport([from - margin, to + margin]);
  }

  destroy(): void {
    for (const dispose of this.disposers) dispose();
    this.disposers.length = 0;
    if (this.frame) cancelAnimationFrame(this.frame);
    if (this.cursorFrame) cancelAnimationFrame(this.cursorFrame);
    this.host.replaceChildren();
  }

  // ---------------------------------------------------------- construction

  private buildPanel(spec: PanelSpec, isLast: boolean): PanelRuntime {
    const height = spec.height + (isLast ? AXIS_H : 0);
    const plotH = spec.height - PAD.top - PAD.bottom;

    const root = document.createElement('section');
    root.className = 'labpanel';

    const head = document.createElement('div');
    head.className = 'labpanel__head';
    const title = document.createElement('h3');
    title.className = 'labpanel__title';
    title.textContent = spec.title;
    const note = document.createElement('span');
    note.className = 'labpanel__note';
    note.textContent = spec.note;
    const readout = document.createElement('span');
    readout.className = 'labpanel__readout';
    head.append(title, note, readout);
    root.appendChild(head);

    const node = svg('svg', {
      class: 'labpanel__svg',
      viewBox: `0 0 ${VIEW_W} ${height}`,
      preserveAspectRatio: 'xMidYMid meet',
      role: 'img',
      'aria-label': `${spec.title} — ${spec.note}`,
    });

    const defs = svg('defs');
    const clip = svg('clipPath', { id: `${this.uid}-${spec.key}` });
    clip.appendChild(
      svg('rect', { x: PAD.left, y: PAD.top, width: PLOT_W, height: Math.max(plotH, 1) }),
    );
    defs.appendChild(clip);
    node.appendChild(defs);

    const gridG = svg('g', { class: 'lab-grid' });
    const plotG = svg('g', { 'clip-path': `url(#${this.uid}-${spec.key})` });
    const seriesG = svg('g', { class: 'lab-series' });
    const eventsG = svg('g', { class: 'lab-events' });
    const cursorG = svg('g', { class: 'lab-cursor' });
    const axisG = svg('g', { class: 'lab-axis' });
    const flagsG = svg('g', { class: 'lab-flags' });

    const cursorLine = svg('line', {
      class: 'lab-cursor__line',
      x1: 0,
      x2: 0,
      y1: PAD.top,
      y2: PAD.top + Math.max(plotH, 1),
    });
    cursorG.appendChild(cursorLine);

    const paths = spec.series.map((series) => {
      const path = svg('path', { class: series.className, d: '' });
      seriesG.appendChild(path);
      return path;
    });
    const cursorDots = spec.series.map((series) => {
      const dot = svg('circle', { class: `lab-cursor__dot ${series.className}`, r: 3, cx: -99, cy: -99 });
      cursorG.appendChild(dot);
      return dot;
    });

    plotG.append(eventsG, seriesG);
    node.append(gridG, plotG, axisG, flagsG, cursorG);
    root.appendChild(node);

    const runtime: PanelRuntime = {
      spec,
      root,
      svg: node,
      gridG,
      seriesG,
      eventsG,
      flagsG,
      axisG,
      cursorG,
      cursorLine,
      cursorDots,
      paths,
      readout,
      plotH: Math.max(plotH, 1),
      height,
      y: () => 0,
      slices: spec.series.map(() => [0, -1] as [number, number]),
    };

    this.attach(node, isLast);
    this.stack.appendChild(root);
    return runtime;
  }

  /** Pointer, wheel and keyboard wiring for one panel's SVG. */
  private attach(node: SVGSVGElement, isLast: boolean): void {
    const onWheel = (event: WheelEvent) => {
      event.preventDefault();
      const t = this.timeAt(node, event.clientX);
      // deltaMode 1 is lines, 2 is pages — normalise so a trackpad and a mouse
      // wheel do not differ by two orders of magnitude.
      const unit = event.deltaMode === 1 ? 16 : event.deltaMode === 2 ? 400 : 1;
      this.zoomAt(t, Math.exp(event.deltaY * unit * 0.002));
    };

    const onPointerDown = (event: PointerEvent) => {
      if ((event.target as Element | null)?.closest('.lab-flag')) return;
      if (event.button !== 0) return;
      node.setPointerCapture(event.pointerId);
      this.drag = { pointerId: event.pointerId, startX: event.clientX, view: this.viewport };
      node.classList.add('is-panning');
    };

    const onPointerMove = (event: PointerEvent) => {
      if (this.drag && this.drag.pointerId === event.pointerId) {
        const rect = node.getBoundingClientRect();
        if (rect.width > 0) {
          const [v0, v1] = this.drag.view;
          // Pixels to milliseconds through the plot area, not the element: the
          // padding is not part of the time axis.
          const perPixel = (v1 - v0) / (rect.width * (PLOT_W / VIEW_W));
          const shift = -(event.clientX - this.drag.startX) * perPixel;
          this.setViewport([v0 + shift, v1 + shift]);
        }
      }
      this.setCursor(this.timeAt(node, event.clientX), event.clientX, event.clientY);
    };

    const onPointerUp = (event: PointerEvent) => {
      if (this.drag?.pointerId === event.pointerId) {
        node.releasePointerCapture(event.pointerId);
        this.drag = null;
        node.classList.remove('is-panning');
      }
    };

    const onLeave = () => this.clearCursor();
    const onDoubleClick = () => this.resetViewport();

    node.addEventListener('wheel', onWheel, { passive: false });
    node.addEventListener('pointerdown', onPointerDown);
    node.addEventListener('pointermove', onPointerMove);
    node.addEventListener('pointerup', onPointerUp);
    node.addEventListener('pointercancel', onPointerUp);
    node.addEventListener('pointerleave', onLeave);
    node.addEventListener('dblclick', onDoubleClick);

    this.disposers.push(() => {
      node.removeEventListener('wheel', onWheel);
      node.removeEventListener('pointerdown', onPointerDown);
      node.removeEventListener('pointermove', onPointerMove);
      node.removeEventListener('pointerup', onPointerUp);
      node.removeEventListener('pointercancel', onPointerUp);
      node.removeEventListener('pointerleave', onLeave);
      node.removeEventListener('dblclick', onDoubleClick);
    });
    void isLast;
  }

  // -------------------------------------------------------------- viewport

  private clamp(next: Viewport): Viewport {
    const [dMin, dMax] = this.domain;
    const full = dMax - dMin;
    let span = Math.min(Math.max(next[1] - next[0], MIN_SPAN_MS), full);
    if (!Number.isFinite(span) || span <= 0) span = Math.max(full, MIN_SPAN_MS);
    let t0 = next[0];
    if (t0 < dMin) t0 = dMin;
    if (t0 + span > dMax) t0 = dMax - span;
    return [t0, t0 + span];
  }

  private zoomAt(anchor: number, factor: number): void {
    const [v0, v1] = this.viewport;
    const span = v1 - v0;
    const fraction = span === 0 ? 0.5 : (anchor - v0) / span;
    const nextSpan = span * factor;
    this.setViewport([anchor - fraction * nextSpan, anchor + (1 - fraction) * nextSpan]);
  }

  /** Client x -> instant, through the viewBox scale. */
  private timeAt(node: SVGSVGElement, clientX: number): number {
    const rect = node.getBoundingClientRect();
    if (rect.width === 0) return this.viewport[0];
    const vx = (clientX - rect.left) * (VIEW_W / rect.width);
    const fraction = (vx - PAD.left) / PLOT_W;
    const [v0, v1] = this.viewport;
    return v0 + Math.min(Math.max(fraction, 0), 1) * (v1 - v0);
  }

  private x(t: number): number {
    const [v0, v1] = this.viewport;
    return PAD.left + (v1 === v0 ? 0.5 : (t - v0) / (v1 - v0)) * PLOT_W;
  }

  // ---------------------------------------------------------------- render

  private schedule(): void {
    if (this.frame) return;
    this.frame = requestAnimationFrame(() => {
      this.frame = 0;
      this.draw();
    });
  }

  /**
   * Redraw the series and grid layers for the current viewport.
   *
   * Called at most once per animation frame, and skipped entirely when the
   * viewport has not moved since the last one — a pointer sitting still over a
   * panel produces no work here at all.
   */
  private draw(): void {
    if (this.lastDrawn && this.lastDrawn[0] === this.viewport[0] && this.lastDrawn[1] === this.viewport[1]) {
      return;
    }
    this.lastDrawn = this.viewport;

    const [v0, v1] = this.viewport;
    const ticks = timeTicks(v0, v1);
    const last = this.panels[this.panels.length - 1];

    for (const panel of this.panels) {
      this.drawPanel(panel, ticks, panel === last);
    }
    if (this.cursorT !== null) this.drawCursor();
  }

  private drawPanel(panel: PanelRuntime, ticks: TimeTick[], isLast: boolean): void {
    const { spec } = panel;
    const top = PAD.top;
    const plotH = panel.plotH;

    // --- visible slices and the y domain over them -------------------------
    let lo = Infinity;
    let hi = -Infinity;
    let anyPoints = false;
    spec.series.forEach((series, i) => {
      const range = visibleRange(series.samples, this.viewport[0], this.viewport[1]);
      panel.slices[i] = range;
      for (let k = range[0]; k <= range[1]; k++) {
        const v = series.samples[k]!.v;
        if (v < lo) lo = v;
        if (v > hi) hi = v;
        anyPoints = true;
      }
    });

    const useLog = Boolean(spec.log) && anyPoints && lo > 0;
    const project = (v: number) => (useLog ? Math.log(v) : v);

    if (!anyPoints) {
      lo = 0;
      hi = 1;
    }
    let pLo = project(useLog ? lo : lo);
    let pHi = project(useLog ? hi : hi);
    if (spec.zeroLine && !useLog) {
      pLo = Math.min(pLo, 0);
      pHi = Math.max(pHi, 0);
    }
    for (const band of spec.bands ?? []) {
      if (useLog) continue;
      pLo = Math.min(pLo, -band);
      pHi = Math.max(pHi, band);
    }
    const gap = pHi - pLo;
    const pad = gap < 1e-9 ? Math.max(Math.abs(pHi) * 0.02, 0.01) : gap * 0.08;
    const yMin = pLo - pad;
    const yMax = pHi + pad;
    const y = (v: number) => top + plotH - ((project(v) - yMin) / (yMax - yMin)) * plotH;
    panel.y = y;

    // --- grid --------------------------------------------------------------
    clear(panel.gridG);
    clear(panel.axisG);

    const values = useLog ? logTicks(Math.exp(yMin), Math.exp(yMax)) : linearTicks(yMin, yMax);
    const maxAbs = values.reduce((acc, v) => Math.max(acc, Math.abs(v)), 0);
    const seriesFormat = spec.series[0]?.format ?? String;
    const formatY = spec.axisFormat ? (v: number) => spec.axisFormat!(v, maxAbs) : seriesFormat;
    for (const value of values) {
      const gy = y(value);
      if (gy < top - 1 || gy > top + plotH + 1) continue;
      panel.gridG.appendChild(
        svg('line', { class: 'lab-grid__line', x1: PAD.left, x2: VIEW_W - PAD.right, y1: gy, y2: gy }),
      );
      const label = svg('text', { class: 'lab-grid__label', x: PAD.left - 7, y: gy + 3, 'text-anchor': 'end' });
      label.textContent = formatY(value);
      panel.axisG.appendChild(label);
    }

    for (const tick of ticks) {
      const tx = this.x(tick.t);
      panel.gridG.appendChild(
        svg('line', { class: 'lab-grid__line lab-grid__line--v', x1: tx, x2: tx, y1: top, y2: top + plotH }),
      );
      if (isLast) {
        const label = svg('text', {
          class: 'lab-grid__label',
          x: tx,
          y: top + plotH + AXIS_LABEL_DY,
          'text-anchor': 'middle',
        });
        label.textContent = tick.label;
        panel.axisG.appendChild(label);
      }
    }

    if (spec.zeroLine && !useLog && yMin < 0 && yMax > 0) {
      panel.gridG.appendChild(
        svg('line', { class: 'lab-grid__zero', x1: PAD.left, x2: VIEW_W - PAD.right, y1: y(0), y2: y(0) }),
      );
    }
    for (const band of spec.bands ?? []) {
      for (const level of [band, -band]) {
        const by = y(level);
        if (by < top || by > top + plotH) continue;
        panel.gridG.appendChild(
          svg('line', { class: 'lab-grid__band', x1: PAD.left, x2: VIEW_W - PAD.right, y1: by, y2: by }),
        );
      }
    }
    panel.axisG.appendChild(
      svg('line', { class: 'lab-axis__line', x1: PAD.left, x2: PAD.left, y1: top, y2: top + plotH }),
    );

    // An empty panel says so. A bare axis with no line on it is indistinguishable
    // from a series that is flat at zero, and on a derivative panel that is a
    // meaningful reading rather than an absence of one.
    if (!anyPoints) {
      const message = svg('text', {
        class: 'lab-empty',
        x: PAD.left + PLOT_W / 2,
        y: top + plotH / 2,
        'text-anchor': 'middle',
      });
      message.textContent = spec.series.some((s) => s.samples.length > 0)
        ? 'no published values in this window'
        : 'not published for this range';
      panel.axisG.appendChild(message);
    }

    // --- series ------------------------------------------------------------
    spec.series.forEach((series, i) => {
      const [from, to] = panel.slices[i]!;
      const drawn = reduce(series.samples, from, to);
      // One string, one attribute write. Building nodes per point is what makes
      // a hand-rolled chart slow; building one path string does not.
      let d = '';
      const maxGap = series.maxGap;
      let previous: number | null = null;
      for (let k = 0; k < drawn.length; k++) {
        const s = drawn[k]!;
        // `M` rather than `L` after a hole wider than the series tolerates, so
        // the line breaks instead of asserting a value across it. With `maxGap`
        // undefined this is exactly the old unconditional `k === 0` test.
        const broken = maxGap !== undefined && previous !== null && s.t - previous > maxGap;
        d += `${k === 0 || broken ? 'M' : 'L'}${this.x(s.t).toFixed(1)},${y(s.v).toFixed(1)}`;
        previous = s.t;
      }
      panel.paths[i]!.setAttribute('d', d);
    });

    // --- events ------------------------------------------------------------
    clear(panel.eventsG);
    clear(panel.flagsG);
    const visible = this.events.filter((e) => {
      const t = isoToTime(e.date);
      return t >= this.viewport[0] && t <= this.viewport[1];
    });
    for (const event of visible) {
      const ex = this.x(isoToTime(event.date));
      panel.eventsG.appendChild(
        svg('line', { class: 'lab-event__line', x1: ex, x2: ex, y1: top, y2: top + plotH }),
      );
      if (!isLast) continue;
      panel.flagsG.appendChild(this.buildFlag(event, ex, top + plotH + FLAG_DY));
    }
  }

  /** A clickable, keyboard-reachable marker under the shared axis. */
  private buildFlag(event: MarketEvent, x: number, y: number): SVGGElement {
    const g = svg('g', {
      class: 'lab-flag',
      transform: `translate(${x.toFixed(1)} ${y.toFixed(1)})`,
      role: 'button',
      tabindex: 0,
      'aria-label': `Jump to ${event.label}`,
    });
    g.appendChild(svg('rect', { class: 'lab-flag__box', x: -17, y: 2, width: 34, height: 13, rx: 2 }));
    const text = svg('text', { class: 'lab-flag__text', x: 0, y: 12, 'text-anchor': 'middle' });
    text.textContent = event.short;
    g.appendChild(text);
    const title = svg('title');
    title.textContent = `${event.label} — ${event.note}`;
    g.appendChild(title);

    const activate = () => {
      this.onEvent?.(event);
      this.jumpTo(event);
    };
    g.addEventListener('click', activate);
    g.addEventListener('keydown', (native) => {
      const key = (native as KeyboardEvent).key;
      if (key === 'Enter' || key === ' ') {
        native.preventDefault();
        activate();
      }
    });
    return g;
  }

  // ---------------------------------------------------------------- cursor

  /**
   * Shared crosshair and tooltip.
   *
   * Scheduled apart from {@link draw} and touching only the cursor layer: moving
   * the pointer must not rebuild a path, because at a century of data that is
   * the difference between a crosshair that tracks and one that lags.
   */
  private setCursor(t: number, clientX: number, clientY: number): void {
    this.cursorT = t;
    this.tipPosition = { x: clientX, y: clientY };
    if (this.cursorFrame) return;
    this.cursorFrame = requestAnimationFrame(() => {
      this.cursorFrame = 0;
      this.drawCursor();
    });
  }

  private tipPosition = { x: 0, y: 0 };

  private clearCursor(): void {
    this.cursorT = null;
    this.tip.hidden = true;
    for (const panel of this.panels) {
      panel.cursorLine.setAttribute('x1', '-99');
      panel.cursorLine.setAttribute('x2', '-99');
      for (const dot of panel.cursorDots) {
        dot.setAttribute('cx', '-99');
        dot.setAttribute('cy', '-99');
      }
      panel.readout.textContent = '';
    }
  }

  private drawCursor(): void {
    const t = this.cursorT;
    if (t === null) return;
    const cx = this.x(t);

    let date: string | null = null;
    const rows: Array<{ label: string; value: string; className: string }> = [];

    for (const panel of this.panels) {
      panel.cursorLine.setAttribute('x1', cx.toFixed(1));
      panel.cursorLine.setAttribute('x2', cx.toFixed(1));

      const readouts: string[] = [];
      panel.spec.series.forEach((series, i) => {
        const sample = nearestSample(series.samples, t);
        const dot = panel.cursorDots[i]!;
        if (!sample) {
          dot.setAttribute('cx', '-99');
          dot.setAttribute('cy', '-99');
          rows.push({ label: series.label, value: 'no data', className: series.className });
          return;
        }
        dot.setAttribute('cx', this.x(sample.t).toFixed(1));
        dot.setAttribute('cy', panel.y(sample.v).toFixed(1));
        // The nearest sample's own date, not the pointer's — with decimated
        // history those differ, and the tooltip must name the row it read.
        if (date === null || Math.abs(sample.t - t) < Math.abs(isoToTime(date) - t)) {
          date = timeToIso(sample.t);
        }
        const formatted = series.format(sample.v);
        readouts.push(formatted);
        rows.push({ label: series.label, value: formatted, className: series.className });
      });
      panel.readout.textContent = readouts.join('  ·  ');
    }

    this.renderTip(date ?? timeToIso(t), rows);
  }

  private renderTip(date: string, rows: Array<{ label: string; value: string; className: string }>): void {
    const parts = [`<div class="labtip__date">${date}</div>`];
    for (const row of rows) {
      parts.push(
        `<div class="labtip__row"><span class="labtip__swatch ${row.className}"></span>` +
          `<span class="labtip__label">${row.label}</span>` +
          `<span class="labtip__value">${row.value}</span></div>`,
      );
    }
    // Every fragment above is either a fixed string or a number formatted by
    // this file — no API text reaches this markup.
    this.tip.innerHTML = parts.join('');
    this.tip.hidden = false;

    const hostRect = this.host.getBoundingClientRect();
    const tipRect = this.tip.getBoundingClientRect();
    let left = this.tipPosition.x - hostRect.left + 16;
    if (left + tipRect.width > hostRect.width) left = this.tipPosition.x - hostRect.left - tipRect.width - 16;
    let top = this.tipPosition.y - hostRect.top + 16;
    if (top + tipRect.height > hostRect.height) top = hostRect.height - tipRect.height - 8;
    this.tip.style.left = `${Math.max(left, 4)}px`;
    this.tip.style.top = `${Math.max(top, 4)}px`;
  }
}

export { PLOT_W, VIEW_W, PAD };
