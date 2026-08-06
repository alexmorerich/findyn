/**
 * FinCrypto page (04-ui-plan.md §P5).
 *
 * Five views over one research engine: the state tiles, a regime timeline strip,
 * the speculation index, the liquidity beta with its R², and the supply
 * schedule.
 *
 * Two things this file does that the other engine pages do not, and both are
 * deliberate:
 *
 * 1. **`expected_return` is rendered as an absence with a reason**, not as a
 *    missing tile. The engine sets it to `null` on purpose and the page has to
 *    say so, because a blank where every other engine shows a number reads as a
 *    bug rather than as a claim being declined.
 * 2. **The regime is drawn as a timeline strip, not a stacked posterior.** Gold
 *    and equity publish distributions and a stack is the honest shape for those.
 *    This engine publishes a *label* from two thresholds — it has no posterior —
 *    and drawing one anyway would invent a precision the model does not have.
 *
 * Empty, stale and 501 states are rendered explicitly everywhere.
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

const ASSET = 'crypto';

/** Ten years of daily observations, matching the engine's nightly window. */
const HISTORY_LIMIT = 4000;

/** Server-side decimation target — a decade of daily points is ~3,650. */
const CHART_POINTS = 1200;

/** Mirrors findynamics/engines/crypto/domain.py::CRYPTO_REGIMES, in wire order. */
const CRYPTO_REGIMES = ['winter', 'normal', 'frenzy'] as const;
type CryptoRegime = (typeof CRYPTO_REGIMES)[number];

const REGIME_COPY: Record<CryptoRegime, string> = {
  winter:
    'More than 45% below the trailing-year peak. A statement about depth, not duration — March 2020 spent a fortnight here on a 50% crash and left again.',
  normal:
    'Neither of the others. The honest label for “nothing in particular is happening”, which is more of the record than the other two combined.',
  frenzy:
    'More than double a year ago AND volatility elevated. Both are required: bitcoin has had 100% years without a blowoff and volatile years without a trend.',
};

const stateHost = document.querySelector('#crypto-state');
const regimeHost = document.querySelector('#crypto-regime');
const speculationHost = document.querySelector('#crypto-speculation');
const betaHost = document.querySelector('#crypto-beta');
const supplyHost = document.querySelector('#crypto-supply');
const signalsHost = document.querySelector('#crypto-signals');

// ------------------------------------------------------------------ charts

const W = 900;
const H = 300;
const PAD = { top: 18, right: 18, bottom: 34, left: 56 };

function noData(svg: SVGSVGElement, message: string): SVGSVGElement {
  svg.appendChild(svgEl('text', { x: W / 2, y: H / 2, 'text-anchor': 'middle' }, message));
  return svg;
}

function axes(svg: SVGSVGElement, ticks: { y: number; label: string }[]): void {
  for (const tick of ticks) {
    svg.appendChild(
      svgEl('line', { class: 'grid', x1: PAD.left, x2: W - PAD.right, y1: tick.y, y2: tick.y }),
    );
    svg.appendChild(
      svgEl('text', { x: PAD.left - 8, y: tick.y + 3, 'text-anchor': 'end' }, tick.label),
    );
  }
  svg.appendChild(
    svgEl('line', { class: 'axis', x1: PAD.left, x2: PAD.left, y1: PAD.top, y2: H - PAD.bottom }),
  );
}

function endLabels(svg: SVGSVGElement, first: string | undefined, last: string | undefined): void {
  if (first) svg.appendChild(svgEl('text', { x: PAD.left, y: H - 12 }, first));
  if (last) {
    svg.appendChild(svgEl('text', { x: W - PAD.right, y: H - 12, 'text-anchor': 'end' }, last));
  }
}

/**
 * The speculation index on a fixed 0-100 axis.
 *
 * Fixed, never auto-scaled. The index is a position on a defined scale that
 * spends most of its life at zero, and letting the axis follow the data would
 * turn a decade of "no" into a chart full of drama.
 */
function speculationChart(points: HistoryPoint[]): SVGSVGElement {
  const svg = svgEl('svg', {
    class: 'chart',
    viewBox: `0 0 ${W} ${H}`,
    role: 'img',
    preserveAspectRatio: 'xMidYMid meet',
    'aria-label': `Bitcoin speculation index over ${points.length} observations, 0 to 100`,
  });

  const usable = points.filter((p) => Number.isFinite(p.value));
  if (usable.length < 2) return noData(svg, 'not enough history to plot');

  const plotW = W - PAD.left - PAD.right;
  const plotH = H - PAD.top - PAD.bottom;
  const x = (i: number) => PAD.left + (i / (usable.length - 1)) * plotW;
  const y = (v: number) => PAD.top + plotH - (v / 100) * plotH;

  axes(
    svg,
    [0, 25, 50, 75, 100].map((v) => ({ y: y(v), label: String(v) })),
  );
  svg.appendChild(
    svgEl('line', {
      class: 'axis',
      x1: PAD.left,
      x2: W - PAD.right,
      y1: y(60),
      y2: y(60),
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

  endLabels(svg, usable[0]?.as_of, usable[usable.length - 1]?.as_of);
  svg.appendChild(svgEl('text', { x: PAD.left, y: 12 }, 'speculation index — 60 is the signal threshold'));
  return svg;
}

/**
 * The liquidity beta over time, auto-scaled.
 *
 * Auto-scaled here, unlike the index above, because a regression coefficient has
 * no defined range — a fixed axis would be inventing one. The zero line is drawn
 * because it is the only meaningful reference the quantity has.
 */
function betaChart(points: HistoryPoint[]): SVGSVGElement {
  const svg = svgEl('svg', {
    class: 'chart',
    viewBox: `0 0 ${W} ${H}`,
    role: 'img',
    preserveAspectRatio: 'xMidYMid meet',
    'aria-label': 'Bitcoin liquidity beta, expanding window',
  });

  const usable = points.filter((p) => Number.isFinite(p.value));
  if (usable.length < 2) return noData(svg, 'not enough history to estimate a beta');

  const values = usable.map((p) => p.value);
  const lo = Math.min(0, ...values);
  const hi = Math.max(0, ...values);
  const span = hi - lo || 1;
  const plotW = W - PAD.left - PAD.right;
  const plotH = H - PAD.top - PAD.bottom;
  const x = (i: number) => PAD.left + (i / (usable.length - 1)) * plotW;
  const y = (v: number) => PAD.top + plotH - ((v - lo) / span) * plotH;

  axes(
    svg,
    [0, 0.25, 0.5, 0.75, 1].map((f) => ({
      y: PAD.top + plotH - f * plotH,
      label: (lo + f * span).toFixed(1),
    })),
  );
  svg.appendChild(
    svgEl('line', {
      class: 'axis',
      x1: PAD.left,
      x2: W - PAD.right,
      y1: y(0),
      y2: y(0),
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

  endLabels(svg, usable[0]?.as_of, usable[usable.length - 1]?.as_of);
  svg.appendChild(
    svgEl('text', { x: PAD.left, y: 12 }, 'monthly log return per unit log change in the money stock'),
  );
  return svg;
}

/**
 * The regime as a timeline strip: one coloured band per contiguous run.
 *
 * A strip rather than a stacked area, because this engine publishes a *label*
 * and not a posterior. Gold's stacked chart is the right shape for a
 * distribution; drawing one from a thresholded label would invent a confidence
 * the model never computed.
 */
function regimeStrip(points: HistoryPoint[]): SVGSVGElement {
  const STRIP_H = 96;
  const svg = svgEl('svg', {
    class: 'chart',
    viewBox: `0 0 ${W} ${STRIP_H}`,
    role: 'img',
    preserveAspectRatio: 'none',
    'aria-label': 'Bitcoin regime over time',
  });

  const usable = points.filter((p) => Number.isFinite(p.value));
  if (usable.length < 2) {
    svg.appendChild(
      svgEl('text', { x: W / 2, y: STRIP_H / 2, 'text-anchor': 'middle' }, 'no regime published yet'),
    );
    return svg;
  }

  const plotW = W - PAD.left - PAD.right;
  const band = 44;
  const top = 20;
  const width = plotW / usable.length;

  usable.forEach((point, i) => {
    // `regime_code` is the index into CRYPTO_REGIMES — engine_output stores
    // REALs, so the label travels as its position in the wire-order vocabulary.
    const name = CRYPTO_REGIMES[Math.round(point.value)];
    if (!name) return;
    svg.appendChild(
      svgEl('rect', {
        class: `regimeband regimeband--${name}`,
        x: (PAD.left + i * width).toFixed(2),
        y: top,
        // +0.6 closes the hairline seams that otherwise appear between
        // consecutive same-coloured days at fractional widths.
        width: (width + 0.6).toFixed(2),
        height: band,
      }),
    );
  });

  svg.appendChild(svgEl('text', { x: PAD.left, y: 12 }, 'regime'));
  svg.appendChild(svgEl('text', { x: PAD.left, y: top + band + 16 }, usable[0]?.as_of ?? ''));
  svg.appendChild(
    svgEl(
      'text',
      { x: W - PAD.right, y: top + band + 16, 'text-anchor': 'end' },
      usable[usable.length - 1]?.as_of ?? '',
    ),
  );
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

function num(value: number | undefined, digits = 2, unit = ''): string {
  if (value === undefined || !Number.isFinite(value)) return '—';
  return `${value.toFixed(digits)}${unit}`;
}

function speculationTone(score: number | undefined): Tone {
  if (score === undefined || !Number.isFinite(score)) return 'idle';
  if (score >= 60) return 'bad';
  if (score >= 20) return 'warn';
  return 'ok';
}

function renderState(result: ApiResult<AssetState>): AssetState | null {
  if (!stateHost) return null;
  if (!result.ok) {
    replace(stateHost, failureBlock(result, 'FinCrypto state'));
    return null;
  }

  const state = result.envelope.data;
  const components = state.components ?? {};

  const tiles = el(
    'div',
    { class: 'tiles' },
    tile('Regime', state.regime, regimeTone(state.regime), `As of ${state.as_of}`),
    tile(
      'Speculation index',
      formatValue(components.speculation_index),
      speculationTone(components.speculation_index),
      `0-100 from ${num(components.speculation_terms, 0)} of 3 terms, combined geometrically.`,
    ),
    tile(
      'Liquidity beta',
      formatValue(components.liquidity_beta),
      'idle',
      `Expanding window over ${num(components.liquidity_beta_months, 0)} months. R² ${num(components.liquidity_beta_r2, 3)} — read that first.`,
    ),
    tile(
      'Risk score',
      formatValue(state.risk_score),
      state.risk_score !== null && state.risk_score >= 60 ? 'warn' : 'ok',
      'Realized volatility and jump intensity, on a bitcoin-calibrated scale. Not comparable with another engine’s risk score.',
    ),
    tile(
      'Confidence',
      formatValue(state.confidence),
      'idle',
      `Capped at ${num(components.confidence_ceiling, 2)} by construction — four cycles of history, a market whose structure changed inside the sample, and a co-movement rather than a mechanism.`,
    ),
  );

  // The absence, stated. A blank where every other engine shows a number reads
  // as a bug; this is a claim being declined and the page has to say which.
  // Routed through stateBlock like every other non-value on the site, so it is
  // styled as a deliberate state rather than as loose prose.
  const noReturn = stateBlock({
    tone: 'warn',
    title: 'No expected return, by design',
    detail:
      'Every other engine here can point at something that generates one — an observable short rate, a fitted curve, an earnings stream, or six hundred months of regime-conditional history. Bitcoin has no cash flow, no issuer and four cycles.',
    hint: 'A mean over four cycles is not a weak estimate of an expected return; it is a description of four events. Publishing it as one would put it into the field an allocation optimises against.',
  });

  const staleness = result.envelope.stale
    ? stateBlock({
        tone: 'warn',
        title: 'This state is stale',
        detail: `Last published ${formatDate(state.as_of)}. The daily run has not reported since.`,
      })
    : null;

  const fallback =
    components.price_from_fallback_source === 1
      ? el(
          'p',
          { class: 'enginecard__detail' },
          'The configured Stooq price was unreachable for this run and the Yahoo fallback carried the series. Same instrument; an unversioned endpoint, so confidence takes a small penalty.',
        )
      : null;

  replace(
    stateHost,
    tiles,
    noReturn,
    staleness,
    fallback,
    el(
      'p',
      { class: 'enginecard__detail' },
      `Model ${state.model_version}. Research only — this engine is excluded from the portfolio layer and nothing outside its own package may import it.`,
    ),
  );
  return state;
}

function renderRegime(state: AssetState | null, result: ApiResult<AssetHistory>): void {
  if (!regimeHost) return;
  if (!result.ok) {
    replace(regimeHost, failureBlock(result, 'Regime history'));
    return;
  }

  const points = result.envelope.data.points ?? [];
  const rows = CRYPTO_REGIMES.map((regime) =>
    el(
      'tr',
      {},
      el('td', { class: 'mono nowrap' }, regime),
      el('td', {}, state?.regime === regime ? badge('current', regimeTone(regime)) : ''),
      el('td', {}, REGIME_COPY[regime]),
    ) as HTMLTableRowElement,
  );

  const legend = el(
    'div',
    { class: 'legend' },
    ...CRYPTO_REGIMES.map((regime) =>
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
    points.length === 0
      ? emptyBlock('regime history', 'The engine has not published this metric yet.')
      : regimeStrip(points),
    legend,
    table(['Regime', '', 'What it means'], rows),
  );
}

function renderHistory(
  host: Element | null,
  result: ApiResult<AssetHistory>,
  what: string,
  chart: (points: HistoryPoint[]) => SVGSVGElement,
  footer?: HTMLElement | null,
): void {
  if (!host) return;
  if (!result.ok) {
    replace(host, failureBlock(result, what));
    return;
  }
  const points = result.envelope.data.points ?? [];
  if (points.length === 0) {
    replace(host, emptyBlock(what, 'The engine has not published this metric yet.'));
    return;
  }
  replace(
    host,
    chart(points),
    result.envelope.data.truncated
      ? stateBlock({
          tone: 'warn',
          title: 'This series is clipped',
          detail:
            'The requested window exceeded the API’s row ceiling, so this is a prefix rather than the whole history.',
        })
      : null,
    footer ?? null,
  );
}

function renderSupply(state: AssetState | null): void {
  if (!supplyHost) return;
  if (!state) {
    replace(supplyHost, emptyBlock('supply schedule', 'The engine has not published a state yet.'));
    return;
  }
  const c = state.components ?? {};
  const projected = c.next_halving_is_projected === 1;

  replace(
    supplyHost,
    el(
      'div',
      { class: 'tiles' },
      tile(
        'Issued supply',
        c.issued_supply === undefined ? '—' : `${(c.issued_supply / 1e6).toFixed(3)}M BTC`,
        'info',
        'Issued, not circulating: coins in blocks whose miner claimed less than the full subsidy, and coins whose keys are lost, are counted here and are not spendable.',
      ),
      tile(
        'Issuance rate',
        num(c.issuance_rate, 3, '%'),
        'info',
        'Annual issuance as a percentage of issued supply, at the current block subsidy.',
      ),
      tile(
        'Stock-to-flow',
        num(c.stock_to_flow, 1),
        'idle',
        'A supply statistic and nothing else. Nothing on this page reads it; the price model that made it famous was falsified in 2022.',
      ),
      tile(
        'Next halving',
        c.days_to_next_halving === undefined ? '—' : `${Math.round(c.days_to_next_halving)} days`,
        'idle',
        projected
          ? 'Projected at the nominal ten-minute block time. Every observed epoch has run 4-10% short of nominal, so this runs late.'
          : 'An observed halving date.',
      ),
    ),
    el(
      'p',
      { class: 'enginecard__detail' },
      'The four halvings that have happened are facts with block heights and dates, hard-coded because they are consensus constants rather than measurements a publisher could revise. Supply inside a completed epoch is exact; inside the current one it is interpolated in time.',
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
  for (const host of [stateHost, regimeHost, speculationHost, betaHost, supplyHost, signalsHost]) {
    if (host) replace(host, loadingBlock('FinCrypto'));
  }

  const [stateResult, regimeResult, speculationResult, betaResult, r2Result] = await Promise.all([
    getAssetState(ASSET),
    getAssetHistory(ASSET, 'regime_code', { limit: HISTORY_LIMIT, points: CHART_POINTS }),
    getAssetHistory(ASSET, 'speculation_index', { limit: HISTORY_LIMIT, points: CHART_POINTS }),
    getAssetHistory(ASSET, 'liquidity_beta', { limit: HISTORY_LIMIT, points: CHART_POINTS }),
    getAssetHistory(ASSET, 'liquidity_beta_r2', { limit: HISTORY_LIMIT, points: CHART_POINTS }),
  ]);

  const state = renderState(stateResult);
  renderRegime(state, regimeResult);
  renderHistory(speculationHost, speculationResult, 'Speculation index', speculationChart);

  // The R² travels with the beta rather than getting a chart of its own: it is
  // the number that says whether the other chart means anything, and putting it
  // in a separate panel would let someone read the coefficient without it.
  const r2Latest = r2Result.ok ? r2Result.envelope.data.points.at(-1)?.value : undefined;
  renderHistory(
    betaHost,
    betaResult,
    'Liquidity beta',
    betaChart,
    r2Latest === undefined
      ? null
      : el(
          'p',
          { class: 'enginecard__detail' },
          `Newest R² ${r2Latest.toFixed(3)} — the share of bitcoin's monthly variance this relationship explains. A coefficient can be well-determined in sign and still explain almost none of the variance, which is the case here and is why no expected return is derived from it.`,
        ),
  );
  renderSupply(state);
  renderSignals(state);
}

void main();
