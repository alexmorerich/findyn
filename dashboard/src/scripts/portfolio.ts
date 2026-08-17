/**
 * The /portfolio page (04-ui-plan.md §P6).
 *
 * A profile switcher over the three risk profiles, a weight *distribution* per
 * asset drawn as a box-and-whisker (never a single number), the templated
 * conditional implication, degraded badges, and a "why" expander listing every
 * input AssetState. Everything comes from one endpoint — the compute plane
 * packed the whole distribution into one row — so a profile switch is one fetch.
 *
 * Empty, stale, 501 and unreachable states are all rendered explicitly: a
 * profile that has never been allocated says "planned" rather than showing a
 * blank chart.
 */
import {
  PROFILES,
  PROFILE_LABELS,
  getPortfolio,
  type ApiResult,
  type PortfolioInput,
  type PortfolioState,
  type WeightBand,
} from '../lib/api';
import {
  badge,
  el,
  failureBlock,
  loadingBlock,
  replace,
  stateBlock,
  svgEl,
  table,
} from '../lib/dom';
import { formatValue, type Tone } from '../lib/format';
import { regimeTone, riskTone } from './engines';

/** Plain-language asset names, mirroring config/portfolio.yaml labels. */
const ASSET_LABELS: Record<string, string> = {
  equity: 'equities',
  rates: 'government bonds',
  gold: 'gold',
  money: 'cash',
  crypto: 'crypto',
};

const switcherHost = document.querySelector('#portfolio-switcher');
const weightsHost = document.querySelector('#portfolio-weights');
const legendHost = document.querySelector('#portfolio-legend');
const implicationHost = document.querySelector('#portfolio-implication');
const whyHost = document.querySelector('#portfolio-why');

function currentProfile(): string {
  const requested = new URLSearchParams(window.location.search).get('profile');
  return requested && (PROFILES as readonly string[]).includes(requested) ? requested : 'balanced';
}

let active = currentProfile();

function percent(value: number | null | undefined, digits = 1): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return '—';
  return `${(value * 100).toFixed(digits)}%`;
}

function signedPercent(value: number, digits = 1): string {
  const sign = value > 0 ? '+' : '';
  return `${sign}${(value * 100).toFixed(digits)}pp`;
}

/** A quantile weight, falling back to the mean if the label is somehow absent. */
function q(band: WeightBand, label: string): number {
  const value = band.quantiles[label];
  return typeof value === 'number' && Number.isFinite(value) ? value : band.mean;
}

// ---------------------------------------------------------------- switcher

function renderSwitcher(): void {
  if (!switcherHost) return;
  replace(
    switcherHost,
    ...PROFILES.map((profile) => {
      const label = PROFILE_LABELS[profile]?.title ?? profile;
      const button = el(
        'button',
        {
          type: 'button',
          class: `switch ${profile === active ? 'switch--on' : ''}`,
          'aria-pressed': profile === active ? 'true' : 'false',
        },
        label,
      );
      button.addEventListener('click', () => {
        if (profile === active) return;
        active = profile;
        // Keep the URL shareable — a linked profile deep-links to itself.
        const url = new URL(window.location.href);
        url.searchParams.set('profile', profile);
        window.history.replaceState({}, '', url);
        renderSwitcher();
        void load();
      });
      return button;
    }),
  );
}

// ------------------------------------------------------------------ chart

const W = 900;
const LEFT = 150;
const RIGHT = 64;
const TOP = 16;
const ROW_H = 46;
const AXIS_H = 34;

function niceCeil(value: number): number {
  if (value <= 0) return 0.1;
  const capped = Math.min(value, 1);
  // Round up to the next 0.05 so the axis lands on readable percentages.
  return Math.min(1, Math.ceil(capped / 0.05) * 0.05);
}

/**
 * The weight distribution as horizontal box-and-whiskers, one row per asset.
 *
 * Box = q25–q75, whiskers = q05–q95, solid marker = median, hollow tick =
 * neutral. A degraded asset has a zero-width distribution (a point mass at
 * neutral); it is drawn as a single hollow marker with a "held at neutral" note
 * rather than an invisible box.
 */
function weightsChart(bands: WeightBand[]): SVGSVGElement {
  const height = TOP + bands.length * ROW_H + AXIS_H;
  const svg = svgEl('svg', {
    class: 'chart',
    viewBox: `0 0 ${W} ${height}`,
    role: 'img',
    preserveAspectRatio: 'xMidYMid meet',
    'aria-label': `Weight distribution across ${bands.length} assets`,
  });

  const xMax = niceCeil(
    Math.max(0.1, ...bands.map((b) => Math.max(q(b, 'q95'), b.neutral, b.mean))),
  );
  const plotW = W - LEFT - RIGHT;
  const x = (w: number) => LEFT + (Math.max(0, Math.min(w, xMax)) / xMax) * plotW;
  const plotBottom = TOP + bands.length * ROW_H;

  // x-axis gridlines + percentage labels.
  const TICKS = 5;
  for (let t = 0; t <= TICKS; t++) {
    const value = (xMax * t) / TICKS;
    const gx = x(value);
    svg.appendChild(svgEl('line', { class: 'grid', x1: gx, x2: gx, y1: TOP, y2: plotBottom }));
    svg.appendChild(
      svgEl(
        'text',
        { x: gx, y: plotBottom + 20, 'text-anchor': 'middle' },
        `${Math.round(value * 100)}%`,
      ),
    );
  }

  bands.forEach((band, i) => {
    const cy = TOP + i * ROW_H + ROW_H / 2;
    const q05 = q(band, 'q05');
    const q25 = q(band, 'q25');
    const q50 = q(band, 'q50');
    const q75 = q(band, 'q75');
    const q95 = q(band, 'q95');

    // Row label: plain asset name.
    svg.appendChild(
      svgEl(
        'text',
        { x: LEFT - 12, y: cy + 4, 'text-anchor': 'end', class: 'chart__rowlabel' },
        ASSET_LABELS[band.asset] ?? band.asset,
      ),
    );

    // Neutral marker: a hollow vertical tick, always drawn.
    const nx = x(band.neutral);
    svg.appendChild(
      svgEl('line', {
        class: 'box-neutral',
        x1: nx,
        x2: nx,
        y1: cy - 16,
        y2: cy + 16,
      }),
    );

    if (band.degraded) {
      // Point mass at neutral — no fabricated spread on a missing input.
      svg.appendChild(svgEl('circle', { class: 'box-degraded', cx: nx, cy, r: 5 }));
      svg.appendChild(
        svgEl(
          'text',
          { x: nx + 12, y: cy + 4, class: 'chart__note' },
          'held at neutral (degraded)',
        ),
      );
      return;
    }

    // Whisker q05–q95.
    svg.appendChild(svgEl('line', { class: 'box-whisker', x1: x(q05), x2: x(q95), y1: cy, y2: cy }));
    for (const edge of [q05, q95]) {
      svg.appendChild(
        svgEl('line', { class: 'box-whisker', x1: x(edge), x2: x(edge), y1: cy - 7, y2: cy + 7 }),
      );
    }

    // Box q25–q75.
    const bx = x(q25);
    const bw = Math.max(x(q75) - bx, 1.5);
    svg.appendChild(
      svgEl('rect', { class: 'box-body', x: bx, y: cy - 11, width: bw, height: 22, rx: 2 }),
    );

    // Median q50.
    const mx = x(q50);
    svg.appendChild(svgEl('line', { class: 'box-median', x1: mx, x2: mx, y1: cy - 11, y2: cy + 11 }));

    // The mean weight as a text tag at the right edge of the plot.
    svg.appendChild(
      svgEl(
        'text',
        { x: W - RIGHT + 8, y: cy + 4, class: 'chart__rowlabel' },
        percent(band.mean),
      ),
    );
  });

  return svg;
}

function chartLegend(): HTMLElement {
  const item = (swatchClass: string, label: string) =>
    el(
      'span',
      { class: 'legend__item' },
      el('span', { class: `legend__swatch ${swatchClass}` }),
      label,
    );
  return el(
    'div',
    { class: 'legend' },
    item('legend__swatch--accent', 'box: 25th–75th percentile'),
    item('legend__swatch--idle', 'whiskers: 5th–95th percentile'),
    item('legend__swatch--info', 'median weight'),
    item('legend__swatch--ghost', 'neutral (strategic) weight'),
  );
}

// --------------------------------------------------------------- rendering

function renderWeights(state: PortfolioState, stale: boolean): void {
  if (!weightsHost) return;

  const staleBanner = stale
    ? stateBlock({
        tone: 'warn',
        title: 'This allocation is stale',
        detail: `Last published ${state.as_of}. The daily run has not reported a newer one — the weights below are the last ones published, not nothing.`,
      })
    : null;

  const header = el(
    'div',
    { class: 'badgerow' },
    badge(PROFILE_LABELS[state.profile]?.title ?? state.profile, 'info'),
    state.degraded ? badge('degraded', 'warn') : badge('all engines reporting', 'ok'),
    el(
      'span',
      { class: 'enginecard__detail' },
      `as of ${state.as_of} · risk-free ${percent(state.risk_free, 2)} · model ${state.model_version}`,
    ),
  );

  // A per-asset table beside the chart: the number for anyone who wants it,
  // never instead of the distribution.
  const rows = state.assets.map((band) => {
    const tone: Tone = band.degraded ? 'warn' : band.delta > 0.005 ? 'ok' : band.delta < -0.005 ? 'bad' : 'idle';
    return el(
      'tr',
      {},
      el('td', {}, ASSET_LABELS[band.asset] ?? band.asset),
      el('td', { class: 'num mono' }, percent(band.mean)),
      el('td', { class: 'num mono' }, percent(band.neutral)),
      el('td', { class: 'num mono' }, band.degraded ? '—' : signedPercent(band.delta)),
      el(
        'td',
        { class: 'num mono' },
        band.degraded ? '—' : `${percent(q(band, 'q05'))} … ${percent(q(band, 'q95'))}`,
      ),
      el('td', {}, band.degraded ? badge('neutral fallback', 'warn') : badge('tilted', tone)),
    ) as HTMLTableRowElement;
  });

  replace(
    weightsHost,
    staleBanner,
    header,
    weightsChart(state.assets),
    table(['Asset', 'Median', 'Neutral', 'Tilt', '5–95% band', 'Status'], rows),
  );

  if (legendHost) replace(legendHost, chartLegend());
}

function renderImplication(state: PortfolioState): void {
  if (!implicationHost) return;
  replace(
    implicationHost,
    el(
      'div',
      { class: `statemsg statemsg--${state.degraded ? 'warn' : 'info'}` },
      el('p', { class: 'statemsg__title' }, 'Conditional implication'),
      el('p', { class: 'statemsg__detail' }, state.implication),
    ),
    state.degraded_reason.length
      ? el(
          'ul',
          { class: 'prose', style: 'margin-top:0.6rem' },
          ...state.degraded_reason.map((reason) => el('li', {}, reason)),
        )
      : null,
  );
}

function renderWhy(state: PortfolioState): void {
  if (!whyHost) return;
  const inputs: PortfolioInput[] = state.inputs ?? [];
  if (inputs.length === 0) {
    replace(whyHost, stateBlock({ tone: 'idle', title: 'No input states recorded for this run.' }));
    return;
  }

  const rows = inputs.map((input) =>
    el(
      'tr',
      {},
      el('td', {}, ASSET_LABELS[input.asset] ?? input.asset),
      el('td', {}, badge(input.regime, input.stale ? 'warn' : regimeTone(input.regime))),
      el('td', { class: 'num mono' }, percent(input.expected_return, 2)),
      el('td', { class: 'num mono' }, formatValue(input.risk_score)),
      el('td', { class: 'num mono' }, formatValue(input.confidence)),
      el(
        'td',
        {},
        input.stale
          ? badge(input.reason || 'stale', 'warn')
          : badge('fresh', 'ok'),
      ),
    ) as HTMLTableRowElement,
  );

  // Risk-score tone lamp beside the table title, so the panel reads at a glance.
  const worst = inputs.reduce((max, i) => Math.max(max, i.stale ? 0 : i.risk_score), 0);
  replace(
    whyHost,
    el(
      'div',
      { class: 'badgerow' },
      el('span', { class: 'enginecard__detail' }, 'Each weight traces to one engine state:'),
      badge(`peak risk ${formatValue(worst)}`, riskTone(worst)),
    ),
    table(
      ['Asset', 'Regime', 'Expected return', 'Risk', 'Confidence', 'Freshness'],
      rows,
    ),
    el(
      'details',
      { class: 'explain' },
      el('summary', {}, 'How a weight is built from these states'),
      el(
        'p',
        { class: 'prose' },
        state.method ||
          'Each asset is scored by its confidence-weighted, risk-adjusted excess return over cash, ' +
            'then the neutral mix is tilted toward the higher scores. The distribution comes from ' +
            'resampling each engine’s expected return at a spread set by its confidence.',
      ),
    ),
  );
}

function renderFailure(result: Exclude<ApiResult<PortfolioState>, { ok: true }>): void {
  const message = failureBlock(result, `Portfolio (${active})`);
  for (const host of [weightsHost, implicationHost, whyHost]) {
    if (host) replace(host, message.cloneNode(true) as HTMLElement);
  }
  if (legendHost) replace(legendHost);
}

// ------------------------------------------------------------------- main

async function load(): Promise<void> {
  for (const host of [weightsHost, implicationHost, whyHost]) {
    if (host) replace(host, loadingBlock(`the ${active} portfolio`));
  }
  if (legendHost) replace(legendHost);

  const result = await getPortfolio(active);
  if (!result.ok) {
    renderFailure(result);
    return;
  }

  const state = result.envelope.data;
  // A stale allocation is still drawn — the last published weights are not
  // nothing — but the banner says so.
  renderWeights(state, result.envelope.stale);
  renderImplication(state);
  renderWhy(state);
}

renderSwitcher();
void load();
