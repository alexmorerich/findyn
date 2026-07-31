/**
 * Home page — the market overview (04-ui-plan.md §P3-C).
 *
 * The home page used to be the system-status board, which answered "is the
 * pipeline healthy" to a reader who arrived asking "where is the market". That
 * page still exists at /status, unchanged; this one answers the question people
 * actually come with, and every reading links to the FinEquity panel that shows
 * how it was derived.
 *
 * Deliberately thin. It is a *summary* of numbers that are published in full
 * elsewhere, so it re-fetches rather than sharing state, and it renders nothing
 * it cannot source — an absent regime says so instead of showing a dash that
 * could be mistaken for a calm reading.
 */
import {
  getAssetState,
  getForecast,
  getInstability,
  getTwoLayerState,
  type ApiResult,
  type AssetState,
  type ForecastResponse,
  type InstabilityHistory,
  type TwoLayerState,
} from '../lib/api';
import { badge, el, emptyBlock, failureBlock, loadingBlock, replace, stateBlock } from '../lib/dom';
import { formatCount, formatDate, formatValue, type Tone } from '../lib/format';

const ASSET = 'equity';

/** Mirrors scripts/equity.ts — the same thresholds must read the same way. */
const RII_ELEVATED = 60;
const RII_HIGH = 80;

const REGIME_TONE: Record<string, Tone> = {
  bull_expansion: 'ok',
  normal_expansion: 'ok',
  late_cycle: 'warn',
  bear: 'bad',
  crisis: 'bad',
};

const HORIZON_LABELS: Record<string, string> = {
  tactical: '6 months',
  strategic: '2 years',
  generational: '12 years',
  educational_30y: '30 years',
  educational_50y: '50 years',
};

const hosts = {
  now: document.querySelector('#overview-now'),
  crash: document.querySelector('#overview-crash'),
  forecast: document.querySelector('#overview-forecast'),
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

function percent(value: number | undefined | null, digits = 1): string {
  if (value === undefined || value === null || !Number.isFinite(value)) return '—';
  return `${(value * 100).toFixed(digits)}%`;
}

function renderNow(
  twoLayer: ApiResult<TwoLayerState>,
  instability: ApiResult<InstabilityHistory>,
): void {
  if (!hosts.now) return;
  if (!twoLayer.ok) {
    replace(hosts.now, failureBlock(twoLayer, 'Market state'));
    return;
  }

  const data = twoLayer.envelope.data;
  const tiles: HTMLElement[] = [];

  if (data.regime) {
    const tone = REGIME_TONE[data.regime.label] ?? 'idle';
    tiles.push(
      tile(
        'Regime',
        data.regime.label.replace(/_/g, ' '),
        tone,
        data.regime.confidence === null
          ? 'Filtered posterior — no smoothing, so each date uses only what was known by then.'
          : `${percent(data.regime.confidence, 0)} posterior probability. Filtered, never smoothed.`,
      ),
    );
  } else {
    tiles.push(
      tile(
        'Regime',
        'not published',
        'idle',
        'The engine declines to publish a regime until the monthly refit has fitted the model.',
      ),
    );
  }

  const velocity = data.kinematics.features.velocity;
  const acceleration = data.kinematics.features.acceleration;
  if (velocity !== undefined) {
    // Velocity is the filter's trend state, annualized in logs — close enough to
    // a percentage return that showing it as one is honest, and far more
    // readable than 0.0873.
    const losing = acceleration !== undefined && velocity > 0 && acceleration < 0;
    tiles.push(
      tile(
        'Velocity',
        percent(velocity, 1),
        velocity < 0 ? 'bad' : losing ? 'warn' : 'ok',
        losing
          ? 'Annualized trend, and it is decelerating — a rally losing its slope.'
          : 'Annualized trend of the Kalman-filtered level, not the last two closes.',
      ),
    );
  }

  const latest = instability.ok
    ? instability.envelope.data.points.filter((p) => p.rii !== null).at(-1)
    : undefined;
  if (latest?.rii != null) {
    const value = latest.rii;
    tiles.push(
      tile(
        'Instability (RII)',
        formatValue(value),
        value >= RII_HIGH ? 'bad' : value >= RII_ELEVATED ? 'warn' : 'ok',
        `0–100 against this series' own expanding history, as of ${formatDate(latest.as_of)}.`,
      ),
    );
  }
  if (latest?.p_transmission != null) {
    tiles.push(
      tile(
        'Fragility',
        percent(latest.p_transmission, 0),
        latest.p_transmission >= 0.6 ? 'bad' : latest.p_transmission >= 0.3 ? 'warn' : 'ok',
        'Whether a shock would propagate or be absorbed: credit, liquidity and the curve.',
      ),
    );
  }

  replace(hosts.now, ...tiles);
}

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

  const factors: Array<[string, number | undefined, string]> = [
    [
      'P(transition)',
      signal('p_transition') ?? (transitionKey ? components[transitionKey] : undefined),
      'The regime enters bear or crisis over the horizon, propagated from today’s posterior.',
    ],
    [
      'P(shock)',
      signal('p_shock') ?? components.p_shock,
      'A tail-sized drawdown begins, from a Pareto fit to declustered historical episodes.',
    ],
    [
      'P(transmission)',
      signal('p_transmission') ?? components.p_transmission,
      'The system propagates rather than absorbs it.',
    ],
  ];
  const available = factors.filter(([, v]) => v !== undefined && Number.isFinite(v)) as Array<
    [string, number, string]
  >;

  if (available.length < 3) {
    replace(
      hosts.crash,
      emptyBlock(
        'a crash decomposition',
        'All three factors are published together or not at all — a composite without its parts is exactly what §4 forbids.',
      ),
    );
    return;
  }

  const composite = signal('crash_risk') ?? components.crash_risk;
  replace(
    hosts.crash,
    el(
      'div',
      { class: 'barlist' },
      ...available.map(([label, value, detail]) => {
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
      }),
    ),
    composite === undefined
      ? null
      : el(
          'p',
          { class: 'enginecard__detail' },
          `Composite: ${formatValue(composite)} / 100 — the product of the three, shown beside them and never instead of them. Full derivation on the equity page.`,
        ),
  );
}

function renderForecast(forecast: ApiResult<ForecastResponse>): void {
  if (!hosts.forecast) return;
  if (!forecast.ok) {
    replace(hosts.forecast, failureBlock(forecast, 'Forecast distribution'));
    return;
  }

  // The overview shows the evaluable horizons only. The 30- and 50-year
  // illustrations are on the equity page behind their own heading; on a summary
  // card, stripped of that context, they would read as forecasts.
  const bands = forecast.envelope.data.horizons.filter((b) => !b.educational_only);
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

  const level = (value: number | undefined) =>
    value === undefined ? '—' : formatCount(Math.round(Math.exp(value)));

  replace(
    hosts.forecast,
    el(
      'div',
      { class: 'barlist' },
      ...bands.map((band) =>
        el(
          'div',
          { class: 'bandrow' },
          el(
            'div',
            { class: 'bandrow__head' },
            el('span', { class: 'barrow__label' }, HORIZON_LABELS[band.horizon] ?? band.horizon),
            badge('quantiles', 'info'),
          ),
          el(
            'dl',
            { class: 'deflist deflist--inline' },
            ...(['0.05', '0.25', '0.5', '0.75', '0.95'] as const).map((q) =>
              el(
                'div',
                { class: 'defrow' },
                el('dt', { class: 'mono' }, `p${Math.round(Number(q) * 100)}`),
                el('dd', { class: 'mono' }, level(band.quantiles[q])),
              ),
            ),
          ),
        ),
      ),
    ),
    stateBlock({
      tone: 'info',
      title: 'A band is not a prediction',
      detail:
        'The 5th and 95th percentiles are what the simulated distribution looks like conditioned on today’s state and the fitted tail model. Half the mass sits outside the inner band by construction, and the model can be wrong about the state as well as the path.',
    }),
  );
}

async function main(): Promise<void> {
  for (const target of Object.values(hosts)) {
    if (target) replace(target, loadingBlock('the market overview'));
  }

  const [twoLayer, state, instability, forecast] = await Promise.all([
    getTwoLayerState(),
    getAssetState(ASSET),
    getInstability({ asset: ASSET, points: 120 }),
    getForecast({ asset: ASSET }),
  ]);

  renderNow(twoLayer, instability);
  renderCrash(state);
  renderForecast(forecast);
}

void main();
