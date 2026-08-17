/**
 * The home page's portfolio summary strip (04-ui-plan.md §P6 final form).
 *
 * A compact read of the balanced profile: a stacked weight bar, the median
 * weight per asset, a degraded badge when a run fell back, and a link into the
 * full /portfolio page with its distributions. It reads the same endpoint the
 * page does and renders every failure state — a profile that has never run says
 * "awaiting first run" rather than showing a blank bar.
 */
import { getPortfolio, type ApiResult, type PortfolioState } from '../lib/api';
import { badge, el, replace, stateBlock } from '../lib/dom';

const ASSET_LABELS: Record<string, string> = {
  equity: 'equities',
  rates: 'gov bonds',
  gold: 'gold',
  money: 'cash',
};

/** A fixed colour per asset, so the strip and its labels agree. */
const ASSET_VAR: Record<string, string> = {
  equity: 'var(--accent)',
  rates: 'var(--info)',
  gold: 'var(--warn)',
  money: 'var(--ok)',
};

function percent(value: number | null | undefined, digits = 1): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return '—';
  return `${(value * 100).toFixed(digits)}%`;
}

function renderStrip(host: Element, result: ApiResult<PortfolioState>): void {
  if (!result.ok) {
    if (result.kind === 'not_implemented') {
      replace(
        host,
        stateBlock({
          tone: 'idle',
          title: 'Portfolio: awaiting first run',
          detail:
            'The balanced allocation appears here after the first portfolio run. It leans a 60/30/10 neutral mix by the engines’ confidence-weighted expected returns.',
        }),
      );
      return;
    }
    replace(
      host,
      stateBlock({
        tone: result.kind === 'unreachable' ? 'warn' : 'bad',
        title: 'Portfolio unavailable',
        detail: result.message,
      }),
    );
    return;
  }

  const state = result.envelope.data;
  const assets = state.assets.filter((a) => a.mean > 0.0005);

  // The stacked bar: one segment per asset, width = median weight.
  const bar = el(
    'div',
    { class: 'pstrip__bar', role: 'img', 'aria-label': 'Median weights, balanced profile' },
    ...assets.map((a) =>
      el('span', {
        class: 'pstrip__seg',
        style: `width:${(a.mean * 100).toFixed(2)}%;background:${ASSET_VAR[a.asset] ?? 'var(--border-strong)'}`,
        title: `${ASSET_LABELS[a.asset] ?? a.asset}: ${percent(a.mean)}`,
      }),
    ),
  );

  const chips = el(
    'div',
    { class: 'pstrip__chips' },
    ...assets.map((a) =>
      el(
        'span',
        { class: 'pstrip__chip' },
        el('span', {
          class: 'pstrip__dot',
          style: `background:${ASSET_VAR[a.asset] ?? 'var(--border-strong)'}`,
        }),
        `${ASSET_LABELS[a.asset] ?? a.asset} ${percent(a.mean)}`,
      ),
    ),
  );

  replace(
    host,
    el(
      'div',
      { class: 'pstrip' },
      el(
        'div',
        { class: 'pstrip__head' },
        badge('balanced', 'info'),
        state.degraded ? badge('degraded', 'warn') : badge('all engines reporting', 'ok'),
        result.envelope.stale ? badge('stale', 'warn') : null,
        el('span', { class: 'enginecard__detail' }, `as of ${state.as_of}`),
        el('a', { class: 'pstrip__link', href: '/portfolio' }, 'All profiles & distributions →'),
      ),
      bar,
      chips,
      el('p', { class: 'pstrip__impl' }, state.implication),
    ),
  );
}

export async function mountPortfolioStrip(selector: string): Promise<void> {
  const host = document.querySelector(selector);
  if (!host) return;
  replace(host, stateBlock({ tone: 'idle', title: 'Loading portfolio…' }));
  renderStrip(host, await getPortfolio('balanced'));
}
