/**
 * The Engines panel — one card per registered engine.
 *
 * Driven entirely by `GET /api/v1/assets`, which enumerates the closed engine
 * vocabulary rather than whatever happens to be in the database. That is the
 * whole design: this file is written once, and every later engine appears on
 * the home page the first time it writes back, with no template change. P4's
 * acceptance is literally "the gold card appears without editing this".
 *
 * Four states, all rendered explicitly, none of them blank:
 *   live         · a state was published recently
 *   stale        · published, but not lately — the engine has stopped reporting
 *   awaiting     · shipped and registered, never run
 *   unreachable  · the API did not answer
 */
import { ENGINE_LABELS, getAssets, type ApiResult, type AssetList, type AssetSummary } from '../lib/api';
import { badge, el, failureBlock, loadingBlock, replace } from '../lib/dom';
import { formatDays, formatValue, type Tone } from '../lib/format';

/**
 * Regime names carry a reading; the colour must not invent one of its own.
 *
 * One table across engines, keyed by the regime name itself. The vocabularies are
 * disjoint by construction (each engine owns its own, per 03-contracts.md §1), so
 * there is no collision to disambiguate — and an unknown name falls through to
 * `info` rather than being coloured by guesswork.
 */
const REGIME_TONE: Record<string, Tone> = {
  // rates
  inverted: 'bad',
  re_steepening: 'warn',
  flat: 'idle',
  steep_tightening: 'warn',
  steep_easing: 'ok',
  // money — the condition of the funding market, not the level of rates
  abundant: 'ok',
  normal: 'idle',
  tightening: 'warn',
  stressed: 'bad',
};

export function regimeTone(regime: string | null): Tone {
  if (!regime) return 'idle';
  return REGIME_TONE[regime] ?? 'info';
}

/** Risk is 0-100 with 100 the most exposed, so the tone runs the other way. */
export function riskTone(score: number | null): Tone {
  if (score === null || !Number.isFinite(score)) return 'idle';
  if (score >= 70) return 'bad';
  if (score >= 40) return 'warn';
  return 'ok';
}

function card(summary: AssetSummary): HTMLElement {
  const label = ENGINE_LABELS[summary.asset];
  const title = label?.title ?? summary.asset;

  const heading = label?.href
    ? el('a', { class: 'enginecard__title', href: label.href }, title)
    : el('span', { class: 'enginecard__title' }, title);

  if (summary.status === 'awaiting_first_run') {
    return el(
      'article',
      { class: 'enginecard enginecard--idle' },
      el('div', { class: 'enginecard__head' }, heading, badge('awaiting first run', 'idle')),
      el('p', { class: 'enginecard__blurb' }, label?.blurb ?? 'Registered engine.'),
      el(
        'p',
        { class: 'enginecard__detail' },
        'Registered, but it has not published a state yet. Its card fills in after the first daily run.',
      ),
    );
  }

  const tone = summary.stale ? 'warn' : regimeTone(summary.regime);

  return el(
    'article',
    { class: `enginecard enginecard--${tone}` },
    el(
      'div',
      { class: 'enginecard__head' },
      heading,
      badge(summary.regime ?? 'unknown', regimeTone(summary.regime)),
    ),
    el('p', { class: 'enginecard__blurb' }, label?.blurb ?? ''),
    el(
      'dl',
      { class: 'enginecard__stats' },
      el('div', {}, el('dt', {}, 'risk'), el('dd', { class: 'mono' }, formatValue(summary.risk_score))),
      el(
        'div',
        {},
        el('dt', {}, 'confidence'),
        el('dd', { class: 'mono' }, formatValue(summary.confidence)),
      ),
      el('div', {}, el('dt', {}, 'as of'), el('dd', { class: 'mono nowrap' }, summary.as_of ?? '—')),
    ),
    el(
      'p',
      { class: 'enginecard__detail' },
      summary.stale
        ? `Last published ${formatDays(summary.freshness_days)} ago — this engine has stopped reporting.`
        : `Model ${summary.model_version ?? 'unknown'}`,
    ),
    summary.stale ? badge('stale', 'warn') : null,
  );
}

export function renderEngines(host: Element, result: ApiResult<AssetList>): void {
  if (!result.ok) {
    replace(host, failureBlock(result, 'Engines'));
    return;
  }

  const assets = result.envelope.data.assets ?? [];
  if (assets.length === 0) {
    replace(
      host,
      failureBlock(
        { kind: 'error', message: 'The API returned no engines at all.', status: 200 },
        'Engines',
      ),
    );
    return;
  }

  // Live engines first, then the ones still waiting: the panel should lead with
  // what it can actually tell you.
  const ordered = [...assets].sort((a, b) => {
    if (a.status !== b.status) return a.status === 'live' ? -1 : 1;
    return 0;
  });

  replace(host, el('div', { class: 'enginegrid' }, ...ordered.map(card)));
}

export async function mountEngines(selector: string): Promise<void> {
  const host = document.querySelector(selector);
  if (!host) return;
  replace(host, loadingBlock('engines'));
  renderEngines(host, await getAssets());
}
