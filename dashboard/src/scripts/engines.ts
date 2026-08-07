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
 * One table across engines, keyed by the regime name itself. Each engine owns its
 * own vocabulary (03-contracts.md §1) and an unknown name falls through to `info`
 * rather than being coloured by guesswork.
 *
 * The vocabularies were disjoint until P5, which introduced `normal` for crypto
 * alongside money's. They collide on a name and agree on a tone — both mean "the
 * ordinary state" and both are `idle` — so one entry serves both. If a future
 * engine reuses a name and *disagrees* about what it means, this table has to
 * become keyed by (asset, regime); it is not, because it does not yet need to be.
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
  // gold — why it is being bid, or why it is not. `hedge_bid` is `idle` rather
  // than `ok` on purpose: it is the ordinary state, and colouring the common
  // case green would make the panel read as an all-clear most of the time.
  hedge_bid: 'idle',
  carry_headwind: 'warn',
  crisis_bid: 'bad',
  // crypto — how much of the price is momentum. `normal` is `idle` for the same
  // reason `hedge_bid` is, and `frenzy` is `bad` rather than `ok` despite being
  // the state in which the price has gone up: this axis is risk, not return.
  winter: 'warn',
  frenzy: 'bad',
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
  // Absent on a Worker older than P5. Treated as false, which is right for every
  // engine that is not crypto and is the state the site had before P5 anyway.
  const experimental = summary.experimental === true;

  const heading = label?.href
    ? el('a', { class: 'enginecard__title', href: label.href }, title)
    : el('span', { class: 'enginecard__title' }, title);

  // The tag rides in the head beside the regime badge, so it is present in both
  // card states — an experimental engine that has never run is still one.
  const tag = experimental ? badge('experimental', 'warn') : null;
  const shell = experimental ? 'enginecard enginecard--experimental' : 'enginecard';

  if (summary.status === 'awaiting_first_run') {
    return el(
      'article',
      { class: `${shell} enginecard--idle` },
      el('div', { class: 'enginecard__head' }, heading, tag, badge('awaiting first run', 'idle')),
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
    { class: `${shell} enginecard--${tone}` },
    el(
      'div',
      { class: 'enginecard__head' },
      heading,
      tag,
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
    // Said on the card, not only in the blurb. Somebody scanning the panel for a
    // number to act on should not have to open the page to find out this one is
    // research output.
    experimental
      ? el(
          'p',
          { class: 'enginecard__detail' },
          'Research only — no expected return, and excluded from the portfolio layer.',
        )
      : null,
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

  // Experimental engines last, whatever their status; then live before waiting.
  // The panel should lead with what it can actually tell you, and an engine that
  // publishes no expected return and is excluded from allocations is not that —
  // 04-ui-plan.md §P5 asks for it to render last and de-emphasized, and this is
  // the "last" half. Sorting rather than special-casing `crypto` keeps the panel
  // the build-once component P1 designed: a second experimental engine needs no
  // change here.
  const ordered = [...assets].sort((a, b) => {
    const experimental = Number(a.experimental === true) - Number(b.experimental === true);
    if (experimental !== 0) return experimental;
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
