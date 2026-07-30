/**
 * The numeraire ribbon — short rate + liquidity chip, on every page.
 *
 * `SOFR 4.31% · liquidity: normal`
 *
 * Deliberately the only place in the dashboard that renders *nothing* on
 * failure. Everywhere else an absent read becomes an explicit message, because a
 * panel the user asked for owes them an explanation. The ribbon is different: it
 * is ambient furniture on pages about something else, and a permanent "money
 * engine unreachable" bar across the top of /rates would be noise on every page
 * that does not depend on it. 04-ui-plan.md §P2 says "hidden if the engine is
 * stale/absent" — this is that rule.
 *
 * The /money page states the same failure loudly, which is where someone who
 * cares about the numeraire will be looking.
 */
import { getAssetState, type AssetState } from '../lib/api';
import { el, replace } from '../lib/dom';
import { formatValue } from '../lib/format';

const host = document.querySelector('#numeraire');

/** Tone per liquidity state — the colour must not invent a reading of its own. */
const LIQUIDITY_TONE: Record<string, string> = {
  abundant: 'ok',
  normal: 'idle',
  tightening: 'warn',
  stressed: 'bad',
};

/**
 * Which series the short rate came from, for the label. The engine records this
 * in `components.primary_rate_share`: 1.0 means the whole path is SOFR, less
 * means the pre-2018 bill fallback is carrying part of it. The *latest* rate is
 * SOFR whenever the share is above zero and the state is recent, so the label
 * says SOFR only when it can.
 */
function rateLabel(state: AssetState): string {
  const share = state.components?.primary_rate_share;
  return share !== undefined && share > 0 ? 'SOFR' : 'short rate';
}

function render(state: AssetState): void {
  if (!host) return;

  const rate = state.expected_return;
  const pct = rate === null || !Number.isFinite(rate) ? null : rate * 100;
  const spread = state.components?.bill_sofr_spread;

  replace(
    host,
    el(
      'div',
      { class: 'ribbon__inner' },
      el(
        'a',
        { class: 'ribbon__rate', href: '/money' },
        el('span', { class: 'ribbon__label' }, rateLabel(state)),
        el('span', { class: 'ribbon__value' }, pct === null ? '—' : `${pct.toFixed(2)}%`),
      ),
      el('span', { class: 'ribbon__sep' }, '·'),
      el(
        'a',
        { class: 'ribbon__liquidity', href: '/money' },
        el('span', { class: 'ribbon__label' }, 'liquidity'),
        el(
          'span',
          { class: `badge badge--${LIQUIDITY_TONE[state.regime] ?? 'info'}` },
          state.regime,
        ),
      ),
      spread === undefined
        ? null
        : el(
            'span',
            { class: 'ribbon__detail mono', title: '3m bill minus overnight, percentage points' },
            `bill−SOFR ${formatValue(spread)}pp`,
          ),
      el('span', { class: 'ribbon__asof mono' }, state.as_of),
    ),
  );
  host.removeAttribute('hidden');
}

async function main(): Promise<void> {
  if (!host) return;

  const result = await getAssetState('money');
  // Not run yet (501), unreachable, or stale: stay out of the way entirely.
  if (!result.ok || result.envelope.stale) return;
  render(result.envelope.data);
}

void main();
