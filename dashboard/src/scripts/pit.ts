/**
 * Point-in-time demonstration.
 *
 * The page exists to show the half of the data that a naive backtest would
 * have used and should not have: figures that describe a period on or before
 * the chosen date, but which nobody could read until weeks later.
 */
import { getPit, type PitSnapshot } from '../lib/api';
import {
  badge,
  el,
  emptyBlock,
  failureBlock,
  loadingBlock,
  replace,
  stateBlock,
  table,
  text,
} from '../lib/dom';
import { formatCount, formatDays, formatValue } from '../lib/format';

const DEFAULT_AS_OF = '2025-03-01';
const DATE_RE = /^\d{4}-\d{2}-\d{2}$/;

const input = document.querySelector<HTMLInputElement>('#pit-date');
const summaryHost = document.querySelector('#pit-summary');
const knownHost = document.querySelector('#pit-known');
const withheldHost = document.querySelector('#pit-withheld');
const knownCount = document.querySelector('#pit-known-count');
const withheldCount = document.querySelector('#pit-withheld-count');

/** Monotonic token so a slow response cannot overwrite a newer one. */
let requestSeq = 0;

function renderAvailable(snapshot: PitSnapshot): void {
  if (!knownHost) return;
  if (knownCount) replace(knownCount, formatCount(snapshot.available.length));

  if (snapshot.available.length === 0) {
    replace(
      knownHost,
      emptyBlock(
        'series knowable on this date',
        `Nothing in the store had been released on or before ${snapshot.as_of}. Try a later date.`,
      ),
    );
    return;
  }

  const rows = snapshot.available.map((a) =>
    el(
      'tr',
      {},
      el(
        'td',
        {},
        el('div', {}, text(a.title, a.series_id)),
        el('div', { class: 'mono', style: 'color:var(--fg-faint);font-size:0.75rem' }, a.series_id),
      ),
      el('td', { class: 'mono nowrap' }, a.obs_date),
      el('td', { class: 'mono nowrap' }, a.release_date),
      el('td', { class: 'num' }, formatValue(a.value)),
      el('td', { class: 'num nowrap' }, formatDays(a.staleness_days)),
    ),
  );

  replace(
    knownHost,
    table(['Series', 'Period', 'Released', 'Value', 'Period age'], rows),
  );
}

function renderWithheld(snapshot: PitSnapshot): void {
  if (!withheldHost) return;
  if (withheldCount) replace(withheldCount, formatCount(snapshot.withheld.length));

  if (snapshot.withheld.length === 0) {
    replace(
      withheldHost,
      emptyBlock(
        'withheld series',
        `Every series in the store had already been published by ${snapshot.as_of}, so nothing was held back at this date.`,
      ),
    );
    return;
  }

  const rows = snapshot.withheld.map((w) =>
    el(
      'tr',
      {},
      el(
        'td',
        {},
        el('div', {}, text(w.title, w.series_id)),
        el('div', { class: 'mono', style: 'color:var(--fg-faint);font-size:0.75rem' }, w.series_id),
      ),
      el('td', { class: 'mono nowrap' }, w.obs_date),
      el('td', { class: 'mono nowrap' }, w.release_date),
      el(
        'td',
        { class: 'num nowrap' },
        badge(`+${formatDays(w.published_days_later)}`, 'bad'),
      ),
      el('td', {}, w.reason),
    ),
  );

  replace(
    withheldHost,
    el(
      'p',
      { class: 'prose', style: 'margin-bottom:0.8rem' },
      'Each row below exists in the database today and describes a period at or before the ',
      'chosen date — and was still unpublished then. A join on period alone would have handed ',
      'every one of these to the model early.',
    ),
    table(['Series', 'Period', 'Actually released', 'Days late', 'Why withheld'], rows),
  );
}

function renderSummary(snapshot: PitSnapshot): void {
  if (!summaryHost) return;
  const covered = snapshot.total_series;
  replace(
    summaryHost,
    el(
      'div',
      { class: 'tiles' },
      el(
        'div',
        { class: 'tile tile--info' },
        el('div', { class: 'tile__label' }, 'Information cutoff'),
        el('div', { class: 'tile__value' }, snapshot.as_of),
        el('div', { class: 'tile__detail' }, 'Only release_date ≤ this date is visible.'),
      ),
      el(
        'div',
        { class: 'tile tile--ok' },
        el('div', { class: 'tile__label' }, 'Knowable then'),
        el('div', { class: 'tile__value' }, formatCount(snapshot.available.length)),
        el('div', { class: 'tile__detail' }, 'Series with at least one released observation.'),
      ),
      el(
        'div',
        { class: 'tile tile--bad' },
        el('div', { class: 'tile__label' }, 'Withheld'),
        el('div', { class: 'tile__value' }, formatCount(snapshot.withheld.length)),
        el('div', { class: 'tile__detail' }, 'Existed in the data, not yet published.'),
      ),
      el(
        'div',
        { class: 'tile tile--idle' },
        el('div', { class: 'tile__label' }, 'Series in catalogue'),
        el('div', { class: 'tile__value' }, formatCount(covered)),
        el('div', { class: 'tile__detail' }, 'Total series registered in the feature store.'),
      ),
    ),
  );
}

async function load(asOf: string): Promise<void> {
  const seq = ++requestSeq;

  if (!DATE_RE.test(asOf)) {
    const msg = stateBlock({
      tone: 'warn',
      title: 'Enter a date as YYYY-MM-DD',
      detail: 'The point-in-time snapshot needs a complete calendar date.',
    });
    if (summaryHost) replace(summaryHost, msg);
    if (knownHost) replace(knownHost, '');
    if (withheldHost) replace(withheldHost, '');
    return;
  }

  if (summaryHost) replace(summaryHost, loadingBlock(`the snapshot for ${asOf}`));
  if (knownHost) replace(knownHost, loadingBlock('released series'));
  if (withheldHost) replace(withheldHost, loadingBlock('withheld series'));

  const result = await getPit(asOf);
  if (seq !== requestSeq) return;

  if (!result.ok) {
    const block = failureBlock(result, `Point-in-time snapshot for ${asOf}`);
    if (summaryHost) replace(summaryHost, block);
    if (knownHost) {
      replace(knownHost, emptyBlock('data', 'Nothing to show while the snapshot is unavailable.'));
    }
    if (withheldHost) {
      replace(
        withheldHost,
        emptyBlock('data', 'Nothing to show while the snapshot is unavailable.'),
      );
    }
    if (knownCount) replace(knownCount, '—');
    if (withheldCount) replace(withheldCount, '—');
    return;
  }

  const snapshot = result.envelope.data;
  renderSummary(snapshot);
  renderAvailable(snapshot);
  renderWithheld(snapshot);
}

function main(): void {
  const initial = input?.value && DATE_RE.test(input.value) ? input.value : DEFAULT_AS_OF;
  if (input) input.value = initial;
  input?.addEventListener('change', () => void load(input.value));
  void load(initial);
}

main();
