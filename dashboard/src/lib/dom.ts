/**
 * Minimal DOM helpers. No framework: the pages are static HTML with small
 * client scripts, and every value that comes back from the API is inserted as
 * a text node rather than as markup.
 */

type Child = Node | string | number | null | undefined | false;
type Attrs = Record<string, string | number | boolean | null | undefined>;

function appendChildren(node: Element, children: Child[]): void {
  for (const child of children) {
    if (child === null || child === undefined || child === false) continue;
    node.appendChild(
      typeof child === 'string' || typeof child === 'number'
        ? document.createTextNode(String(child))
        : child,
    );
  }
}

export function el<K extends keyof HTMLElementTagNameMap>(
  tag: K,
  attrs: Attrs = {},
  ...children: Child[]
): HTMLElementTagNameMap[K] {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (value === null || value === undefined || value === false) continue;
    node.setAttribute(key, value === true ? '' : String(value));
  }
  appendChildren(node, children);
  return node;
}

const SVG_NS = 'http://www.w3.org/2000/svg';

export function svgEl<K extends keyof SVGElementTagNameMap>(
  tag: K,
  attrs: Attrs = {},
  ...children: Child[]
): SVGElementTagNameMap[K] {
  const node = document.createElementNS(SVG_NS, tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (value === null || value === undefined || value === false) continue;
    node.setAttribute(key, value === true ? '' : String(value));
  }
  appendChildren(node, children);
  return node;
}

/** Replace a container's contents with the given nodes. */
export function replace(container: Element, ...children: Child[]): void {
  container.replaceChildren();
  appendChildren(container, children);
}

export function mustFind<T extends Element = HTMLElement>(selector: string): T | null {
  return document.querySelector<T>(selector);
}

export function text(value: string | number | null | undefined, fallback = '—'): string {
  if (value === null || value === undefined || value === '') return fallback;
  return String(value);
}

/**
 * A row of `label: value` definition pairs — used for tiles and metadata blocks.
 */
export function defRow(label: string, value: Child): HTMLElement {
  return el('div', { class: 'defrow' }, el('dt', {}, label), el('dd', {}, value));
}

export function badge(label: string, tone: 'ok' | 'warn' | 'bad' | 'idle' | 'info'): HTMLElement {
  return el('span', { class: `badge badge--${tone}` }, label);
}

/**
 * The one place a failed or empty read is turned into something on screen.
 * Every panel routes through this, so no page can silently render blank.
 */
export function stateBlock(opts: {
  tone: 'idle' | 'info' | 'warn' | 'bad';
  title: string;
  detail?: string;
  hint?: string;
}): HTMLElement {
  return el(
    'div',
    { class: `statemsg statemsg--${opts.tone}`, role: opts.tone === 'bad' ? 'alert' : 'status' },
    el('p', { class: 'statemsg__title' }, opts.title),
    opts.detail ? el('p', { class: 'statemsg__detail' }, opts.detail) : null,
    opts.hint ? el('p', { class: 'statemsg__hint' }, opts.hint) : null,
  );
}

export function loadingBlock(what: string): HTMLElement {
  return stateBlock({ tone: 'idle', title: `Loading ${what}…` });
}

/**
 * Render an API failure. `not_implemented` is presented as roadmap
 * information, not as a fault — the endpoint is reserved, not broken.
 */
export function failureBlock(
  failure: { kind: string; message: string; milestone?: string; status?: number },
  what: string,
): HTMLElement {
  if (failure.kind === 'not_implemented') {
    return stateBlock({
      tone: 'info',
      title: `${what}: planned, milestone ${failure.milestone ?? 'unknown'}`,
      detail: failure.message,
    });
  }
  if (failure.kind === 'unreachable') {
    return stateBlock({
      tone: 'warn',
      title: `${what}: API unreachable`,
      detail: failure.message,
      hint: 'The dashboard is static and reads the serving plane from your browser. Check PUBLIC_FINDYN_API and that the Worker is deployed and reachable.',
    });
  }
  if (failure.kind === 'not_found') {
    return stateBlock({ tone: 'warn', title: `${what}: not found`, detail: failure.message });
  }
  return stateBlock({
    tone: 'bad',
    title: `${what}: request failed${failure.status ? ` (HTTP ${failure.status})` : ''}`,
    detail: failure.message,
  });
}

export function emptyBlock(what: string, detail: string): HTMLElement {
  return stateBlock({ tone: 'idle', title: `No ${what} yet`, detail });
}

/** Build a table from a header row and pre-built body rows. */
export function table(headers: string[], rows: HTMLTableRowElement[]): HTMLElement {
  return el(
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
          ...headers.map((h) => el('th', { scope: 'col' }, h)),
        ),
      ),
      el('tbody', {}, ...rows),
    ),
  );
}
