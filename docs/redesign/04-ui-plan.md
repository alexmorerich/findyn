# UI Evolution Plan — visible features per phase

This plan makes **every phase end with a visible upgrade to the deployed
site**, not just green tests.

As of P3-C the home page is a market overview — regime, velocity, instability,
the three crash factors and the forecast bands — and the system-status
instrument panel it replaced lives at `/status`, intact. Every endpoint
`FINDYN_V1_SPEC.md` §13 names now answers; the `501`-with-a-milestone
convention below stays for the engines still to come, and an unpublished engine
(`/assets/gold/state`) still uses it.

## Global rules (all phases)

- **Deploy is part of Done.** Each phase's acceptance now includes:
  `cd dashboard && npm run build`, `cd serving && npm run deploy`, then verify
  the new UI live at the workers.dev URL. A phase whose UI is only visible
  locally is not complete.
- Keep the existing visual language: the dark instrument-panel style,
  `lib/api.ts` for all data access, client-side rendering from the live API
  (nothing baked into pages), staleness badges everywhere, the disclaimer
  footer.
- Charts: lightweight, no heavy chart framework — SVG rendered by small
  TypeScript modules under `dashboard/src/scripts/` (follow the existing
  pattern). Every chart must handle the empty/501/stale states with an
  explicit message, never a blank box.
- Every score shown must be explainable: clicking (or expanding) a score
  reveals its `components` breakdown from the API.
- Navigation: the header nav gains one entry per shipped engine page; entries
  for unshipped engines are not shown (no dead links).

## P1 — FinRates ships, and the home page gets its Engines panel

- **Home page**: new "Engines" panel driven by `GET /api/v1/assets` — one card
  per registered engine showing regime badge, risk score, `as_of`, staleness.
  Engines that exist but have no data render as "awaiting first run"; engines
  not yet registered simply don't appear. This panel is built once and never
  needs template changes when later engines ship — cards appear as the
  registry grows.
- **`/rates` page**:
  - Yield-curve snapshot chart: today's fitted NS curve + raw tenor points,
    with ghost curves for 1 month ago and 1 year ago.
  - NS factor history: level / slope / curvature as three stacked sparkline
    rows (from `/assets/rates/history?metric=ns_*`).
  - Regime timeline strip: colored band of the rate regime over time.
  - Signals table: `curve_inversion`, `term_premium_trend` with direction
    arrows.

## P2 — FinMoney: the numeraire ribbon

- **Header ribbon** (all pages): current annualized short rate + liquidity
  state chip (e.g. `SOFR 4.31% · liquidity: normal`), from
  `/assets/money/state`. Hidden if the engine is stale/absent.
- **`/money` page**: wealth-index chart (1$ compounded), trailing carry table
  (1m/3m/12m), discount-factor curve, liquidity state badge with its
  component spread values.

## P3 — FinEquity: the flagship page (three sub-milestones, ship UI with each)

- **A (features land)**: `/equity` page v1 — filtered price vs raw price
  chart, velocity/acceleration panel, jerk indicator lamp (off/amber/red by
  threshold).
- **B (regimes land)**: regime posterior stacked-area chart over time; current
  regime hero badge; calibrated transition-probability dials for 3m/6m/12m
  with their SHAP top-contributors list.
- **C (RII/forecast land)**: RII gauge (0–100) with 90-day sparkline; crash
  decomposition shown as **three separate bars** (P(transition), P(shock),
  P(transmission)) — never a single composite alone; forecast fan chart
  (quantile bands per horizon, `educational_only` horizons visually
  separated); conditional-implication text block.
- Home page hero switches from "system status" to "market overview" (equity
  regime + RII + rates regime), status panel moves to `/status`.

## P4 — FinGold

- **`/gold` page**: regime badge, hedge-score history chart, drivers panel
  (real rate, USD trend, liquidity stress, jump intensity as labeled tiles).
- Home Engines panel: gold card appears automatically (no template change —
  verify this, it proves the P1 panel design).

## P5 — FinCrypto

- **`/crypto` page**: persistent EXPERIMENTAL banner, speculation index,
  liquidity-beta chart, regime badge. Card on home carries an
  `EXPERIMENTAL` tag and renders last, visually de-emphasized.

## P6 — Portfolio: the payoff page

- **`/portfolio` page**: profile switcher (conservative/balanced/growth),
  weight-distribution fan/box chart per asset (distributions, never single
  numbers), implication text, degraded badges per input engine, and a
  "why" expander listing each input `AssetState`.
- Home page final form: portfolio summary strip on top, Engines panel below,
  header ribbon from P2, status relegated to `/status`.

## Acceptance addendum (applies to every phase prompt)

1. Dashboard builds clean (`npm run build`) and the new page passes a manual
   smoke check via the deployed URL.
2. Kill-switch behavior: with the phase's engine data absent, its page and
   card render an explicit awaiting/stale state — no blank panels, no
   uncaught fetch errors in the console.
3. Deployed live and verified at the workers.dev URL before the phase is
   reported complete.
