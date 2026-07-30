# P4 — FinGold engine

Copy everything below this line to the coder agent. Requires P3 merged.

---

You are working in the `findyn` repo. Read `docs/redesign/01-target-architecture.md`
and `docs/redesign/03-contracts.md`. No-lookahead law applies.

## Task

Build FinGold in `findynamics/engines/gold/`: hard-asset / crisis-protection
engine. Gold has no cash flow — the engine models *drivers*, not valuation:
real interest rate, USD strength, liquidity stress, crisis premium.

## 1. Data

Add under `engines.gold.series`: gold price — LBMA/London PM fix via FRED
(`GOLDPMGBD228NLBM`, daily, lag 1). Factor inputs `real_rate` and
`usd_strength` already exist from P1; `liquidity` factor from v1 set. If a
needed stress series is missing (e.g. `NFCI`), add it to `factors:` in
`series.yaml` — config change, not code.

## 2. Model

- `drivers.py`: driver panel = real 10y rate (level + 12m change), USD index
  trend, liquidity/NFCI stress score, equity RII **read from the
  `instability_index` table via `WorldState.series`, not by importing the
  equity engine**.
- `regime.py`: 2–3 state Markov regime switching (statsmodels
  `MarkovRegression`) on gold monthly returns with driver exogs; vocabulary
  `hedge_bid | carry_headwind | crisis_bid` in `engines/gold/domain.py`.
  Expanding-window fit, frozen between monthly refits.
- `jumps.py`: jump detection on daily returns (threshold on robust z, e.g.
  Lee-Mykland-style); rolling jump intensity feeds the crisis premium.
- `hedge.py`: hedge score 0–100 = rolling conditional correlation of gold vs
  equity drawdown periods + regime posterior weighting; formula documented in
  the module docstring.
- `engine.py`: `GoldEngine(AssetEngine)`, `name="gold"`. `AssetState`:
  `regime`; `expected_return` = regime-conditional historical mean (clearly
  labelled low-confidence in `components`); `risk_score` from realized vol +
  jump intensity; `signals` ⊇ `hedge_score`, `real_rate_headwind`,
  `crisis_premium`. `outputs()` → `hedge_score`, `jump_intensity`,
  `regime_posterior_*`.

## 3. Integration

Enable flag config, daily orchestrator pickup with zero job-code edits.
Dashboard per `docs/redesign/04-ui-plan.md` §P4: `gold.astro` (regime badge,
hedge-score history, drivers panel) and verify the gold card appears on the
home Engines panel with **no template change**. Build, deploy, verify live.

## 4. Tests (acceptance)

- Regime fixtures: 2008H2, 2011, and Mar-2020 windows classify as
  `crisis_bid`/`hedge_bid`; 2013 and 2022 rate-shock windows as
  `carry_headwind` — assert on those windows in a walk-forward (never
  full-sample-fit) backtest.
- Jump detector on synthetic jump-diffusion fixtures (known jump dates).
- PIT replay test at ≥2 cutoffs.
- `lint-imports`: gold imports no other engine. Full suites green.
