# P6 — Portfolio decision engine + multi-asset dashboard

Copy everything below this line to the coder agent. Requires P4 merged
(P5/crypto optional and excluded by default).

---

You are working in the `findyn` repo. Read `docs/redesign/01-target-architecture.md`
and `docs/redesign/03-contracts.md`. The v1 non-goals are binding here more
than anywhere: **no trade commands, no deterministic targets** — outputs are
weight *distributions* and templated conditional implications.

## Task

Build `findynamics/portfolio/` consuming the latest `AssetState` per enabled
non-experimental engine (via the registry + D1 `asset_state`, never engine
internals) plus `WorldState` factors.

## 1. Model

- `inputs.py`: assemble the decision panel — per-asset expected_return,
  risk_score, confidence, regime; discount factors and risk-free benchmark
  from the money engine's `engine_output`.
- `allocate.py`: regime-conditional strategic weights. v1 method: for each
  asset, map (expected_return − risk_free, risk_score, confidence) into a
  score; convert scores to weights with a risk-budget normalization; produce
  a **distribution** over weights by resampling engine confidence
  (documented method, no black box). Three risk profiles
  (`conservative | balanced | growth`) parameterized in
  `config/portfolio.yaml`.
- `implications.py`: templated conditional-implication text generated from
  the weight deltas vs the profile's neutral allocation — same template
  mechanism/disclaimer as the v1 spec §12.
- `guardrails.py`: hard constraints — weights sum to 1, per-asset caps from
  config, crypto excluded unless `include_experimental: true`, missing/stale
  engine ⇒ fall back to that asset's neutral weight and flag `degraded`.

## 2. Persistence & API

- Migration `0005_portfolio.sql`: `portfolio_state (profile, as_of,
  model_version, weights TEXT/*JSON quantiles per asset*/, implication TEXT,
  degraded INTEGER, PRIMARY KEY(profile, as_of, model_version))`.
- Write-back kind `portfolio_state` in `admin/writeback.ts`.
- `GET /api/v1/portfolio?profile=` — latest weights distribution +
  implication + input `AssetState` references; standard envelope +
  disclaimer.

## 3. Dashboard

Per `docs/redesign/04-ui-plan.md` §P6:
- `portfolio.astro`: profile switcher, weight-distribution fan/box chart per
  asset (distributions, never single numbers), implication text, degraded
  badges, and a "why" expander listing each input `AssetState`.
- Home page final form: portfolio summary strip + Engines panel + numeraire
  ribbon; status page fully relegated to `/status`.
- Build, deploy, verify live — deploy is part of Done.

## 4. Backtest

Walk-forward backtest of the balanced profile vs a static 60/30/10
(equity/rates/gold) benchmark over 2000–2024 at monthly rebalance, using only
PIT `asset_state` history (replayed via `backtest/replay.py`). Report: max
drawdown, vol, return, and — most important — behavior in 2008 and 2020
windows. Commit the report artifact. The goal is *robustness evidence*, not
performance marketing.

## 5. Acceptance

- Chaos test: kill one engine's data (simulate staleness) → API still 200,
  weights fall back to neutral for that asset, `degraded` flagged end-to-end
  to the dashboard badge.
- Guardrail unit tests (caps, sum-to-1, experimental exclusion).
- PIT replay of portfolio state at ≥2 cutoffs.
- `lint-imports`: portfolio imports no `engines.*` module internals.
- Full compute + serving suites green; dashboard builds.
