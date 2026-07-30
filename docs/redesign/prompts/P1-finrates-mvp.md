# P1 — FinRates MVP (first real engine)

Copy everything below this line to the coder agent. Requires P0 merged.

---

You are working in the `findyn` repo. Before writing any code, read:
`docs/redesign/01-target-architecture.md`, `docs/redesign/03-contracts.md`,
and `FINDYN_V1_SPEC.md` §14.1 (no-lookahead law) — all engine data access goes
through `WorldState.series` / `pit_join`, expanding-window fits only, no
centered filters.

## Task

Build the FinRates MVP in `findynamics/engines/rates/`:

```
Treasury data → yield curve → Nelson-Siegel → rate regime → D1 → API → dashboard
```

## 1. Data (extend config, not code)

Add to `compute/config/series.yaml` under `engines.rates.series`: FRED
constant-maturity yields `DGS1MO DGS3MO DGS6MO DGS1 DGS2 DGS3 DGS5 DGS7 DGS10
DGS20 DGS30` (daily, lag 1 day) and `T10YIE` (breakeven). Add factor inputs
under `factors:`: `real_rate` (DGS10 − T10YIE) and `usd_strength` (DTWEXBGS)
with the standard 0–100 scoring. The FRED provider already exists — no
provider code changes. Extend `jobs/backfill.py` to backfill these series
through the existing ingestion path with correct `release_date` synthesis.

## 2. Model (`engines/rates/`)

- `curve.py`: assemble the PIT yield curve for a date from the DGS series;
  handle missing tenors (holidays, DGS1MO starts 2001; fit whatever tenors
  exist that day, require ≥5).
- `nelson_siegel.py`: static Nelson-Siegel fit per date. β0/β1/β2 by OLS on a
  λ grid (fix λ by in-sample RMSE on the training window, then freeze — do
  not refit λ daily). Outputs level (β0), slope (−β1), curvature (β2), rmse.
- `regime.py`: rule-based v1 regimes from PIT-safe transforms of level/slope
  (e.g. slope sign + 12m level trend). Vocabulary (document in
  `engines/rates/domain.py`):
  `steep_easing | steep_tightening | flat | inverted | re_steepening`.
  Thresholds in `config/engines/rates.yaml`, not code. HMM/Kalman is Phase-2
  of this engine — do not build it now.
- `engine.py`: `RatesEngine(AssetEngine)`, `name="rates"`, registered via
  `@register_engine`. `predict` → `AssetState` with: `regime`;
  `expected_return` = curve-implied 12m expected return of a 10y
  constant-maturity position (carry + rolldown from the fitted curve — cite
  formula in docstring); `risk_score` = duration risk proxy scaled 0–100;
  `signals` ⊇ `curve_inversion`, `term_premium_trend`. `outputs()` → per-date
  `ns_level, ns_slope, ns_curvature, ns_rmse` as `engine_output` rows.
- Enable in `config/engines/rates.yaml` (`enabled: true`).

## 3. Orchestration

Rewrite `jobs/daily.py` per `01-target-architecture.md` §8: build `WorldState`
(compute factors via `factors/compute.py`) → loop `enabled_engines()` →
`predict` + `outputs` → HMAC write-back to serving. `jobs/monthly_refit.py`
calls `fit` (for rates: λ selection + threshold calibration). Keep jobs as
thin CLIs; all logic in the package.

## 4. Serving plane

- Migration `serving/migrations/0004_multi_asset.sql` exactly per
  `01-target-architecture.md` §6 (`asset_state`, `engine_output`, additive
  `asset` columns).
- Write-back payloads per `03-contracts.md` §6 in `admin/writeback.ts`, with
  validation and idempotent upserts, tested in vitest.
- New routes in `src/api/`: `GET /api/v1/assets`,
  `GET /api/v1/assets/:asset/state`,
  `GET /api/v1/assets/:asset/history?metric=&from=&to=` — existing response
  envelope, staleness flags, `501` + phase tag for engines with no data.

## 5. Dashboard

Implement `docs/redesign/04-ui-plan.md` §P1 exactly:
- Home page "Engines" panel driven by `GET /api/v1/assets` (registry-driven
  cards — built once, later engines appear without template changes).
- New page `dashboard/src/pages/rates.astro`: NS curve snapshot with ghost
  curves, level/slope/curvature history, regime timeline strip, signals table.
Follow the existing pages' structure (`lib/api.ts`, `scripts/`), handle
501/stale/empty states explicitly, then **build and deploy** (`npm run build`,
`npm run deploy`) and verify live — deploy is part of Done.

## 6. Tests (acceptance)

- Unit: NS fit recovers known synthetic curves (β within tolerance); curve
  assembly handles missing tenors; regime rules on constructed fixtures.
- **PIT replay test**: recompute rates `AssetState` at ≥3 historical cutoffs
  using only `release_date ≤ cutoff` data and assert it matches the stored
  run (the §14.1 rule-5 pattern; reusable helper goes in
  `findynamics/backtest/replay.py`).
- Sanity backtest: regimes over 2000–2024 mark the 2000, 2006–07 and 2019 and
  2022 inversions as `inverted` — assert on those windows.
- `lint-imports`, ruff, full pytest, serving typecheck + vitest all green.

## Constraints

- No ML dependencies. `numpy/pandas/scipy` only for this phase.
- The rates engine must not import any other engine or anything in
  `portfolio/`.
- No deterministic rate forecasts — outputs are states, scores, signals.
