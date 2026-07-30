# FinDynamics — Target Architecture

## 1. Vision

FinDynamics is a financial physics engine: it transforms macro state into asset
dynamics and portfolio decisions. It is **not** a collection of price
predictors.

```
World State → Risk Factors → Asset Physics Engines → Portfolio Engine → Dashboard
```

Each engine models a distinct stochastic process:

| Engine | Financial force | Core mathematics |
|---|---|---|
| FinMoney | Time value | Money-market account `M(t)=M(0)·exp(∫r dt)` |
| FinRates | Interest-rate dynamics | Nelson-Siegel → state space/Kalman → Vasicek/CIR/Hull-White |
| FinEquity | Growth & risk premium | Kalman kinematics, HMM regimes, ERP/valuation, calibrated transition probabilities |
| FinGold | Trust & crisis protection | Regime switching + jump diffusion on real rates/USD/stress |
| FinCrypto | Network scarcity (experimental) | Jump diffusion + adoption/liquidity beta |

## 2. Reconciliation with the existing repo

Two decisions the pure FinDynamics doc leaves open are settled here:

**D1 — Keep the two-plane deployment.** The Cloudflare serving plane
(D1/KV/R2, Hono API, HMAC write-back, Astro dashboard) and the Python compute
plane (GitHub Actions cron) are a *deployment* topology, orthogonal to the
FinDynamics *logical* architecture. They stay. The FinDynamics layer diagram
lives entirely inside the compute plane; the serving plane is the "app" layer.

**D2 — The current FinDyn v1 S&P500 spec becomes the FinEquity engine.** The
existing infrastructure (providers, `pit_join`, quality engine, `series.yaml`,
PIT schema) is asset-agnostic and is promoted to the shared data/core layers.
The S&P500-specific model stack (Kalman kinematics, 5-state HMM, RII, crash
decomposition) is built later, inside `engines/equity`, per the original
`FINDYN_V1_SPEC.md`.

Repo name stays `findyn`. Product name is **FinDynamics**; the Python package
is `findynamics`.

## 3. Layered architecture and dependency rules

```
        core          (contracts, WorldState, registry, config, no-lookahead law)
          ↑
   data · factors     (providers, pit, quality  ·  shared risk factors)
          ↑
       engines        (money | rates | equity | gold | crypto — mutually isolated)
          ↑
      portfolio
          ↑
   app (jobs CLIs, serving plane, dashboard)
```

Hard rules, CI-enforced with `import-linter` from Phase 0:

1. `core` imports nothing from `data`, `factors`, `engines`, `portfolio`.
2. Engines import `core`, `data`, `factors` — never another engine.
3. `portfolio` imports engine **outputs** (`AssetState`) via the registry,
   never engine internals.
4. Engines register themselves in `core.registry`; discovery is by name, not
   by import.
5. `crypto` is experimental: nothing outside `engines/crypto` may import it,
   and the portfolio engine excludes it unless explicitly configured in.

## 4. Repository layout (target)

```
findyn/
├── FINDYN_V1_SPEC.md                # historical spec → governs engines/equity
├── docs/redesign/                   # this design pack
├── compute/                         # Python 3.11 — the FinDynamics framework
│   ├── pyproject.toml               # package renamed findyn → findynamics
│   ├── config/
│   │   ├── series.yaml              # global series map (PIT lags) — extended per engine
│   │   └── engines/                 # one yaml per engine (enable flags, params)
│   ├── findynamics/
│   │   ├── core/
│   │   │   ├── contracts/           # AssetState, WorldState, FactorState, Signal
│   │   │   ├── engine.py            # AssetEngine ABC
│   │   │   ├── registry.py          # engine + provider registries
│   │   │   └── config.py            # strict config loading (from findyn/config.py)
│   │   ├── data/
│   │   │   ├── providers/           # fred, shiller, stooq, bls, bea, mock, resilience
│   │   │   ├── pit.py               # pit_join — sole gateway to macro_series
│   │   │   └── quality.py
│   │   ├── factors/                 # Layer 0 — shared risk factors
│   │   │   ├── definitions.py       # factor registry built from series.yaml
│   │   │   └── compute.py           # winsorized z → expanding percentile → 0-100
│   │   ├── engines/
│   │   │   ├── money/
│   │   │   ├── rates/
│   │   │   ├── equity/              # FinDyn v1 spec lives here when built
│   │   │   ├── gold/
│   │   │   └── crypto/
│   │   ├── portfolio/
│   │   └── backtest/                # walk-forward harness, replay test helpers
│   ├── jobs/                        # thin CLIs: backfill · daily · weekly · monthly_refit
│   └── tests/
├── serving/                         # TypeScript · Cloudflare Workers (unchanged role)
│   ├── migrations/                  # + 0004_multi_asset.sql
│   └── src/{api,admin,ingest,providers,lib}
├── dashboard/                       # Astro — grows one page per engine
└── .github/workflows/
```

## 5. Layer specifications

### Layer 0 — Factors (shared risk factors)

The nine v1 "forces" (valuation, earnings, liquidity, rates, credit, inflation,
labor, risk_appetite, sentiment) generalize into **factors**: global financial
variables owned by no engine. Factors are computed once per run from
`series.yaml` definitions, PIT-correct, scored 0–100 with component
breakdowns, and served to every engine through `WorldState`.

Additions needed by the new engines (extend `series.yaml`, never hard-code):
real rates (DGS10 − T10YIE), USD strength (DTWEXBGS), curve points
(DGS1MO…DGS30), vol (VIXCLS), global liquidity/M2 (M2SL, WALCL).

### Layer 1 — Asset engines

Every engine implements the `AssetEngine` interface (`03-contracts.md`) and
produces an `AssetState`. Per-engine model roadmaps:

- **money**: integrate short rate from FinRates/factor data → cash carry,
  discount factors D(t,h), risk-free benchmark, liquidity state. No ML.
- **rates**: Phase 1 Nelson-Siegel (level/slope/curvature) + rule-based rate
  regimes; Phase 2 Kalman state space; Phase 3 Vasicek/CIR/Hull-White.
  *Two different slopes, deliberately.* The published `ns_slope` factor is
  `-b1`, which is a long-end-versus-instantaneous read of the fitted curve. The
  regime rules do **not** branch on it: they read the empirical 10y-3m spread
  off the same fit, because that is the spread the inversion literature is
  written about and the one the sanity backtest asserts against. They are
  correlated but not interchangeable — around a 2019-style inversion they can
  disagree in sign. Both are published; anything consuming "the slope" has to
  say which it means.
- **equity**: `FINDYN_V1_SPEC.md` unchanged in substance — Kalman kinematics,
  FFD, 5-state HMM, XGBoost calibration, RII, crash decomposition, Monte
  Carlo. No deep-learning price prediction.
- **gold**: regime switching driven by real rate, USD, liquidity stress;
  jump/crisis premium; outputs hedge score + crisis probability.
- **crypto**: research-only; jump diffusion + network/adoption metrics;
  quarantined by the import rules.

### Portfolio layer

Consumes the latest `AssetState` per registered engine plus `WorldState`;
produces target-weight *distributions* and conditional implications (same
non-goals as v1: no trade commands, no deterministic targets).

## 6. Data model evolution (D1)

`macro_series` (the PIT table) is already asset-agnostic — untouched.
Migration `0004_multi_asset.sql` adds:

```sql
-- One row per engine per run: the serialized AssetState.
CREATE TABLE asset_state (
  asset          TEXT NOT NULL,      -- 'rates' | 'money' | 'equity' | 'gold' | 'crypto'
  as_of          TEXT NOT NULL,
  model_version  TEXT NOT NULL,
  regime         TEXT NOT NULL,
  expected_return REAL,
  risk_score     REAL,               -- 0-100
  confidence     REAL,               -- 0-1
  signals        TEXT NOT NULL,      -- JSON array of Signal
  components     TEXT,               -- JSON explanation trace
  PRIMARY KEY (asset, as_of, model_version)
);

-- Engine-specific wide outputs (e.g. NS level/slope/curvature per date).
CREATE TABLE engine_output (
  asset     TEXT NOT NULL,
  metric    TEXT NOT NULL,           -- 'ns_level' | 'ns_slope' | 'carry_1y' | ...
  as_of     TEXT NOT NULL,
  value     REAL NOT NULL,
  meta      TEXT,                    -- JSON
  PRIMARY KEY (asset, metric, as_of)
);
```

Existing S&P500-shaped tables (`derived_features`, `force_scores`,
`regime_state`, `instability_index`, `forecast_distribution`) are kept and
become the equity engine's private output tables; `force_scores` doubles as
the factor store (code-level name: factors; table rename not worth the churn).
Where a `symbol`/`asset` column is missing, 0004 adds it with default `'SPX'`.

## 7. API evolution (serving plane)

Existing endpoints stay. New namespace, same envelope
(`as_of`, `model_version`, `stale`, disclaimer):

| Endpoint | Returns |
|---|---|
| `GET /api/v1/assets` | Registered engines + status/staleness |
| `GET /api/v1/assets/:asset/state` | Latest `AssetState` |
| `GET /api/v1/assets/:asset/history?metric=` | `engine_output` time series |
| `GET /api/v1/factors` | Factor scores + component breakdowns (generalizes `/forces`) |
| `GET /api/v1/portfolio` | Portfolio engine output (Phase 6) |

Unbuilt endpoints return `501` + phase tag (existing convention).
Write-back: the existing HMAC admin route gains `POST /admin/v1/results`
payload types for `asset_state` / `engine_output` batches.

## 8. Jobs and scheduling

`jobs/daily.py` becomes an orchestrator: load config → build `WorldState`
(factors) → for each **enabled** engine in the registry: `predict` → write
back. `weekly` runs simulation-grade work (Monte Carlo), `monthly_refit` runs
`fit` with expanding windows and bumps `model_version`. Engines are enabled
per `config/engines/*.yaml`, so shipping a new engine never edits job code.

## 9. Development roadmap

| Phase | Content | Exit criterion |
|---|---|---|
| P0 | Package restructure + core contracts + registry + import-linter | All existing tests pass from new paths; CI enforces layer rules |
| P1 | FinRates MVP | NS factors in D1, `/assets/rates/state` live, dashboard page, PIT replay test for rates |
| P2 | FinMoney | Carry/discount factors published; rates engine consumes its benchmark via WorldState |
| P3 | FinEquity | FINDYN_V1_SPEC M2–M4 delivered inside `engines/equity` |
| P4 | FinGold | Regime + hedge score, backtested against 2008/2020 stress windows |
| P5 | FinCrypto | Isolated research module; import quarantine verified in CI |
| P6 | Portfolio + multi-asset dashboard | Weight distributions from ≥3 engines; chaos test stays green |

Principle preserved from both source documents: **do not build framework
beyond what the next engine needs.** P0 creates only the contracts P1
consumes.
