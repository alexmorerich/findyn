# FinDyn v1.0 — S&P500 Dynamic State Engine

> **Re-scoped:** this spec now governs `findynamics/engines/equity`; see `docs/redesign/`.

**Final Build Specification (for Coder Agent)**
Status: APPROVED · Version: 1.0 · Parent system: FinOS / FinDyn

---

## 0. Role & Mission

You are acting as a senior quantitative researcher, financial engineer, ML engineer, and Cloudflare architect. Build the first production-grade module of FinDyn: the **S&P500 Dynamic State Engine**.

The purpose of this system is **NOT** to predict exact S&P500 prices.

The purpose is to model the S&P500 as a **nonlinear dynamic system** and estimate:

- Current market position
- Market velocity (direction and speed of movement)
- Market acceleration (strengthening/weakening momentum)
- Market instability (transition risk)
- Market regime and regime transition probability
- Future **probability distributions** (never point predictions)

Behave like a **navigation system for investors**: it tells you where the market is, which direction it is moving, whether momentum is strengthening or weakening, and whether regime-transition risk is rising.

### Non-Goals (hard constraints)

1. No deterministic price targets. All forward-looking output is a distribution.
2. No trading commands ("sell stocks", "buy bonds"). Output conditional implications only (§12).
3. No claim of crisis prediction. The engine detects **proximity to critical points**, not the shape of the next crisis (§4).
4. Jerk/Snap are **risk-state indicators**, not precise physical quantities (§3.1).

---

## 1. Position in the FinOS Ecosystem

```
FinOS
├── FinDyn   — Financial Dynamics Intelligence
│     └── S&P500 Dynamic State Engine  ← THIS PROJECT (V1)
├── FinArk   — Portfolio Construction Engine
├── FinWar   — Geopolitical Risk Engine
└── FinArt   — Investment Research Intelligence
```

Roadmap: **V1** S&P500 Engine → **V2** Multi-Asset Regime Engine → **V3** World Macro Engine.
Design every interface (DB schema, API, feature pipeline) so that `symbol`/`asset` is a parameter, never a hard-coded constant — V2 must be a data extension, not a rewrite.

---

## 2. Core Model Design — Two Separate State Layers

**DO NOT mix price derivatives and economic drivers.** They live in two layers:

| Layer | Question it answers | State vector |
|---|---|---|
| A. Market Kinematics | *What is the market doing?* | `K(t)` |
| B. Market Forces | *Why is it doing this?* | `F(t)` |

### 2.1 Layer A — Market Kinematic State

`K(t) = [ Price, Velocity, Acceleration, Jerk*, RII ]`

| Component | Definition | Computation (causal only, §7) |
|---|---|---|
| Price | Filtered log index level | Kalman-smoothed log price (filtered estimate, not smoother — smoother uses future data) |
| Velocity | Market movement speed | Kalman state slope (annualized trend), plus multi-window momentum (21/63/252d) on FFD series |
| Acceleration | Change of momentum | First difference of Kalman velocity state; earnings-momentum acceleration as secondary input |
| Jerk* | Change of acceleration → **Trend Instability Indicator** | Rolling z-score of Δacceleration. Published only as a normalized z-score with regime context, never as a raw derivative |
| Snap → **RII** | **Regime Instability Index** — probability-like score [0,100] of imminent state transition | Composite, §3.2. Never computed as a raw 4th derivative |

### 2.2 Layer B — Market Force State

`F(t) = [ Valuation, Earnings, Liquidity, Rates, Credit, Inflation, Labor, RiskAppetite, Sentiment ]`

Each force is a 0–100 percentile score built from concrete series (mapping in §5.2), computed point-in-time, with a JSON component breakdown stored for explainability (§16).

---

## 3. Risk-State Indicators (Jerk / RII)

### 3.1 Why high-order derivatives are demoted

Financial series are noisy; raw 3rd/4th derivatives explode noise. Therefore:

- Jerk = z-scored change of the *filtered* acceleration → interpreted as "trend instability", thresholded (|z| > 2 = elevated), never fed raw into forecasts.
- Snap is **replaced entirely** by the Regime Instability Index.

### 3.2 Regime Instability Index (RII)

Weighted composite (weights configurable, default equal; normalized to 0–100):

| Component | Signal |
|---|---|
| HMM posterior entropy | `−Σ p·log(p)` over regime posteriors — high = model uncertain = unstable |
| 1 − max posterior | Regime confidence deficit |
| Jerk \|z\| | Trend instability (§3.1) |
| Vol-of-vol | Realized vol of rolling realized vol (63d window of 21d RV); VIX change rate when available |
| Correlation breakdown | Rolling 63d stock–bond return correlation vs its 3y baseline |
| Credit velocity | 21d rate of change of HY OAS |
| Liquidity stress | NFCI level and 13-week change |

---

## 4. Crash / Black-Swan Risk Decomposition

**Do NOT treat "Snap/black swan" as one variable and do NOT overfit 2008/2020.** The engine detects whether the system is entering a region where state jumps become likely — not what the next crisis looks like. (Reference frame: 2007 = high valuation + end of credit expansion + tightening liquidity + high leverage occurring *together*.)

```
P(Crash, h) = P(Regime Transition, h) × P(Shock | fragile state) × P(Transmission)
```

| Factor | Meaning | Estimation |
|---|---|---|
| `P(Regime Transition)` | Probability of moving to Bear/Crisis within horizon h | HMM transition matrix + XGBoost-calibrated transition probability (§8), modulated by RII |
| `P(Shock \| fragile)` | Baseline hazard of an exogenous/endogenous shock while fragile | EVT: fit Generalized Pareto Distribution to the tail of long-history (1871+) monthly drawdowns; hazard rate scaled by RII decile |
| `P(Transmission)` | Whether a shock propagates into a systemic move | Fragility score: FINRA margin debt trend, HY OAS level+velocity, NFCI, yield-curve state, (optional) index concentration |

Publish `crash_risk ∈ [0,100]` **and** the three factors separately — the decomposition is part of the explainability contract. Theoretical framing: nonlinear dynamical systems, risk state-space models, macro-quant fund practice — not a stock-prediction AI.

---

## 5. Data Sources

### 5.1 Source priority & constraints

| # | Source | Use | Constraints (design around these) |
|---|---|---|---|
| 1 | **FRED API** | Rates, inflation, GDP, labor, money supply, credit, stress indices | Free key. ⚠️ FRED `SP500` daily series covers only ~10 trailing years (licensing) — not usable for deep backfill |
| 2 | **Shiller dataset** (`ie_data.xls`, monthly, 1871+) | CAPE, real price, real earnings, dividends — the deep-history backbone | Monthly only; download quarterly via R2-cached copy |
| 3 | **BLS API** | Employment detail, average hourly earnings | Free tier: registration key, 500 req/day |
| 4 | **BEA API** | Corporate profits (NIPA Table 1.12), national accounts | Quarterly, ~2-month publication lag |
| 5 | **Treasury FiscalData / par-yield API** | Daily yield curve | No key required |
| 6 | **Alpha Vantage (free tier)** | Daily market prices (SPY as S&P500 proxy) — **preferred price source** | ⚠️ 25 requests/day free — fine for incremental daily updates, NOT for backfill; backfill from Stooq CSV (`^spx` daily, free bulk) |
| 7 | **Yahoo Finance** | **Fallback only** | Unofficial/unstable. Isolate behind `providers/yahoo.ts` implementing the same `PriceProvider` interface so it can be deleted without touching anything else |

Every provider is an isolated adapter with a shared interface (`fetchSeries(seriesId, from, to)`), a per-provider rate limiter, and a circuit breaker.

### 5.2 Series map (initial set — extend via config, not code)

| Force / Layer | Series (source: ID) | Freq | Pub. lag |
|---|---|---|---|
| Price | AlphaVantage: SPY; Stooq: ^spx (backfill); Shiller: real price (1871+) | D / M | same day / ~1 month |
| Valuation | Shiller CAPE; excess CAPE yield (CAPE⁻¹ − 10y real rate) | M | ~1 month |
| Earnings | Shiller real EPS; BEA corporate profits | M / Q | ~1m / ~2m |
| Liquidity | FRED: M2SL, WALCL, RRPONTSYD, NFCI | W–M | 1–2 weeks |
| Rates | FRED: DGS2, DGS10, DFII10, T10Y2Y, T10Y3M | D | same day |
| Credit | FRED: BAMLH0A0HYM2 (HY OAS), BAMLC0A0CM (IG OAS), BUSLOANS, DRTSCILM (SLOOS, quarterly) | D–Q | 0d–1q |
| Inflation | FRED: CPIAUCSL, CPILFESL, T5YIFR | M / D | ~2 weeks / same day |
| Labor | FRED: UNRATE, PAYEMS, ICSA; BLS: CES0500000003 (AHE) | W–M | 5d–1 week |
| Risk appetite | FRED: VIXCLS; HY−IG spread differential; STLFSI4 | D–W | same day–1w |
| Sentiment | FRED: UMCSENT | M | ~2 weeks |

**Every macro observation is stored with `release_date`** (first date the value was publicly known). Where FRED/ALFRED provides vintages, store the vintage. Where not, apply a conservative per-series lag from the table above — configured in `compute/config/series.yaml`, never hard-coded.

---

## 6. Architecture — Two Planes

**Critical constraint:** Cloudflare Workers cannot run the Python scientific stack (Kalman/HMM/XGBoost/EVT). Do not attempt ML inside Workers. Split:

```
┌─────────────────────── SERVING PLANE (Cloudflare) ────────────────────────┐
│  Cron Triggers ─→ Ingestion Workers ─→ validate ─→ normalize ─→ D1        │
│  KV: real-time cache (VIX, SPY, yields) + last-known-good state snapshot  │
│  R2: raw source snapshots, Shiller xls, model artifacts, MC path bundles  │
│  Hono API Worker: /api/v1/* (read-only, public)                           │
│  Admin Worker: /admin/v1/results (HMAC-authed write-back from compute)    │
│  Astro dashboard (Workers static assets)                                  │
└───────────────────────────────▲───────────────────────────────────────────┘
                                │ HMAC-signed JSON write-back
┌───────────────────────────────┴───────────────────────────────────────────┐
│               COMPUTE PLANE (Python 3.11, GitHub Actions cron)             │
│  daily job:  pull point-in-time data from D1 → features → Kalman → HMM    │
│              → XGBoost calibration → RII → crash decomposition → write back│
│  weekly job: Monte Carlo (10k paths) → forecast distributions → R2 + D1   │
│  monthly job: refit models (expanding window), version bump, backtest CI  │
└────────────────────────────────────────────────────────────────────────────┘
```

- Compute runs on **GitHub Actions scheduled workflows** (free, zero-ops). Alternative if runs exceed GA limits: Cloudflare Containers (paid) — the compute code must not assume either host (plain Python CLI entrypoints).
- Stack: TypeScript + Hono (serving), Python 3.11 with `pandas`, `numpy`, `statsmodels`, `hmmlearn`, `xgboost`, `scipy` (compute), Astro (dashboard), Wrangler + D1 migrations (infra).
- Repo layout:

```
findyn/
├── FINDYN_V1_SPEC.md
├── serving/                 # TypeScript, Cloudflare
│   ├── wrangler.jsonc
│   ├── migrations/          # D1 SQL migrations (§7)
│   └── src/
│       ├── providers/       # fred.ts, shiller.ts, bls.ts, bea.ts, treasury.ts, alphavantage.ts, stooq.ts, yahoo.ts (isolated)
│       ├── ingest/          # cron handlers, validation, normalization
│       ├── api/             # public read API (§13)
│       └── admin/           # HMAC write-back endpoint
├── compute/                 # Python 3.11
│   ├── pyproject.toml
│   ├── config/series.yaml   # series map, publication lags, force weights
│   ├── findyn/              # features/, models/, mc/, backtest/, pit.py (point-in-time joins)
│   └── jobs/                # backfill.py, daily.py, weekly.py, monthly_refit.py
├── dashboard/               # Astro
└── .github/workflows/       # ci.yml, compute-daily.yml, compute-weekly.yml
```

### Ingestion strategy

Historical **backfill once** (`jobs/backfill.py` + one-off Worker routes, sources: Stooq + Shiller + full FRED history), then **incremental updates** via Cron Triggers (daily 22:30 UTC for market data; staggered for macro releases). Real-time-ish values (VIX, SPY, 10y yield) cached in KV with 15-min TTL.

---

## 7. Database Design (D1, SQLite dialect)

All migrations under `serving/migrations/`, applied via `wrangler d1 migrations apply`.

```sql
-- 0001_core.sql
CREATE TABLE market_price (
  date        TEXT NOT NULL,            -- observation date YYYY-MM-DD
  symbol      TEXT NOT NULL,            -- 'SP500', 'SPY', ...
  close       REAL NOT NULL,
  volume      REAL,
  source      TEXT NOT NULL,            -- provider id
  ingested_at TEXT NOT NULL,
  PRIMARY KEY (date, symbol, source)
);

CREATE TABLE macro_series (
  series_id    TEXT NOT NULL,           -- e.g. 'FRED:CPIAUCSL'
  obs_date     TEXT NOT NULL,           -- period the value describes
  release_date TEXT NOT NULL,           -- POINT-IN-TIME KEY: first public availability
  value        REAL NOT NULL,
  vintage      TEXT,                    -- ALFRED vintage when available
  source       TEXT NOT NULL,
  ingested_at  TEXT NOT NULL,
  PRIMARY KEY (series_id, obs_date, release_date)
);
CREATE INDEX idx_macro_pit ON macro_series (series_id, release_date);

CREATE TABLE derived_features (
  date          TEXT NOT NULL,
  feature       TEXT NOT NULL,          -- 'price_filtered','velocity','acceleration','jerk_z','ffd_price', ...
  value         REAL NOT NULL,
  model_version TEXT NOT NULL,
  computed_at   TEXT NOT NULL,
  PRIMARY KEY (date, feature, model_version)
);

CREATE TABLE force_scores (
  date          TEXT NOT NULL,
  force         TEXT NOT NULL,          -- valuation|earnings|liquidity|rates|credit|inflation|labor|risk_appetite|sentiment
  score         REAL NOT NULL,          -- 0-100 percentile, point-in-time
  components    TEXT,                   -- JSON breakdown for explainability
  model_version TEXT NOT NULL,
  PRIMARY KEY (date, force, model_version)
);

CREATE TABLE regime_state (
  date          TEXT NOT NULL,
  regime        TEXT NOT NULL,          -- bull_expansion|normal_expansion|late_cycle|bear|crisis
  probability   REAL NOT NULL,
  model_version TEXT NOT NULL,
  PRIMARY KEY (date, regime, model_version)
);

CREATE TABLE instability_index (
  date           TEXT NOT NULL,
  rii            REAL NOT NULL,         -- 0-100
  p_transition   REAL,                  -- crash decomposition factors (§4)
  p_shock        REAL,
  p_transmission REAL,
  crash_risk     REAL,                  -- 0-100 composite
  components     TEXT,                  -- JSON
  model_version  TEXT NOT NULL,
  PRIMARY KEY (date, model_version)
);

CREATE TABLE forecast_distribution (
  as_of          TEXT NOT NULL,
  horizon        TEXT NOT NULL,         -- tactical|strategic|generational|educational_30y|educational_50y
  quantile       REAL NOT NULL,         -- 0.05,0.10,0.25,0.50,0.75,0.90,0.95
  value          REAL NOT NULL,         -- projected real log index level
  educational_only INTEGER NOT NULL DEFAULT 0,
  model_version  TEXT NOT NULL,
  PRIMARY KEY (as_of, horizon, quantile, model_version)
);

CREATE TABLE tradable_proxy_mapping (
  analysis_asset  TEXT NOT NULL,        -- 'SP500'
  jurisdiction    TEXT NOT NULL,        -- 'EU_UCITS','US','LATAM_DEFAULT'
  tradable_ticker TEXT NOT NULL,        -- 'SPYL','CSPX','VUAA','SPY','VOO'
  name            TEXT NOT NULL,
  notes           TEXT,
  PRIMARY KEY (analysis_asset, jurisdiction, tradable_ticker)
);

CREATE TABLE ingestion_log (
  id           INTEGER PRIMARY KEY AUTOINCREMENT,
  run_at       TEXT NOT NULL,
  source       TEXT NOT NULL,
  series_id    TEXT,
  status       TEXT NOT NULL,           -- ok|degraded|failed
  rows_written INTEGER DEFAULT 0,
  error        TEXT
);
```

Seed `tradable_proxy_mapping`: SP500 → SPYL (SPDR S&P500 UCITS acc), CSPX (iShares Core S&P500 UCITS), VUAA (Vanguard S&P500 UCITS) for `EU_UCITS`; SPY/VOO for `US`. The engine **analyzes** SP500; the output layer maps to jurisdiction-appropriate tradable instruments.

---

## 8. Feature Engineering Pipeline

**Never compute high-order derivatives from raw prices.**

```
Raw price → gap/outlier handling → CAUSAL noise filtering → Fractional Differentiation
         → state estimation → derivative extraction → z-scoring
```

1. **Filtering — causal only.** Primary: Kalman filter (local linear trend model: level + slope states). Use the *filtered* estimate at t (information ≤ t), never the RTS smoother output, in any feature that feeds models.
   ⚠️ **Savitzky-Golay uses a centered window = lookahead bias. It is permitted ONLY in the dashboard display layer, never in features, model inputs, or backtests.** Enforce via module separation: `findyn/features/` must not import the SG function.
2. **Fractional Differentiation (FFD).** Fixed-width window fractional differencing (López de Prado method). For each series choose the **minimum d** whose FFD series passes ADF at 95% confidence — searched on the training window only, then frozen for the out-of-sample period. Purpose: stationarity while preserving long memory. Store d per series per model_version.
3. **Derivatives.** Velocity = Kalman slope state; Acceleration = Δslope; Jerk = z-score of Δacceleration (risk indicator only). All published with rolling z-normalization (expanding window, min 10y).
4. **Force scores.** Each force = weighted average of constituent indicator z-scores (winsorized at ±3σ, expanding-window percentile → 0–100). Weights in `series.yaml`.

---

## 9. Model Engine (three layers)

| Layer | Method | Output | Notes |
|---|---|---|---|
| L1 State Estimation | Kalman filter / linear-Gaussian state space (`statsmodels`) | Smoothed (filtered) kinematic state K(t) | Refit monthly, expanding window |
| L2 Regime Detection | Gaussian HMM, 5 states (`hmmlearn`) on FFD returns + realized vol + credit spread + curve slope | Posterior P(regime) for Bull Expansion / Normal Expansion / Late Cycle / Bear / Crisis | States are unlabeled by HMM — map to named regimes by sorting on (mean return, vol) with documented rules; assert label stability across refits |
| L3 Probability Refinement | XGBoost classifier on [K(t), F(t), RII] → P(transition to Bear/Crisis within h) for h ∈ {3m, 6m, 12m} | Calibrated transition probabilities (isotonic calibration) | Walk-forward training only; purged K-fold CV with 6-month embargo (López de Prado) |

**Deferred to v1.5 (do NOT build in v1):** LSTM / Transformer time-series refinement. V1 ships Kalman + FFD + HMM + XGBoost — deliverable and backtestable.

---

## 10. Forecast Engine

Output = probability distributions (quantiles of real log index level), regime-conditioned. Never a single number.

| Horizon | Range | Purpose | Method |
|---|---|---|---|
| Tactical | 3–6 months | Liquidity management | Regime-conditional return distributions + MC |
| Strategic | 1–3 years | Stock/bond/gold allocation input | Regime-switching MC with force-score conditioning |
| Generational | 10–15 years | Long-term valuation cycle | CAPE-anchored expected-return band + MC |
| Ultra-long | 30–50 years | **`educational_only=1`** — scenario simulation, excluded from all accuracy evaluation | Scenario trees |

---

## 11. Monte Carlo Simulator

Generate ≥10,000 paths per horizon (weekly job, paths summary → D1, full bundles → R2).

- **Base process:** regime-switching model using HMM-estimated transition matrix and per-regime return/vol distributions.
- **Shock overlay:** independent shock process — do not replay 2008/2020. Shock taxonomy: financial crisis, liquidity crisis, technology disruption, geopolitical shock, policy regime change. For each: frequency distribution (Poisson/Hawkes intensity scaled by RII), severity distribution (EVT/GPD tail fitted on 1871+ drawdowns), recovery distribution (duration model on historical recoveries).
- Report per-horizon: quantile fan, max-drawdown distribution, P(drawdown > 20%/30%/50%), time-under-water distribution.
- Goal: **robustness, not historical replay.**

---

## 12. Output Contract (portfolio layer)

No trading commands. Output shape:

```json
{
  "as_of": "2026-07-29",
  "regime": {"late_cycle": 0.50, "bear": 0.35, "bull_expansion": 0.15},
  "rii": 62,
  "crash_risk": {"composite": 41, "p_transition": 0.35, "p_shock": 0.18, "p_transmission": 0.65},
  "conditional_implication": "Under this distribution, investors with low risk tolerance may consider reducing equity concentration and increasing liquidity.",
  "tradable_proxies": {"EU_UCITS": ["SPYL", "CSPX", "VUAA"], "US": ["SPY", "VOO"]},
  "disclaimer": "Not investment advice. Final allocation depends on investor constraints, risk tolerance, and tax jurisdiction."
}
```

Conditional-implication strings are templated per (regime × crash-risk band) in config — reviewed text, not model-generated.

---

## 13. API Contract (public, read-only, Hono)

| Endpoint | Returns |
|---|---|
| `GET /api/v1/state` | Latest K(t) + F(t) snapshot + data-freshness flags |
| `GET /api/v1/regime?from&to` | Regime probability history |
| `GET /api/v1/forces?from&to` | Force score history with component breakdowns |
| `GET /api/v1/instability?from&to` | RII + crash decomposition history |
| `GET /api/v1/forecast?horizon=tactical\|strategic\|generational` | Quantile distribution (educational horizons flagged) |
| `GET /api/v1/simulate?horizon=&paths=` | MC summary statistics |
| `GET /api/v1/health` | Per-source ingestion status from `ingestion_log` |

All responses include `model_version`, `as_of`, and `stale: true/false` per data block.

---

## 14. Engineering Safety Requirements

### 14.1 STRICT NO-LOOKAHEAD BIAS (hard rules, CI-enforced)

1. Every feature/model value at date t uses only information with `release_date ≤ t` (market prices: close of t−1 or t per explicit convention `INFO_SET = "t-1"` in config — one convention, applied everywhere).
2. All macro joins go through one function: `findyn/pit.py::pit_join(series, as_of)` — the **only** allowed way to read `macro_series` in compute. Code review + lint rule: no direct `macro_series` reads elsewhere.
3. No centered filters/windows in the feature path (§8.1).
4. Model fitting: expanding-window walk-forward; hyperparameters chosen by purged CV with embargo; FFD `d` frozen from training window.
5. **CI replay test:** pick N random historical cutoff dates; recompute the full state using only `release_date ≤ cutoff`; diff against the stored history for those dates. Any mismatch beyond float tolerance fails the build.

### 14.2 Graceful degradation

- Provider failure → circuit breaker → fallback chain: live API → D1 last-known → KV cached state snapshot. The forecast/API layer **never crashes**; it serves the last consistent state with `stale: true` and the staleness age.
- Alpha Vantage quota exhausted → automatic Stooq/Yahoo fallback via the `PriceProvider` interface; log `degraded` in `ingestion_log`.

### 14.3 Explainability

- Every regime change record ships reasons: top-N feature contributions (XGBoost SHAP values for L3; posterior-driver decomposition for HMM) stored in the `components` JSON columns.
- Crash risk always published as its three factors (§4), never only as a composite.

---

## 15. Backtesting Protocol

Walk-forward event studies (data granularity: monthly Shiller before ~1950, daily after):

| Event | Window | Data granularity |
|---|---|---|
| 1929 crash | 1927–1933 | monthly |
| 1970s inflation | 1972–1976 | monthly |
| 1987 crash | 1986–1988 | daily |
| 2000 dot-com | 1998–2003 | daily |
| 2008 GFC | 2006–2010 | daily |
| 2020 COVID | 2019–2021 | daily |
| 2022 rate shock | 2021–2023 | daily |

Metrics (reported per event and aggregate, committed as `backtest/report.md` artifacts):

- Regime detection accuracy and **lead/lag in days/months** vs NBER + drawdown-defined bear markets
- Drawdown warning ability: P(crash_risk elevated | drawdown > 20% within 6m)
- Recovery detection lag
- **False alarm rate**: P(no 10% drawdown within 12m | crash_risk elevated)
- Brier score + reliability diagram for transition probabilities

---

## 16. Dashboard (Astro)

Panels: (1) regime probability gauge + history ribbon, (2) kinematics chart — filtered price, velocity, acceleration bands (SG smoothing allowed here only), (3) force scores bar/radar with component drill-down, (4) RII time series with alert thresholds, (5) crash-risk decomposition (three factors), (6) forecast fan charts per horizon (educational horizons visually separated), (7) data health / staleness panel.

---

## 17. Deliverables & Milestones

| M | Deliverable | Acceptance criteria |
|---|---|---|
| M0 | Repo scaffold, wrangler config, D1 migrations, CI skeleton | `wrangler deploy` succeeds; migrations apply cleanly; CI green |
| M1 | Provider adapters + ingestion workers + full backfill | All §5.2 series backfilled with `release_date` populated; `ingestion_log` clean; Yahoo isolated behind interface |
| M2 | Feature pipeline (Kalman, FFD, forces) + write-back | Features reproducible bit-for-bit from raw data; **no-lookahead CI replay test passes** |
| M3 | HMM regime engine + XGBoost calibration | Walk-forward backtest report covering 2000/2008/2020/2022 with §15 metrics |
| M4 | RII + crash decomposition + Monte Carlo + EVT | GPD fit diagnostics committed; shock taxonomy configured; full backtest incl. monthly-era events |
| M5 | Public API + dashboard + degradation handling | All §13 endpoints live; kill-a-provider chaos test serves stale-flagged state without error |
| M6 | Documentation + runbook | Fresh clone → bootstrap → deployed system following README only; architecture diagram included |

---

## 18. Disclaimer (must appear in API + dashboard)

FinDyn is a market navigation system. It estimates market position, velocity, acceleration, instability, and regime-transition probability. It does **not** claim to predict the future with certainty and does **not** provide investment advice. Final allocation decisions depend on investor constraints, risk tolerance, and tax jurisdiction.
