# FinDyn v1.0 — S&P500 Dynamic State Engine

A **market navigation system**, not a price predictor.

FinDyn models the S&P500 as a nonlinear dynamic system and estimates where the market is,
how fast it is moving, whether momentum is strengthening or weakening, and whether the
system is approaching a region where regime transitions become likely. Every forward-looking
output is a probability distribution.

Full specification: **[`FINDYN_V1_SPEC.md`](./FINDYN_V1_SPEC.md)** — the single source of truth.
This README is the design map and the operator's guide.

---

## Non-goals

These are hard constraints, enforced in code and CI — not aspirations.

| Constraint | Enforcement |
|---|---|
| No deterministic price targets | Forecasts are stored and served only as quantiles (`forecast_distribution`) |
| No trading commands ("sell stocks") | The API emits templated *conditional implications*; no endpoint returns an action |
| No claim of crisis prediction | Crash risk is decomposed into three published factors, never a single oracle number |
| No lookahead bias | Point-in-time joins, causal filters only, and a CI replay test that recomputes history from `release_date`-filtered data |

Jerk and Snap are **risk-state indicators**, not precise physical quantities. Raw 3rd and 4th
derivatives of a noisy price series are noise amplifiers; they are demoted accordingly (§3).

---

## Position in the FinOS ecosystem

```
FinOS
├── FinDyn   — Financial Dynamics Intelligence
│     └── S&P500 Dynamic State Engine  ← THIS REPO (V1)
├── FinArk   — Portfolio Construction Engine
├── FinWar   — Geopolitical Risk Engine
└── FinArt   — Investment Research Intelligence
```

Roadmap: **V1** S&P500 Engine → **V2** Multi-Asset Regime Engine → **V3** World Macro Engine.
Every interface treats `symbol`/`asset` as a parameter, so V2 is a data extension rather than
a rewrite.

---

## The two-layer state model

The core design decision is that price derivatives and economic drivers **never mix**.

### Layer A — Market Kinematics: *what is the market doing?*

`K(t) = [ Price, Velocity, Acceleration, Jerk, RII ]`

| Component | Meaning | How it is computed |
|---|---|---|
| Price | Filtered log index level | Kalman **filtered** estimate (not the smoother — that reads the future) |
| Velocity | Speed and direction | Kalman slope state + multi-window momentum on the fractionally differenced series |
| Acceleration | Momentum change | First difference of the Kalman velocity state |
| Jerk | **Trend instability indicator** | Rolling z-score of Δacceleration; thresholded, never fed raw into a model |
| Snap → **RII** | **Regime Instability Index**, 0–100 | Composite of HMM posterior entropy, confidence deficit, jerk, vol-of-vol, stock–bond correlation breakdown, credit velocity, liquidity stress |

### Layer B — Market Forces: *why is it doing this?*

`F(t) = [ Valuation, Earnings, Liquidity, Rates, Credit, Inflation, Labor, RiskAppetite, Sentiment ]`

Each force is a 0–100 point-in-time percentile score built from concrete series, stored with a
JSON component breakdown so any score can be explained back to its inputs.

The full series map — FRED/Shiller/BLS/BEA/Treasury/Alpha Vantage ids, frequencies, and
publication lags — lives in [`compute/config/series.yaml`](./compute/config/series.yaml).
Adding a data source is a config change, never a code change.

---

## Crash risk: proximity to a critical point

The engine does not ask *"will the next crisis look like 2008?"* It asks *"is the system
entering a region where state jumps become likely?"* — the 2007 signature was high valuation,
the end of credit expansion, tightening liquidity, and high leverage occurring **together**.

```
P(Crash, h) = P(Regime Transition, h) × P(Shock | fragile state) × P(Transmission)
```

| Factor | Source |
|---|---|
| `P(Regime Transition)` | HMM transition matrix + XGBoost-calibrated transition probability, modulated by RII |
| `P(Shock \| fragile)` | Extreme Value Theory — GPD tail fitted to 1871+ monthly drawdowns, hazard scaled by RII decile |
| `P(Transmission)` | Fragility score: margin debt trend, HY OAS level and velocity, NFCI, yield-curve state |

All three are published separately. A composite alone would be unfalsifiable.

---

## Architecture — two planes

Cloudflare Workers cannot host the scientific Python stack (Kalman, HMM, XGBoost, EVT), so
compute is deliberately split out rather than compromised.

```
┌─────────────────────── SERVING PLANE (Cloudflare) ────────────────────────┐
│  Cron Triggers ─→ Ingestion Workers ─→ validate ─→ normalize ─→ D1        │
│  KV: real-time cache (VIX, SPY, yields) + last-known-good state snapshot  │
│  R2: raw source snapshots, Shiller xls, model artifacts, MC path bundles  │
│  Hono API Worker: /api/v1/*  (public, read-only)                          │
│  Admin Worker:    /admin/v1/results  (HMAC-authenticated write-back)      │
│  Astro dashboard (Workers static assets)                                  │
└───────────────────────────────▲───────────────────────────────────────────┘
                                │ HMAC-signed JSON
┌───────────────────────────────┴───────────────────────────────────────────┐
│              COMPUTE PLANE (Python 3.11, GitHub Actions cron)             │
│  daily   → point-in-time pull → features → Kalman → HMM → XGBoost → RII   │
│  weekly  → Monte Carlo (10k paths) → forecast distributions → R2 + D1     │
│  monthly → expanding-window refit, version bump, backtest artifacts       │
└───────────────────────────────────────────────────────────────────────────┘
```

The compute jobs are plain Python CLIs with no host assumptions, so the scheduler can move
from GitHub Actions to Cloudflare Containers without touching job code.

### Repository layout

```
findyn/
├── FINDYN_V1_SPEC.md            # specification — source of truth
├── serving/                     # TypeScript · Cloudflare Workers
│   ├── wrangler.jsonc           # bindings, cron triggers, compat date
│   ├── worker-configuration.d.ts# generated by `npm run types` (CI checks for drift)
│   ├── migrations/              # D1 schema (§7)
│   ├── scripts/                 # bootstrap.sh, check-crons.mjs
│   ├── src/
│   │   ├── index.ts             # fetch + scheduled entrypoints
│   │   ├── domain.ts            # regimes, forces, horizons, disclaimer
│   │   ├── api/                 # public read API
│   │   ├── admin/               # HMAC-authenticated compute write-back
│   │   ├── ingest/              # cron dispatch, ingestion logging
│   │   ├── providers/           # one isolated adapter per source
│   │   └── lib/                 # response envelope, staleness
│   └── test/                    # vitest, runs inside workerd
├── compute/                     # Python 3.11 · models
│   ├── config/series.yaml       # series map + publication lags
│   ├── findyn/
│   │   ├── pit.py               # point-in-time joins — the no-lookahead choke point
│   │   ├── config.py            # strict config validation
│   │   └── domain.py            # vocabulary mirrored from serving/src/domain.ts
│   ├── jobs/                    # backfill · daily · weekly · monthly_refit
│   └── tests/
├── dashboard/                   # Astro (M5)
└── .github/workflows/           # ci.yml · compute-daily.yml · compute-weekly.yml
```

---

## Data model

Nine tables (`serving/migrations/0001_core.sql`). The one that shapes everything else:

```sql
CREATE TABLE macro_series (
  series_id    TEXT NOT NULL,   -- 'FRED:CPIAUCSL'
  obs_date     TEXT NOT NULL,   -- the period the value describes
  release_date TEXT NOT NULL,   -- when it first became knowable  ← point-in-time key
  value        REAL NOT NULL,
  vintage      TEXT,            -- ALFRED vintage where available
  ...
  PRIMARY KEY (series_id, obs_date, release_date)
);
```

March CPI describes March but is not knowable until mid-April. Storing only `obs_date` would
make every backtest quietly clairvoyant. `release_date` is in the primary key so revisions
coexist with original prints, and `pit_join` picks whichever was current at the cutoff.

Other tables: `market_price`, `derived_features`, `force_scores`, `regime_state`,
`instability_index`, `forecast_distribution`, `tradable_proxy_mapping`, `ingestion_log`.

---

## Feature pipeline

```
Raw price → gap/outlier handling → CAUSAL filtering → fractional differentiation
          → state estimation → derivative extraction → z-scoring
```

1. **Kalman filter** (local linear trend). The *filtered* estimate at t only — the RTS smoother
   is banned from the feature path.
2. **Fractional differentiation** (fixed-width window). Minimum `d` passing ADF at 95%, searched
   on the training window and then frozen. Stationarity without discarding long memory.
3. **Derivatives** from the state, not from raw prices.
4. **Force scores**: winsorized z-scores → expanding-window percentile → 0–100.

> **Savitzky–Golay is a lookahead trap.** Its centered window reads data after *t*. It is
> permitted in the dashboard display layer and forbidden everywhere else; `findyn/features/`
> must not import it. This was corrected during design review — the original plan had it in
> the feature pipeline, which would have silently inflated every backtest.

## Model engine

| Layer | Method | Output |
|---|---|---|
| L1 State estimation | Kalman / linear-Gaussian state space (`statsmodels`) | Filtered kinematic state K(t) |
| L2 Regime detection | 5-state Gaussian HMM (`hmmlearn`) | Posterior over Bull Expansion, Normal Expansion, Late Cycle, Bear, Crisis |
| L3 Probability refinement | XGBoost + isotonic calibration | Calibrated P(transition to Bear/Crisis within 3m / 6m / 12m) |

HMM states are unlabeled; they are mapped to named regimes by sorting on (mean return, vol)
with documented rules, and label stability is asserted across refits.

**LSTM / Transformer refinement is deferred to v1.5.** V1 ships a stack that can be fully
backtested and explained.

## Forecast and simulation

| Horizon | Range | Purpose |
|---|---|---|
| Tactical | 3–6 months | Liquidity management |
| Strategic | 1–3 years | Stock/bond/gold allocation input |
| Generational | 10–15 years | Long-term valuation cycle |
| Ultra-long | 30–50 years | `educational_only` — scenario simulation, excluded from all accuracy evaluation |

Monte Carlo: ≥10,000 regime-switching paths per horizon, with an independent shock overlay
drawn from a taxonomy (financial, liquidity, technology, geopolitical, policy) rather than a
replay of 2008 or 2020. Each class carries a frequency, severity (EVT/GPD) and recovery
distribution. The goal is robustness, not historical fidelity.

## Backtesting

Walk-forward event studies over 1929, 1973–74, 1987, 2000, 2008, 2020 and 2022 — monthly
granularity before ~1950, daily after. Reported metrics: regime-detection lead/lag against
NBER and drawdown-defined bears, drawdown warning rate, recovery detection lag, **false alarm
rate**, and Brier score with a reliability diagram for transition probabilities.

---

## API

Public, read-only, JSON. Every response carries `as_of`, `model_version`, `stale` and the
disclaimer.

| Endpoint | Returns | Milestone |
|---|---|---|
| `GET /api/v1/health` | Per-source ingestion status and staleness | ✅ M0 |
| `GET /api/v1/state` | Latest K(t) + F(t) snapshot | M2 |
| `GET /api/v1/forces` | Force score history with component breakdowns | M2 |
| `GET /api/v1/regime` | Regime probability history | M3 |
| `GET /api/v1/instability` | RII + crash decomposition history | M4 |
| `GET /api/v1/forecast` | Quantile distribution per horizon | M4 |
| `GET /api/v1/simulate` | Monte Carlo summary statistics | M4 |

Unbuilt endpoints return `501` with their milestone rather than an empty payload, so a consumer
can distinguish "not built" from "no data".

Example of the output contract (§12) — note the shape of the advice:

```json
{
  "regime": {"late_cycle": 0.50, "bear": 0.35, "bull_expansion": 0.15},
  "rii": 62,
  "crash_risk": {"composite": 41, "p_transition": 0.35, "p_shock": 0.18, "p_transmission": 0.65},
  "conditional_implication": "Under this distribution, investors with low risk tolerance may consider reducing equity concentration and increasing liquidity.",
  "tradable_proxies": {"EU_UCITS": ["SPYL", "CSPX", "VUAA"], "US": ["SPY", "VOO"]}
}
```

The engine analyzes SP500; `tradable_proxy_mapping` translates that into jurisdiction-appropriate
instruments. It is a mapping table, not a recommendation.

---

## Safety engineering

**No lookahead (§14.1)** — five rules, the last one automated:

1. Every value at date *t* uses only information with `release_date ≤ t`; one convention
   (`INFO_SET = "t-1"`) applied everywhere.
2. All macro reads go through `compute/findyn/pit.py::pit_join`. Nothing else may touch
   `macro_series`.
3. No centered filters or windows in the feature path.
4. Expanding-window walk-forward fitting; purged K-fold CV with a 6-month embargo; FFD `d`
   frozen from the training window.
5. **CI replay test**: recompute the full state at random historical cutoffs using only
   `release_date ≤ cutoff` and diff against stored history. Mismatch fails the build. (Lands
   with M2.)

**Graceful degradation (§14.2)** — provider failure trips a circuit breaker and falls back
live API → D1 last-known → KV cached snapshot. The API serves a stale-flagged state; it never
5xxs. Quota exhaustion on Alpha Vantage routes to Stooq/Yahoo through the same `PriceProvider`
interface and records `degraded` in `ingestion_log`.

**Explainability (§14.3)** — every regime change stores its top-N feature contributions (SHAP
for L3, posterior decomposition for L2). Crash risk is always published as its three factors.

---

## Milestones

| M | Deliverable | Acceptance | Status |
|---|---|---|---|
| M0 | Repo scaffold, wrangler config, D1 migrations, CI | Deployable bundle validates; migrations apply; CI green | ✅ **Done** |
| M1 | Provider adapters, ingestion workers, full backfill | All series backfilled with `release_date`; Yahoo isolated | ⬜ Next |
| M2 | Feature pipeline (Kalman, FFD, forces) + write-back | Features reproducible; **no-lookahead replay test passes** | ⬜ |
| M3 | HMM regime engine + XGBoost calibration | Walk-forward backtest over 2000/2008/2020/2022 | ⬜ |
| M4 | RII, crash decomposition, Monte Carlo, EVT | GPD diagnostics committed; full backtest incl. monthly era | ⬜ |
| M5 | Public API, dashboard, degradation handling | All endpoints live; kill-a-provider chaos test stays up | ⬜ |
| M6 | Documentation + runbook | Fresh clone → deployed system from the README alone | ⬜ |

### M0 status

Delivered: repo structure across both planes; nine-table D1 schema with the point-in-time key;
Hono API skeleton with a working `/health` and milestone-tagged stubs; HMAC write-back
authentication with a cross-language test vector; cron dispatch for four ingestion cadences;
strict `series.yaml` validation; **`pit_join` implemented and tested** (12 cases — it is the
safety linchpin and was worth building before anything depends on it); GitHub Actions CI for
both planes.

Verified locally: `tsc` clean, 22 serving tests, 28 compute tests, `ruff check`/`format` clean,
migrations applied via wrangler, deployable bundle validated with all five bindings resolved.
A real `wrangler deploy` needs Cloudflare credentials — `npm run deploy:check` is the
credential-free equivalent and runs in CI.

---

## Getting started

**Requirements:** Node 24+, Python 3.11+, a Cloudflare account (for deployment only).

```bash
git clone https://github.com/alexmorerich/findyn.git && cd findyn

# Serving plane
cd serving
npm install
npm run db:migrate:local      # apply schema to a local D1
npm test                      # 22 tests, inside workerd
npm run dev                   # http://localhost:8787/api/v1/health

# Compute plane
cd ../compute
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/pytest              # 28 tests
```

### Deploying

```bash
cd serving
npx wrangler login
./scripts/bootstrap.sh        # creates D1 + KV + R2, prints the ids
# paste the ids into wrangler.jsonc, then:
npm run types                 # regenerate binding types
npm run db:migrate:remote
npx wrangler secret put FRED_API_KEY      # and BLS / BEA / ALPHAVANTAGE
npx wrangler secret put ADMIN_HMAC_SECRET # openssl rand -hex 32
npm run deploy
```

Set `ADMIN_HMAC_SECRET` and `FINDYN_ADMIN_URL` as GitHub Actions secrets so the compute plane
can write results back.

### Verification

```bash
cd serving && npm run typecheck && npm run check:crons && npm test && npm run deploy:check
cd compute && .venv/bin/ruff check . && .venv/bin/ruff format --check . && .venv/bin/pytest
```

---

## Implementation decisions

Where the spec left room, these choices were made and are recorded here per its rule 5.

| Decision | Reason |
|---|---|
| Compute runs on GitHub Actions, not Cloudflare Containers | Free and zero-ops; jobs stay host-agnostic so the move is a workflow change |
| Binding types generated by `wrangler types`, `@cloudflare/workers-types` removed | Cloudflare superseded the package; CI diffs the generated file so types cannot drift from `wrangler.jsonc` |
| `compatibility_date` pinned to the newest date the bundled workerd supports | A future date fails the runtime at startup |
| Cron↔handler consistency checked by a Node script, not a test | `wrangler.jsonc` cannot be parsed inside workerd (no filesystem, no JSONC parser) |
| `pit_join` fully implemented at M0 | Everything downstream depends on it; a stub would have made M0's CI vacuous |
| Compute jobs exit `2` when their milestone has not landed | Distinguishes "not built" from "ran and found nothing" |
| Static assets wired at M5, not M0 | `assets.directory` must exist for a deploy to validate; `dashboard/dist` does not exist yet |
| Python 3.11 in CI; 3.13 works locally | Spec pins 3.11; `requires-python = ">=3.11"` keeps both viable |

---

## Disclaimer

FinDyn is a market navigation system. It estimates market position, velocity, acceleration,
instability, and regime-transition probability. It does **not** claim to predict the future
with certainty and does **not** provide investment advice. Final allocation decisions depend
on investor constraints, risk tolerance, and tax jurisdiction.
