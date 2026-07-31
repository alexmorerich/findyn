# P3 — FinEquity engine (the original FinDyn v1 spec, as an engine)

Copy everything below this line to the coder agent. Requires P2 merged.
This is the largest phase — split into the three sub-milestones below and
land each as its own PR (they map to the original M2/M3/M4).

---

You are working in the `findyn` repo. Read `docs/redesign/01-target-architecture.md`,
`docs/redesign/03-contracts.md`, and — for this phase, in full —
`FINDYN_V1_SPEC.md`. The v1 spec is the authoritative model specification for
this engine; the redesign changed *where* it lives
(`findynamics/engines/equity/`), not *what* it does.

**One exception to the spec's authority.** §5.1/§5.2 name sources that were
never implemented (`alphavantage`, `yahoo`, `treasury`) and a `PRICE:<symbol>`
series-id convention that no adapter accepts. Both are gone. `series.yaml` and
`core/config.py::VALID_PROVIDERS` are the current truth for data; the spec
remains authoritative for the *model*. Adding a source now means writing its
adapter first — `VALID_PROVIDERS` is asserted equal to the provider registry,
so a config-only addition fails at load, by design.

## Data — settle this before writing model code

Four series feed this engine, verified live 2026-07-31. Their spans are not
interchangeable, and the difference decides sub-milestone B:

| `series.yaml` role | series | frequency | span | obs | engine role (`prices.py`) |
|---|---|---|---|---|---|
| `primary` | `FRED:SP500` | daily | 2016-08 → now | 2,512 | `publication` |
| `backfill` | `STOOQ:^SPX` | daily | pending probe | — | `calibration` if available |
| `regime_proxy` | `FRED:NASDAQ100` | daily | 1986-01 → now | 11,994 | `calibration` otherwise |
| `deep_history` | `SHILLER:NOMINAL_PRICE` | monthly | 1871 → now | ~1,860 | `deep_history` |

Two vocabularies, deliberately: `series.yaml` roles describe *where data comes
from*, the engine roles describe *what each series is for*. `prices.py` is the
single place the former becomes the latter, and everything downstream speaks
only the engine's names.

Consequences you must design around rather than discover:

- **`FRED:SP500` is licence-capped to a rolling ~10-year window.** It contains
  no 2000, no 2008, and only the tail of 2020. It is the current-state
  backbone, not a calibration set.
- **`FRED:NASDAQ100` is the daily series with crisis history** — 1987, 2000,
  2008, 2020 all in-sample. It is a proxy, not the S&P: higher vol, tech-heavy.
  Say so wherever a fitted parameter depends on it.
- **`SHILLER:NOMINAL_PRICE` is monthly.** It is the only 1871+ source and the
  only basis for the spec's "1871+ drawdowns".
- **`STOOQ:^SPX` would be daily S&P back to ~1928** and is the ideal backbone,
  but the endpoint bot-filters some networks. A CI probe
  (`.github/workflows/compute-backfill.yml`, run it with `provider: stooq`)
  settles whether it is available. **Check that run's artifact before starting
  sub-milestone B.** If it succeeded, prefer `STOOQ:^SPX` as the daily
  calibration series and demote `regime_proxy` to a cross-check; if it failed,
  the split below stands.
- **`FRED:SP500` exists in FRED but not ALFRED**, so it carries no true
  vintages: its release dates are synthesized from the configured 1-day lag by
  `data/vintages.py`. That is expected, not a bug — but it means the rule-5
  replay test proves *lag discipline* for this series, not vintage fidelity.

Flip `enabled: true` in `config/engines/equity.yaml` when the engine wrapper
lands. That will fail `tests/core/test_config.py::test_only_engines_whose_phase_has_landed_are_enabled`,
which pins `enabled_engine_names() == ("money", "rates")` — update it in the
same commit, it is an intentional tripwire.

## Scope

Implement the S&P500 Dynamic State Engine inside `engines/equity/`, consuming
shared factors from Layer 0 and PIT data via `WorldState.series`. The nine v1
forces already exist as factors (P0); do not recompute them in the engine.

### Sub-milestone A — feature pipeline (spec M2)

- `prices.py`: resolve the four roles above into the three inputs the rest of
  the engine names — `publication` (always `primary`), `calibration`
  (`backfill` where the probe landed it, else `regime_proxy`), and
  `deep_history` — before any feature code runs. This is the only place a role
  becomes a series; nothing downstream reads `series.yaml` roles directly. The
  choice is a fixed precedence recorded in `model_version`, never "whichever
  role has rows today": test that resolution is a pure function of what D1
  holds, so ingesting a lower-precedence role cannot silently move the training
  window mid-phase. Whatever the probe recovers is an ingested D1 fact — the
  engine never calls Stooq at runtime.
- `features/kalman.py`: local-linear-trend Kalman (statsmodels), **filtered**
  estimates only — the RTS smoother is banned from the feature path.
- `features/ffd.py`: fixed-width fractional differentiation; minimum `d`
  passing ADF at 95% searched on the training window, then frozen.
- `features/kinematics.py`: velocity/acceleration from the Kalman state;
  jerk as rolling z-score of Δacceleration (thresholded indicator, never raw).
- The feature pipeline takes its series as a parameter and runs over all three
  roles, never hard-coding one: `publication` is what the engine's live state
  describes, and sub-milestone B runs the identical transform over
  `calibration`. `d` from `ffd.py` is searched and frozen **per series** — one
  shared `d` across series with different memory is a silent error.
- Savitzky-Golay must not appear anywhere under `engines/equity/` — add the
  spec's CI grep guard.
- Write-back of `derived_features` rows through the existing admin route.
- **The spec §14.1 rule-5 replay test lands here** using
  `backtest/replay.py`: recompute full feature state at random historical
  cutoffs; mismatch fails CI.

### Sub-milestone B — regime engine (spec M3)

**Fit on `calibration`, infer on `publication`.** A 5-state HMM whose `crisis`
state is fitted on 2,512 observations containing one drawdown has no crisis
regime — it has an outlier. Fit the HMM and the transition classifier on the
`calibration` series, then apply the fitted model to the `publication` feature
path for the published state. Record which series supplied the parameters in
`model_version` and in the backtest report; a reader must never have to guess
whether a number came from the S&P or a proxy.

**The transfer is only valid if the features are scale-free — prove it, don't
assume it.** When `calibration` is `FRED:NASDAQ100` it is a more volatile index
than `publication`, and a Gaussian HMM fitted in raw feature space encodes that
volatility in its state means and covariances: applied to S&P features it will
under-call `crisis` for a mechanical reason that looks like a market judgement.
Standardize every feature against a rolling window of its *own* series before
it reaches the HMM, and add a test asserting the fitted state means are within
tolerance when the pipeline is fitted on each series separately. If the probe
landed `STOOQ:^SPX`, `calibration` and `publication` are the same index and
this reduces to a sanity check — which is the main reason to prefer it.

- `regime/hmm.py`: 5-state Gaussian HMM (hmmlearn) on the FFD/kinematic
  features; states mapped to the vocabulary in `engines/equity/domain.py`
  (bull_expansion … crisis) by sorting on (mean return, vol); label-stability
  assertion across refits.
- `regime/calibrate.py`: XGBoost + isotonic calibration for P(transition to
  bear/crisis) at 3m/6m/12m; purged K-fold with 6-month embargo; SHAP
  contributions stored per prediction.
- Walk-forward backtest over 2000/2008/2020/2022 — **on `calibration`, since
  `publication` does not reach 2000 or 2008.** Report lead/lag vs NBER,
  false-alarm rate, Brier score. Commit the backtest report artifact.
  Add a monthly-resolution cross-check on `deep_history` over the same
  episodes: it is genuinely the S&P across all four, so agreement between it
  and a proxy `calibration` is what licenses the transfer. Disagreement there
  is a stop sign for sub-milestone C, not a footnote.

### Sub-milestone C — RII, crash decomposition, Monte Carlo (spec M4)

- `rii.py`: Regime Instability Index 0–100 per spec composite.
- `crash.py`: the three published factors — P(transition) from B,
  P(shock|fragile) via EVT/GPD on 1871+ drawdowns, P(transmission) fragility
  score. Never a single composite alone. The 1871+ drawdowns are
  `deep_history`, which is **monthly** — fit the GPD on monthly
  drawdown magnitudes and convert to the daily horizon explicitly, documenting
  the scaling in the docstring. Reading a monthly tail as if it were daily
  understates crash frequency by roughly the square root of 21.
- `simulate.py`: ≥10k regime-switching Monte Carlo paths per horizon with the
  shock-class overlay (taxonomy from `engines/equity/domain.py`); quantiles
  only into `forecast_distribution`; path bundles to R2.
- **Discounting and risk-free benchmark come from the money engine's
  `engine_output` via `WorldState.series`** — no engine import.

### Engine wrapper

`EquityEngine(AssetEngine)`, `name="equity"`. `predict` → `AssetState`:
`regime` = argmax HMM posterior; `expected_return` = distribution median at
the strategic horizon; `risk_score` = RII; `confidence` = posterior max;
`signals` ⊇ calibrated transition probabilities and crash factors;
`components` = top SHAP/posterior contributions. `fit` = monthly expanding
refit + `model_version` bump. Legacy v1 endpoints (`/state`, `/forces`,
`/regime`, `/instability`, `/forecast`) flip from `501` to live, backed by the
same tables, alongside `/assets/equity/state`.

## Dependencies

`statsmodels`, `hmmlearn`, `xgboost`, `shap`, `scipy` — added to
`compute/pyproject.toml` in this phase only. Keep them out of `core/`.

## Dashboard

Ship UI with **each sub-milestone** per `docs/redesign/04-ui-plan.md` §P3:
A → `/equity` kinematics page; B → regime posterior chart + transition dials;
C → RII gauge, three-bar crash decomposition, forecast fan chart, and the
home-page hero switch to market overview (status moves to `/status`).
Deploy after each sub-milestone and verify live.

## Acceptance

- Replay test green in CI (rule 5).
- Walk-forward backtest artifacts committed with the metrics table from the
  spec (§ backtesting): lead/lag, drawdown warning rate, false-alarm rate,
  Brier + reliability diagram. The report names its `calibration` series and,
  where that is a proxy, carries the `deep_history` cross-check beside it.
- `prices.py` role resolution is tested as a pure function of D1 contents, and
  the scale-free transfer assertion from sub-milestone B is green.
- All v1 non-goals hold: quantiles only, no trade commands, crash risk always
  published as three factors.
- `lint-imports` (no equity→rates/money imports), full pytest, ruff, serving
  vitest, dashboard builds.
