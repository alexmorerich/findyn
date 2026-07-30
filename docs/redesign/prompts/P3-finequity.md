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

## Scope

Implement the S&P500 Dynamic State Engine inside `engines/equity/`, consuming
shared factors from Layer 0 and PIT data via `WorldState.series`. The nine v1
forces already exist as factors (P0); do not recompute them in the engine.

### Sub-milestone A — feature pipeline (spec M2)

- `features/kalman.py`: local-linear-trend Kalman (statsmodels), **filtered**
  estimates only — the RTS smoother is banned from the feature path.
- `features/ffd.py`: fixed-width fractional differentiation; minimum `d`
  passing ADF at 95% searched on the training window, then frozen.
- `features/kinematics.py`: velocity/acceleration from the Kalman state;
  jerk as rolling z-score of Δacceleration (thresholded indicator, never raw).
- Savitzky-Golay must not appear anywhere under `engines/equity/` — add the
  spec's CI grep guard.
- Write-back of `derived_features` rows through the existing admin route.
- **The spec §14.1 rule-5 replay test lands here** using
  `backtest/replay.py`: recompute full feature state at random historical
  cutoffs; mismatch fails CI.

### Sub-milestone B — regime engine (spec M3)

- `regime/hmm.py`: 5-state Gaussian HMM (hmmlearn) on the FFD/kinematic
  features; states mapped to the vocabulary in `engines/equity/domain.py`
  (bull_expansion … crisis) by sorting on (mean return, vol); label-stability
  assertion across refits.
- `regime/calibrate.py`: XGBoost + isotonic calibration for P(transition to
  bear/crisis) at 3m/6m/12m; purged K-fold with 6-month embargo; SHAP
  contributions stored per prediction.
- Walk-forward backtest over 2000/2008/2020/2022; report lead/lag vs NBER,
  false-alarm rate, Brier score. Commit the backtest report artifact.

### Sub-milestone C — RII, crash decomposition, Monte Carlo (spec M4)

- `rii.py`: Regime Instability Index 0–100 per spec composite.
- `crash.py`: the three published factors — P(transition) from B,
  P(shock|fragile) via EVT/GPD on 1871+ drawdowns, P(transmission) fragility
  score. Never a single composite alone.
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
  Brier + reliability diagram.
- All v1 non-goals hold: quantiles only, no trade commands, crash risk always
  published as three factors.
- `lint-imports` (no equity→rates/money imports), full pytest, ruff, serving
  vitest, dashboard builds.
