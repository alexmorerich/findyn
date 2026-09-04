# KK2 — KK coupling `C` and financial curvature `K`

Copy everything below this line to the coder agent. Requires KK1 merged.

---

You are working in the `findyn` repo. Read `docs/research/prompts/MASTER-kk-omega.md`
and `docs/research/kk-omega-design.md`. No-lookahead law applies.

Branch: `research/kk2-coupling`.

## Task

Implement the cross-dimensional coupling between observable price dynamics and
latent Ω dynamics, and the curvature composite built on top of it. Both are
**research features with documented definitions** — neither is a signal yet.

## 1. Coupling (`coupling.py`)

The target quantity is `C_t ≈ ∂²P/∂t∂Ω`: how much the price trend moves per unit
of latent-state motion. Implement **three** estimators, because the naive one is
numerically hostile and the report needs to show that it was not the only thing
tried.

### 1a. `ratio_coupling` — the direct discrete approximation

    C_t ≈ Δv_t / ΔΩ_t

Numerical safeguards, all in config, all documented:

- **Denominator floor.** `ΔΩ_t` passes through zero constantly; an unguarded
  ratio produces ±10^9 spikes that dominate every downstream z-score. Require
  `|ΔΩ_t| >= eps`, where `eps` is a **percentile of `|ΔΩ|` over the expanding
  history** (config, default 25th) — not a fixed constant, because Ω's scale
  depends on which columns survived §2 of KK1.
- Below the floor, emit `NaN`, never 0. Zero is a claim; NaN is the absence of
  one, and a run of zeros would bias every correlation toward zero.
- Winsorize the result at expanding percentiles (config, default 1/99). State in
  the docstring that this is a variance-control decision that *can* destroy the
  signal it is protecting, and report the share of winsorized observations.
- Report `coupling_undefined_share` in diagnostics. If it exceeds a configured
  ceiling the estimator declines rather than publishing a mostly-NaN series.

### 1b. `regression_coupling` — the estimator the report will actually use

Expanding-window OLS, closed form (five running sums, as `engines/crypto/
liquidity_beta.py` does — no statsmodels dependency needed for the point
estimate):

    v_t = β0 + β1·Ω̇_t + β2·Ω_t + ε_t
    a_t = β0 + β1·Ω̇_t + β2·Ω̈_t + β3·Ω_t + ε_t

- Expanding window with a `min_observations` floor from config. Below it,
  publish nothing.
- `C_t` = the fitted `β1` on `Ω̇` at date `t` — i.e. a **time series of
  coefficients**, each estimated only on data up to `t`. This is the coupling
  field `A_i` in the design note's mapping table.
- Publish alongside: `coupling_tstat`, `coupling_r2`, `coupling_n`. A β with no
  standard error is not a finding.
- Use Newey–West standard errors (lag from config, default `floor(4·(n/100)^0.25)`)
  because both regressors are persistent and OLS standard errors will otherwise
  be optimistic by a factor of several. `statsmodels` is already a dependency.

### 1c. `interaction_coupling` — the sanity control

    C_t = Z(v_t) · Z(Ω̇_t)

with both z-scores **expanding**. Cheap, always defined, and a useful check that
1a and 1b are not measuring an artifact of their own denominator.

All three are exposed; config chooses which is the `primary` for KK3.

## 2. Curvature (`curvature.py`)

    K_t = w1·Z(a_t) + w2·Z(j_t) + w3·Z(Ω̇_t) + w4·Z(Ω̈_t) + w5·Z(C_t)

Non-negotiable rules:

- **Every `Z` is an expanding z-score** with the same minimum-baseline convention
  as `jerk_z` (`features/kinematics.py`). A full-sample z here would leak the
  future into every historical curvature value.
- Terms enter as **magnitudes where direction is not meaningful** — follow the
  RII's treatment of jerk (`rii.py`: "jerk enters as a magnitude"). Document per
  term whether it is signed or absolute and why.
- **Weights are equal by default and are NOT fitted on the full sample.** Config
  offers exactly two modes:
  - `weights: equal` (default) — `w_i = 1/5`, the documented rule;
  - `weights: fitted` — estimated on the **first walk-forward training window
    only** and then frozen for the entire out-of-sample path, with the fitting
    window recorded in the artifact.
  Any third mode is lookahead. If you find yourself wanting to optimize weights
  on the whole history, that is the failure this rule exists to prevent.
- Missing terms degrade gracefully with renormalized weights, and the surviving
  term list is reported — the pattern `compute_rii` uses ("six components, with
  the seventh named").
- Publish `financial_curvature` and `financial_curvature_percentile` (expanding).

## 3. Diagnostics (`diagnostics.py`)

A `render_report`-style markdown renderer, following
`engines/equity/diagnostics.py`. It must produce:

- **Ω panel:** explained variance ratio per walk-forward window, loading table,
  columns dropped, sign-reference column, share of the path where the sign
  reference changed.
- **Coupling panel:** for each of the three estimators — defined share, β1 path
  with Newey–West t-stats, `R²`, winsorized share.
- **Curvature panel:** component correlation matrix, weight table, and the
  contribution decomposition on the five largest `K` dates.
- **The redundancy panel — required.** Correlation and rank correlation of Ω,
  Ω̇, `C`, `K` against **each existing published quantity**: `velocity`,
  `acceleration`, `jerk_z`, `risk_score`/RII and each RII component from
  `rii.py`. If `K` correlates > 0.9 with the RII, say so in one sentence at the
  top of the panel: *"K is a re-derivation of the existing instability index."*
  That sentence is a legitimate result and must not be softened.
- **Stationarity/scale note:** ADF or variance-ratio reading on Ω and `C`, so the
  KK3 regressions are not run on something with an obvious unit root without
  saying so.

## 4. Tests

- `ratio_coupling`: synthetic `ΔΩ` crossing zero produces NaN, not ±inf; the
  undefined share is reported; a run with a configured ceiling exceeded declines.
- `regression_coupling`: synthetic data with **known β1** recovers it to 1e-6 at
  the full window; expanding path is monotone in `n`; the coefficient at date `t`
  is unchanged when data after `t` is appended (**the truncation test again** —
  this is the leakage surface a regression introduces).
- Newey–West lag selection is a pure function of `n`, tested at three `n`.
- `curvature`: with `weights: equal` and one term missing, weights renormalize to
  sum to 1 (assert with `math.fsum`); the fitted mode records its fitting window
  and produces the *same* weights when later data is appended.
- Determinism: full `C` and `K` paths byte-identical across two processes.
- Guard: the source grep from KK0 still passes over the new modules.

## Acceptance

- [ ] Three coupling estimators implemented, each with its safeguards and
      diagnostics.
- [ ] `K` implemented with the equal-weight default and the frozen-fit mode;
      no full-sample weight fitting is reachable from config.
- [ ] Truncation test green for the expanding regression coefficients.
- [ ] Redundancy panel produced, including the Ω-vs-RII correlations.
- [ ] `ruff check`, `ruff format --check`, `pytest`, `lint-imports` green.
- [ ] Diffstat still touches nothing outside `findynamics/research/`,
      `config/research/`, `tests/research/`, `docs/research/`.
