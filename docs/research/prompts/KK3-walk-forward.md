# KK3 — Walk-forward model comparison (the actual experiment)

Copy everything below this line to the coder agent. Requires KK2 merged.

---

You are working in the `findyn` repo. Read `docs/research/prompts/MASTER-kk-omega.md`
and `docs/research/kk-omega-design.md` §8 (the pre-registered falsification plan).
No-lookahead law applies, and this is the phase where breaking it would be
invisible.

Branch: `research/kk3-walkforward`.

## Task

Answer Q1–Q5 out of sample. Everything reported here must come from a model that
had never seen the observation it is being graded on.

## 1. The walk-forward harness (`backtest.py`)

Reuse `findynamics/backtest/replay.py::world_at` for the PIT gateway — do not
build a second one. At each rebalance date `t` in a fixed cadence (config,
default monthly), on an **expanding** window:

1. Build `Z` from data with `release_date <= t`.
2. Fit the scaler and the Ω estimator on that window. **Fit, then freeze.**
3. Transform forward only: Ω on `(t, t+cadence]` uses the spec fitted at `t`.
4. Compute Ω̇, Ω̈, `C`, `K` from the stitched causal path.
5. Fit the predictive model of §2 on the window; predict `(t, t+cadence]`.
6. Record `OmegaSpec`, the model coefficients, the config hash and the model
   version for that window.

The stitched out-of-sample path is the only thing any metric may be computed on.
Store it as CSV under `compute/backtests/omega/` so the report is reproducible
from a clone.

**Warm-up.** Discard the first `min_train_years` (config, default 5) of
rebalance dates entirely — an Ω fitted on 200 observations is a statement about
its own start-up, exactly as `kalman_burn_in_years` is for velocity.

## 2. The model arms

Five arms, identical in everything except the feature set:

| Arm | Features |
|---|---|
| **A — baseline** | `P, v, a, j` (jerk as `jerk_z`, matching the engine) |
| **A′ — RII control** | `P, v, a, j, RII` and its published components |
| **B** | `A + Ω` |
| **C** | `A + Ω, Ω̇, Ω̈` |
| **D** | `A + C` (coupling) |
| **E — full** | `A + Ω, Ω̇, Ω̈, C, K` |

**A′ is mandatory.** Without it, "Ω adds information over price alone" is a claim
about a strawman: `rii.py` already publishes a composite built from nearly the
same observables. Any incremental result that survives A but not A′ must be
reported as *no incremental information over what the engine already publishes*.

Estimator per arm: **ridge regression for continuous targets, logistic
regression for binary targets**, both from scikit-learn, both with the
regularization strength chosen on an **inner expanding split of the training
window only**. Not XGBoost: the arms differ by 1–5 columns and a high-variance
learner will make the comparison a study of its own tuning. If you want a
non-linear arm, add it as a clearly-labelled secondary result after the linear
one is reported.

## 3. Targets and horizons

Horizons `h ∈ {1, 5, 21, 63}` trading days (config).

| Question | Target | Statistic | Report |
|---|---|---|---|
| Q1 | forward return `r_{t+h}` | out-of-sample R², ΔR² vs A and vs A′ | per arm, per h |
| Q2 | forward realized vol `σ_{t+h}` | OOS R² of `\|Ω̇_t\|` → `σ_{t+h}`, and ΔR² | per h |
| Q3 | regime transition within `h` (label from the **existing** equity HMM posterior, read as data, computed causally) | AUC, precision/recall, Brier | per h |
| Q4 | `r_{t+h}` and `σ_{t+h}` | OOS R² and rank IC of `C_t` | per h |
| Q5 | a simple long/flat or scaled-exposure rule | CAGR, Sharpe, Sortino, MaxDD, Calmar, vol, hit rate, turnover, **cost-adjusted** return | per arm |

Rules that decide whether these numbers mean anything:

- **OOS R² is computed against the training-window mean, not the test-window
  mean.** The latter is lookahead and produces flattering positives.
- **Overlapping horizons destroy naive standard errors.** For `h > 1`, report
  Newey–West (`lag = h - 1` minimum) or Hansen–Hodrick corrected t-stats, and say
  which. An uncorrected t = 3.0 at `h = 63` is roughly noise.
- **Multiple testing.** You are running 5 arms × 4 horizons × 5 questions.
  Report a Benjamini–Hochberg adjusted q-value alongside every p-value, and state
  the family you adjusted within. A result that survives only unadjusted is
  reported as not surviving.
- **Transaction costs** for Q5: config, default 5 bps per unit turnover, and
  report the break-even cost level at which the arm's edge disappears. An edge
  that dies at 3 bps is a finding, not a footnote.
- Turnover is reported for every arm including the baseline.

## 4. Stability

Split the out-of-sample path into non-overlapping sub-periods (config, default:
2005–2009, 2010–2014, 2015–2019, 2020–2024, 2025–). Report every headline
statistic per sub-period. A result present in exactly one sub-period is
`UNSTABLE`, not `SUPPORTED` — the design note's verdict rule already says so;
enforce it in the renderer rather than leaving it to the reader.

## 5. Ω regime vs the existing HMM regime

Build the diagnostic matrix the track brief asks for, on out-of-sample data only:

| | future vol (h) | future drawdown (h) |
|---|---|---|
| HMM regime | | |
| `omega_regime` | | |
| both agree | | |
| they disagree | | |

Then answer, in prose, with numbers attached:

- **Earlier detection?** Lead/lag in trading days of the first `latent_stress`
  call vs the first `bear|crisis` call, per episode, using the episode dates
  already defined in `engines/equity/backtest.py::EPISODES`. Reuse that list;
  do not invent a second set of crisis dates.
- **Different information?** Conditional on the HMM regime, does `omega_regime`
  still separate future vol?
- **Redundant?** Cramér's V / mutual information between the two labels.
- **False positives / false negatives?** Count them against the same 20%-drawdown
  criterion `engines/equity/backtest.py` uses for its false-alarm rate, so the
  two reports are comparable.

Do not assume Ω improves the HMM. A finding that it fires later and more often is
a complete answer.

## 6. Leakage tests for this phase

The unit tests from KK1/KK2 do not cover the harness itself, which is where
leakage actually enters:

- **Harness truncation test.** Run the whole walk-forward on data truncated at
  `T`; run it again on the full data. Every recorded prediction, Ω value and
  coefficient dated `<= T` must be **bit-identical**.
- **Shuffled-target control.** Re-run the full pipeline with the target series
  randomly permuted (fixed seed) and assert every arm's OOS R² is within noise of
  zero and no AUC exceeds a configured ceiling (default 0.55). If the shuffled
  control shows skill, the harness is leaking and every other number in this
  phase is void. **This test is the acceptance gate for the phase.**
- **Future-spike test.** Insert an extreme value after `T`; assert nothing dated
  `<= T` changes.
- **Window-boundary test.** Assert that at each refit boundary the Ω path has no
  discontinuity larger than a configured multiple of its own trailing std —
  catches a sign flip or a scaler reset that the KK1 unit test missed at the
  harness level.

## 7. Job entry point

`compute/jobs/omega_research.py`, following the shape of `jobs/backtest.py`:

- reads the committed fixture by default, no network, no API keys;
- `--out` writes CSV artifacts under `compute/backtests/omega/`;
- refuses to run unless `config/research/omega.yaml` has `enabled: true` **or**
  `--force-experimental` is passed, and prints the experimental banner either way;
- is **not** wired into any cron, workflow or `[project.scripts]` entry.

## Acceptance

- [ ] Walk-forward harness implemented on top of `replay.world_at`, with per-window
      spec/coefficient/version recording.
- [ ] All five arms (including **A′**) run end to end and produce a CSV.
- [ ] Q1–Q5 statistics produced at all four horizons with multiplicity-adjusted
      q-values and autocorrelation-corrected standard errors.
- [ ] Sub-period stability table produced.
- [ ] Ω-regime vs HMM-regime matrix and lead/lag per `EPISODES` produced.
- [ ] **Shuffled-target control passes** — paste the actual numbers.
- [ ] Harness truncation test bit-identical; future-spike and window-boundary
      tests green.
- [ ] `ruff check`, `ruff format --check`, `pytest`, `lint-imports` green.
- [ ] No cron, workflow, `serving/` or `dashboard/` file touched.
