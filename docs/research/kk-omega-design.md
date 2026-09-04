# KK-Ω — design note and pre-registered falsification plan

Status: **pre-registration**. Written in KK0, before any Ω value exists.
Sections §8–§10 are the binding part: they fix the statistics, the nulls and the
decision rules *before* the numbers, and KK5 quotes §8 and §9 verbatim above the
verdict so a reader can check that the rule was not moved after seeing the data.

Implemented by `compute/findynamics/research/omega/`, parameterized by
`compute/config/research/omega.yaml`, quarantined by the `Research is
quarantined` import-linter contract in `compute/pyproject.toml`.

---

## 1. Hypothesis and scope

FinDynamics represents an asset as `(P, v, a, j)` — price, and three derivatives
of the Kalman-filtered log price (`FINDYN_V1_SPEC.md` §8.1, §8.3). This track
tests whether extending that representation with a latent market-state
coordinate Ω and its dynamics

```
(P, v, a, j)  →  (P, v, a, j, Ω, Ω̇, Ω̈, C, K)
```

carries **statistically significant out-of-sample** information about future
returns, volatility, regime transitions and drawdowns.

A fifth-dimensional latent-state representation inspired by Kaluza–Klein
geometry is being tested as a quantitative modelling hypothesis. It is not a
claim that markets have a physical fifth dimension, and no sentence in this
track may assert one.

### 1.1 The five things a reader must be able to tell apart

The whole risk of a physics-inspired model is that the inspiration gets read as
evidence. So the layers are separated explicitly, and the report is required to
keep them separated:

**(a) Mathematical inspiration.** The Kaluza–Klein line element
`ds² = g_ij dx^i dx^j + φ²(dΩ + A_i dx^i)²` is used as *guidance for the model's
shape* and for nothing else. It says: carry an observable block, carry one extra
coordinate, and carry an explicit term coupling the two. That shape is the only
thing borrowed. No field equation is solved, no metric is inverted, and no
result in this track depends on the analogy being apt — if the analogy is
wrong, the arms in §8 still measure exactly what they measure.

**(b) Model assumptions.** Ω is a scalar, latent, and inferred from observables
by a linear projection (§4). Its dynamics are assumed smooth enough that a
local-linear-trend filter is the right way to read Ω̇ off it. The coupling `C` is
assumed to be a slowly varying coefficient rather than a constant. `K` is
assumed to be an interpretable composite rather than a fitted object. Each of
these is an assumption that could be wrong, and each has a diagnostic in KK2.

**(c) Empirical observables.** Nine candidate transforms of seven configured
series (§3). Nothing in this track adds a data dependency.

**(d) Fitted parameters.** The scaler mean/scale, the PC1 loading vector, the
Kalman variances on the Ω path, the expanding-OLS coupling coefficients, and —
in the optional `weights: fitted` mode only — the five curvature weights. Every
one of them is fitted on a training window and frozen forward (§6), and every
one of them is recorded per walk-forward window in `OmegaSpec`.

**(e) Results.** Out-of-sample statistics only, computed on the stitched causal
path, against the pre-registered nulls in §8. Nothing computed in-sample is a
result; it is a diagnostic, and the report labels it as one.

### 1.2 The primary falsification target

`engines/equity/rii.py::compute_rii` already publishes a composite built from
posterior entropy, confidence deficit, `|jerk_z|`, vol-of-vol, equity/bond
correlation breakdown, credit velocity and liquidity stress. Ω's candidate
feature vector (§3) overlaps that list almost completely: realized vol, vol of
vol, drawdown, dispersion, rate level change, curve slope, NFCI and the HY OAS
change are, component for component, close to the same observables the RII is
scored from.

So the research question is **not** "does Ω beat price alone". That is nearly
guaranteed — any stress composite beats four derivatives of one series — and it
is nearly worthless. The question is:

> Does Ω carry information beyond `(P, v, a, j)` **and beyond the RII the engine
> already publishes**?

Every model comparison in KK3 therefore carries an RII control arm (`A′`), and
§9 reserves a distinct verdict — `NO INCREMENTAL INFORMATION` — for the outcome
where Ω beats `A` and loses to `A′`. That outcome is the *expected* one, and
reporting it clearly is the main thing this track is for.

### 1.3 Scope boundaries

- Ω is a **latent market-state / liquidity-information coordinate**, inferred
  from observables. It is deliberately not hard-coded as "complexity",
  "entropy", "information" or any other named quantity: naming it would be
  asserting the interpretation this track exists to test.
- Ω is **not a sixth asset engine**. `core/contracts/vocab.py::ASSETS` is a
  closed 5-tuple mirrored in `serving/src/domain.ts`; nothing here touches it,
  registers an engine, or writes `asset_state` / `engine_output` /
  `derived_features` / D1.
- Outputs are states, scores, quantiles and signals. Never a price target, never
  a trade command (`FINDYN_V1_SPEC.md` §12).
- **The model is allowed to fail.** A well-evidenced `REJECTED` closes this track
  successfully. There is no tuning budget for making it pass.

---

## 2. The KK mapping table

Read the third column first. Every row is an analogy that has already been
over-read once by somebody, and the column exists so the report cannot quietly
promote a metaphor into a claim.

| KK concept | FinDynamics implementation | What it is **not** |
|---|---|---|
| 4D spacetime | the observable state `(P, v, a, j)` — filtered log price and its three derivatives | not spacetime; the four components are one series and its derivatives, not four independent directions |
| 5th dimension | the latent coordinate Ω, PC1 of a standardized causal feature block (§4) | not a physical dimension; a scalar summary of nine observables |
| metric `g_ij` | the state-space covariance of the observable block — the Kalman filter's own error covariance over `(level, slope)` | not a spacetime metric; no signature, no curvature invariants, and it is not inverted |
| `A_i` | the coupling field between observable and latent dynamics: the expanding-OLS coefficient on Ω̇ in the velocity/acceleration regression (KK2 §1b) | not a gauge field; no gauge freedom, no covariant derivative |
| `F_ij` | the cross-state interaction term, realized as `interaction_coupling` = `Z(v_t)·Z(Ω̇_t)` (KK2 §1c) | not a field strength; an antisymmetrized derivative is not defined here and is not attempted |
| `φ` | the latent-state scale — the standard deviation of Ω over the fit window, carried in `OmegaSpec.scale` and the explained-variance ratio | not a scalar field; a normalization constant per window |
| geodesic | the unforced trajectory of the fitted linear system — what `(v, Ω̇)` does when the regression residual is set to zero | not a prediction; nothing is published from it, and it appears only as a KK2 diagnostic |
| curvature | `K`, a documented instability composite of `Z(a)`, `Z(j)`, `Z(Ω̇)`, `Z(Ω̈)`, `Z(C)` (KK2 §2) | not Riemann curvature; no connection, no second derivatives of a metric — the word is a label on a composite |
| motion in the 5th dimension | a latent regime transition — `omega_regime` moving between `latent_calm`, `latent_transition`, `latent_stress` | not motion; a banding of one continuous variable's expanding percentile |

The RII precedent applies to the last row in particular. §3.1 of the spec
replaced *snap* — the fourth derivative of price — with a composite, because a
fourth derivative of a daily index is essentially all microstructure. `K` is the
same kind of object: a composite that stands for an unmeasurable quantity, and
it inherits the same obligation to be documented rather than named after the
thing it evokes.

---

## 3. Ω's observable inputs

Every series below is **already declared** in `compute/config/series.yaml`.
Adding a data dependency is out of scope for this track; if the report concludes
one is needed, it must first show that no configured series can stand in for it.

"In fixture from" is the first observation in the committed fixture
`compute/tests/fixtures/equity_prices.csv`, which is what every test and the
default research run read. Lags are the `publication_lag_days` declared in
`series.yaml`.

| Series id | Freq | Lag (d) | Declared at | In fixture from | Note |
|---|---|---|---|---|---|
| `YAHOO:^GSPC` | daily | 0 | `engines.equity.series.backfill` | **1927-12-30** | the long price path |
| `FRED:SP500` | daily | 1 | `engines.equity.series.primary` | 2016-08-01 | publication path, short |
| `FRED:NASDAQ100` | daily | 1 | `engines.equity.series.regime_proxy` | 1986-01-02 | cross-asset / dispersion proxy |
| `FRED:DGS10` | daily | 1 | `engines.equity.series.bond_yield` | 2000-01-03 | rates level |
| `FRED:T10Y3M` | daily | 1 | `engines.equity.series.curve_slope` | 2000-01-03 | curve slope |
| `FRED:NFCI` | weekly | 7 | `engines.equity.series.liquidity_stress` | 2000-01-07 | liquidity / financial conditions |
| `FRED:BAMLH0A0HYM2` | daily | 1 | `engines.equity.series.credit_spread` | **2023-08-01** | credit spread — *thin*, must be optional |

### 3.1 The conclusion this table forces

**The primary research window is 2000-01 → 2026-07, daily.** That is where the
rates, curve and liquidity columns all exist together, and it is the widest
window on which the full feature block is defined.

**The HY OAS column is available for about three years only** (2023-08 →
2026-07, 787 observations). This is the single most consequential fact in the
recon, because the obvious implementation — build the frame, `dropna()`, fit —
would silently start the research window in August 2023 and produce an Ω with no
2008 and no 2020 in it. Every subsequent number would be a statement about
thirty-five months of a bull market, and nothing in the pipeline would say so.

Therefore Ω is defined over a **configurable, availability-checked feature set**:

- Each candidate column declares itself; at each fit date the estimator counts
  the knowable rows per column and **drops** any column below
  `min_column_observations`.
- Dropped columns are named in `OmegaSpec.dropped` and logged at INFO. They are
  never imputed and never zero-filled — a zero on a stress axis means "maximally
  calm", and absence means nothing of the kind. This is the rule `compute_rii`
  already applies to its seven components ("six components, with the seventh
  named"), reused rather than reinvented.
- A column is never allowed to decide the start of the window. The window start
  is a config value; the column set adapts to it.

**Any wider window is a different specification.** The price-only Ω runs back to
1927-12-30 on `YAHOO:^GSPC` and is implemented as `PriceOnlyOmegaEstimator`
(KK1 §3). It is reported as its **own row in every table**, never merged with
the full-column specification, because the two do not have the same columns and
an average of them is a number about nothing.

### 3.2 Why these transforms and not the levels

Ω must not inherit the level of the S&P. A PC1 taken over raw levels is
dominated by whichever column has the largest variance in its own units, which
on this block is the index. So every column is a *scale-free causal transform*:
realized vol, vol of vol, `|log return|`, drawdown from a trailing running max,
a differenced rolling correlation, an n-day change in a yield, a curve level in
percentage points, the NFCI level, an n-day change in a spread. Windows are
declared in months or years and converted against each series' own
`periods_per_year`, following `features/kinematics.py`, so the same config
number means the same span on a daily and a monthly path.

---

## 4. Estimator choice

**Option A — PCA first component — is the MVP.** `PCAOmegaEstimator` fits
`sklearn.decomposition.PCA(n_components=1, svd_solver="full")` on the
window-standardized feature block and reports `Ω_t` as the standardized row
dotted with the stored loadings.

The four MVP criteria, and how the three candidates score:

| | deterministic | explainable | cheap | easy to backtest |
|---|---|---|---|---|
| **A — PCA PC1** | yes, with `svd_solver="full"` and a pinned thread pool (§7) | yes — one loading per column, printable as a table | yes — one SVD of a 9-column matrix per walk-forward window | yes — `transform` is a pure function of `(row, spec)` |
| B — HMM latent state | only with the same seed + threadpool treatment `regime/hmm.py` already documents | partly — states are unlabeled and need a documented labelling rule | moderate — EM over 200 iterations per window | harder — a refit can permute state labels, which rewrites history without changing a number |
| C — autoencoder | no — GPU/BLAS nondeterminism, initialization, and dropout all move the answer | no — a bottleneck activation has no loading table | no — a fit per walk-forward window is the whole compute budget | no — and it would need a new dependency, which §6 of the master prompt forbids in `dependencies` |

Option C is out on all four counts and on the dependency rule. Option B is the
interesting one and it is **deferred, not rejected**: `engines/equity/regime/hmm.py`
already exists, already solves the determinism problem, and already carries a
documented labelling rule, so the marginal cost of a second specification later
is small. What it does not do is give a *continuous* Ω, and Ω̇ / Ω̈ / `C` all need
one — a state index is not differentiable, and dressing one up as a coordinate
would make every derivative in the track a difference of integers.

The deciding argument for A is the sign problem in §5. A PCA loading vector is a
single object whose orientation can be pinned by one documented rule. An HMM's
five states can permute in `5! = 120` ways, and pinning *that* is a labelling
rule with its own failure modes — `regime/hmm.py` is on its third such rule, and
the two it rejected each failed on a real series. Starting the track with the
estimator that has one degree of orientation freedom rather than 120 is not
timidity; it is putting the harder problem after the first out-of-sample result.

The estimator is swappable behind a `Protocol`:

```python
class OmegaEstimator(Protocol):
    def fit(self, z: pd.DataFrame) -> OmegaSpec: ...
    def transform(self, z: pd.DataFrame, spec: OmegaSpec) -> pd.Series: ...
```

`OmegaSpec` is the frozen fit, and it round-trips through `as_dict()` /
`from_dict()` exactly as `KalmanParams` and `FfdFit` do, because KK3 stores one
per walk-forward window and the report must be able to print the loading history.

**A second estimator without a first out-of-sample result is speculation**, so
KK1 ships exactly two implementations: `PCAOmegaEstimator` and the
`PriceOnlyOmegaEstimator` that §3.1 requires for the 1927 specification. Option
B is the planned second specification, after KK3 has produced a number.

---

## 5. The sign problem

A principal component's sign is arbitrary: `(loadings, scores)` and
`(−loadings, −scores)` are the same decomposition, and which one an SVD returns
depends on the LAPACK driver, the BLAS build and the data.

Under walk-forward refits this is not a cosmetic issue, it is a **signal
factory**. Suppose the window ending 2011-06 returns `+w` and the window ending
2011-07 returns `−w`. Ω does not drift across that boundary; it reflects through
zero. Ω̇ — a derivative of Ω — then registers the largest excursion in its
history on exactly the refit date, every refit date, and `Ω̈` registers a larger
one. The resulting series is a spike train perfectly aligned to a monthly
calendar, and it would look like a signal: it is large, it is regular, it
survives winsorization, and it correlates with anything else that has monthly
structure. It would also be entirely an artifact of the linear-algebra library.

### 5.1 The pinning rule (implemented in KK1 §3, required test in KK1 §6)

> The loading on the `sign_reference` column is forced non-negative. If it is
> negative, the entire loading vector is negated. `sign_reference` defaults to
> `realized_vol`; if that column was dropped by the availability check, the
> reference falls to the next surviving column in the fixed precedence order
> declared in `omega.yaml`, and the column actually used is recorded in
> `OmegaSpec.sign_reference`.

Two consequences, both intended:

1. **Ω increases with market stress by construction.** That is an orientation
   convention, not a finding, and the report must say so wherever Ω's sign is
   interpreted. It is the same move `rii.py` makes when it fixes `direction=+1`
   so that 100 is maximally unstable — the axis is named by what it measures.
2. **Consecutive windows produce a continuous path.** The refit boundary
   introduces the ordinary discontinuity of a re-estimated model, not a
   reflection.

The precedence list is fixed and ordered in config rather than computed, because
"the column with the largest absolute loading" — the tempting alternative — is
itself unstable: two columns with near-equal loadings would hand the reference
back and forth between windows and reintroduce the flip through a side door.

### 5.2 Required tests

- **Sign stability (KK1 §6).** Fit on `[t0, t1]` and on `[t0, t1 + 250]`; assert
  both specs have the same sign on `sign_reference` and that Ω over the shared
  dates correlates > 0.9 across the two fits.
- **Window-boundary test (KK3 §6).** Over the full walk-forward path, assert no
  refit boundary produces a jump in Ω larger than a configured multiple of its
  own trailing standard deviation. This is the harness-level version, and it is
  the one that would catch a sign flip that the unit test's two windows missed.

A failure of either means every Ω̇, Ω̈, `C` and `K` result downstream is an
artifact, and the phase stops until it is fixed.

---

## 6. Anti-lookahead protocol

`FINDYN_V1_SPEC.md` §14.1 is law. Ω at date *t* is a function of data with
`release_date <= t` and nothing else. The literal sequence, per walk-forward
rebalance date `t`:

1. **Expanding window.** Take every observation with `release_date <= t`, via
   `backtest/replay.py::world_at` and a `PandasPITAccessor` bound to `t`. The
   accessor binds the cutoff, so nothing downstream can widen it. No second PIT
   gateway is built.
2. **Fit the scaler on that window.** Mean and scale are computed on the window
   and stored in `OmegaSpec.mean` / `OmegaSpec.scale`.
3. **Fit the PCA on that window**, inside `threadpool_limits(1)`, and pin the
   sign (§5).
4. **Transform forward only.** Ω on `(t, t + cadence]` uses the spec fitted at
   `t`. `transform` reads the stored mean/scale/loadings and never recomputes
   them — it is a pure function of `(row, spec)`, and KK1 asserts that.
5. **Derivatives from the stitched causal path.** Ω̇ and Ω̈ are read off the
   stitched out-of-sample Ω via the same local-linear-trend Kalman filter price
   goes through (`features/kalman.py::filter_state`, `filter()` and never
   `smooth()`), with the same burn-in as `kinematics.py` applies to velocity.

Banned outright, and greppable by the KK0 guard test:

- a full-sample `StandardScaler().fit_transform` or `PCA(...).fit_transform`;
- a centred window (`.rolling(center=True)`), a negative shift (`.shift(-n)`),
  the RTS smoother (`.smooth(`, `smoothed_state`), Savitzky–Golay;
- `.expanding(...).apply(...)`, which is causal in principle and trivially
  non-causal in practice depending on what the applied function closes over;
- an expanding-window statistic used as if it were a rolling one at the fit
  boundary.

The grep is the cheap layer. The expensive and load-bearing layer is the
**truncation test**: compute Ω over data truncated at `T`, then over data through
`T + 500`, and assert the values on all dates `<= T` are bit-identical
(`check_exact=True`), at three cutoffs spanning a calm window, 2008, and a
recent window. A source grep catches lookahead-*shaped* code; the truncation test
catches lookahead, including the kind that arrives through a library.

**Warm-up.** The first `min_train_years` (config, default 5) of rebalance dates
are discarded entirely. An Ω fitted on 200 observations is a statement about its
own start-up, exactly as `kalman_burn_in_years` is for velocity —
`kinematics.py` documents the case where the first published velocity on the
century path was **+142% annualized**, and an unfiltered start-up on Ω would be
the same artifact wearing a new name.

---

## 7. Determinism protocol

Same data + same config + same model version ⇒ byte-identical output. This is
not a nicety: KK3 stores one `OmegaSpec` per window and the report is required
to regenerate byte-identically, and `tests/engines/equity/test_reproducibility.py`
exists because "almost the same model" cost this repo a monthly job that went red
with a 409 nobody could act on (issue #6).

The rules, all inherited from `engines/equity/regime/hmm.py` rather than
reinvented:

1. **A fixed seed constant** in the module, not in config — changing it is a
   model change and belongs in a version bump.
2. **`threadpool_limits(1)` around every sklearn fit.** The seed alone does not
   buy reproducibility. `hmm.py` documents the measurement: a threaded
   floating-point reduction sums its partials in whatever order the threads
   finish, which moved fitted parameters by a relative ~1e-9 and re-serialized a
   340 kB booster. PCA's SVD is BLAS-parallel for the same reason k-means is.
3. **`svd_solver="full"`.** The randomized solver draws a random projection and
   is not bit-reproducible across BLAS builds; the block is nine columns wide, so
   there is no performance argument on the other side.
4. **`math.fsum` for every order-sensitive sum** — the renormalized curvature
   weights above all. CPython 3.12 gave `sum` Neumaier compensation and 3.11 sums
   naively, so the same expression is exactly 1.0 on a developer's interpreter
   and 0.9999999999999999 on the pinned CI one (commit `bdf5e2d`).
5. **No dict-iteration-order dependence.** Surviving columns are kept in a
   sorted, deterministic order — never a set, never insertion order.
6. **`np.random.default_rng(SEED)`** for anything stochastic, never the legacy
   global `np.random` functions.
7. **No wall clock in any artifact.** `OmegaSpec` carries `fit_start`,
   `fit_end`, `n_observations` and `model_version`, and nothing derived from
   `datetime.now()` — the failure `test_the_artifact_carries_no_wall_clock_timestamp`
   guards against for the equity engine.

Required test (KK1 §6): fit twice in the same process **and** in a subprocess,
and assert `OmegaSpec.as_dict()` serializes to identical JSON bytes.

---

## 8. Falsification plan — pre-registered

Everything below is fixed before any number exists. KK5 quotes this section
verbatim above the verdict.

**Horizons.** `h ∈ {1, 5, 21, 63}` trading days, for every question (config).

**Arms.** A (baseline `P,v,a,j`), **A′ (RII control** — `A` + the published RII
and its components), B (`A + Ω`), C (`A + Ω, Ω̇, Ω̈`), D (`A + C`), E (full).
Ridge for continuous targets, logistic for binary, regularization chosen on an
inner expanding split of the training window only.

**Statistical rules that decide whether any of these numbers mean anything:**

- **OOS R² is computed against the training-window mean**, never the test-window
  mean. The latter is lookahead and produces flattering positives.
- **Overlapping horizons destroy naive standard errors.** For `h > 1`, t-stats
  are Newey–West with `lag >= h - 1`, and the report states which correction was
  used. An uncorrected `t = 3.0` at `h = 63` is approximately noise.
- **Multiple testing.** The family is *all* tests reported in KK3 — 6 arms × 4
  horizons × 5 questions. Every p-value is reported beside a Benjamini–Hochberg
  adjusted q-value, and the family adjusted within is named. **A result that
  survives only unadjusted is reported as not surviving.**
- **The significance threshold is `q < 0.10`** throughout. Fixed here, not later.
- **Sub-period stability.** The out-of-sample path is split into 2005–2009,
  2010–2014, 2015–2019, 2020–2024, 2025– (config). Every headline statistic is
  reported per sub-period.
- **The shuffled-target control is the gate.** The full pipeline is re-run with
  the target permuted under a fixed seed. Every arm's OOS R² must be within noise
  of zero and no AUC may exceed 0.55. **If the shuffled control shows skill, the
  harness is leaking and every number in KK3 is void** — no partial credit, no
  "but the real result was stronger".

### Q1 — Does Ω carry incremental information about forward returns?

- **Target:** forward log return `r_{t+h}`.
- **Statistic:** out-of-sample R² per arm; `ΔR²` of each Ω arm against **A** and
  against **A′**; Clark–West / Diebold–Mariano on the difference of squared
  errors with Newey–West `lag >= h - 1`.
- **Null:** `ΔR² <= 0` against A′.
- **Decision rule:** Q1 **passes** iff at least one Ω arm has `ΔR² > 0` versus
  **A′** at `q < 0.10` on at least **two of the four** horizons, **and** its
  `ΔR²` versus A′ is non-negative on all four. A gain over A alone does not
  count for Q1 and is reported under §9's `NO INCREMENTAL INFORMATION`.

### Q2 — Does Ω̇ carry incremental information about forward volatility?

- **Target:** forward realized volatility `σ_{t+h}` over `(t, t+h]`.
- **Statistic:** OOS R² of `|Ω̇_t|` added to A and to A′; `ΔR²`; same corrections.
- **Null:** `ΔR² <= 0` against A′.
- **Decision rule:** Q2 **passes** iff `ΔR² > 0` versus A′ at `q < 0.10` on at
  least two of the four horizons.

### Q3 — Does Ω detect regime transitions earlier than the existing HMM?

- **Target:** binary — the existing equity HMM posterior enters `bear` or
  `crisis` within `h`. The label is read as **data**, computed causally; the HMM
  is not re-fitted by this track.
- **Statistics:** AUC, precision/recall, Brier; and the **lead/lag in trading
  days** of the first `latent_stress` call versus the first `bear|crisis` call
  per episode, using `engines/equity/backtest.py::EPISODES` — the same four
  episodes the existing report uses, not a second list.
- **Nulls:** `ΔAUC <= 0` against A′; median lead `<= 0` days.
- **Decision rule:** Q3 **passes** iff *all three* hold: (i) `ΔAUC > 0` versus A′
  at `q < 0.10` on at least two horizons; (ii) the median lead across the four
  `EPISODES` is strictly positive; and (iii) the false-alarm rate of
  `latent_stress`, measured on the same `MATERIAL_DRAWDOWN = 0.20` criterion
  `engines/equity/backtest.py::false_alarm_rate` uses, exceeds the equity
  engine's own published false-alarm rate by no more than **10 percentage
  points**. Earlier and much noisier is not earlier; it is a lower threshold.

### Q4 — Does the coupling `C` carry information its parts do not?

- **Targets:** `r_{t+h}` and `σ_{t+h}`.
- **Statistics:** OOS R² and rank IC of `C_t`; `ΔR²` of arm D against A and A′;
  and the KK2 §3 redundancy panel — Spearman of `C` against `velocity`,
  `acceleration`, `jerk_z`, the RII and each RII component.
- **Nulls:** `ΔR² <= 0` against A′; and `C` is redundant if `|ρ_s| >= 0.9`
  against any single existing published quantity.
- **Decision rule:** Q4 **passes** iff `ΔR²(D vs A′) > 0` at `q < 0.10` on at
  least two horizons **and** `C`'s largest `|ρ_s|` against an existing published
  quantity is `< 0.9`. If the correlation gate fails, the result is reported as
  a re-derivation regardless of the R².

### Q5 — Does any of it survive as a decision rule after costs?

- **Target:** a simple long/flat or scaled-exposure rule per arm.
- **Statistics:** CAGR, Sharpe, Sortino, MaxDD, Calmar, volatility, hit rate,
  turnover (**reported for every arm including the baseline**), cost-adjusted
  return at the configured 5 bps per unit turnover, and the **break-even cost**
  at which each arm's edge vanishes.
- **Metric helpers.** `findynamics/backtest/portfolio.py` supplies `_metrics`
  and `_max_drawdown`, and KK3 reuses them rather than writing a second Sharpe.
  Two caveats the recon found and which KK3 must handle rather than discover:
  they are private (`_`-prefixed), and `_metrics` is **hard-coded to a monthly
  cadence** — it annualizes with `12.0 / n` and `sqrt(12)` because the P6
  backtest rebalances monthly. It also returns only `cagr`, `vol`, `sharpe`,
  `max_drawdown`, `total_return` and `months`; **Sortino, Calmar, hit rate and
  turnover do not exist yet**. So "reuse the helpers" is only partly available,
  and KK3 has a decision to make that this note deliberately does not make for
  it: either widen `_metrics` with an explicit periods-per-year argument
  (backwards-compatible, but it edits a module the P6 backtest depends on), or
  wrap it in `research/` and compute the four missing statistics there. What
  KK3 must **not** do is write a second Sharpe with a different annualization —
  that is exactly how two reports come to disagree about the same strategy, and
  whichever route it takes, the report states which one and why.
- **Null:** cost-adjusted Sharpe of the best Ω arm `<=` that of A′.
- **Decision rule:** Q5 **passes** iff the best Ω arm's cost-adjusted Sharpe
  exceeds A′'s at 5 bps in at least **three of the five sub-periods** and its
  break-even cost is at least **10 bps**. An edge that dies at 3 bps is a
  finding, not a footnote, and is reported as a failure of Q5.

---

## 9. Verdict vocabulary

Exactly one verdict, chosen by the rules below and by nothing else. KK5 quotes
this section verbatim above the verdict it reaches.

| Verdict | Condition |
|---|---|
| **SUPPORTED** | Q1 passes, **and** at least two of Q2–Q5 pass, **and** every passing result is present in at least **three of the five** sub-periods. |
| **PARTIALLY SUPPORTED** | Q1 fails, but at least two of Q2, Q3, Q4 pass with sub-period stability of at least 3/5. Ω says nothing usable about returns and something measurable about volatility, transitions or coupling. |
| **NO INCREMENTAL INFORMATION** | Some Ω arm beats **A** at `q < 0.10`, but **no** question passes against **A′**. The information exists and `engines/equity/rii.py` already publishes it. |
| **UNSTABLE** | A question passes on the pooled out-of-sample path but the result appears in **two or fewer** of the five sub-periods, **or** the headline flips sign under the §F sensitivity table (coupling floor 25th → 40th percentile, or winsorization 1/99 → 5/95, or curvature weights `equal` → `fitted`). |
| **REJECTED** | No question passes against **either** A or A′; **or** the shuffled-target control shows skill and the cause is not found; **or** Ω's Spearman correlation against a single existing published quantity is `>= 0.95`, in which case Ω is that quantity under another name. |

Precedence, so two rules cannot both fire: **REJECTED** is checked first, then
**UNSTABLE**, then **NO INCREMENTAL INFORMATION**, then **PARTIALLY SUPPORTED**,
then **SUPPORTED**. A result cannot be promoted past a failed gate by any
argument made after the numbers exist.

---

## 10. What would make us delete this module

If this section cannot be written, the hypothesis is not falsifiable and the
track should stop at KK0. It can be written, and here it is.

**Delete `compute/findynamics/research/omega/`, `compute/config/research/`,
`compute/tests/research/` and `compute/backtests/omega/` — keeping
`docs/research/kk-omega-report.md` as the record — when any one of these is
true:**

1. **The verdict is `REJECTED`.** Nothing else is required. The report is the
   deliverable; the code was the instrument.
2. **The verdict is `NO INCREMENTAL INFORMATION` and the KK2 redundancy panel
   shows `|ρ_s(K, RII)| > 0.9`.** Then `K` is a re-derivation of the existing
   instability index computed a more expensive way, and keeping a second
   implementation of one quantity guarantees the two eventually disagree about
   what 50 means — the exact reason `rii.py` reuses `factors.compute.score_series`
   rather than writing a second scoring pipeline.
3. **The shuffled-target control shows skill** and the leak is not found within
   one phase's work. A harness that scores on permuted targets cannot be trusted
   with real ones, and a leak that resists one phase of searching is cheaper to
   delete than to keep hunting.
4. **The truncation test cannot be made to pass bit-identically.** Not "passes to
   a tolerance" — bit-identically. A research module that cannot reproduce its own
   history has no way to distinguish a model improvement from a bug, and it sits
   in a repo whose storage contract is content addressing.
5. **The verdict is `UNSTABLE` and one sensitivity table flips the headline.** A
   result that depends on the coupling denominator floor being at the 25th rather
   than the 40th percentile is a result about the floor.
6. **Twelve months after KK5 merges, the module is still `enabled: false` with no
   new result.** Dead research code is worse than absent research code: it
   accumulates maintenance in CI, it is a live import target for anyone who does
   not read the quarantine contract, and its presence implies a conclusion that
   was never reached.

**What is deliberately *not* on this list:** "Ω did not beat the RII." That is
the expected outcome (§1.2), it is a real finding, and it produces the report
that justifies the track. Deleting the code afterwards is the plan, not a
punishment.

**What deletion does not touch.** Nothing. That is the point of the quarantine:
the `Research is quarantined` contract makes `findynamics.research` unreachable
from `portfolio`, `engines`, `factors`, `core` and `data`, so `rm -r` on this
module cannot break a production import. If a future phase makes that untrue,
the deletion plan above is the thing it broke.
