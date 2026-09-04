# KK1 — Ω estimator, Ω̇, Ω̈, feature schema

Copy everything below this line to the coder agent. Requires KK0 merged.

---

You are working in the `findyn` repo. Read `docs/research/prompts/MASTER-kk-omega.md`
and `docs/research/kk-omega-design.md` (written in KK0). No-lookahead law applies.

Branch: `research/kk1-omega`.

## Task

Implement the latent market-state coordinate Ω and its two derivatives inside
`findynamics/research/omega/`, causally, deterministically, and with the
estimator swappable.

## 1. Contracts (`contracts.py`)

Frozen dataclasses with strict `__post_init__` validation, in the style of
`core/contracts/state.py`:

```python
@dataclass(frozen=True)
class OmegaSpec:          # what a fit froze — travels with every artifact
    estimator: str        # "pca" for the MVP
    columns: tuple[str, ...]      # feature columns actually used, in order
    dropped: tuple[str, ...]      # columns dropped for insufficient history
    mean: tuple[float, ...]       # scaler mean, per column
    scale: tuple[float, ...]      # scaler scale, per column
    loadings: tuple[float, ...]   # PC1 loadings, sign already pinned
    explained_variance_ratio: float
    sign_reference: str           # the column whose loading pins the sign
    fit_start: date
    fit_end: date
    n_observations: int
    model_version: str

@dataclass(frozen=True)
class OmegaPath:          # the computed path at one information set
    omega: pd.Series
    omega_velocity: pd.Series
    omega_acceleration: pd.Series
    omega_volatility: pd.Series
    omega_zscore: pd.Series
    omega_regime: pd.Series       # ordered categorical, see §5
    spec: OmegaSpec
    diagnostics: dict[str, float]
```

`OmegaSpec` must round-trip through `as_dict()` / `from_dict()` exactly like
`KalmanParams` and `FfdFit` do — the walk-forward in KK3 stores one per window.

## 2. Feature vector Z_t (`features.py`)

Build Z_t **only** from series already in `config/series.yaml`, read through
`WorldState.series` / `PandasPITAccessor`. Every column is a *transform* of an
observable, and every transform must be causal and scale-free (Ω must not inherit
the level of the S&P):

Default column set (all configurable in `omega.yaml`, all optional):

| Column | Definition | Source |
|---|---|---|
| `realized_vol` | trailing realized vol of log returns, annualized, window from config | price |
| `vol_of_vol` | trailing std of `realized_vol` | derived |
| `abs_return` | \|log return\|, EWMA-smoothed | price |
| `drawdown` | current drawdown from trailing running max (expanding max, never global) | price |
| `dispersion` | rolling correlation of S&P vs NASDAQ100 returns, differenced | `FRED:NASDAQ100` |
| `rate_level_chg` | n-day change in `FRED:DGS10` | rates |
| `curve_slope` | `FRED:T10Y3M` level | rates |
| `liquidity_stress` | `FRED:NFCI`, forward-filled from its weekly release | liquidity |
| `credit_velocity` | n-day change in `FRED:BAMLH0A0HYM2` | credit — **optional, thin** |

Rules:

- **Availability check, not silent truncation.** A column with fewer than
  `min_column_observations` knowable rows at the fit date is *dropped* and named
  in `OmegaSpec.dropped`. Never let one thin column decide the start of the
  research window. Log at INFO which columns survived.
- All windows in months or years in config, converted to periods against the
  series' own frequency — the convention `features/kinematics.py` uses, so the
  same number means the same span on daily and monthly paths.
- Weekly/monthly inputs are forward-filled to the daily index **as of their
  release date**, never their observation date. The PIT accessor already enforces
  this; do not re-implement a lag.
- No column may be a function of the future. `test_no_lookahead_guards` will grep
  for the shapes, but the real proof is the KK1 §6 truncation test.

## 3. Estimator (`estimator.py`)

```python
class OmegaEstimator(Protocol):
    def fit(self, z: pd.DataFrame) -> OmegaSpec: ...
    def transform(self, z: pd.DataFrame, spec: OmegaSpec) -> pd.Series: ...
```

`PCAOmegaEstimator` is the MVP implementation:

1. Drop unavailable columns (§2), keep the survivors in a **sorted, deterministic
   order** — never a set, never dict insertion order.
2. Standardize with mean/std computed **on the fit window only**; store them in
   the spec. `transform` uses the stored values and never re-computes.
3. Fit PCA (`sklearn.decomposition.PCA(n_components=1, svd_solver="full",
   random_state=SEED)`) inside `threadpool_limits(1)`. `svd_solver="full"` on
   purpose: the randomized solver is not bit-reproducible across BLAS builds and
   this module's artifacts are compared by content.
4. **Pin the sign.** The rule, which must be in the module docstring and in the
   design note: *the loading on `sign_reference` (default `realized_vol`) is
   forced non-negative; if it is negative, negate the whole loading vector.* If
   `sign_reference` was dropped, fall back to the next column in a documented
   fixed precedence list, and record which one in `OmegaSpec.sign_reference`.
   Ω therefore increases with market stress by construction, and consecutive
   walk-forward windows produce a continuous path rather than a sign-flip spike.
5. `transform` returns `Ω_t` = the standardized row dotted with the stored
   loadings. It is a pure function of `(row, spec)` — assert this in a test.

Also implement a trivial `PriceOnlyOmegaEstimator` (PC1 of the price-derived
columns only) so KK3 has a specification that runs back to 1927 on
`YAHOO:^GSPC`. It is a *different specification*, reported separately.

Do **not** implement the HMM or autoencoder estimator in this phase. The
`Protocol` is the extension point; a second estimator without a first
out-of-sample result is speculation.

## 4. Dynamics (`dynamics.py`)

Ω̇ and Ω̈ must use the repo's existing derivative conventions, not raw finite
differences of a noisy latent series. Follow `features/kinematics.py`:

- Run Ω through the same **local linear trend Kalman filter**
  (`engines/equity/features/kalman.py::filter_state`) that price goes through.
  `Ω̇` is the filtered slope, annualized by `periods_per_year`. `Ω̈` is its first
  difference. This is not decoration — differencing a PC1 twice produces
  microstructure, which is exactly the failure that docstring documents for price.
- Apply the same **burn-in**: discard the leading `kalman_burn_in_years` of
  filtered slope, for the same reason (diffuse prior, not a market signal).
  Report `burn_in_periods` in `OmegaPath.diagnostics`.
- `omega_volatility` = trailing std of Ω̇, window from config.
- `omega_zscore` = **expanding** z-score of Ω, minimum baseline in years from
  config, exactly like `jerk_z`. Never a rolling or full-sample z.
- If you cannot reuse `filter_state` directly (its signature assumes a price
  path), extract the shared part rather than copying it, and say so in the report.
  Copying it is the one thing that guarantees the two drift apart.

## 5. `omega_regime`

A discrete reading over Ω's expanding percentile — **not** a second regime model
and not an HMM. Three states with thresholds in config, vocabulary in
`findynamics/research/omega/domain.py`:

`latent_calm | latent_transition | latent_stress`

Thresholds are on the **expanding** percentile of Ω (e.g. <60, 60–85, >85). The
docstring must state plainly that this is a display/diagnostic banding of one
continuous variable and carries no dynamics of its own; the comparison against
the real HMM regime happens in KK3 §5.

## 6. Tests (`compute/tests/research/omega/`)

**Unit**
- Z_t transforms on synthetic frames with known answers (a constant series has
  zero realized vol; a monotone ramp has zero drawdown; a single spike moves
  `abs_return` on exactly one date).
- Column dropping: a frame where `credit_velocity` has 10 rows drops it, names it
  in `spec.dropped`, and the surviving Ω is unchanged from the run without it.
- Scaler/PCA round-trip: `transform` on the fit window reproduces the fit's own
  scores to `1e-12`.
- Derivatives: on a synthetic Ω = sin(t), `Ω̇` leads Ω by ~90° and `Ω̈` is
  ~antiphase — assert on correlation signs, not exact values.
- Missing data: NaN gaps in one column do not propagate NaN into every Ω value;
  document the fill rule and test it.
- Boundary: a frame shorter than the minimum fit window raises a named error, it
  does not return a two-point Ω.

**Sign stability (the important one)**
- Fit on `[t0, t1]` and on `[t0, t1 + 250]`. Assert the two specs' loading
  vectors have the **same sign** on `sign_reference`, and that Ω over the shared
  dates correlates > 0.9 between the two fits. A test that fails here means the
  sign pinning is broken and every Ω̇ result downstream is an artifact.

**Leakage**
- **Truncation test:** compute Ω over data truncated at date `T`, then over data
  through `T + 500`. Assert the values on all dates `<= T` are **bit-identical**
  (`pd.testing.assert_series_equal(..., check_exact=True)`). This is the single
  most valuable test in the track. Run it at ≥3 cutoffs spanning a calm window, a
  crisis window (2008), and a recent window.
- Assert no NaN-filling ever pulls a value backwards: plant a future spike after
  `T` and re-run the truncation test.

**Determinism**
- Fit twice in the same process and in a subprocess (`python -c`); assert
  `OmegaSpec.as_dict()` serializes to **identical JSON bytes**, following
  `tests/engines/equity/test_reproducibility.py`.

Use the committed fixture `tests/fixtures/equity_prices.csv` — no network, no new
fixture files unless you can show the existing one cannot express the case.

## 7. Config additions (`config/research/omega.yaml`)

Every window, threshold and column list from §2–§5, with a comment per entry
explaining *why that value* in house style. Keep `enabled: false`.

## Acceptance

- [ ] `OmegaEngine`-style public interface exists: `fit`, `transform`,
      `fit_transform`, `diagnostics` (name it `OmegaEngine` in `__init__.py`,
      but it does **not** subclass `AssetEngine` and is **not** registered).
- [ ] Ω, Ω̇, Ω̈, `omega_volatility`, `omega_zscore`, `omega_regime` all produced.
- [ ] Sign pinning implemented, documented, and covered by the stability test.
- [ ] Truncation/leakage test green at ≥3 cutoffs, `check_exact=True`.
- [ ] Determinism test green across processes.
- [ ] Column-availability handling green; HY OAS drop path exercised.
- [ ] `ruff check`, `ruff format --check`, `pytest`, `lint-imports` all green.
- [ ] Production behaviour unchanged: `git diff --stat` touches nothing under
      `findynamics/{core,data,factors,engines,portfolio}/`, `jobs/`, `serving/`
      or `dashboard/`. Paste the diffstat.
