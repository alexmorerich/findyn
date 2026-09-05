"""``C ≈ ∂²P/∂t∂Ω`` — how price dynamics respond to latent-state motion.

This is the ``A_i`` term of the design note's mapping table (§2): the coupling
between the observable block and the latent coordinate. It is a **research
feature with a documented definition, not a signal**, and nothing downstream
trades on it.

Three estimators, because the obvious one is numerically hostile and the report
has to be able to show it was not the only thing tried. Config chooses which is
``primary`` for KK3; all three are computed and all three are reported.

1a. ``ratio_coupling`` — the direct discrete approximation
----------------------------------------------------------

``C_t ≈ Δv_t / ΔΩ_t``, and the denominator passes through zero constantly. An
unguarded ratio produces ±10⁹ spikes that dominate every downstream z-score and
every correlation, so the denominator carries a floor.

The floor is a **percentile of ``|ΔΩ|`` over the expanding history** rather than
a fixed constant. Ω's scale depends on which columns survived the availability
check and on the window the projection was fitted over, so a constant would mean
a different thing in every walk-forward window — and would silently stop
binding, or start binding on everything, as the spec moved.

Below the floor the estimator emits ``NaN`` and never ``0``. Zero is a claim
about the coupling; absence is not one, and a run of zeros would bias every
correlation computed on the series toward zero while looking perfectly healthy.

**Winsorization is a variance-control decision that can destroy the signal it is
protecting.** Clipping at the expanding 1st and 99th percentiles keeps one
denominator near the floor from setting the scale for the whole series — and if
the coupling's information *lives* in its extremes, which is exactly what a
regime-transition story would predict, this removes it. There is no way to have
both, so the share of winsorized observations is published beside the series and
the report is required to show the sensitivity.

With a 25th-percentile floor roughly a quarter of observations are undefined **by
construction**. ``max_undefined_share`` is therefore not about the floor; it is
about pathology — an Ω that barely moves at all — and the estimator declines
rather than publishing a mostly-NaN column.

1b. ``regression_coupling`` — the one the report will actually use
------------------------------------------------------------------

Expanding-window OLS in closed form, the way
``engines/crypto/liquidity_beta.py`` does it, generalized from one regressor to
several::

    v_t = β0 + β1·Ω̇_t + β2·Ω_t + ε_t
    a_t = β0 + β1·Ω̇_t + β2·Ω̈_t + β3·Ω_t + ε_t

``C_t`` is the fitted ``β1`` on Ω̇ **at date t, estimated only on data up to t** —
a time series of coefficients, not one number. Both specifications are computed
and reported; ``coupling.specification`` chooses which one ``C`` is.

Closed form rather than a fit per date, for the reason that module gives: the
coefficients are a function of running sums, so the whole path costs one pass,
and — more usefully — it is exactly the normal equations, so there is no
optimizer whose tolerance could make two runs differ in the last bits. The
running sums are ``np.cumsum``, which is sequential and therefore deterministic;
a threaded reduction here would put the same ~1e-9 wobble into the coefficients
that ``regime/hmm.py`` documents for the HMM.

A near-singular Gram matrix is masked out rather than solved. Ω and Ω̇ are
strongly related early in a window, and a solve through a badly conditioned
matrix does not raise — it returns enormous coefficients that look like a
finding.

**Standard errors are Newey–West.** Both regressors are persistent and plain OLS
standard errors on persistent regressors are optimistic by a factor of several,
so a β published with an OLS t-stat would be a β published with the wrong
answer. A β with no standard error is not a finding.

The t-stat is computed on a **cadence** (config, default every 21 observations,
and always on the last date) and is ``NaN`` between. An exact daily Newey–West
path is O(n²) in the sample length because the residuals are refitted at every
date, and the quantity moves on the scale of months. Left ``NaN`` rather than
forward-filled: a chart of a forward-filled significance reading implies a test
was run on a day it was not.

1c. ``interaction_coupling`` — the sanity control
--------------------------------------------------

``C_t = Z(v_t) · Z(Ω̇_t)``, both z-scores expanding. Cheap, always defined, and
the check that 1a and 1b are not measuring an artifact of their own denominator.
If the three disagree about when the market was coupled, the report says so.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from findynamics.engines.equity.features.kinematics import baseline_window, expanding_z
from findynamics.research.omega.config import OmegaConfig

log = logging.getLogger("findynamics.research.omega.coupling")

#: The specifications ``regression_coupling`` fits, and their regressors in
#: order. The intercept is added by the estimator; ``omega_velocity`` is first
#: in both because ``C`` is its coefficient and the position is read by name
#: rather than by index, but keeping it first makes the printed tables line up.
REGRESSION_SPECIFICATIONS: dict[str, tuple[str, ...]] = {
    "velocity": ("omega_velocity", "omega"),
    "acceleration": ("omega_velocity", "omega_acceleration", "omega"),
}

#: The regressor whose coefficient *is* the coupling.
COUPLING_REGRESSOR = "omega_velocity"

#: Estimator ids, in the order the diagnostics panel prints them.
ESTIMATOR_NAMES: tuple[str, ...] = ("ratio", "regression", "interaction")


class OmegaCouplingError(ValueError):
    """Raised when a coupling estimator is configured or called incoherently.

    Distinct from *declining*: a decline is a result — the data could not
    support an estimate and the estimator says so — and comes back on
    :class:`CouplingResult`. This is a mistake.
    """


@dataclass(frozen=True)
class CouplingParams:
    """The coupling half of ``omega.yaml``."""

    #: Which estimator KK3 treats as ``C``.
    primary: str = "regression"
    #: Which regression specification the regression estimator publishes as ``C``.
    specification: str = "velocity"
    #: Percentile of ``|ΔΩ|`` below which the ratio estimator declines to divide.
    denominator_percentile: float = 25.0
    #: Expanding winsorization bounds for the ratio estimator, in percent.
    winsor_lower: float = 1.0
    winsor_upper: float = 99.0
    #: Above this share of undefined observations the ratio estimator declines.
    max_undefined_share: float = 0.50
    #: Observations before any expanding statistic is published.
    min_observations: int = 504
    #: Fixed Newey-West lag, or ``None`` for the ``floor(4·(n/100)^0.25)`` rule.
    newey_west_lag: int | None = None
    #: Observations between Newey-West evaluations.
    tstat_cadence: int = 21
    #: Reciprocal-condition floor on the Gram matrix before it is solved.
    max_condition: float = 1e10
    #: Baseline for the interaction estimator's expanding z-scores, in years.
    zscore_min_years: float = 10.0

    @classmethod
    def from_config(cls, config: OmegaConfig) -> CouplingParams:
        block = config.block("coupling")
        defaults = cls()

        primary = str(block.get("primary", defaults.primary))
        if primary not in ESTIMATOR_NAMES:
            raise OmegaCouplingError(
                f"coupling.primary must be one of {list(ESTIMATOR_NAMES)}, got {primary!r}"
            )
        specification = str(block.get("specification", defaults.specification))
        if specification not in REGRESSION_SPECIFICATIONS:
            raise OmegaCouplingError(
                f"coupling.specification must be one of "
                f"{sorted(REGRESSION_SPECIFICATIONS)}, got {specification!r}"
            )

        denominator = float(block.get("denominator_percentile", defaults.denominator_percentile))
        if not 0.0 < denominator < 100.0:
            raise OmegaCouplingError(
                f"coupling.denominator_percentile must be within (0, 100), got {denominator}"
            )
        lower = float(block.get("winsor_lower", defaults.winsor_lower))
        upper = float(block.get("winsor_upper", defaults.winsor_upper))
        if not 0.0 <= lower < upper <= 100.0:
            raise OmegaCouplingError(
                "coupling winsorization must satisfy 0 <= winsor_lower < winsor_upper <= 100, "
                f"got {lower} and {upper}"
            )

        undefined = float(block.get("max_undefined_share", defaults.max_undefined_share))
        if not 0.0 < undefined <= 1.0:
            raise OmegaCouplingError(
                f"coupling.max_undefined_share must be within (0, 1], got {undefined}"
            )

        lag = block.get("newey_west_lag")
        if lag is not None and (not isinstance(lag, int) or isinstance(lag, bool) or lag < 0):
            raise OmegaCouplingError(
                f"coupling.newey_west_lag must be a non-negative int or absent, got {lag!r}"
            )

        cadence = int(block.get("tstat_cadence", defaults.tstat_cadence))
        if cadence < 1:
            raise OmegaCouplingError(f"coupling.tstat_cadence must be >= 1, got {cadence}")

        minimum = int(block.get("min_observations", defaults.min_observations))
        if minimum < 3:
            raise OmegaCouplingError(
                f"coupling.min_observations must be >= 3, got {minimum}; a regression with "
                "an intercept and two regressors has no residual degrees of freedom below it"
            )

        return cls(
            primary=primary,
            specification=specification,
            denominator_percentile=denominator,
            winsor_lower=lower,
            winsor_upper=upper,
            max_undefined_share=undefined,
            min_observations=minimum,
            newey_west_lag=lag,
            tstat_cadence=cadence,
            max_condition=float(block.get("max_condition", defaults.max_condition)),
            zscore_min_years=float(block.get("zscore_min_years", defaults.zscore_min_years)),
        )


@dataclass(frozen=True)
class CouplingResult:
    """One estimator's coupling path, its detail columns and what it could not do."""

    estimator: str
    #: ``C_t``. Empty when the estimator declined.
    coupling: pd.Series
    #: Per-date supporting columns — coefficients, t-stats, R², sample size.
    #: Empty for estimators that have none to give.
    detail: pd.DataFrame = field(default_factory=pd.DataFrame)
    diagnostics: dict[str, float] = field(default_factory=dict)
    #: True when the data could not support an estimate. A *result*, not an error.
    declined: bool = False
    decline_reason: str = ""

    @property
    def defined_share(self) -> float:
        if self.coupling.empty:
            return 0.0
        return float(self.coupling.notna().mean())

    def latest(self) -> float | None:
        usable = self.coupling.dropna()
        return None if usable.empty else float(usable.iloc[-1])


def newey_west_lag(n: int) -> int:
    """``floor(4·(n/100)^0.25)`` — a pure function of the sample size.

    Pure and tested at three ``n`` on purpose: a lag that depended on anything
    but the count would make two runs on the same data report different
    significance. Floored at zero so a tiny sample gives plain White standard
    errors rather than a negative lag.
    """
    if n <= 0:
        return 0
    return max(int(math.floor(4.0 * (n / 100.0) ** 0.25)), 0)


def _align(**series: pd.Series) -> pd.DataFrame:
    """The named series on the dates they *all* have a finite value.

    Intersection, not union. The price kinematics run from 1927 and Ω from 2000
    on the shipped configuration, so a union index would leave three quarters of
    the frame empty and every "defined share" in the diagnostics would be a
    statement about how far back the price series reaches rather than about the
    estimator. Consecutive rows here are consecutive trading days wherever the
    inputs are daily, which is what makes a ``diff()`` over this frame a
    one-day change.
    """
    return pd.concat(
        [value.rename(name) for name, value in series.items()], axis=1, sort=True
    ).dropna(how="any")


def _expanding_quantile(series: pd.Series, percentile: float, min_periods: int) -> pd.Series:
    """The ``percentile``-th value of the history up to and including each row.

    ``expanding().quantile()`` and never ``expanding().apply()`` — the applied
    form is banned by the guard because the function it applies can close over
    anything, and this one cannot see past its own window.
    """
    return series.expanding(min_periods=max(min_periods, 2)).quantile(percentile / 100.0)


def ratio_coupling(
    velocity: pd.Series,
    omega: pd.Series,
    *,
    params: CouplingParams,
) -> CouplingResult:
    """``Δv / ΔΩ`` with an expanding-percentile denominator floor."""
    frame = _align(v=velocity, omega=omega)
    if frame.empty:
        return CouplingResult(
            estimator="ratio",
            coupling=pd.Series(dtype=float, name="coupling"),
            declined=True,
            decline_reason="v and Ω share no dates",
        )
    delta_v = frame["v"].diff()
    delta_omega = frame["omega"].diff()

    magnitude = delta_omega.abs()
    floor = _expanding_quantile(magnitude, params.denominator_percentile, params.min_observations)

    # Both legs finite and the floor established: the rows on which the
    # estimator is entitled to an opinion. Everything else is warm-up, not a
    # decline, and must not count against the undefined share.
    eligible = delta_v.notna() & delta_omega.notna() & floor.notna()
    passes = eligible & (magnitude >= floor)

    raw = pd.Series(np.nan, index=frame.index, dtype=float)
    # NaN below the floor, never 0: zero is a claim about the coupling and
    # absence is not one.
    raw[passes] = delta_v[passes] / delta_omega[passes]

    eligible_count = int(eligible.sum())
    undefined_share = float((eligible & ~passes).sum()) / eligible_count if eligible_count else 1.0

    if eligible_count == 0 or undefined_share > params.max_undefined_share:
        reason = (
            f"{undefined_share:.1%} of {eligible_count} eligible observation(s) fall below the "
            f"{params.denominator_percentile:g}th-percentile denominator floor, above the "
            f"{params.max_undefined_share:.0%} ceiling"
            if eligible_count
            else "no observation has both a finite Δv and a finite ΔΩ"
        )
        log.info("omega coupling: the ratio estimator declines — %s", reason)
        return CouplingResult(
            estimator="ratio",
            coupling=pd.Series(dtype=float, name="coupling"),
            diagnostics={
                "coupling_undefined_share": undefined_share,
                "coupling_eligible": float(eligible_count),
            },
            declined=True,
            decline_reason=reason,
        )

    lower = _expanding_quantile(raw, params.winsor_lower, params.min_observations)
    upper = _expanding_quantile(raw, params.winsor_upper, params.min_observations)
    clipped = raw.clip(lower=lower, upper=upper)
    winsorized = int(((clipped != raw) & raw.notna()).sum())

    log.info(
        "omega coupling (ratio): %d of %d eligible observation(s) defined (%.1f%% below the "
        "floor); %d winsorized at the %g/%g percentiles",
        int(clipped.notna().sum()),
        eligible_count,
        undefined_share * 100.0,
        winsorized,
        params.winsor_lower,
        params.winsor_upper,
    )
    return CouplingResult(
        estimator="ratio",
        coupling=clipped.rename("coupling"),
        detail=pd.DataFrame({"coupling_raw": raw, "coupling_floor": floor}),
        diagnostics={
            "coupling_undefined_share": undefined_share,
            "coupling_eligible": float(eligible_count),
            "coupling_winsorized_share": (
                float(winsorized) / float(max(int(raw.notna().sum()), 1))
            ),
            "coupling_defined": float(clipped.notna().sum()),
        },
    )


def _expanding_gram(design: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Running ``X'X`` and ``X'y`` — the normal equations, accumulated once.

    ``np.cumsum`` is a sequential reduction and therefore deterministic; a
    threaded one would put the same last-bit wobble into the coefficients that
    ``regime/hmm.py`` documents for the HMM's k-means initialization.
    """
    outer = design[:, :, None] * design[:, None, :]
    return np.cumsum(outer, axis=0), np.cumsum(design * target[:, None], axis=0)


def _newey_west_se(
    design: np.ndarray,
    target: np.ndarray,
    beta: np.ndarray,
    gram_inverse: np.ndarray,
    lag: int,
) -> np.ndarray:
    """Heteroskedasticity- and autocorrelation-consistent standard errors.

    ``V = (X'X)^{-1} S (X'X)^{-1}`` with the Bartlett-weighted
    ``S = Γ₀ + Σ_j (1 − j/(L+1))(Γ_j + Γ_j')``. Returns ``sqrt(diag(V))``.
    """
    residual = target - design @ beta
    weighted = design * residual[:, None]
    s = weighted.T @ weighted
    for j in range(1, lag + 1):
        if j >= len(weighted):
            break
        gamma = weighted[j:].T @ weighted[:-j]
        s = s + (1.0 - j / (lag + 1.0)) * (gamma + gamma.T)
    covariance = gram_inverse @ s @ gram_inverse
    return np.sqrt(np.clip(np.diag(covariance), 0.0, None))


def _fit_expanding(
    target: pd.Series,
    regressors: pd.DataFrame,
    *,
    params: CouplingParams,
) -> pd.DataFrame:
    """Expanding-window OLS of ``target`` on ``regressors`` plus an intercept.

    Returns one row per date with a column per coefficient, plus ``r_squared``,
    ``n``, and the Newey-West standard error and t-statistic of the coupling
    regressor on the evaluation cadence.
    """
    usable = _align(__y=target, **{name: regressors[name] for name in regressors.columns})
    names = list(regressors.columns)
    columns = ["intercept", *names]
    empty = pd.DataFrame(columns=[*columns, "r_squared", "n", "se", "tstat"], dtype=float)
    if len(usable) < params.min_observations:
        return empty

    y = usable["__y"].to_numpy(dtype=float)
    x = np.column_stack([np.ones(len(usable)), usable[names].to_numpy(dtype=float)])
    gram, cross = _expanding_gram(x, y)
    count = np.arange(1, len(usable) + 1, dtype=float)

    # A near-singular Gram does not raise, it returns enormous coefficients that
    # look like a finding. Ω and Ω̇ are strongly related early in a window, so
    # this mask is doing real work rather than guarding a theoretical case.
    with np.errstate(divide="ignore", invalid="ignore"):
        condition = np.linalg.cond(gram)
    valid = (count >= params.min_observations) & np.isfinite(condition)
    valid &= condition <= params.max_condition
    if not valid.any():
        return empty

    beta = np.full((len(usable), x.shape[1]), np.nan)
    solved = np.linalg.solve(gram[valid], cross[valid][:, :, None])[:, :, 0]
    beta[valid] = solved

    # R² from the running sums: 1 − RSS/TSS, with both accumulated the same way
    # the coefficients were so the three numbers describe one sample.
    sum_y = np.cumsum(y)
    sum_yy = np.cumsum(y * y)
    total = sum_yy - sum_y**2 / count
    fitted_ss = np.einsum("ij,ij->i", beta, cross, optimize=True) - sum_y**2 / count
    with np.errstate(invalid="ignore", divide="ignore"):
        r_squared = np.clip(fitted_ss / np.where(total > 0.0, total, np.nan), 0.0, 1.0)

    coupling_column = columns.index(COUPLING_REGRESSOR)
    se = np.full(len(usable), np.nan)
    positions = np.flatnonzero(valid)
    # The cadence, and always the last valid date — the report quotes that one.
    chosen = set(positions[:: params.tstat_cadence].tolist())
    if len(positions):
        chosen.add(int(positions[-1]))
    for position in sorted(chosen):
        window = slice(0, position + 1)
        lag = (
            params.newey_west_lag
            if params.newey_west_lag is not None
            else newey_west_lag(position + 1)
        )
        se[position] = _newey_west_se(
            x[window],
            y[window],
            beta[position],
            np.linalg.inv(gram[position]),
            lag,
        )[coupling_column]

    with np.errstate(invalid="ignore", divide="ignore"):
        tstat = beta[:, coupling_column] / np.where(se > 0.0, se, np.nan)

    out = pd.DataFrame(beta, index=usable.index, columns=columns)
    out["r_squared"] = r_squared
    out["n"] = count
    out["se"] = se
    out["tstat"] = tstat
    return out.where(pd.Series(valid, index=usable.index), other=np.nan)


def regression_coupling(
    velocity: pd.Series,
    acceleration: pd.Series,
    omega: pd.Series,
    omega_velocity: pd.Series,
    omega_acceleration: pd.Series,
    *,
    params: CouplingParams,
) -> CouplingResult:
    """Expanding-window OLS ``β1`` on Ω̇, with Newey-West t-statistics.

    Both specifications in :data:`REGRESSION_SPECIFICATIONS` are fitted and
    reported; ``coupling.specification`` chooses which one ``C`` is. Fitting
    both costs one extra pass and stops the report from being a claim about the
    one specification that happened to be configured.
    """
    available = {
        "omega": omega,
        "omega_velocity": omega_velocity,
        "omega_acceleration": omega_acceleration,
    }
    targets = {"velocity": velocity, "acceleration": acceleration}

    detail_parts: list[pd.DataFrame] = []
    diagnostics: dict[str, float] = {}
    primary: pd.DataFrame | None = None

    for name, regressor_names in REGRESSION_SPECIFICATIONS.items():
        fitted = _fit_expanding(
            targets[name],
            pd.DataFrame({key: available[key] for key in regressor_names}),
            params=params,
        )
        if name == params.specification:
            primary = fitted
        if fitted.empty:
            diagnostics[f"regression_{name}_rows"] = 0.0
            continue
        detail_parts.append(fitted.add_prefix(f"{name}__"))
        published = fitted[COUPLING_REGRESSOR].dropna()
        diagnostics[f"regression_{name}_rows"] = float(len(published))
        if not published.empty:
            diagnostics[f"regression_{name}_beta_latest"] = float(published.iloc[-1])
            r_squared = fitted["r_squared"].dropna()
            if not r_squared.empty:
                diagnostics[f"regression_{name}_r2_latest"] = float(r_squared.iloc[-1])
            tstat = fitted["tstat"].dropna()
            if not tstat.empty:
                diagnostics[f"regression_{name}_tstat_latest"] = float(tstat.iloc[-1])

    if primary is None or primary.empty:
        reason = (
            f"the {params.specification!r} specification has fewer than "
            f"{params.min_observations} complete observation(s)"
        )
        log.info("omega coupling: the regression estimator declines — %s", reason)
        return CouplingResult(
            estimator="regression",
            coupling=pd.Series(dtype=float, name="coupling"),
            diagnostics=diagnostics,
            declined=True,
            decline_reason=reason,
        )

    detail = pd.concat(detail_parts, axis=1) if detail_parts else pd.DataFrame()
    coupling = primary[COUPLING_REGRESSOR].rename("coupling")
    log.info(
        "omega coupling (regression, %s): β1 on Ω̇ over %d date(s), latest %.4g (t=%s, R²=%.3f)",
        params.specification,
        int(coupling.notna().sum()),
        diagnostics.get(f"regression_{params.specification}_beta_latest", float("nan")),
        f"{diagnostics[f'regression_{params.specification}_tstat_latest']:.2f}"
        if f"regression_{params.specification}_tstat_latest" in diagnostics
        else "n/a",
        diagnostics.get(f"regression_{params.specification}_r2_latest", float("nan")),
    )
    return CouplingResult(
        estimator="regression",
        coupling=coupling,
        detail=detail,
        diagnostics=diagnostics,
    )


def interaction_coupling(
    velocity: pd.Series,
    omega_velocity: pd.Series,
    *,
    params: CouplingParams,
    periods_per_year: float,
) -> CouplingResult:
    """``Z(v)·Z(Ω̇)``, both expanding. The control.

    Always defined wherever both legs are, which is the point: if the ratio and
    the regression disagree with this one about when the market was coupled, at
    least one of them is measuring its own denominator.
    """
    frame = _align(v=velocity, omega_velocity=omega_velocity)
    if frame.empty:
        return CouplingResult(
            estimator="interaction",
            coupling=pd.Series(dtype=float, name="coupling"),
            declined=True,
            decline_reason="v and Ω̇ share no dates",
        )

    baseline, is_short = baseline_window(
        len(frame), periods_per_year, min_years=params.zscore_min_years
    )
    if is_short:
        log.info(
            "omega coupling (interaction): %d observation(s) is under %.0f years; the "
            "z-score baseline uses %d periods instead",
            len(frame),
            params.zscore_min_years,
            baseline,
        )

    z_velocity = expanding_z(frame["v"], baseline)
    z_omega = expanding_z(frame["omega_velocity"], baseline)
    product = (z_velocity * z_omega).rename("coupling")

    return CouplingResult(
        estimator="interaction",
        coupling=product,
        detail=pd.DataFrame({"z_velocity": z_velocity, "z_omega_velocity": z_omega}),
        diagnostics={
            "interaction_baseline_periods": float(baseline),
            "interaction_baseline_is_short": float(is_short),
            "coupling_defined": float(product.notna().sum()),
        },
    )


def all_couplings(
    velocity: pd.Series,
    acceleration: pd.Series,
    omega: pd.Series,
    omega_velocity: pd.Series,
    omega_acceleration: pd.Series,
    *,
    params: CouplingParams,
    periods_per_year: float,
) -> dict[str, CouplingResult]:
    """All three estimators, keyed by :data:`ESTIMATOR_NAMES`.

    Computed together rather than on demand so the diagnostics panel and KK3
    are looking at the same three objects, and so a decline by one of them is
    visible beside the two that did not decline.
    """
    return {
        "ratio": ratio_coupling(velocity, omega, params=params),
        "regression": regression_coupling(
            velocity,
            acceleration,
            omega,
            omega_velocity,
            omega_acceleration,
            params=params,
        ),
        "interaction": interaction_coupling(
            velocity,
            omega_velocity,
            params=params,
            periods_per_year=periods_per_year,
        ),
    }


def primary_coupling(results: dict[str, CouplingResult], params: CouplingParams) -> pd.Series:
    """The series KK3 treats as ``C``, or an empty one if that estimator declined."""
    result = results.get(params.primary)
    if result is None:
        raise OmegaCouplingError(
            f"no result for the configured primary estimator {params.primary!r}; "
            f"have {sorted(results)}"
        )
    return result.coupling


__all__ = [
    "COUPLING_REGRESSOR",
    "ESTIMATOR_NAMES",
    "REGRESSION_SPECIFICATIONS",
    "CouplingParams",
    "CouplingResult",
    "OmegaCouplingError",
    "all_couplings",
    "interaction_coupling",
    "newey_west_lag",
    "primary_coupling",
    "ratio_coupling",
    "regression_coupling",
]
