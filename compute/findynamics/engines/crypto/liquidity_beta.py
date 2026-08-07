"""Bitcoin's sensitivity to the money supply, estimated on an expanding window.

The one macro relationship this engine is willing to estimate. Bitcoin has no
cash flow and no issuer, so the standard toolkit has nothing to grip; what it
demonstrably *does* have is a long co-movement with the quantity of money, which
is at least a measurable relationship between two published series.

The model
---------

Monthly, on log changes::

    r_t = alpha + beta * dL_t + e_t

``r_t`` is bitcoin's monthly log return and ``dL_t`` the monthly log change in
global liquidity (US M2 plus total Fed assets, both in USD billions). Monthly
rather than daily because the regressor is monthly: M2 has twelve observations a
year, and a daily regression would be twenty-one copies of each of them
pretending to be independent, which inflates the t-statistic by roughly sqrt(21)
and tells you nothing the monthly regression does not.

``beta`` is published with units: a beta of 8 says a 1% month-on-month rise in
the money stock has historically accompanied an 8% bitcoin month. That is a
statement about *co-movement in a sample*, not a causal claim and not a
forecast — the engine publishes no expected return, and this coefficient is
neither an input to one nor a substitute for one.

Expanding, never rolling
------------------------

§14.1 rule 4. The estimate on date *t* uses every month up to and including *t*
and no others, so a beta published for 2017 is the beta a run in 2017 would have
computed. A rolling window would be defensible statistics and indefensible
point-in-time hygiene here — it is the expanding form that makes the replay test
able to prove anything.

The arithmetic is expanding sums rather than a regression per date: the OLS
slope is a function of five running totals, so the whole path costs one pass
instead of one fit per month, and — more usefully — it is exactly the closed
form, so there is no optimizer whose tolerance could make two runs differ in the
last bits.

The implied band
----------------

The regression is on returns, so its residuals are monthly. Turning them into a
statement about the *level* — which is what "how extended is the price" needs —
is a cumulative sum: starting from the first month of the sample and adding each
month's residual gives how far the log price has drifted from where the fitted
relationship would have carried it.

That cumulative sum is a random walk, so its band widens with the square root of
the sample::

    band_half_width(t) = band_sigma * residual_sigma(t) * sqrt(n(t))

Widening is the honest shape and not a nicety: the further from the anchor, the
less the implied level says, and a constant-width band would claim a precision
about 2024 that it earned in 2015. It is **not** a valuation and must never be
drawn as one — an R² of this size on this many monthly observations describes a
cloud, and the band is the width of the cloud rather than a target the price is
expected to return to.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

log = logging.getLogger("findynamics.engines.crypto.liquidity_beta")

MONTHS_PER_YEAR = 12


@dataclass(frozen=True)
class BetaRules:
    """Regression settings, from ``config/engines/crypto.yaml``."""

    #: Months required before a coefficient is published at all. Three years:
    #: below that the slope is a line through a handful of points and its
    #: standard error is wider than any reading anybody would take from it.
    min_observations: int = 36
    #: Residual standard deviations that make up the half-width of the implied
    #: band. 2.0 rather than 1.0 because the residuals are fat-tailed and a
    #: one-sigma band would put the price outside it a third of the time.
    band_sigma: float = 2.0
    #: USD-billions conversion for each leg. M2SL is published in billions and
    #: WALCL in millions, and adding them without this produces a composite in
    #: which the Fed's balance sheet outweighs all of M2 by a factor of a
    #: thousand — a mistake that changes the beta's sign nowhere and its meaning
    #: entirely, which is why it is config rather than a literal.
    m2_scale: float = 1.0
    central_bank_assets_scale: float = 0.001

    @classmethod
    def from_params(cls, params: dict[str, Any]) -> BetaRules:
        raw = params.get("liquidity_beta") or {}
        if not isinstance(raw, dict):
            raise ValueError("engines/crypto.yaml: 'liquidity_beta' must be a mapping")
        defaults = cls()
        return cls(
            min_observations=int(raw.get("min_observations", defaults.min_observations)),
            band_sigma=float(raw.get("band_sigma", defaults.band_sigma)),
            m2_scale=float(raw.get("m2_scale", defaults.m2_scale)),
            central_bank_assets_scale=float(
                raw.get("central_bank_assets_scale", defaults.central_bank_assets_scale)
            ),
        )


@dataclass(frozen=True)
class BetaResult:
    """The expanding regression's path, one row per month."""

    #: Slope per month. NaN before ``min_observations``.
    beta: pd.Series
    alpha: pd.Series
    r_squared: pd.Series
    #: Residual standard deviation, for the band.
    residual_sigma: pd.Series
    #: Months in each date's estimation sample.
    n_observations: pd.Series
    #: Actual minus fitted monthly return.
    residual: pd.Series
    #: Whether the composite used both legs, and which were present.
    legs: dict[str, bool]

    @property
    def empty(self) -> bool:
        return self.beta.empty

    def latest(self) -> float | None:
        """Newest published beta, or ``None`` before the sample floor."""
        if self.beta.empty:
            return None
        clean = self.beta.dropna()
        return None if clean.empty else float(clean.iloc[-1])

    def latest_r_squared(self) -> float | None:
        if self.r_squared.empty:
            return None
        clean = self.r_squared.dropna()
        return None if clean.empty else float(clean.iloc[-1])


def composite_liquidity(
    frame: pd.DataFrame,
    ids: dict[str, str],
    rules: BetaRules,
) -> tuple[pd.Series, dict[str, bool]]:
    """US M2 plus total Fed assets, in USD billions, forward-filled to daily.

    Returns ``(level, legs_present)``. A missing leg is dropped rather than
    zero-filled: a zero would be a claim that the Fed's balance sheet is empty,
    which is a much stronger statement than "we could not read it".
    """
    legs: dict[str, bool] = {}
    parts: list[pd.Series] = []

    for role, scale in (
        ("m2", rules.m2_scale),
        ("central_bank_assets", rules.central_bank_assets_scale),
    ):
        series_id = ids.get(role)
        present = bool(series_id and series_id in frame and frame[series_id].notna().any())
        legs[role] = present
        if present:
            parts.append(frame[series_id].ffill() * scale)

    if not parts:
        return pd.Series(dtype=float), legs

    total = parts[0]
    for part in parts[1:]:
        total = total.add(part, fill_value=np.nan)
    return total.dropna(), legs


def _expanding_ols(x: pd.Series, y: pd.Series, min_observations: int) -> pd.DataFrame:
    """Expanding-window OLS of ``y`` on ``x`` with an intercept.

    Closed form from running sums, so date *t*'s coefficients are a function of
    observations 1..t and of nothing else. ``expanding()`` includes the current
    row, which is correct — both values were knowable on their own date.
    """
    usable = pd.concat([x.rename("x"), y.rename("y")], axis=1).dropna()
    if usable.empty:
        return pd.DataFrame(columns=["beta", "alpha", "r_squared", "sigma", "n"], dtype=float)

    n = usable["x"].expanding().count()
    sum_x = usable["x"].expanding().sum()
    sum_y = usable["y"].expanding().sum()
    sum_xx = (usable["x"] ** 2).expanding().sum()
    sum_xy = (usable["x"] * usable["y"]).expanding().sum()
    sum_yy = (usable["y"] ** 2).expanding().sum()

    # Centred moments: S_xx = sum(x^2) - n*mean(x)^2, and likewise.
    s_xx = sum_xx - sum_x**2 / n
    s_xy = sum_xy - sum_x * sum_y / n
    s_yy = sum_yy - sum_y**2 / n

    beta = s_xy / s_xx.replace(0.0, np.nan)
    alpha = sum_y / n - beta * sum_x / n
    # Explained over total. Clipped at 0 rather than left slightly negative by
    # floating-point cancellation on a near-flat fit.
    r_squared = (beta * s_xy / s_yy.replace(0.0, np.nan)).clip(lower=0.0, upper=1.0)
    # Residual variance with the two estimated parameters accounted for.
    residual_ss = (s_yy - beta * s_xy).clip(lower=0.0)
    sigma = np.sqrt(residual_ss / (n - 2).where(n > 2))

    out = pd.DataFrame(
        {"beta": beta, "alpha": alpha, "r_squared": r_squared, "sigma": sigma, "n": n}
    )
    # Below the floor the numbers exist but mean nothing; blanking them is what
    # stops a two-point "beta" reaching a dashboard.
    return out.where(out["n"] >= min_observations)


def estimate(
    monthly_returns: pd.Series,
    liquidity_level: pd.Series,
    rules: BetaRules,
    *,
    legs: dict[str, bool] | None = None,
) -> BetaResult:
    """Expanding regression of monthly bitcoin returns on liquidity growth.

    ``monthly_returns`` and ``liquidity_level`` are both month-end indexed;
    the regressor is the log change of the level.
    """
    if monthly_returns.empty or liquidity_level.empty:
        empty = pd.Series(dtype=float)
        return BetaResult(
            beta=empty,
            alpha=empty,
            r_squared=empty,
            residual_sigma=empty,
            n_observations=empty,
            residual=empty,
            legs=legs or {},
        )

    positive = liquidity_level[liquidity_level > 0]
    growth = np.log(positive).diff().reindex(monthly_returns.index)

    fitted = _expanding_ols(growth, monthly_returns, rules.min_observations)
    if fitted.empty:
        empty = pd.Series(dtype=float)
        return BetaResult(
            beta=empty,
            alpha=empty,
            r_squared=empty,
            residual_sigma=empty,
            n_observations=empty,
            residual=empty,
            legs=legs or {},
        )

    fitted = fitted.reindex(monthly_returns.index)
    predicted = fitted["alpha"] + fitted["beta"] * growth
    residual = monthly_returns - predicted

    log.info(
        "crypto liquidity beta: %s over %s months",
        f"{fitted['beta'].dropna().iloc[-1]:.2f}"
        if fitted["beta"].notna().any()
        else "unavailable",
        f"{int(fitted['n'].dropna().iloc[-1])}" if fitted["n"].notna().any() else "0",
    )

    return BetaResult(
        beta=fitted["beta"],
        alpha=fitted["alpha"],
        r_squared=fitted["r_squared"],
        residual_sigma=fitted["sigma"],
        n_observations=fitted["n"],
        residual=residual,
        legs=legs or {},
    )


def level_deviation(result: BetaResult) -> pd.Series:
    """Cumulative residual: how far the log price has drifted from the fit.

    A level, in log points. Published as ``liquidity_residual`` so the page can
    show the thing the band is drawn around rather than only the distance from
    it.
    """
    if result.residual.empty:
        return pd.Series(dtype=float)
    # Only months inside the estimation sample contribute — before the floor
    # there are no coefficients, so there is no residual to accumulate.
    inside = result.residual.where(result.beta.notna())
    return inside.cumsum()


def band_half_width(result: BetaResult, rules: BetaRules) -> pd.Series:
    """``band_sigma * residual_sigma * sqrt(n)`` — the random walk's spread."""
    if result.residual_sigma.empty:
        return pd.Series(dtype=float)
    return rules.band_sigma * result.residual_sigma * np.sqrt(result.n_observations)


def excess_over_band(result: BetaResult, rules: BetaRules) -> pd.Series:
    """How far above the implied band the price sits, in band half-widths.

    0 inside the band or below it, growing above. One-sided on purpose: the
    speculation index asks how much momentum is in the price, and a price
    *below* what the money supply would imply is not evidence of speculation. It
    is not evidence of cheapness either, which is why this returns a distance
    rather than a signal and why nothing downstream inverts it.
    """
    deviation = level_deviation(result)
    width = band_half_width(result, rules)
    if deviation.empty or width.empty:
        return pd.Series(dtype=float)
    return (deviation / width.replace(0.0, np.nan)).clip(lower=0.0)


__all__ = [
    "MONTHS_PER_YEAR",
    "BetaResult",
    "BetaRules",
    "band_half_width",
    "composite_liquidity",
    "estimate",
    "excess_over_band",
    "level_deviation",
]
