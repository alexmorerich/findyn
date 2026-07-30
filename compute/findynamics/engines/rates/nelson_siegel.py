"""Static Nelson-Siegel fit — the curve reduced to three numbers.

A yield curve is eleven noisy quotes that move almost together. Nelson-Siegel
(1987) replaces them with three interpretable factors and one shape parameter:

    y(tau) = b0 + b1 * (1 - exp(-L*tau)) / (L*tau)
                + b2 * [ (1 - exp(-L*tau)) / (L*tau) - exp(-L*tau) ]

* ``b0`` is the **level**: the loading is 1 at every maturity, so b0 is where the
  curve settles at the long end. ``y(inf) = b0``.
* ``b1`` carries the **slope**: its loading is 1 at tau=0 and decays to 0, so
  ``y(0) = b0 + b1``. The published slope factor is ``-b1``, i.e. long minus
  short, so that a positive number means an upward-sloping curve.
* ``b2`` is the **curvature**: its loading is 0 at both ends and peaks in the
  belly, which is exactly the hump a two-factor model cannot represent.
* ``L`` (lambda) fixes *where* that hump sits. It is the only nonlinear
  parameter, and it is what makes the model awkward: refit daily, the three
  factors stop being comparable across dates because they are loadings on a
  moving basis. So lambda is chosen once over a grid (:func:`select_lambda`),
  frozen, and the betas are plain OLS given it (:func:`fit_curve`).

No lookahead is possible inside this module — it sees one cross-section at a
time and never a date. Keeping the information-set discipline is the caller's
job (:mod:`findynamics.engines.rates.curve`).
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np

#: Diebold-Li (2006) convention: curvature loading peaks at 30 months.
DEFAULT_LAMBDA = 0.609

#: Three betas need at least three points; below five the fit starts describing
#: quote noise rather than curve shape.
MIN_TENORS = 5


@dataclass(frozen=True)
class NelsonSiegelFit:
    """One day's curve, as three factors plus the residual they leave behind."""

    level: float  # b0
    slope: float  # -b1, so positive means upward-sloping
    curvature: float  # b2
    rmse: float  # percentage points
    lambda_: float
    n_tenors: int

    @property
    def beta1(self) -> float:
        """The raw b1 coefficient. ``short_rate = level + beta1``."""
        return -self.slope

    def yield_at(self, maturity: float | np.ndarray) -> float | np.ndarray:
        """Fitted yield at one or more maturities, in percent."""
        scalar = np.ndim(maturity) == 0
        loadings = design_matrix(np.atleast_1d(np.asarray(maturity, dtype=float)), self.lambda_)
        fitted = loadings @ np.array([self.level, self.beta1, self.curvature])
        return float(fitted[0]) if scalar else fitted


def design_matrix(maturities: np.ndarray, lambda_: float) -> np.ndarray:
    """The three Nelson-Siegel loadings evaluated at ``maturities``.

    Columns are [level, slope, curvature]. The slope and curvature loadings both
    contain ``(1 - exp(-L*tau)) / (L*tau)``, which is 0/0 at tau=0; the limit is
    1, and it is substituted directly rather than left to produce a NaN.
    """
    if lambda_ <= 0:
        raise ValueError(f"lambda must be positive, got {lambda_}")

    tau = np.asarray(maturities, dtype=float)
    if np.any(tau < 0):
        raise ValueError("maturities must be non-negative")

    lt = lambda_ * tau
    with np.errstate(divide="ignore", invalid="ignore"):
        decay = np.where(lt > 1e-12, (1.0 - np.exp(-lt)) / np.where(lt > 0, lt, 1.0), 1.0)

    return np.column_stack([np.ones_like(tau), decay, decay - np.exp(-lt)])


def fit_curve(
    maturities: Sequence[float] | np.ndarray,
    yields: Sequence[float] | np.ndarray,
    *,
    lambda_: float = DEFAULT_LAMBDA,
    min_tenors: int = MIN_TENORS,
) -> NelsonSiegelFit | None:
    """Least-squares fit of the three betas at a fixed ``lambda_``.

    Returns ``None`` when the cross-section is too thin or degenerate. A thin day
    (a holiday, or any date before DGS1MO began in 2001) is a normal event and
    must not abort a 25-year replay, so it is reported as "no fit" rather than
    raised.
    """
    tau = np.asarray(maturities, dtype=float)
    y = np.asarray(yields, dtype=float)
    if tau.shape != y.shape:
        raise ValueError(f"maturities and yields differ in length: {tau.shape} vs {y.shape}")

    usable = np.isfinite(tau) & np.isfinite(y)
    tau, y = tau[usable], y[usable]
    if len(tau) < min_tenors:
        return None

    design = design_matrix(tau, lambda_)
    try:
        betas, *_ = np.linalg.lstsq(design, y, rcond=None)
    except np.linalg.LinAlgError:
        return None
    if not np.all(np.isfinite(betas)):
        return None

    residuals = y - design @ betas
    rmse = float(np.sqrt(np.mean(residuals**2)))
    if not math.isfinite(rmse):
        return None

    b0, b1, b2 = (float(b) for b in betas)
    return NelsonSiegelFit(
        level=b0,
        slope=-b1,
        curvature=b2,
        rmse=rmse,
        lambda_=float(lambda_),
        n_tenors=int(len(tau)),
    )


def select_lambda(
    curves: Iterable[tuple[Sequence[float], Sequence[float]]],
    grid: Sequence[float],
    *,
    min_tenors: int = MIN_TENORS,
) -> tuple[float, dict[float, float]]:
    """Pick the ``lambda`` with the lowest in-sample RMSE across a training set.

    Selection is a property of the *window*, not of any one day: the winner is
    the value that fits the whole training period best, and it is then frozen so
    that level, slope and curvature mean the same thing on every subsequent
    date. Refitting per day would make the factor histories incomparable —
    monthly_refit is the only thing that moves it.

    Returns the chosen lambda and the full grid of mean RMSEs, so a refit can be
    audited rather than trusted.
    """
    candidates = [float(x) for x in grid if float(x) > 0]
    if not candidates:
        raise ValueError("lambda grid is empty")

    materialized = [(np.asarray(m, dtype=float), np.asarray(y, dtype=float)) for m, y in curves]
    scores: dict[float, float] = {}

    for lambda_ in candidates:
        errors = [
            fit.rmse
            for maturities, yields in materialized
            if (fit := fit_curve(maturities, yields, lambda_=lambda_, min_tenors=min_tenors))
            is not None
        ]
        if errors:
            scores[lambda_] = float(np.mean(errors))

    if not scores:
        # No day in the window produced a fit at any lambda. Returning the
        # convention beats raising: the caller keeps running on the default.
        return DEFAULT_LAMBDA, {}

    best = min(scores, key=lambda k: scores[k])
    return best, scores
