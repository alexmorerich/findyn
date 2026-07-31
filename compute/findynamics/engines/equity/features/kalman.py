"""Local-linear-trend Kalman filter over log price (FINDYN_V1_SPEC.md §8.1, §9 L1).

The model is the standard two-state local linear trend::

    y_t     = mu_t + eps_t                 eps_t   ~ N(0, sigma2_irregular)
    mu_t    = mu_{t-1} + beta_{t-1} + xi_t  xi_t   ~ N(0, sigma2_level)
    beta_t  = beta_{t-1} + zeta_t           zeta_t ~ N(0, sigma2_trend)

``mu`` is the filtered log price and ``beta`` is the per-period log drift the
kinematics are read off. Three variances, estimated by maximum likelihood.

**Filtered, never smoothed.** ``a_{t|t}`` conditions on observations up to and
including *t*; the RTS smoother's ``a_{t|T}`` conditions on the whole sample and
would put tomorrow's price into today's velocity. The spec bans it from the
feature path, so this module calls ``filter()`` rather than ``smooth()`` and
reads ``filtered_state`` only. A results object from ``filter()`` has no
smoothed state to reach for, which makes the rule structural rather than a
comment someone can edit away.

Estimating the variances by MLE over the whole knowable sample and then filtering
with them is expanding-window practice, not lookahead: the sample ends at the
information-set cutoff, so a value recomputed at cutoff *t* is identical to the
value published on the run whose ``as_of`` was *t*. That equality is what the
rule-5 replay test checks. It does mean the value stored for a date is not
frozen forever — a later run re-estimates the variances on more data and the
history shifts slightly. That is the correct behaviour for an expanding-window
estimator, and it is why the replay test compares against a state recomputed at
its own cutoff rather than against whatever is in D1 today.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.structural import UnobservedComponents

log = logging.getLogger("findynamics.engines.equity.features.kalman")

#: statsmodels' parameter names for this specification, in its own order.
PARAM_NAMES: tuple[str, ...] = ("sigma2.irregular", "sigma2.level", "sigma2.trend")

#: Too short a sample cannot identify three variances; MLE on it returns
#: whatever the optimizer wandered into. Two years of trading days.
MIN_OBSERVATIONS = 504

#: Optimizer iteration cap. Reached rather than converged is logged, not raised:
#: a filtered path from slightly-off variances is far more useful than no state.
DEFAULT_MAXITER = 200


class InsufficientHistoryError(ValueError):
    """Raised when a series is too short to identify the model."""


@dataclass(frozen=True)
class KalmanParams:
    """The three fitted variances, in the units of squared log price."""

    irregular: float
    level: float
    trend: float

    def to_array(self) -> np.ndarray:
        """In ``PARAM_NAMES`` order, which is the order statsmodels wants."""
        return np.array([self.irregular, self.level, self.trend], dtype=float)

    def as_dict(self) -> dict[str, float]:
        return {"irregular": self.irregular, "level": self.level, "trend": self.trend}

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> KalmanParams:
        return cls(
            irregular=float(raw["irregular"]),
            level=float(raw["level"]),
            trend=float(raw["trend"]),
        )


@dataclass(frozen=True)
class KalmanState:
    """The filtered path. Both series share the index of the input."""

    #: Filtered log price ``mu_{t|t}``.
    level: pd.Series
    #: Filtered per-period log drift ``beta_{t|t}``. Per *observation*, not per
    #: year — :mod:`.kinematics` annualizes it, because only that module knows
    #: how many observations a year of this series holds.
    slope: pd.Series
    params: KalmanParams
    #: False when the optimizer hit its iteration cap. Surfaced on the state as a
    #: confidence input rather than swallowed.
    converged: bool

    def __len__(self) -> int:
        return len(self.level)


def _model(log_price: pd.Series) -> UnobservedComponents:
    """The specification, over a positional index.

    A ``DatetimeIndex`` of trading days has no regular frequency, and statsmodels
    responds by warning on every construction and guessing a frequency for
    anything date-aware. The state space model has no use for the dates — it is a
    recursion over observations in order — so it is given a clean positional
    index and the caller reattaches the real one.
    """
    endog = pd.Series(np.asarray(log_price, dtype=float))
    return UnobservedComponents(endog, level="local linear trend")


def fit_params(
    log_price: pd.Series,
    *,
    maxiter: int = DEFAULT_MAXITER,
    label: str | None = None,
) -> tuple[KalmanParams, bool]:
    """Maximum-likelihood variances over the whole series handed in.

    Returns the parameters and whether the optimizer reported convergence. The
    caller is responsible for having clipped the series to its information set —
    this function has no notion of a cutoff and will fit whatever it is given.

    **Not converging is common here and is usually not a problem.** An index
    price is close to a random walk with a very slowly moving drift, so the
    likelihood is nearly flat in ``sigma2_trend`` and the optimizer stops on a
    line-search failure at a boundary solution — a local *linear* trend
    collapsing toward a local *level* model. The filtered path from that solution
    is perfectly usable, and it is stable: the same data returns the same
    parameters regardless of the iteration cap, which is what the replay test
    needs. The flag is carried onto the state as a confidence input rather than
    being treated as a failure.
    """
    if len(log_price) < MIN_OBSERVATIONS:
        raise InsufficientHistoryError(
            f"local linear trend needs at least {MIN_OBSERVATIONS} observations to "
            f"identify three variances, got {len(log_price)}"
        )

    model = _model(log_price)
    with warnings.catch_warnings():
        # The convergence warning is re-reported below with the reason attached;
        # statsmodels' version says only "check mle_retvals".
        warnings.simplefilter("ignore")
        result = model.fit(disp=False, maxiter=maxiter)

    retvals = result.mle_retvals if isinstance(result.mle_retvals, dict) else {}
    converged = bool(retvals.get("converged", True))
    if not converged:
        log.info(
            "kalman: MLE on %s stopped after %s iteration(s) without converging "
            "(warnflag=%s); this is the usual boundary solution for a near-random-walk "
            "price and the filtered path is still used",
            label or log_price.name or "series",
            retvals.get("iterations", "?"),
            retvals.get("warnflag", "?"),
        )

    values = np.asarray(result.params, dtype=float)
    return (
        KalmanParams(irregular=float(values[0]), level=float(values[1]), trend=float(values[2])),
        converged,
    )


def filter_state(
    log_price: pd.Series,
    params: KalmanParams | None = None,
    *,
    maxiter: int = DEFAULT_MAXITER,
    label: str | None = None,
) -> KalmanState:
    """Filtered level and slope over ``log_price``.

    ``params`` frozen from a previous fit keeps a daily run cheap and its history
    stable; ``None`` re-estimates them on the series as given, which is the
    correct expanding-window behaviour for a refit.
    """
    if log_price.empty:
        empty = pd.Series(dtype="float64", index=log_price.index)
        return KalmanState(
            level=empty,
            slope=empty.copy(),
            params=KalmanParams(0.0, 0.0, 0.0),
            converged=False,
        )

    converged = True
    if params is None:
        params, converged = fit_params(log_price, maxiter=maxiter, label=label)
    elif len(log_price) < 2:
        raise InsufficientHistoryError(
            f"the filter needs at least two observations, got {len(log_price)}"
        )

    model = _model(log_price)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # filter(), never smooth(): the smoother conditions on the whole sample.
        result = model.filter(params.to_array())

    # filtered_state is (k_states, nobs); rows are [level, trend] for this spec.
    filtered = np.asarray(result.filtered_state, dtype=float)
    index = log_price.index
    return KalmanState(
        level=pd.Series(filtered[0], index=index, name="price_filtered"),
        slope=pd.Series(filtered[1], index=index, name="slope"),
        params=params,
        converged=converged,
    )


__all__ = [
    "DEFAULT_MAXITER",
    "MIN_OBSERVATIONS",
    "PARAM_NAMES",
    "InsufficientHistoryError",
    "KalmanParams",
    "KalmanState",
    "filter_state",
    "fit_params",
]
