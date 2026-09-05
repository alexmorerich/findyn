"""Forward-looking **labels**. The one module in this package allowed to shift back.

Everything else here is a feature path and is bound by ``FINDYN_V1_SPEC.md``
§14.1 rule 3: no centred windows, no negative shifts, nothing at date *t* that
depends on a row after *t*. This module is the label path, which is a different
path, and grading a call without knowing what happened next is not possible.

The separation is enforced rather than asserted. ``TARGET_MODULES`` in
``tests/research/omega/test_no_lookahead_guards.py`` names this file, and
``TARGET_EXEMPT`` names the single pattern it may trip — ``.shift(-n)``. It is
still bound by the other seven: a label module has a reason to look forward and
no reason at all to fit a scaler on the whole sample or to reach for the RTS
smoother, and a file-level exemption would have handed it those for free.

The obligation that comes with the exemption
--------------------------------------------

A target must never reach a feature. The walk-forward fits its arms on features
dated ``<= t`` against targets that were **fully realized by t** — a target at
``t - h + 1`` is not yet observable at *t* and is dropped, not imputed. That is
the rule ``regime/calibrate.py`` states for its transition labels: "rows near the
end of the sample whose horizon runs past the data are dropped rather than
labelled 0, because 'no transition was observed' and 'the window has not
happened yet' are different statements, and conflating them teaches the model
that the present is always safe."

:func:`realized_by` is how that rule is applied here, and the walk-forward calls
it rather than trimming by hand.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger("findynamics.research.omega.targets")


def forward_return(log_price: pd.Series, horizon: int) -> pd.Series:
    """Log return over the ``horizon`` observations *after* each date.

    ``r_{t→t+h}``, so the value on row *t* describes what happened next and is
    unknowable at *t*. That is the point of a target.
    """
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}")
    return (log_price.shift(-horizon) - log_price).rename(f"forward_return_{horizon}")


def forward_volatility(
    log_price: pd.Series,
    horizon: int,
    *,
    periods_per_year: float,
) -> pd.Series:
    """Annualized realized volatility of the returns over ``(t, t+h]``.

    A minimum of two returns, so ``h = 1`` has no volatility target and comes
    back empty rather than as a column of zeros — one return has no dispersion,
    and calling that zero volatility would tell a model that every single day is
    perfectly calm.
    """
    if horizon < 2:
        return pd.Series(np.nan, index=log_price.index, name=f"forward_vol_{horizon}")
    returns = log_price.diff()
    # Reverse the series, take a *trailing* window, reverse back: the result on
    # row t is the standard deviation of rows t+1..t+h. Written this way rather
    # than with a forward rolling window because pandas has none, and it is the
    # construction `engines/equity/backtest.py::false_alarm_rate` uses.
    forward = returns[::-1].rolling(window=horizon, min_periods=horizon).std()[::-1].shift(-1)
    return (forward * np.sqrt(periods_per_year)).rename(f"forward_vol_{horizon}")


def forward_drawdown(log_price: pd.Series, horizon: int) -> pd.Series:
    """Worst decline from today's level over the next ``horizon`` observations.

    Non-negative, in simple-return units, so 0.2 is a 20% drawdown — the same
    scale ``engines/equity/backtest.py::MATERIAL_DRAWDOWN`` is quoted on, so the
    two reports can be compared without a conversion in between.
    """
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}")
    trough = log_price[::-1].rolling(window=horizon, min_periods=1).min()[::-1].shift(-1)
    return (1.0 - np.exp(trough - log_price)).clip(lower=0.0).rename(f"forward_drawdown_{horizon}")


def transition_within(labels: pd.Series, horizon: int, adverse: frozenset[str]) -> pd.Series:
    """1 where an adverse regime is *entered* within the next ``horizon`` rows.

    An **entry**, not a state: a market already in crisis has transitioned, and
    labelling every day of it as a fresh transition would teach a classifier to
    predict the present. ``regime/calibrate.py`` makes the same distinction and
    for the same reason.
    """
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}")
    state = labels.isin(adverse)
    entry = (state & ~state.shift(1, fill_value=False)).astype(float)
    forward = entry[::-1].rolling(window=horizon, min_periods=horizon).max()[::-1].shift(-1)
    return forward.rename(f"transition_{horizon}")


def realized_by(target: pd.Series, cutoff: pd.Timestamp, horizon: int) -> pd.Series:
    """``target`` restricted to rows whose horizon had closed by ``cutoff``.

    The rule that keeps a target out of a fit that could not have seen it. A row
    at ``t`` describes ``(t, t+h]``, so it is knowable only once ``t+h`` has
    happened; rows nearer the cutoff than ``h`` observations are **dropped**,
    never imputed.
    """
    usable = target.loc[:cutoff].dropna()
    if len(usable) <= horizon:
        return usable.iloc[:0]
    return usable.iloc[:-horizon]


__all__ = [
    "forward_drawdown",
    "forward_return",
    "forward_volatility",
    "realized_by",
    "transition_within",
]
