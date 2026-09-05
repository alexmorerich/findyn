"""The label path — the one module allowed to look forward, and its obligation.

Every function here is unknowable at the row it describes. The tests are about
alignment: a target that is off by one is a target that leaks a day, and it
would not look wrong in a chart.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from findynamics.research.omega.targets import (
    forward_drawdown,
    forward_return,
    forward_volatility,
    realized_by,
    transition_within,
)


def _series(values) -> pd.Series:
    return pd.Series(values, index=pd.bdate_range(date(2020, 1, 1), periods=len(values)))


def test_the_forward_return_describes_what_happens_next():
    log_price = _series(np.log([100.0, 110.0, 121.0, 133.1]))

    forward = forward_return(log_price, 1)

    assert forward.iloc[0] == pytest.approx(np.log(1.1))
    # The last row has no next observation and is NaN rather than zero.
    assert np.isnan(forward.iloc[-1])


def test_the_forward_return_is_never_knowable_at_its_own_row():
    """A level shift shows up on the row *before* it, never on or after it."""
    log_price = _series(np.log([100.0] * 5 + [200.0] * 5))

    forward = forward_return(log_price, 1)

    assert forward.iloc[4] == pytest.approx(np.log(2.0))
    assert forward.iloc[5] == pytest.approx(0.0)


def test_a_one_day_horizon_has_no_volatility_target():
    """One return has no dispersion, and calling that zero would be a claim.

    The alternative — a column of zeros — teaches a model that every single day
    is perfectly calm, and it is why the Q2 decision rule counts the horizons it
    was actually tested at.
    """
    log_price = _series(np.log(np.linspace(100.0, 120.0, 50)))

    assert forward_volatility(log_price, 1, periods_per_year=252.0).isna().all()
    assert forward_volatility(log_price, 5, periods_per_year=252.0).notna().any()


def test_the_forward_drawdown_is_non_negative_and_on_the_engine_scale():
    """0.2 means a 20% drawdown, matching ``MATERIAL_DRAWDOWN``.

    Same units as ``engines/equity/backtest.py``, so Ω's false-alarm rate and
    the engine's are comparable without a conversion in between.
    """
    log_price = _series(np.log([100.0, 100.0, 80.0, 100.0, 100.0]))

    drawdown = forward_drawdown(log_price, 2)

    assert (drawdown.dropna() >= 0.0).all()
    assert drawdown.iloc[0] == pytest.approx(0.2, abs=1e-9)
    assert drawdown.iloc[2] == pytest.approx(0.0)


def test_a_transition_label_marks_the_entry_and_not_the_state():
    """A market already in crisis has transitioned.

    Labelling every day of it would teach a classifier to predict the present,
    which is the distinction ``regime/calibrate.py`` draws for the same reason.
    """
    labels = pd.Series(
        ["bull_expansion"] * 3 + ["crisis"] * 4 + ["bull_expansion"] * 3,
        index=pd.bdate_range(date(2020, 1, 1), periods=10),
    )

    within = transition_within(labels, 2, frozenset({"crisis"}))

    # The entry is on row 3, so rows 1 and 2 see it coming.
    assert within.iloc[1] == pytest.approx(1.0)
    assert within.iloc[2] == pytest.approx(1.0)
    # Row 4 is inside the crisis and there is no *fresh* entry ahead of it.
    assert within.iloc[4] == pytest.approx(0.0)


def test_realized_by_drops_rows_whose_horizon_has_not_closed():
    """Not imputed — dropped. "No transition observed" and "the window has not
    happened yet" are different statements."""
    target = _series(np.arange(20.0))
    cutoff = target.index[14]

    usable = realized_by(target, cutoff, horizon=5)

    assert usable.index[-1] == target.index[9]
    assert len(usable) == 10


def test_realized_by_returns_nothing_when_the_horizon_exceeds_the_sample():
    target = _series(np.arange(4.0))

    assert realized_by(target, target.index[-1], horizon=10).empty


@pytest.mark.parametrize("horizon", [0, -1])
def test_a_non_positive_horizon_is_refused(horizon):
    log_price = _series(np.log(np.linspace(100.0, 110.0, 20)))

    with pytest.raises(ValueError, match="horizon must be >= 1"):
        forward_return(log_price, horizon)
