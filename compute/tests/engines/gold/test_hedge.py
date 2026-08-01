"""The hedge score.

Synthetic series carry most of this file, because the properties being asserted
are *definitional* — a perfectly negatively correlated asset must score high, an
identical one must score low — and those have exact expected values that no real
series can offer.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from findynamics.engines.gold import hedge as hedge_mod
from findynamics.engines.gold.hedge import HedgeRules
from tests.engines.gold.conftest import EQUITY_PROXY, PRICE, wide_frame

RULES = HedgeRules()


def _paths(n: int = 1500, seed: int = 5) -> tuple[pd.DatetimeIndex, pd.Series]:
    """A business-day index and an equity path with a long drawdown in the middle."""
    rng = np.random.default_rng(seed)
    index = pd.bdate_range("2010-01-04", periods=n)
    steps = rng.normal(0.0004, 0.011, size=n)
    # A 30% drawdown across the middle third, so drawdown days actually exist.
    steps[n // 3 : n // 2] -= 0.004
    return index, pd.Series(1000.0 * np.exp(np.cumsum(steps)), index=index)


# --------------------------------------------------------------------------
# Drawdown detection
# --------------------------------------------------------------------------


def test_drawdown_is_measured_from_a_trailing_peak():
    index, equity = _paths()
    flags = hedge_mod.equity_drawdown(equity, RULES)

    assert flags.dtype == bool
    assert flags.any(), "the fixture has a 30% drawdown and none was found"
    # A new all-time high is never a drawdown.
    peaks = equity[equity == equity.cummax()]
    assert not flags.reindex(peaks.index).fillna(False).any()


def test_drawdown_uses_no_future_peak():
    """The peak must be trailing: the 2008 crash cannot mark 2005 as a drawdown."""
    index, equity = _paths()
    full = hedge_mod.equity_drawdown(equity, RULES)
    early = hedge_mod.equity_drawdown(equity.iloc[:800], RULES)
    pd.testing.assert_series_equal(full.iloc[:800], early, check_names=False)


def test_no_equity_history_means_no_drawdown():
    assert hedge_mod.equity_drawdown(pd.Series(dtype=float), RULES).empty


# --------------------------------------------------------------------------
# The score's definitional endpoints
# --------------------------------------------------------------------------


def test_a_perfect_diversifier_scores_high():
    """Gold = -1 x equity on every day: conditional correlation -1, score 100."""
    index, equity = _paths()
    equity_returns = np.log(equity).diff()
    gold_returns = -equity_returns

    support = pd.Series(1.0, index=index)
    result = hedge_mod.compute(gold_returns, equity, support, RULES)

    assert result.latest_correlation() == pytest.approx(-1.0, abs=1e-6)
    assert result.latest() == pytest.approx(100.0, abs=1e-6)


def test_an_identical_asset_scores_zero():
    index, equity = _paths()
    equity_returns = np.log(equity).diff()

    support = pd.Series(0.0, index=index)
    result = hedge_mod.compute(equity_returns, equity, support, RULES)

    assert result.latest_correlation() == pytest.approx(1.0, abs=1e-6)
    assert result.latest() == pytest.approx(0.0, abs=1e-6)


def test_the_score_stays_inside_its_own_range():
    index, equity = _paths()
    rng = np.random.default_rng(3)
    gold = pd.Series(rng.normal(0.0, 0.01, len(index)), index=index)
    support = pd.Series(rng.random(len(index)), index=index)

    score = hedge_mod.compute(gold, equity, support, RULES).score.dropna()
    assert score.between(0.0, 100.0).all()


def test_the_blend_weights_do_what_they_say():
    """correlation_weight: 0 must reduce the score to the regime term exactly."""
    index, equity = _paths()
    gold = -np.log(equity).diff()
    support = pd.Series(0.25, index=index)

    rules = HedgeRules(correlation_weight=0.0, regime_weight=1.0)
    result = hedge_mod.compute(gold, equity, support, rules)
    assert result.latest() == pytest.approx(25.0, abs=1e-6)


def test_the_weights_must_be_a_convex_blend():
    with pytest.raises(ValueError, match="convex"):
        HedgeRules.from_params({"hedge": {"correlation_weight": 0.8, "regime_weight": 0.8}})


# --------------------------------------------------------------------------
# Conditioning — the point of the whole module
# --------------------------------------------------------------------------


def test_the_correlation_is_conditional_on_the_drawdown():
    """Calm-market co-movement must not reach the score.

    The fixture is built so the two answers disagree by construction: gold tracks
    equities exactly on the majority of days and inverts on the drawdown
    minority. An unconditional correlation is therefore strongly positive and the
    conditional one is -1 — which is the difference the whole module exists to
    preserve, and the one an average over both states destroys.

    The drawdown mask is supplied rather than derived, because the function under
    test takes it as an argument and deriving it here would make the assertion
    depend on ``equity_drawdown`` as well (which has its own tests above).
    """
    index, equity = _paths()
    equity_returns = np.log(equity).diff()

    drawdown = pd.Series(False, index=index)
    drawdown.iloc[-120:] = True
    assert drawdown.mean() < 0.2, "the drawdown must be the minority for this to prove anything"

    gold_returns = equity_returns.where(~drawdown, -equity_returns)

    unconditional = gold_returns.corr(equity_returns)
    correlation, days = hedge_mod.conditional_correlation(
        gold_returns, equity_returns, drawdown, RULES
    )

    assert unconditional > 0.3
    assert correlation.dropna().iloc[-1] == pytest.approx(-1.0, abs=1e-6)
    assert days.iloc[-1] >= RULES.min_drawdown_days


def test_too_few_drawdown_days_yields_no_correlation():
    """A correlation over eleven points is noise with a decimal point."""
    index, equity = _paths()
    equity_returns = np.log(equity).diff()
    drawdown = pd.Series(False, index=index)
    drawdown.iloc[-5:] = True

    correlation, days = hedge_mod.conditional_correlation(
        equity_returns, equity_returns, drawdown, RULES
    )
    assert correlation.dropna().empty
    assert days.iloc[-1] == 5


# --------------------------------------------------------------------------
# Degradation
# --------------------------------------------------------------------------


def test_without_an_equity_series_the_score_is_the_regime_term():
    index, equity = _paths()
    gold = pd.Series(0.001, index=index)
    support = pd.Series(0.8, index=index)

    result = hedge_mod.compute(gold, None, support, RULES)
    assert not result.correlation_available
    assert result.latest() == pytest.approx(80.0)


def test_an_absent_regime_reads_as_neutral_not_as_zero():
    """ "No regime yet" is not evidence that gold has stopped hedging."""
    index, equity = _paths()
    gold = -np.log(equity).diff()

    result = hedge_mod.compute(gold, equity, pd.Series(np.nan, index=index), RULES)
    assert result.regime_support.eq(0.5).all()
    assert result.latest() == pytest.approx(100.0 * (0.6 * 1.0 + 0.4 * 0.5))


# --------------------------------------------------------------------------
# On the real snapshot
# --------------------------------------------------------------------------


def test_real_gold_scores_as_a_middling_hedge(gold_observations):
    """Sanity, not calibration.

    Gold's conditional correlation with equities is genuinely close to zero and
    wanders either side of it, so the score should sit in the middle of the range
    and not be pinned at an end. A score of 0 or 100 on real data would mean the
    scale is wrong.
    """
    frame = wide_frame(gold_observations)
    gold_returns = np.log(frame[PRICE].dropna()).diff()
    support = pd.Series(0.6, index=gold_returns.index)

    result = hedge_mod.compute(gold_returns, frame[EQUITY_PROXY], support, RULES)
    assert result.correlation_available
    assert -0.6 < result.latest_correlation() < 0.6
    assert 25.0 < result.latest() < 85.0
