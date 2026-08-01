"""Lee-Mykland jump detection.

The synthetic fixtures carry the load here, because they are the only inputs
whose true jump set is known. History can show the detector fires on 15 April
2013, which is evidence it is not broken; only a planted jump can show it finds
the ones that are there and does not invent ones that are not.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from findynamics.engines.gold import jumps as jumps_mod
from findynamics.engines.gold.jumps import JumpRules
from tests.engines.gold.conftest import PRICE, jump_diffusion, wide_frame

RULES = JumpRules()


# --------------------------------------------------------------------------
# The statistic
# --------------------------------------------------------------------------


def test_gumbel_threshold_grows_with_the_sample():
    """The maximum of n draws grows with n, so the bar has to as well.

    This is the property that makes the detector usable at all: a fixed 3-sigma
    cut over 14,000 daily returns admits ~40 jumps by arithmetic alone.
    """
    small = jumps_mod.gumbel_threshold(250, 0.01)
    large = jumps_mod.gumbel_threshold(14000, 0.01)
    assert 3.0 < small < large < 8.0


def test_gumbel_threshold_tightens_as_alpha_falls():
    assert jumps_mod.gumbel_threshold(1000, 0.05) < jumps_mod.gumbel_threshold(1000, 0.01)


def test_gumbel_threshold_rejects_an_impossible_alpha():
    with pytest.raises(ValueError, match="alpha"):
        jumps_mod.gumbel_threshold(1000, 1.5)


def test_bipower_sigma_ignores_the_jump_it_is_scaling():
    """The whole reason for bipower rather than realized variance.

    A sum of squares is dominated by the largest observation in its window, so
    the day of a crash raises its own threshold and hides itself. A product of
    adjacent absolute returns is barely moved, because the jump is multiplied by
    an ordinary return in both terms it enters.
    """
    returns = pd.Series(np.full(200, 0.01), index=pd.bdate_range("2020-01-01", periods=200))
    quiet = jumps_mod.bipower_sigma(returns, RULES.window).iloc[-1]

    spiked = returns.copy()
    spiked.iloc[-1] = 0.20  # twenty sigma
    shocked = jumps_mod.bipower_sigma(spiked, RULES.window).iloc[-1]

    realized = math.sqrt((spiked.iloc[-RULES.window :] ** 2).mean())
    # A twenty-sigma day inflates realized volatility by 170% and bipower by 16%.
    # The absolute numbers matter less than the ratio: the estimator the jump
    # would have to beat barely notices it, which is what makes the test have
    # any power against the observation it is testing.
    bipower_inflation = shocked / quiet - 1.0
    realized_inflation = realized / quiet - 1.0
    assert bipower_inflation < 0.25
    assert realized_inflation > 1.0
    assert bipower_inflation < realized_inflation / 5.0


def test_bipower_window_must_be_usable():
    with pytest.raises(ValueError, match="at least 3"):
        jumps_mod.bipower_sigma(pd.Series([0.01, 0.02]), window=2)


# --------------------------------------------------------------------------
# Detection on synthetic jump diffusions — known jump dates
# --------------------------------------------------------------------------


def test_finds_the_planted_jumps():
    """Recall on a Merton jump diffusion whose jump dates are known."""
    returns, planted = jump_diffusion()
    result = jumps_mod.detect(returns, RULES)

    found = set(result.dates())
    hits = [day for day in planted if day in found]
    assert len(hits) >= len(planted) - 1, (
        f"detected {len(hits)} of {len(planted)} planted jumps; missed "
        f"{[str(d.date()) for d in planted if d not in found]}"
    )


def test_does_not_invent_jumps_in_a_pure_diffusion():
    """Specificity: with no jumps planted, a 1% test should stay near-silent.

    The Gumbel threshold controls the probability that the *maximum* over the
    sample exceeds it, so the expected number of false positives over one path
    is about alpha — not alpha x n. A handful would mean the calibration is
    wrong, not that the path was unlucky.
    """
    returns, _ = jump_diffusion(n_jumps=0)
    result = jumps_mod.detect(returns, RULES)
    assert len(result.dates()) <= 2, f"false positives at {[str(d.date()) for d in result.dates()]}"


def test_the_statistic_keeps_the_jump_direction():
    """A downward jump is a negative statistic — the sign is information."""
    returns, planted = jump_diffusion(n_jumps=4, jump_size=0.12, seed=11)
    result = jumps_mod.detect(returns, RULES)
    for day in planted:
        if day in set(result.dates()):
            assert np.sign(result.statistic[day]) == np.sign(returns[day])


def test_intensity_is_annualized_and_bounded():
    returns, _ = jump_diffusion()
    intensity = jumps_mod.detect(returns, RULES).intensity.dropna()

    assert not intensity.empty
    assert (intensity >= 0).all()
    # One jump inside the trailing year reads as roughly one per year, so the
    # ceiling is a jump every session.
    assert intensity.max() <= jumps_mod.TRADING_DAYS


def test_detection_is_causal():
    """Truncating the input must not change any date's answer.

    The sharpest statement of the no-lookahead law available here, and stronger
    than asserting the intensity is zero before the first planted jump — which
    only holds when the detector happens to produce no earlier false positive,
    making a real property depend on a lucky seed.
    """
    returns, planted = jump_diffusion()
    cutoff = planted[len(planted) // 2]

    full = jumps_mod.detect(returns, RULES)
    truncated = jumps_mod.detect(returns.loc[:cutoff], RULES)

    shared = truncated.statistic.index
    pd.testing.assert_series_equal(
        full.statistic.loc[shared], truncated.statistic, check_names=False
    )
    pd.testing.assert_series_equal(
        full.intensity.loc[shared], truncated.intensity, check_names=False
    )
    assert set(truncated.dates()) == {d for d in full.dates() if d <= cutoff}


def test_short_history_declines_rather_than_guessing():
    returns, _ = jump_diffusion(days=50, n_jumps=0)
    result = jumps_mod.detect(returns, RULES)
    assert result.empty
    assert result.dates() == []
    assert result.latest_intensity() is None


# --------------------------------------------------------------------------
# Detection on the real fix — the events everyone agrees about
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("day", "what"),
    [
        ("2013-04-15", "the -9.3% single-session collapse"),
        ("2020-03-12", "the COVID dash for cash"),
    ],
)
def test_detects_the_known_gold_crashes(gold_observations, day, what):
    frame = wide_frame(gold_observations)
    returns = np.log(frame[PRICE].dropna()).diff()
    result = jumps_mod.detect(returns, RULES)

    found = {d.date().isoformat() for d in result.dates()}
    assert day in found, f"missed {day} ({what})"


def test_real_detections_stay_rare(gold_observations):
    """A detector that fires weekly cannot be the input to a crisis premium."""
    frame = wide_frame(gold_observations)
    returns = np.log(frame[PRICE].dropna()).diff()
    result = jumps_mod.detect(returns, RULES)

    years = (returns.index[-1] - returns.index[0]).days / 365.25
    per_year = len(result.dates()) / years
    assert 0.5 < per_year < 5.0, f"{per_year:.1f} jumps a year is not a jump detector"


# --------------------------------------------------------------------------
# The crisis premium
# --------------------------------------------------------------------------


def test_crisis_premium_is_bounded_and_lifted_by_stress():
    index = pd.bdate_range("2020-01-01", periods=10)
    intensity = pd.Series(3.0, index=index)

    calm = jumps_mod.crisis_premium(
        intensity, pd.Series(-1.0, index=index), intensity_reference=6.0, stress_weight=0.5
    )
    tight = jumps_mod.crisis_premium(
        intensity, pd.Series(2.0, index=index), intensity_reference=6.0, stress_weight=0.5
    )

    assert calm.between(0.0, 1.0).all()
    assert tight.between(0.0, 1.0).all()
    # Negative stress does not argue the jumps away; positive stress lifts.
    assert calm.iloc[-1] == pytest.approx(0.5)
    assert tight.iloc[-1] > calm.iloc[-1]


def test_crisis_premium_needs_jumps_before_stress_matters():
    """Stress alone is a tightening, not a crisis. Zero jumps means zero premium."""
    index = pd.bdate_range("2020-01-01", periods=10)
    premium = jumps_mod.crisis_premium(
        pd.Series(0.0, index=index),
        pd.Series(3.0, index=index),
        intensity_reference=6.0,
        stress_weight=0.5,
    )
    assert (premium == 0.0).all()
