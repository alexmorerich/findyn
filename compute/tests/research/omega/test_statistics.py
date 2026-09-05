"""The statistics, on cases with known answers.

Each of these exists because its absence produces a flattering number rather
than a wrong one, which is the harder failure to notice.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from findynamics.research.omega.statistics import auc as auc_score
from findynamics.research.omega.statistics import (
    benjamini_hochberg,
    brier,
    delong_auc_test,
    delta_r2_test,
    hac_tstat,
    newey_west_lag,
    oos_r2,
    performance,
    rank_ic,
)


def _series(values) -> pd.Series:
    """A daily-indexed series from raw values.

    ``np.asarray`` first, deliberately: ``pd.Series(other_series, index=...)``
    *reindexes* rather than relabels, so handing it a series with an integer
    index and a date index silently produces a column of NaN.
    """
    values = np.asarray(values, dtype=float)
    return pd.Series(values, index=pd.bdate_range(date(2000, 1, 3), periods=len(values)))


@pytest.mark.parametrize(
    "n,horizon,expected",
    [(100, 1, 4), (6661, 1, 11), (6661, 63, 62), (504, 21, 20), (0, 1, 0), (0, 63, 62)],
)
def test_the_lag_is_floored_at_the_overlap_the_horizon_creates(n, horizon, expected):
    """At ``h = 63`` the rule alone leaves 51 lags of overlap uncorrected.

    Every observation shares 62 days with its neighbour at that horizon, so the
    correction has to reach at least that far or the t-statistic is describing a
    sample several times larger than the one that exists.
    """
    assert newey_west_lag(n, horizon) == expected


def test_oos_r2_is_measured_against_the_training_mean():
    """The one line most likely to be "simplified" into the lookahead form.

    A forecast equal to the training mean scores exactly zero. The same forecast
    scored against the *test* mean would score positive whenever the test window
    happened to drift — which is a model with no skill being paid for the drift.
    """
    actual = _series([1.0, 2.0, 3.0, 4.0, 5.0])
    training_mean = 0.0
    flat = _series([training_mean] * 5)

    assert oos_r2(actual, flat, training_mean) == pytest.approx(0.0)
    # A perfect forecast scores 1 regardless of the benchmark.
    assert oos_r2(actual, actual, training_mean) == pytest.approx(1.0)
    # And the lookahead form would have scored the flat forecast positively.
    assert oos_r2(actual, flat, float(actual.mean())) < 0.0


def test_hac_widens_the_standard_error_on_overlapping_data():
    """Autocorrelated data has fewer independent observations than rows.

    A 63-day overlapping series is built by rolling a mean over white noise; the
    naive t-statistic treats every row as independent and the HAC one does not.
    """
    rng = np.random.default_rng(3)
    overlapping = _series(pd.Series(rng.normal(size=2_000) + 0.05).rolling(63).mean().dropna())

    naive = float(overlapping.mean() / (overlapping.std(ddof=1) / np.sqrt(len(overlapping))))
    corrected, _, lag = hac_tstat(overlapping, horizon=63)

    assert lag >= 62
    assert abs(corrected) < abs(naive) / 2.0, (
        f"HAC t={corrected:.2f} is not meaningfully smaller than naive t={naive:.2f}"
    )


def test_the_delta_r2_test_is_one_sided_and_rewards_the_better_forecast():
    """A worse challenger must not come back significant."""
    rng = np.random.default_rng(5)
    actual = _series(rng.normal(size=1_500))
    good = actual * 0.5
    bad = _series(rng.normal(size=1_500))

    better = delta_r2_test(actual, good, bad, horizon=1)
    worse = delta_r2_test(actual, bad, good, horizon=1)

    assert better["dm_p"] < 0.01
    assert worse["dm_p"] > 0.99


def test_benjamini_hochberg_is_monotone_and_never_below_its_p():
    values = pd.Series([0.001, 0.01, 0.02, 0.5, 0.9], index=list("abcde"))

    q = benjamini_hochberg(values)

    assert (q >= values - 1e-12).all()
    assert q.is_monotonic_increasing
    assert q.max() <= 1.0
    # The smallest p over five tests is multiplied by five before the step-up.
    assert q.iloc[0] == pytest.approx(0.005)


def test_benjamini_hochberg_ignores_tests_that_could_not_run():
    """A NaN is not a passing test and must not shrink the family."""
    values = pd.Series([0.01, np.nan, 0.02])

    q = benjamini_hochberg(values)

    assert np.isnan(q.iloc[1])
    assert q.iloc[0] == pytest.approx(0.02)


def test_auc_is_one_for_a_perfect_ranking_and_a_half_for_noise():
    labels = _series([0.0] * 50 + [1.0] * 50)

    assert auc_score(labels, labels) == pytest.approx(1.0)
    assert auc_score(labels, -labels) == pytest.approx(0.0)
    assert auc_score(labels, _series([1.0] * 100)) == pytest.approx(0.5)


def test_the_delong_test_separates_a_real_gap_from_a_noisy_one():
    """The test Q3 needs, and the reason it needed one.

    The shuffled-target control reached ΔAUC > 0 on two horizons by noise and
    would have "passed" Q3 on that. A ΔAUC with no standard error is a coin flip
    dressed as a finding.
    """
    rng = np.random.default_rng(7)
    labels = _series(np.repeat([0.0, 1.0], 500))
    informative = labels + rng.normal(scale=0.6, size=1_000)
    noise = _series(rng.normal(size=1_000))

    real = delong_auc_test(labels, pd.Series(informative, index=labels.index), noise)
    fake = delong_auc_test(labels, noise, _series(rng.normal(size=1_000)))

    assert real["delta_auc"] > 0.2
    assert real["p"] < 0.001
    assert fake["p"] > 0.05


def test_brier_rewards_a_calibrated_probability():
    labels = _series([0.0, 0.0, 1.0, 1.0])

    assert brier(labels, labels) == pytest.approx(0.0)
    assert brier(labels, _series([0.5] * 4)) == pytest.approx(0.25)


def test_rank_ic_is_one_for_a_monotone_transform():
    values = _series(np.linspace(1.0, 10.0, 200))

    assert rank_ic(values, np.exp(values)) == pytest.approx(1.0)
    assert rank_ic(values, -values) == pytest.approx(-1.0)


def test_performance_charges_for_turnover_and_reports_the_break_even():
    """An edge that dies at 3 bps is a finding, not a footnote."""
    rng = np.random.default_rng(11)
    returns = _series(rng.normal(0.0004, 0.01, size=1_000))
    flipping = _series(np.tile([1.0, 0.0], 500))

    free = performance(flipping, returns, cost_bps=0.0)
    costly = performance(flipping, returns, cost_bps=50.0)

    assert costly.cost_adjusted_cagr < free.cost_adjusted_cagr
    assert free.turnover > 400
    assert np.isfinite(free.break_even_bps)
    # Charging exactly the break-even cost should wipe the gross edge out.
    at_break_even = performance(flipping, returns, cost_bps=free.break_even_bps)
    assert abs(at_break_even.cost_adjusted_cagr) < abs(free.cagr) + 1e-9


def test_buy_and_hold_has_almost_no_turnover():
    returns = _series(np.full(500, 0.0004))
    always = _series(np.ones(500))

    metrics = performance(always, returns, cost_bps=5.0)

    assert metrics.turnover <= 1.0
    # One unit of turnover at 5 bps over two years is a few basis points of
    # CAGR, not zero — asserted in absolute terms so the tolerance means
    # something rather than tracking whatever the CAGR happens to be.
    assert abs(metrics.cost_adjusted_cagr - metrics.cagr) < 1e-3
    assert metrics.hit_rate == pytest.approx(1.0)
