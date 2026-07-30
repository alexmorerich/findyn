"""Discount factors: discount_mod(t,0)=1, monotonicity, and reading the curve as data.

The independence contract is the interesting part. These tests build the NS
factors as *observation rows* — the shape the ``engine_output`` provider
delivers — and never import ``engines.rates``, which is what a consumer of the
published curve actually has to do.
"""

from __future__ import annotations

import math
from datetime import date, timedelta

import pandas as pd
import pytest

from findynamics.core.contracts.vocab import DISCOUNT_HORIZON_YEARS, DISCOUNT_HORIZONS
from findynamics.data.accessor import PandasPITAccessor
from findynamics.engines.money import discount as discount_mod
from tests.engines.money.conftest import NS_IDS, curve_rows

#: A plausible upward-sloping curve: 4.5% long, 1.2pp of slope, mild hump.
UPWARD = discount_mod.NelsonSiegelFactors(level=4.5, slope=1.2, curvature=-0.5, lambda_=0.609)

#: An inverted curve, of the kind that exists and must not be smoothed away.
INVERTED = discount_mod.NelsonSiegelFactors(level=3.6, slope=-1.4, curvature=0.8, lambda_=0.609)

#: The published-factor payload matching :data:`UPWARD`, in wire role names.
UPWARD_ROLES = {"level": 4.5, "slope": 1.2, "curvature": -0.5, "lambda": 0.609}


def accessor_with_curve(factors: dict[str, float], as_of: date) -> PandasPITAccessor:
    days = [as_of - timedelta(days=i) for i in range(5)]
    frame = pd.DataFrame(curve_rows(factors, days))
    return PandasPITAccessor(frame, as_of + timedelta(days=2))


class TestDiscountFactor:
    def test_a_zero_horizon_is_worth_exactly_one(self):
        """discount_mod(t, 0) = 1. A dollar today is a dollar, at every rate."""
        for rate in (-2.0, 0.0, 0.5, 5.0, 20.0):
            assert discount_mod.discount_factor(rate, 0.0) == 1.0

    def test_a_zero_rate_discounts_nothing_at_any_horizon(self):
        for years in (0.25, 1.0, 10.0, 30.0):
            assert discount_mod.discount_factor(0.0, years) == 1.0

    def test_it_is_exp_minus_z_h(self):
        assert discount_mod.discount_factor(5.0, 10.0) == pytest.approx(math.exp(-0.5), rel=1e-15)

    def test_a_negative_rate_puts_the_factor_above_one(self):
        """Not clamped: a negative zero rate really does mean that."""
        assert discount_mod.discount_factor(-1.0, 5.0) > 1.0


class TestNelsonSiegelReconstruction:
    """The published betas must evaluate to the curve they were fitted from."""

    def test_the_instantaneous_rate_is_level_minus_slope(self):
        """Published slope is -b1, so y(0) = b0 + b1 = level - slope."""
        assert UPWARD.yield_at(0.0) == pytest.approx(UPWARD.level - UPWARD.slope)

    def test_the_long_end_approaches_the_level(self):
        curve = discount_mod.NelsonSiegelFactors(
            level=4.5, slope=1.2, curvature=-0.5, lambda_=0.609
        )
        assert curve.yield_at(100.0) == pytest.approx(4.5, abs=0.1)

    def test_an_upward_sloping_fit_rises_with_maturity(self):
        curve = discount_mod.NelsonSiegelFactors(level=4.5, slope=1.2, curvature=0.0, lambda_=0.609)
        yields = [curve.yield_at(m) for m in (0.5, 1, 2, 5, 10, 30)]
        assert yields == sorted(yields)


class TestBuildCurve:
    def test_every_standard_horizon_is_published(self):
        curve = discount_mod.build_curve(4.3, None)
        assert tuple(curve.factors) == DISCOUNT_HORIZONS

    def test_short_horizons_come_from_the_short_rate_and_long_ones_from_the_fit(self):
        curve = discount_mod.build_curve(4.3, UPWARD)

        for horizon in ("1m", "3m", "6m", "1y"):
            assert curve.sources[horizon] == "short_rate"
        for horizon in ("2y", "3y", "10y", "30y"):
            assert curve.sources[horizon] == "ns"

    def test_the_short_end_is_exactly_the_short_rate_flat(self):
        curve = discount_mod.build_curve(4.3, None)
        assert curve.factors["1y"] == pytest.approx(math.exp(-0.043), rel=1e-15)
        assert curve.factors["3m"] == pytest.approx(math.exp(-0.043 * 0.25), rel=1e-15)

    def test_the_long_end_uses_the_fitted_yield_at_that_maturity(self):
        curve = discount_mod.build_curve(4.3, UPWARD)
        expected = math.exp(-(UPWARD.yield_at(10.0) / 100.0) * 10.0)
        assert curve.factors["10y"] == pytest.approx(expected, rel=1e-15)

    def test_a_positive_curve_gives_monotonically_falling_factors(self):
        """The property a present value depends on."""
        factors = [discount_mod.build_curve(4.3, UPWARD).factors[h] for h in DISCOUNT_HORIZONS]
        assert factors == sorted(factors, reverse=True)
        assert all(0.0 < f <= 1.0 for f in factors)

    def test_monotonicity_holds_with_no_curve_at_all(self):
        """The flat fallback must not break the invariant either."""
        for rate in (0.0, 0.5, 4.3, 12.0):
            factors = [discount_mod.build_curve(rate, None).factors[h] for h in DISCOUNT_HORIZONS]
            assert factors == sorted(factors, reverse=True)

    def test_monotonicity_across_a_range_of_realistic_curves(self):
        """Holds for every curve whose zero rates are all non-negative."""
        for level, slope, curvature in (
            (4.5, 1.2, -0.5),
            (6.0, 0.2, -1.5),
            (2.5, 1.0, 0.4),
            (0.9, 0.4, 0.3),
        ):
            fit = discount_mod.NelsonSiegelFactors(
                level=level, slope=slope, curvature=curvature, lambda_=0.609
            )
            short_rate = fit.yield_at(0.25)
            assert short_rate >= 0.0, "fixture must not have a negative short end"

            factors = [
                discount_mod.build_curve(short_rate, fit).factors[h] for h in DISCOUNT_HORIZONS
            ]
            assert factors == sorted(factors, reverse=True), (level, slope, curvature)

    def test_an_ordinary_inversion_stays_monotone(self):
        """The surprising direction, and the reason the invariant is robust.

        discount_mod compares ``z(h)·h``, so doubling the horizon outweighs a yield a point
        or two lower. An inverted curve therefore does *not* break monotonicity,
        which is worth pinning: it is easy to assume otherwise and then "fix"
        something that was never wrong.
        """
        overnight = 5.4
        assert INVERTED.yield_at(2.0) < overnight

        factors = [
            discount_mod.build_curve(overnight, INVERTED).factors[h] for h in DISCOUNT_HORIZONS
        ]
        assert factors == sorted(factors, reverse=True)

    def test_a_deep_inversion_at_the_stitch_is_not_smoothed_away(self):
        """The one case that does break it, left uncorrected on purpose.

        The overnight rate is pinned at 6% while the curve prices the 2y at 2% —
        early 2020's shape. Then ``z(2y)·2 < z(1y)·1`` and discount_mod(2y) really is above
        discount_mod(1y). Forcing the sequence down would be inventing a curve nobody quoted.
        """
        deep = discount_mod.NelsonSiegelFactors(level=2.2, slope=0.4, curvature=-0.6, lambda_=0.609)
        overnight = 6.0
        assert deep.yield_at(2.0) < overnight / 2.0

        curve = discount_mod.build_curve(overnight, deep)
        assert curve.sources["1y"] == "short_rate"
        assert curve.sources["2y"] == "ns"
        assert curve.factors["2y"] > curve.factors["1y"]

    def test_a_negative_short_rate_lifts_the_near_end_above_one(self):
        """The other documented exception: not clamped, because it is real."""
        curve = discount_mod.build_curve(-0.4, None)
        near = [curve.factors[h] for h in ("1m", "3m", "6m", "1y")]
        assert all(f > 1.0 for f in near)
        assert near == sorted(near), "a negative rate makes the factor grow with horizon"

    def test_a_non_finite_horizon_is_dropped_not_published(self):
        fit = discount_mod.NelsonSiegelFactors(
            level=float("nan"), slope=0.0, curvature=0.0, lambda_=0.609
        )
        curve = discount_mod.build_curve(4.0, fit)
        assert "10y" not in curve.factors
        assert curve.factors["1y"] == pytest.approx(math.exp(-0.04))

    def test_an_unknown_horizon_is_a_configuration_error(self):
        with pytest.raises(ValueError, match="no year count configured"):
            discount_mod.build_curve(4.0, None, horizons=("42y",))


class TestReadingTheCurveAsPublishedData:
    """The independence contract in practice: rows in, factors out."""

    def test_the_latest_published_factors_are_read_back(self):
        as_of = date(2026, 7, 20)
        accessor = accessor_with_curve(
            {"level": 4.5, "slope": 1.2, "curvature": -0.5, "lambda": 0.609}, as_of
        )
        curve = discount_mod.read_curve_factors(accessor, NS_IDS)
        assert curve is not None
        assert (curve.level, curve.slope, curve.curvature, curve.lambda_) == (
            4.5,
            1.2,
            -0.5,
            0.609,
        )

    def test_three_of_four_metrics_is_no_curve_rather_than_a_guessed_lambda(self):
        """A wrong lambda misplaces the whole belly of the curve."""
        as_of = date(2026, 7, 20)
        rows = curve_rows({"level": 4.5, "slope": 1.2, "curvature": -0.5}, [as_of])
        accessor = PandasPITAccessor(pd.DataFrame(rows), as_of + timedelta(days=2))
        assert discount_mod.read_curve_factors(accessor, NS_IDS) is None

    def test_a_non_positive_lambda_is_rejected(self):
        as_of = date(2026, 7, 20)
        accessor = accessor_with_curve(
            {"level": 4.5, "slope": 1.2, "curvature": -0.5, "lambda": 0.0}, as_of
        )
        assert discount_mod.read_curve_factors(accessor, NS_IDS) is None

    def test_factors_published_after_the_cutoff_are_invisible(self):
        """The point of routing the curve through the PIT gateway.

        A run standing before FinRates published cannot see the curve, however
        recently the rows were written.
        """
        obs = date(2026, 7, 20)
        rows = curve_rows(
            {"level": 4.5, "slope": 1.2, "curvature": -0.5, "lambda": 0.609},
            [obs],
            published_on=date(2026, 7, 25),
        )
        frame = pd.DataFrame(rows)
        assert (
            discount_mod.read_curve_factors(PandasPITAccessor(frame, date(2026, 7, 24)), NS_IDS)
            is None
        )
        assert (
            discount_mod.read_curve_factors(PandasPITAccessor(frame, date(2026, 7, 25)), NS_IDS)
            is not None
        )

    def test_missing_role_ids_is_a_programming_error(self):
        accessor = accessor_with_curve(
            {"level": 4.5, "slope": 1.2, "curvature": -0.5, "lambda": 0.609}, date(2026, 7, 20)
        )
        with pytest.raises(ValueError, match="missing"):
            discount_mod.read_curve_factors(accessor, {"level": NS_IDS["level"]})

    def test_the_history_keeps_only_dates_carrying_all_four_metrics(self):
        complete = [date(2026, 7, 15), date(2026, 7, 16)]
        rows = curve_rows(
            {"level": 4.5, "slope": 1.2, "curvature": -0.5, "lambda": 0.609}, complete
        )
        rows += curve_rows({"level": 4.6, "slope": 1.1}, [date(2026, 7, 17)])
        accessor = PandasPITAccessor(pd.DataFrame(rows), date(2026, 7, 20))

        history = discount_mod.curve_factor_history(accessor, NS_IDS)
        assert list(history.index.date) == complete
        assert set(history.columns) == {"level", "slope", "curvature", "lambda_"}

    def test_an_empty_information_set_gives_an_empty_history(self):
        accessor = PandasPITAccessor(
            pd.DataFrame(columns=["series_id", "obs_date", "release_date", "value"]),
            date(2026, 7, 20),
        )
        assert discount_mod.curve_factor_history(accessor, NS_IDS).empty


def test_the_horizon_grid_and_its_year_counts_agree():
    """Two constants that would silently disagree if either were edited alone."""
    assert set(DISCOUNT_HORIZONS) == set(DISCOUNT_HORIZON_YEARS)
    years = [DISCOUNT_HORIZON_YEARS[h] for h in DISCOUNT_HORIZONS]
    assert years == sorted(years), "horizons must be listed in ascending maturity"
