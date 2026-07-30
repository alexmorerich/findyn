"""The money-market account: does M(t) actually equal M(0)·exp(Σ r·Δt)?

Every assertion here is against a number computed independently of the code
under test — a closed form, or arithmetic written out in the test itself. account_mod test
that asserts the implementation equals itself proves only that it is
deterministic.
"""

from __future__ import annotations

import math
from datetime import date, timedelta

import pandas as pd
import pytest

from findynamics.data.accessor import PandasPITAccessor
from findynamics.engines.money import account as account_mod
from tests.engines.money.conftest import (
    BILL_3M,
    DTB3,
    SOFR,
    constant_rate_frame,
    rate_rows,
)


def path_from(observations: pd.DataFrame, as_of: date, **kwargs) -> account_mod.RatePath | None:
    accessor = PandasPITAccessor(observations, as_of)
    wide = accessor.wide()
    return account_mod.splice_rate_path(wide, **kwargs)


class TestDayCount:
    """ACT/360 is a decision with consequences; they are asserted, not assumed."""

    def test_one_year_at_a_constant_rate_matches_the_closed_form(self):
        """365 days at 5% ACT/360 compounds to exp(0.05 · 364/360).

        364 and not 365: the index spans 365 observations, so 364 intervals
        accrue. Getting this off by one is the classic way an accrual is silently
        wrong by a day's interest.
        """
        frame = constant_rate_frame(5.0, start=date(2020, 1, 1), days=365)
        path = path_from(frame, date(2021, 1, 1), primary=SOFR)
        assert path is not None

        index = account_mod.wealth_index(path)
        expected = math.exp(0.05 * 364 / 360.0)
        assert index.iloc[-1] == pytest.approx(expected, rel=1e-12)

    def test_the_360_basis_is_not_365(self):
        """account_mod year of 5% earns 5.07% on ACT/360, not 5.00%. That is real money."""
        frame = constant_rate_frame(5.0, start=date(2020, 1, 1), days=366)
        path = path_from(frame, date(2021, 1, 5), primary=SOFR)
        assert path is not None
        earned = account_mod.wealth_index(path).iloc[-1] - 1.0
        # 365 intervals / 360 basis, continuously compounded.
        assert earned == pytest.approx(math.exp(0.05 * 365 / 360.0) - 1.0, rel=1e-12)
        assert earned > 0.05

    def test_the_index_starts_at_exactly_one(self):
        frame = constant_rate_frame(4.0, days=10)
        path = path_from(frame, date(2020, 2, 1), primary=SOFR)
        assert path is not None
        assert account_mod.wealth_index(path).iloc[0] == 1.0

    def test_a_zero_rate_earns_nothing(self):
        """Bills traded at 0.00% for years; that must compound to exactly 1.0."""
        frame = constant_rate_frame(0.0, days=200)
        path = path_from(frame, date(2021, 1, 1), primary=SOFR)
        assert path is not None
        index = account_mod.wealth_index(path)
        assert index.iloc[-1] == 1.0
        assert (index == 1.0).all()

    def test_a_negative_rate_shrinks_the_account(self):
        """Not clamped. Negative money-market rates have happened."""
        frame = constant_rate_frame(-0.5, start=date(2020, 1, 1), days=181)
        path = path_from(frame, date(2020, 7, 1), primary=SOFR)
        assert path is not None
        assert account_mod.wealth_index(path).iloc[-1] == pytest.approx(
            math.exp(-0.005 * 180 / 360.0), rel=1e-12
        )


class TestAccrualConvention:
    """Which rate earns over which interval, and what happens across a weekend."""

    def test_accrual_uses_the_rate_at_the_start_of_each_interval(self):
        """Two dates, two rates: only the first can have earned anything.

        The rate stamped on the closing date has not applied to anything yet. If
        this ever reads 6%, the accrual has become right-endpoint and is paying
        today's balance at a rate published tomorrow.
        """
        rows = rate_rows(SOFR, {date(2020, 6, 1): 2.0, date(2020, 6, 2): 6.0})
        path = path_from(pd.DataFrame(rows), date(2020, 6, 4), primary=SOFR)
        assert path is not None

        index = account_mod.wealth_index(path)
        assert index.iloc[-1] == pytest.approx(math.exp(0.02 * 1 / 360.0), rel=1e-12)

    def test_a_weekend_accrues_three_days_at_fridays_rate(self):
        """Cash does not stop earning because the desk is shut."""
        friday, monday = date(2020, 6, 5), date(2020, 6, 8)
        rows = rate_rows(SOFR, {friday: 3.0, monday: 3.0})
        path = path_from(pd.DataFrame(rows), date(2020, 6, 10), primary=SOFR)
        assert path is not None
        assert account_mod.wealth_index(path).iloc[-1] == pytest.approx(
            math.exp(0.03 * 3 / 360.0), rel=1e-12
        )

    def test_a_gap_wider_than_the_limit_restarts_rather_than_bridging(self):
        """account_mod 200-day hole is missing data, not a long holiday.

        Paying one stale overnight rate across it would produce a balance that
        looks plausible and is fiction. The restart is visible instead.
        """
        rows = rate_rows(
            SOFR,
            {date(2020, 1, 1): 5.0, date(2020, 1, 2): 5.0, date(2020, 7, 20): 5.0},
        )
        path = path_from(pd.DataFrame(rows), date(2020, 7, 25), primary=SOFR)
        assert path is not None

        index = account_mod.wealth_index(path, max_gap_days=40)
        assert index.iloc[1] == pytest.approx(math.exp(0.05 / 360.0), rel=1e-12)
        assert index.iloc[2] == 1.0
        assert account_mod.accrual_base(index) == date(2020, 7, 20)


class TestIndexResolutionIsUnitSafe:
    """pandas 2 keeps whatever datetime resolution a frame was built at.

    This is a regression test for a real bug: the accrual originally scaled the
    index's raw integers as nanoseconds, so a ``datetime64[us]`` frame — which is
    what ``read_csv`` produces — understated every interest payment by a factor
    of a thousand while still looking like a plausible wealth index.
    """

    @pytest.mark.parametrize("unit", ["s", "ms", "us", "ns"])
    def test_every_datetime_resolution_gives_the_same_accrual(self, unit):
        frame = constant_rate_frame(5.0, start=date(2020, 1, 1), days=100)
        frame["obs_date"] = frame["obs_date"].astype(f"datetime64[{unit}]")

        path = path_from(frame, date(2020, 5, 1), primary=SOFR)
        assert path is not None
        assert account_mod.wealth_index(path).iloc[-1] == pytest.approx(
            math.exp(0.05 * 99 / 360.0), rel=1e-12
        )


class TestDiscountToInvestmentYield:
    """DTB3 is a discount quote; an accrual needs an investment yield."""

    def test_conversion_matches_the_closed_form(self):
        """d / (1 - d·n/360). At 5% over 91 days that is 5.0645%."""
        assert account_mod.investment_yield_from_discount(5.0, days=91) == pytest.approx(
            5.0 / (1.0 - 0.05 * 91 / 360.0), rel=1e-12
        )

    def test_the_investment_yield_always_exceeds_the_discount_rate(self):
        for discount in (0.5, 2.0, 5.0, 15.0):
            assert account_mod.investment_yield_from_discount(discount, days=91) > discount

    def test_zero_is_a_fixed_point(self):
        assert account_mod.investment_yield_from_discount(0.0) == 0.0

    def test_an_impossible_quote_degrades_instead_of_returning_infinity(self):
        """account_mod discount rate implying a non-positive price would blow up the index."""
        assert account_mod.investment_yield_from_discount(500.0, days=91) == 500.0


class TestSplice:
    """SOFR primary, DTB3 fallback — at the real 2018-04-03 boundary."""

    def test_the_splice_prefers_sofr_and_falls_back_only_where_it_is_missing(
        self, money_observations
    ):
        """On the real snapshot the handover lands on SOFR's actual first day."""
        path = path_from(
            money_observations,
            date(2018, 5, 1),
            primary=SOFR,
            fallback=DTB3,
        )
        assert path is not None

        sources = path.sources
        assert sources.loc[: pd.Timestamp("2018-04-02")].eq(DTB3).all()
        assert sources.loc[pd.Timestamp("2018-04-03") :].eq(SOFR).all()
        assert path.start == date(2018, 1, 2)

    def test_the_spliced_path_accrues_across_the_boundary_without_a_reset(self, money_observations):
        """The whole point of splicing: one continuous dollar, not two."""
        path = path_from(money_observations, date(2018, 6, 1), primary=SOFR, fallback=DTB3)
        assert path is not None
        index = account_mod.wealth_index(path)
        assert account_mod.accrual_base(index) == date(2018, 1, 2)
        assert index.iloc[-1] > 1.0
        # Strictly increasing: 2018 had no negative money-market rates.
        assert index.is_monotonic_increasing

    def test_the_fallback_leg_is_converted_before_it_accrues(self):
        """Hand-computed across the splice date, both legs, in one number.

        Three dates. The first two accrue on the converted bill, the third on
        SOFR itself. Written out longhand precisely so the test does not agree
        with the code by construction.
        """
        rows = rate_rows(DTB3, {date(2018, 4, 1): 1.80, date(2018, 4, 2): 1.80})
        rows += rate_rows(SOFR, {date(2018, 4, 3): 1.75})
        path = path_from(pd.DataFrame(rows), date(2018, 4, 6), primary=SOFR, fallback=DTB3)
        assert path is not None
        assert path.sources.tolist() == [DTB3, DTB3, SOFR]

        bill_yield = 1.80 / (1.0 - 0.018 * 91 / 360.0)
        expected = math.exp((bill_yield / 100.0) * 1 / 360.0 + (bill_yield / 100.0) * 1 / 360.0)
        assert account_mod.wealth_index(path).iloc[-1] == pytest.approx(expected, rel=1e-12)

    def test_the_fallback_fills_a_single_missing_day_not_just_an_era(self):
        """The rule is per date. account_mod SOFR publication gap is the same case."""
        rows = rate_rows(SOFR, {date(2020, 6, 1): 0.08, date(2020, 6, 3): 0.08})
        rows += rate_rows(DTB3, {date(2020, 6, 2): 0.15})
        path = path_from(pd.DataFrame(rows), date(2020, 6, 6), primary=SOFR, fallback=DTB3)
        assert path is not None
        assert path.sources.tolist() == [SOFR, DTB3, SOFR]

    def test_share_from_reports_the_composition(self, money_observations):
        path = path_from(money_observations, date(2018, 5, 1), primary=SOFR, fallback=DTB3)
        assert path is not None
        assert 0.0 < path.share_from(SOFR) < 1.0
        assert path.share_from(SOFR) + path.share_from(DTB3) == pytest.approx(1.0)

    def test_no_series_at_all_is_no_path_rather_than_an_exception(self):
        rows = rate_rows(BILL_3M, {date(2020, 1, 1): 1.5})
        assert path_from(pd.DataFrame(rows), date(2020, 2, 1), primary=SOFR) is None


class TestRealizedCarry:
    """Carry is the exact inverse of the accrual, and says so when it cannot be."""

    def test_a_constant_rate_returns_that_exact_rate(self):
        """The property that makes the convention worth choosing.

        ln(M(t)/M(t-h))·360/Δdays over an unchanged 4.25% path returns 4.25%, to
        the last decimal. account_mod simple-interest annualization would return 4.31% and
        invite the reader to think the engine has a view.
        """
        frame = constant_rate_frame(4.25, start=date(2020, 1, 1), days=500)
        path = path_from(frame, date(2021, 6, 1), primary=SOFR)
        assert path is not None
        index = account_mod.wealth_index(path)

        for window in (30, 91, 365):
            assert account_mod.realized_carry(index, window) == pytest.approx(0.0425, rel=1e-12)

    def test_carry_is_a_decimal_not_a_percent(self):
        frame = constant_rate_frame(5.0, start=date(2020, 1, 1), days=400)
        path = path_from(frame, date(2021, 3, 1), primary=SOFR)
        assert path is not None
        assert account_mod.realized_carry(account_mod.wealth_index(path), 365) == pytest.approx(
            0.05, rel=1e-12
        )

    def test_a_window_the_path_cannot_reach_is_none_not_a_guess(self):
        """Reporting a 3-month carry as a 12-month one is worse than reporting none."""
        frame = constant_rate_frame(3.0, start=date(2020, 1, 1), days=90)
        path = path_from(frame, date(2020, 4, 5), primary=SOFR)
        assert path is not None
        index = account_mod.wealth_index(path)
        assert account_mod.realized_carry(index, 30) is not None
        assert account_mod.realized_carry(index, 365) is None

    def test_a_window_spanning_an_accrual_reset_is_none(self):
        """The quotient across a reset is not a growth factor."""
        values = {date(2020, 1, 1) + timedelta(days=i): 5.0 for i in range(20)}
        values.update({date(2020, 9, 1) + timedelta(days=i): 5.0 for i in range(40)})
        path = path_from(pd.DataFrame(rate_rows(SOFR, values)), date(2020, 10, 15), primary=SOFR)
        assert path is not None
        index = account_mod.wealth_index(path)

        assert account_mod.realized_carry(index, 30) == pytest.approx(0.05, rel=1e-9)
        # 365 days back lands before the gap; the ratio would be meaningless.
        assert account_mod.realized_carry(index, 365) is None

    def test_carry_tracks_a_rate_change_in_the_right_direction(self):
        """Halve the rate for the last month and the 1m carry must follow."""
        values = {date(2020, 1, 1) + timedelta(days=i): 4.0 for i in range(200)}
        values.update({date(2020, 7, 19) + timedelta(days=i): 2.0 for i in range(40)})
        path = path_from(pd.DataFrame(rate_rows(SOFR, values)), date(2020, 8, 30), primary=SOFR)
        assert path is not None
        index = account_mod.wealth_index(path)

        assert account_mod.realized_carry(index, 30) == pytest.approx(0.02, rel=1e-9)
        assert account_mod.realized_carry(index, 365) is None

        # account_mod window spanning both rates blends them by days, so it must land
        # strictly between — and nearer 4%, which most of the window earned.
        blended = account_mod.realized_carry(index, 200)
        assert blended is not None
        assert 0.02 < blended < 0.04
        assert blended > 0.03

    def test_an_empty_index_is_none(self):
        assert account_mod.realized_carry(pd.Series(dtype=float), 30) is None
