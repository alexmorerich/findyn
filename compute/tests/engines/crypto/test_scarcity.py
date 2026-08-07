"""The supply schedule — the one part of FinCrypto that is not an estimate.

Asserted against the real issuance record, because there is one: unlike every
other quantity this engine publishes, bitcoin's supply on a given date is a fact
that can be checked rather than a model output that can only be judged
plausible.
"""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import pytest

from findynamics.engines.crypto import scarcity


class TestTheHalvingCalendar:
    def test_the_four_observed_halvings_are_the_ones_that_happened(self):
        """Block heights and dates, not parameters. See the module docstring."""
        observed = (
            (210_000, date(2012, 11, 28)),
            (420_000, date(2016, 7, 9)),
            (630_000, date(2020, 5, 11)),
            (840_000, date(2024, 4, 20)),
        )
        halvings = scarcity.HALVINGS
        assert halvings == observed

    def test_the_subsidy_halves_every_epoch(self):
        assert [scarcity.subsidy_for_epoch(i) for i in range(6)] == [
            50.0,
            25.0,
            12.5,
            6.25,
            3.125,
            1.5625,
        ]

    def test_each_halving_date_opens_the_epoch_that_pays_half(self):
        for index, (_, day) in enumerate(scarcity.HALVINGS):
            epoch = scarcity.current_epoch(day)
            assert epoch.index == index + 1
            assert epoch.subsidy == scarcity.subsidy_for_epoch(index + 1)
            assert epoch.start == day

    def test_the_epoch_after_the_last_observed_one_is_flagged_as_projected(self):
        """A projection and a fact must not be published as the same thing."""
        current = scarcity.current_epoch(date(2026, 8, 5))
        assert current.index == 4
        assert current.end_is_projected is True

        historical = scarcity.current_epoch(date(2021, 1, 1))
        assert historical.end_is_projected is False

    def test_the_projection_uses_the_protocol_interval_not_a_fitted_one(self):
        """Nominal, and therefore late — which is documented rather than tuned out.

        Every observed epoch has run short of 210,000 * 10 minutes because hash
        rate grows within an epoch. Correcting for that would mean forecasting
        hash-rate growth, which is not a consensus constant and does not belong
        in this module.
        """
        nominal = scarcity.NOMINAL_EPOCH_DAYS
        assert nominal == pytest.approx(1458.33, abs=0.01)

        projected = scarcity.project_halving(date(2024, 4, 20))
        assert projected.year == 2028

        observed = [
            (b - a).days
            for (_, a), (_, b) in zip(scarcity.HALVINGS, scarcity.HALVINGS[1:], strict=False)
        ]
        assert all(days < scarcity.NOMINAL_EPOCH_DAYS for days in observed), (
            "every observed epoch ran short of nominal; if that stops being true "
            "the projection's documented bias has reversed"
        )

    def test_a_date_before_genesis_is_refused(self):
        with pytest.raises(ValueError, match="precedes the genesis block"):
            scarcity.current_epoch(date(2008, 10, 31))


class TestIssuedSupply:
    @pytest.mark.parametrize(
        ("day", "expected"),
        [
            # Each halving is a closed form: 210,000 blocks at the epoch's subsidy.
            (date(2012, 11, 28), 10_500_000),
            (date(2016, 7, 9), 15_750_000),
            (date(2020, 5, 11), 18_375_000),
            (date(2024, 4, 20), 19_687_500),
        ],
    )
    def test_supply_at_each_halving_is_exact(self, day, expected):
        assert scarcity.issued_supply(day) == pytest.approx(expected, abs=1.0)

    def test_supply_mid_epoch_is_within_a_tenth_of_a_percent_of_the_truth(self):
        """The interpolation's error bound, asserted rather than asserted-about.

        Real issued supply on 2026-08-05 was ~20.06M BTC. The interpolation
        cannot be exact — block height is not one of this engine's inputs — so
        what is tested is that it is close enough that the difference is smaller
        than the difference between issued and spendable supply, which nobody
        knows anyway.
        """
        modelled = scarcity.issued_supply(date(2026, 8, 5))
        assert modelled == pytest.approx(20_060_000, rel=0.001)

    def test_supply_never_decreases(self):
        days = pd.date_range("2009-01-03", "2030-01-01", freq="17D")
        supply = [scarcity.issued_supply(ts.date()) for ts in days]
        assert all(b >= a for a, b in zip(supply, supply[1:], strict=False))

    def test_supply_stays_under_the_twenty_one_million_cap(self):
        assert scarcity.issued_supply(date(2040, 1, 1)) < 21_000_000

    def test_issuance_rate_falls_at_every_halving(self):
        rates = [
            scarcity.issuance_rate(day - timedelta(days=1)) for _, day in scarcity.HALVINGS[1:]
        ]
        after = [
            scarcity.issuance_rate(day + timedelta(days=1)) for _, day in scarcity.HALVINGS[1:]
        ]
        for before, following in zip(rates, after, strict=True):
            assert following < before

    def test_the_current_issuance_rate_is_under_one_percent(self):
        rate = scarcity.issuance_rate(date(2026, 8, 5))
        assert 0.7 < rate < 0.9


class TestStockToFlow:
    def test_it_is_stock_over_flow_and_nothing_more(self):
        day = date(2026, 8, 5)
        assert scarcity.stock_to_flow(day) == pytest.approx(
            scarcity.issued_supply(day) / scarcity.annual_issuance(day)
        )

    def test_it_doubles_across_a_halving(self):
        """Because the flow halves and the stock barely moves in a day.

        Worth pinning because this discontinuity is exactly what the falsified
        price model turned into a forecast. The ratio is published as a supply
        statistic; this test asserts the arithmetic, not a price implication.
        """
        before = scarcity.stock_to_flow(date(2024, 4, 19))
        after = scarcity.stock_to_flow(date(2024, 4, 21))
        assert after == pytest.approx(2 * before, rel=0.01)

    def test_nothing_in_predict_reads_it(self):
        """The guard on §0's first non-goal.

        `stock_to_flow` is published for the page. If it ever becomes an input to
        the regime, the risk score or a forecast, this engine has started making
        the claim its docstring says it does not.
        """
        import inspect

        from findynamics.engines.crypto import engine as engine_mod

        for method in (
            engine_mod.CryptoEngine._risk_score,
            engine_mod.CryptoEngine._confidence,
            engine_mod.CryptoEngine._signals,
        ):
            assert "stock_to_flow" not in inspect.getsource(method)


class TestTheSchedule:
    def test_it_is_a_pure_function_of_the_calendar(self):
        """No market data enters, so no PIT treatment is needed — and none is used.

        The consequence is what matters: a run on any cutoff computes the same
        supply for the same past date, because the halvings that had happened by
        that date had happened.
        """
        index = pd.date_range("2015-01-01", "2026-08-05", freq="D")
        first = scarcity.schedule(index)
        second = scarcity.schedule(index[::-1].sort_values())
        pd.testing.assert_frame_equal(first, second)

    def test_an_empty_index_gives_an_empty_frame_with_the_right_columns(self):
        empty = scarcity.schedule(pd.DatetimeIndex([]))
        assert list(empty.columns) == [
            "issued_supply",
            "issuance_rate",
            "stock_to_flow",
            "subsidy",
        ]
        assert empty.empty
