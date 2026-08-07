"""The vol/drawdown regime, asserted against the windows everyone agrees on.

Which periods count as a frenzy and which as a winter is not a question a
synthetic series can answer — you can always invent one that crosses whichever
line you wrote — so these run against the real 2010-2026 record. The named
windows are the acceptance criteria for this phase: 2017Q4 and 2021 must be
``frenzy``, the 2018 and 2022 drawdowns must be ``winter``.

The synthetic tests below them cover the ordering rule and the boundaries, where
real data cannot isolate one condition from the other.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from findynamics.engines.crypto import regime as regime_mod
from findynamics.engines.crypto.domain import CRYPTO_REGIMES
from findynamics.engines.crypto.regime import RegimeRules
from tests.engines.crypto.conftest import price_series


@pytest.fixture(scope="module")
def labels(request) -> pd.Series:
    """Regime per date over the whole snapshot, on the shipped configuration."""
    from findynamics.core.config import load_series_config

    observations = request.getfixturevalue("crypto_observations")
    config = load_series_config()
    rules = RegimeRules.from_params(config.engines["crypto"].params)
    return regime_mod.classify(price_series(observations), rules).label.dropna()


def at(labels: pd.Series, day: str) -> str:
    key = pd.Timestamp(day)
    assert key in labels.index, f"{day} has no published regime"
    return str(labels.loc[key])


class TestTheAcceptanceWindows:
    """The four windows this phase is graded on."""

    @pytest.mark.parametrize(
        "day",
        [
            "2017-10-31",  # the run-up
            "2017-11-30",
            "2017-12-15",  # near the top
            "2017-12-31",
        ],
    )
    def test_2017q4_is_a_frenzy(self, labels, day):
        assert at(labels, day) == "frenzy"

    @pytest.mark.parametrize(
        "day",
        [
            "2021-02-26",
            "2021-04-14",  # the April peak
            "2021-10-29",
            "2021-11-09",  # the November peak
        ],
    )
    def test_2021_is_a_frenzy(self, labels, day):
        assert at(labels, day) == "frenzy"

    @pytest.mark.parametrize(
        "day",
        [
            "2018-02-28",
            "2018-06-29",
            "2018-09-28",
            "2018-12-31",
        ],
    )
    def test_the_2018_drawdown_is_a_winter(self, labels, day):
        assert at(labels, day) == "winter"

    @pytest.mark.parametrize(
        "day",
        [
            "2022-06-30",
            "2022-09-30",
            "2022-11-30",
            "2022-12-31",
        ],
    )
    def test_the_2022_drawdown_is_a_winter(self, labels, day):
        assert at(labels, day) == "winter"


class TestWhatTheLabelsSayAboutTheRecord:
    def test_every_label_is_in_the_vocabulary(self, labels):
        assert set(labels.unique()) <= set(CRYPTO_REGIMES)

    def test_normal_is_the_most_common_state(self, labels):
        """The honest label for "nothing in particular is happening".

        If a threshold change ever makes frenzy or winter the modal state, the
        labels have stopped carrying information and this asks why.
        """
        counts = labels.value_counts()
        assert counts.idxmax() == "normal"
        assert counts["normal"] > 0.4 * len(labels)

    def test_all_three_states_are_visited(self, labels):
        counts = labels.value_counts()
        for name in CRYPTO_REGIMES:
            assert counts.get(name, 0) > 100, f"{name} is nearly never used"

    def test_march_2020_is_a_winter_for_a_fortnight_and_then_is_not(self, labels):
        """Depth, not duration — documented in regime.py and asserted here.

        A 50% crash is a winter for as long as the price is 50% down. Smoothing
        that away would be smoothing away the thing worth reporting; leaving it
        means the label has to be read as what it is.
        """
        assert at(labels, "2020-03-16") == "winter"
        # And it leaves again once the drawdown closes, rather than latching.
        assert at(labels, "2020-06-30") != "winter"

    def test_the_2024_bull_market_is_not_a_frenzy(self, labels):
        """The discrimination that makes the frenzy label worth having.

        2024 more than doubled year on year, so the trend leg fired — but
        realized volatility ran in the mid-40s against 2017's 90s. Requiring both
        conditions is what separates an ETF-era bull market from a blowoff, and a
        rule on the 12-month return alone would call them the same thing.
        """
        assert at(labels, "2024-03-13") == "normal"
        assert at(labels, "2024-12-31") == "normal"


class TestTheOrderingRule:
    """Winter is evaluated before frenzy, and the order is the rule."""

    def _path(self, closes: list[float]) -> pd.Series:
        index = pd.date_range("2020-01-01", periods=len(closes), freq="D")
        return pd.Series(closes, index=index, dtype=float)

    def test_a_market_that_is_both_is_called_a_winter(self):
        """November 2021 in miniature: up hugely on the year, far off the peak.

        Calling this a frenzy would describe the top of a market as its middle.
        """
        rules = RegimeRules(
            window=100,
            min_observations=100,
            drawdown_threshold=0.45,
            frenzy_return=0.69,
            # Zeroed so the vol leg cannot be the reason frenzy loses — the point
            # is that winter wins on ordering, not on the other rule failing.
            frenzy_vol=0.0,
        )
        # Flat, then a 5x spike inside the window, then a fall that leaves the
        # price still more than double where it was 100 days ago and 56% off the
        # peak it made in between. Both conditions hold at once, which is the
        # only configuration where the ordering is observable.
        closes = [100.0] * 100 + list(np.linspace(100, 500, 60)) + list(np.linspace(500, 220, 30))
        view = regime_mod.classify(self._path(closes), rules)
        latest = view.latest()

        # Both conditions hold at the end...
        values = view.latest_values()
        assert values["drawdown"] <= -0.45
        assert values["return_12m"] >= 0.69
        # ...and winter wins.
        assert latest == "winter"

    def test_frenzy_needs_both_legs(self):
        """A doubling with no volatility is not a frenzy, and neither is the reverse."""
        rules = RegimeRules(window=100, min_observations=100, frenzy_return=0.69, frenzy_vol=60.0)

        # Smooth doubling: the trend leg fires, the vol leg does not.
        smooth = self._path(list(np.geomspace(100, 400, 220)))
        assert regime_mod.classify(smooth, rules).latest() == "normal"

        # Volatile but flat: the vol leg fires, the trend leg does not.
        rng = np.random.default_rng(20260805)
        wobble = 100 * np.exp(np.cumsum(rng.normal(0.0, 0.06, 220)))
        view = regime_mod.classify(self._path(list(wobble)), rules)
        assert view.latest_values()["realized_vol"] > 60.0
        assert view.latest() != "frenzy"


class TestCausalityAndReadiness:
    def test_no_label_is_published_before_the_windows_are_full(self):
        """ "Not enough history" and "nothing is happening" are different statements.

        Defaulting the early years to `normal` would publish the second when only
        the first is true.
        """
        rules = RegimeRules(window=365, min_observations=400)
        index = pd.date_range("2020-01-01", periods=500, freq="D")
        price = pd.Series(np.linspace(100, 200, 500), index=index)

        view = regime_mod.classify(price, rules)
        assert view.label.iloc[:400].isna().all()
        assert view.label.iloc[400:].notna().any()

    def test_a_label_never_changes_once_the_windows_have_passed(self, crypto_observations):
        """The no-lookahead property at the regime level.

        Every window here is trailing, so extending the price series must not
        move a label that was already published. A centred window or a `bfill`
        would fail this on the dates near the earlier cutoff, which is exactly
        the fingerprint to look for.
        """
        from findynamics.core.config import load_series_config

        rules = RegimeRules.from_params(load_series_config().engines["crypto"].params)
        price = price_series(crypto_observations)

        early = regime_mod.classify(price.loc[: pd.Timestamp("2022-12-31")], rules).label.dropna()
        late = regime_mod.classify(price, rules).label.dropna()

        shared = early.index.intersection(late.index)
        assert len(shared) > 1500
        pd.testing.assert_series_equal(early.loc[shared], late.loc[shared], check_names=False)

    def test_an_empty_price_series_gives_an_empty_view_rather_than_an_exception(self):
        view = regime_mod.classify(pd.Series(dtype=float), RegimeRules())
        assert view.empty
        assert view.latest() is None


class TestConfigLoading:
    def test_a_non_mapping_block_is_refused_at_load(self):
        with pytest.raises(ValueError, match="must be a mapping"):
            RegimeRules.from_params({"regime": "45%"})

    def test_every_threshold_the_model_branches_on_comes_from_config(self, config):
        """A rule that only exists in Python cannot be recalibrated without a deploy."""
        rules = RegimeRules.from_params(config.engines["crypto"].params)
        assert rules.window == 365
        assert rules.drawdown_threshold == 0.45
        assert rules.frenzy_return == 0.69
        assert rules.frenzy_vol == 60.0
        assert rules.min_observations == 400

    def test_the_calendar_is_365_and_not_252(self):
        """Bitcoin has no exchange calendar.

        At 252 every annualized volatility this module computes would be
        understated by about 20%, silently, in a number the frenzy leg compares
        against a configured threshold.
        """
        assert regime_mod.CALENDAR_DAYS == 365

        index = pd.date_range("2020-01-01", periods=800, freq="D")
        rng = np.random.default_rng(1)
        returns = pd.Series(rng.normal(0, 0.03, 800), index=index)
        vol = regime_mod.realized_volatility(returns, 365).dropna()
        # 3% daily on a 365-day year is ~57% annualized; on 252 it would be ~48%.
        assert 50.0 < vol.iloc[-1] < 65.0


def test_the_snapshot_starts_where_bitcoin_had_a_price(crypto_observations):
    """Guard on the fixture, so a truncated regeneration is loud rather than subtle.

    2010-08-18 is not a chosen window — it is the first date bitcoin had a market
    price at all. Before it the chart is padded with zeros, which the adapter
    drops because "no market yet" is not a price of zero.

    The count matters as much as the start: the whole point of the splice is that
    the sample now covers 2011 and 2013, and a regeneration that silently lost
    the history role would still start in 2014 and still pass a start-date-only
    check.
    """
    price = price_series(crypto_observations)
    assert price.index[0].date() == date(2010, 8, 18)
    assert price.index[-1].date() >= date(2026, 8, 1)
    assert len(price) > 5700, "the record is missing the pre-2014 extension"
