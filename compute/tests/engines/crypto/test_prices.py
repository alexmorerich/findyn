"""The stitched price record.

Two kinds of test, split by what each can prove.

The **real snapshot** answers whether this particular splice is honest: the three
validity numbers are measurements of two published series and nothing invented
can tell you whether blockchain.info's daily average tracks Yahoo's close.

**Synthetic pairs** answer whether the *guard* works, which the real data cannot:
the shipped pair passes, so running only against it would test that a splice
happens and never that a bad one is refused. Each refusal branch gets a pair
built to trip exactly one check.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from findynamics.data.accessor import PandasPITAccessor
from findynamics.engines.crypto import prices as prices_mod
from findynamics.engines.crypto.prices import PriceUnavailable
from tests.engines.crypto.conftest import (
    PRICE,
    PRICE_FALLBACK,
    PRICE_HISTORY,
    SERIES_IDS,
    SNAPSHOT_AS_OF,
    close_leg,
    history_leg,
)


@pytest.fixture(scope="module")
def wide(crypto_observations) -> pd.DataFrame:
    return PandasPITAccessor(crypto_observations, SNAPSHOT_AS_OF).wide()


@pytest.fixture(scope="module")
def record(wide) -> prices_mod.PriceRecord:
    return prices_mod.build(wide, SERIES_IDS)


class TestTheRealSplice:
    def test_the_record_reaches_back_to_the_first_market_price(self, record):
        assert record.spliced
        assert record.price.index[0].date() == date(2010, 8, 18)
        assert len(record.price) > 5700

    def test_it_gains_the_two_cycles_the_closes_cannot_see(self, record):
        """The whole reason for the phase.

        Yahoo starts 2014-09-17, which cuts 2011 and 2013 out of a sample that
        has six cycles in it. Asserting on the *count* of pre-2014 observations
        rather than only on the start date, because a one-row extension would
        also move the start date.
        """
        pre_2014 = record.price.loc[: pd.Timestamp("2014-09-16")]
        assert len(pre_2014) > 1400
        assert pre_2014.index[0].date() == date(2010, 8, 18)

    def test_the_three_validity_numbers_are_what_the_docstring_claims(self, wide):
        """The measurements the splice decision rests on, pinned.

        These are quoted in ``prices.py`` and in ``series.yaml`` as the
        justification for joining a daily average to a daily close. Quoting a
        number in a docstring and never asserting it is how a docstring becomes
        wrong, so they are asserted here — loosely, because they move slightly as
        the snapshot is regenerated, and a regeneration that moved them a lot
        would be telling us the sources had changed.
        """
        check = prices_mod.measure_splice(close_leg_from(wide), history_leg_from(wide))
        assert check.overlap > 4000
        assert abs(check.seam_return) < 0.05
        assert abs(check.level_bias) < 0.01
        assert 0.95 < check.volatility_ratio < 1.10
        assert check.refusal() is None

    def test_the_closes_are_never_restated_by_the_extension(self, record, wide):
        """The extension supplies dates before the closes and nothing else.

        A splice that let an average overwrite a close would silently restate
        published figures — and the two disagree by a median 1.5%, so it would do
        so visibly and wrongly.
        """
        closes = close_leg_from(wide)
        shared = record.price.index.intersection(closes.index)
        assert len(shared) > 4000
        pd.testing.assert_series_equal(
            record.price.loc[shared], closes.loc[shared], check_names=False
        )

    def test_provenance_is_recorded_per_date(self, record):
        """Per date, like gold's ex-post real-rate flag — not summarised away."""
        assert record.is_daily_average.loc[pd.Timestamp("2012-06-01")]
        assert not record.is_daily_average.loc[pd.Timestamp("2020-03-12")]
        assert 0.2 < record.average_share < 0.3

    def test_the_record_has_no_duplicate_dates(self, record):
        """Trailing windows read this index positionally.

        One duplicated date would shift every window after it by one, silently.
        """
        assert not record.price.index.has_duplicates
        assert record.price.index.is_monotonic_increasing
        assert (record.price > 0).all()

    def test_the_identity_names_both_vendors(self, record):
        assert record.series_id == f"{PRICE_FALLBACK}+{PRICE_HISTORY}"

    def test_the_configured_primary_is_absent_and_that_is_reported(self, record):
        """Stooq is bot-filtered from every automated egress this project has."""
        assert record.close_role == "price_fallback"
        assert record.from_fallback is True


def close_leg_from(wide: pd.DataFrame) -> pd.Series:
    values = wide[PRICE_FALLBACK].dropna()
    return values[values > 0]


def history_leg_from(wide: pd.DataFrame) -> pd.Series:
    values = wide[PRICE_HISTORY].dropna()
    return values[values > 0]


def _pair(
    *,
    closes_from: str = "2014-09-17",
    n_closes: int = 800,
    n_history: int = 1200,
    level: float = 400.0,
    overlap_drift: float = 0.0,
    overlap_smoothing: int = 1,
    seam_jump: float = 1.0,
    seed: int = 7,
) -> pd.DataFrame:
    """A close leg and a history leg, perturbed one property at a time.

    The default pair is a *good* splice: the history is the closes over the
    overlap and a continuous random walk before them, so it trips nothing. Each
    keyword breaks exactly one of the three checks and leaves the other two
    intact, which is the only way to know which branch a refusal came from.

    * ``overlap_drift`` — the history drifts away from the closes across their
      shared span while still joining cleanly at the seam. A drifting index.
    * ``overlap_smoothing`` — the history is a rolling mean of the closes: same
      level, materially lower volatility. An exaggerated daily average.
    * ``seam_jump`` — the prefix is rebased, so joining invents a price move.
    """
    rng = np.random.default_rng(seed)
    close_index = pd.date_range(closes_from, periods=n_closes, freq="D")
    closes = pd.Series(
        level * np.exp(np.cumsum(rng.normal(0.0, 0.03, n_closes))), index=close_index
    )

    history_index = pd.date_range(end=close_index[-1], periods=n_history, freq="D")
    shared = history_index.intersection(close_index)

    # Over the overlap the history starts as a copy of the closes, then takes
    # whichever single perturbation was asked for.
    overlap = closes.loc[shared].copy()
    if overlap_smoothing > 1:
        smoothed = overlap.rolling(overlap_smoothing, min_periods=1).mean()
        # Re-levelled onto the closes. A rolling mean of a trending walk lags it,
        # which would introduce a level offset and trip the *bias* check first —
        # realistic, but then the test would not be exercising the branch it
        # names. Matching the means leaves volatility as the only difference.
        overlap = smoothed * (overlap.mean() / smoothed.mean())
    if overlap_drift:
        overlap = overlap * np.linspace(1.0, 1.0 + overlap_drift, len(overlap))

    # The prefix walks backwards from the first close, so by construction it
    # joins without a step unless `seam_jump` puts one there.
    before = history_index[history_index < close_index[0]]
    if len(before):
        walk = np.exp(np.cumsum(rng.normal(0.0, 0.03, len(before))))
        prefix = pd.Series(closes.iloc[0] * seam_jump * (walk / walk[-1]), index=before)
    else:
        prefix = pd.Series(dtype=float)

    history = pd.concat([prefix, overlap]).sort_index()
    return pd.DataFrame({PRICE_FALLBACK: closes, PRICE_HISTORY: history}).sort_index()


IDS = {"price": PRICE, "price_fallback": PRICE_FALLBACK, "price_history": PRICE_HISTORY}


class TestTheGuardRefusesABadSplice:
    """The shipped pair passes, so every refusal branch needs a built one."""

    def test_a_step_at_the_seam_is_refused(self):
        """A rebasing, not a market day. Joining it would invent a price move."""
        frame = _pair(seam_jump=3.0)
        record = prices_mod.build(frame, IDS)

        assert not record.spliced
        assert record.declined_reason is not None
        assert "across the seam" in record.declined_reason
        # And the record falls back to the closes rather than to nothing.
        assert record.price.index[0] == frame[PRICE_FALLBACK].dropna().index[0]

    def test_a_persistent_level_offset_is_refused(self):
        """A different series, not a spread between venues."""
        record = prices_mod.build(_pair(overlap_drift=0.4), IDS)

        assert not record.spliced
        assert "on average" in (record.declined_reason or "")

    def test_a_differently_volatile_extension_is_refused(self):
        """The regime and risk score would inherit it.

        This engine's whole output is a function of the return process, so an
        extension that is calm where the asset was not would publish a calm 2011.
        """
        record = prices_mod.build(_pair(overlap_smoothing=30), IDS)

        assert not record.spliced
        assert "volatility" in (record.declined_reason or "")

    def test_too_little_overlap_is_refused(self):
        """Below a few hundred shared dates the comparison is anecdote."""
        record = prices_mod.build(_pair(n_closes=100, n_history=150), IDS)

        assert not record.spliced
        assert "share only" in (record.declined_reason or "")

    def test_a_refusal_is_carried_on_the_record_not_swallowed(self):
        """The engine takes a confidence penalty for it and the page says so.

        A silently-shortened sample is the failure this whole module exists to
        make visible: a run that quietly published 2014-onwards where 2010 was
        expected looks exactly like a run that worked.
        """
        record = prices_mod.build(_pair(overlap_drift=0.4), IDS)
        assert record.history_series_id == PRICE_HISTORY
        assert record.declined_reason is not None
        assert record.average_share == 0.0


class TestDegradation:
    def test_the_primary_wins_when_it_is_present(self):
        """Stooq is unreachable today; the day it is not, it takes over."""
        frame = _pair()
        frame[PRICE] = frame[PRICE_FALLBACK] * 1.0
        record = prices_mod.build(frame, IDS)

        assert record.close_role == "price"
        assert record.from_fallback is False

    def test_the_history_role_alone_still_produces_a_record(self):
        """Better than nothing, and labelled as entirely average-based."""
        frame = _pair()[[PRICE_HISTORY]]
        record = prices_mod.build(frame, IDS)

        assert not record.price.empty
        assert record.average_share == 1.0
        assert record.is_daily_average.all()

    def test_no_price_role_at_all_raises_rather_than_returning_empty(self):
        with pytest.raises(PriceUnavailable, match="no configured price role"):
            prices_mod.build(pd.DataFrame(index=pd.DatetimeIndex([])), IDS)

    def test_an_extension_that_reaches_no_further_back_is_a_no_op(self):
        frame = _pair(n_history=400)
        frame[PRICE_HISTORY] = frame[PRICE_HISTORY].where(
            frame.index >= frame[PRICE_FALLBACK].dropna().index[0]
        )
        record = prices_mod.build(frame, IDS)

        assert not record.spliced
        assert record.declined_reason is None  # nothing was wrong; there was nothing to add


class TestCausality:
    def test_the_record_at_a_past_cutoff_is_a_prefix_of_the_record_today(self, crypto_observations):
        """No lookahead through the splice.

        The decision to splice is made from the overlap knowable at the cutoff,
        so in principle a later run could decide differently and restate history.
        This asserts it does not: every price a 2018 run published is the price a
        2026 run publishes for that date.
        """
        early = PandasPITAccessor(crypto_observations, date(2018, 12, 31)).wide()
        late = PandasPITAccessor(crypto_observations, SNAPSHOT_AS_OF).wide()

        early_record = prices_mod.build(early, SERIES_IDS)
        late_record = prices_mod.build(late, SERIES_IDS)

        shared = early_record.price.index.intersection(late_record.price.index)
        assert len(shared) > 2500
        pd.testing.assert_series_equal(
            early_record.price.loc[shared],
            late_record.price.loc[shared],
            check_names=False,
        )

    def test_the_provenance_flags_also_replay(self, crypto_observations):
        early = PandasPITAccessor(crypto_observations, date(2018, 12, 31)).wide()
        late = PandasPITAccessor(crypto_observations, SNAPSHOT_AS_OF).wide()

        early_record = prices_mod.build(early, SERIES_IDS)
        late_record = prices_mod.build(late, SERIES_IDS)

        shared = early_record.price.index.intersection(late_record.price.index)
        pd.testing.assert_series_equal(
            early_record.is_daily_average.loc[shared],
            late_record.is_daily_average.loc[shared],
            check_names=False,
        )


def test_the_fixture_carries_both_legs(crypto_observations):
    """Guard: a regeneration that lost the history role must fail loudly here."""
    assert not close_leg(crypto_observations).empty
    assert not history_leg(crypto_observations).empty
    assert history_leg(crypto_observations).index[0].date() == date(2010, 8, 18)
