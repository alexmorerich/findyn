"""No-lookahead replay for FinCrypto (FINDYN_V1_SPEC.md §14.1 rule 5).

The one test worth more than every other guard combined: it is the only one that
catches lookahead which entered through *model* code rather than through the data
layer. If a value secretly depends on the future, recomputing it from an old
information set gives a different answer than recomputing it today.

For this engine the risk sits in three places, and all three are exercised below:

* **The expanding regression.** A rolling window, or an accidental full-sample
  mean in the running totals, would move every historical beta whenever a new
  month arrived — and would be invisible in the output, because the numbers
  would look entirely reasonable.
* **The regime's trailing windows.** A centred rolling maximum would let the
  peak a drawdown is measured from be one the market had not made yet.
* **The jump detector's local volatility.** A centred window would let the day
  after a crash inform the day of it.

Unlike gold's replay there is nothing to freeze first. This engine holds no
fitted parameters — every estimator is an expanding closed form recomputed per
run — so the "fit once, then replay" dance that gold needs has no analogue here,
and its absence is itself asserted in ``test_engine.py``.
"""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from findynamics.backtest.replay import assert_replays, compare, replay, world_at
from findynamics.engines.crypto.engine import CryptoEngine

#: Cutoffs spanning the branches the engine has: a frenzy, a winter, a calm
#: recovery and the end of the snapshot. More than the two the phase asks for,
#: because the interesting failures are regime-specific.
CUTOFFS = (
    date(2017, 12, 29),  # frenzy
    date(2018, 12, 31),  # winter
    date(2021, 11, 9),  # frenzy, the November peak
    date(2023, 12, 29),  # normal
    date(2026, 8, 5),  # the snapshot's end
)

#: Days before a cutoff within which a published value may still be restated.
#:
#: The guarantee an engine can make is not "a published value never changes" — it
#: is "a published value stops changing once every input covering that date has
#: been released". M2 is monthly with a 14-day lag from month end, so a run on 31
#: December forward-fills the last money-stock reading it has, and a run in 2026
#: asked about that same December uses the January print. Both are correct on
#: their own information set; demanding they agree would be demanding the first
#: one see the future.
#:
#: 45 days clears the longest configured lag with room to spare. Everything older
#: than this is compared exactly, and the assertion on ``compared`` below fails
#: if the carve-out ever grows into the substance of the test.
SETTLEMENT_DAYS = 45


@pytest.fixture
def engine(config, artifacts) -> CryptoEngine:
    return CryptoEngine(config, artifacts)


def state_at(engine: CryptoEngine, observations, cutoff: date, config):
    engine._cache = None
    world = world_at(observations, cutoff, config=config, with_factors=False)
    return engine.predict(world)


class TestReplay:
    def test_every_cutoff_produces_a_state(self, engine, crypto_observations, config):
        points = replay(engine, crypto_observations, CUTOFFS, config=config, with_factors=False)
        assert len(points) == len(CUTOFFS)
        for point in points:
            assert point.ok, f"{point.cutoff}: {point.error}"

    def test_a_stored_run_replays_field_for_field(self, engine, crypto_observations, config):
        """The §14.1 rule 5 shape: store, then prove it is reproducible."""
        stored = {
            cutoff: state_at(engine, crypto_observations, cutoff, config) for cutoff in CUTOFFS
        }
        engine._cache = None
        assert_replays(engine, crypto_observations, stored, config=config, with_factors=False)

    def test_a_later_run_does_not_rewrite_an_earlier_state(
        self, engine, crypto_observations, config
    ):
        """The sharp version: run the newest cutoff first, then the oldest.

        An expanding statistic that had quietly become a full-sample one would
        show up here and nowhere else, because the engine has already seen 2026
        by the time it is asked about 2017.
        """
        first = state_at(engine, crypto_observations, CUTOFFS[0], config)

        state_at(engine, crypto_observations, CUTOFFS[-1], config)
        again = state_at(engine, crypto_observations, CUTOFFS[0], config)

        mismatches = compare(again, first, cutoff=CUTOFFS[0])
        assert not mismatches, "\n".join(str(m) for m in mismatches)

    def test_the_regime_at_a_past_cutoff_is_the_one_that_was_published(
        self, engine, crypto_observations, config
    ):
        """The acceptance windows, recomputed from their own information sets.

        A run standing in December 2017 must call it a frenzy using only what
        December 2017 knew — not with the benefit of having watched 2018.
        """
        assert state_at(engine, crypto_observations, date(2017, 12, 29), config).regime == "frenzy"
        assert state_at(engine, crypto_observations, date(2018, 12, 31), config).regime == "winter"
        assert state_at(engine, crypto_observations, date(2021, 11, 9), config).regime == "frenzy"
        assert state_at(engine, crypto_observations, date(2022, 12, 30), config).regime == "winter"

    def test_published_outputs_replay_too(self, engine, crypto_observations, config):
        """The state is one row; the charts are tens of thousands, and they replay too.

        ``AssetState`` only carries the newest date, so a transform that reached
        forward on every date *except* the last would replay perfectly at the
        state level. This compares the per-date series, which is where such a bug
        would actually live.
        """
        early_cutoff, late_cutoff = date(2023, 12, 29), date(2026, 8, 5)

        engine._cache = None
        early = {
            (row.metric, row.as_of): row.value
            for row in engine.outputs(
                world_at(crypto_observations, early_cutoff, config=config, with_factors=False)
            )
        }
        engine._cache = None
        late = {
            (row.metric, row.as_of): row.value
            for row in engine.outputs(
                world_at(crypto_observations, late_cutoff, config=config, with_factors=False)
            )
        }

        settled = early_cutoff - timedelta(days=SETTLEMENT_DAYS)
        compared = 0
        failures = []
        for key, value in early.items():
            if key not in late or key[1] > settled:
                continue
            compared += 1
            if abs(value - late[key]) > 1e-9:
                failures.append(f"{key[0]} on {key[1]}: {value!r} then {late[key]!r}")

        assert compared > 10_000, f"only {compared} rows overlapped; the replay compared nothing"
        assert not failures, (
            f"{len(failures)} published value(s) changed when later data arrived — "
            "this is what lookahead looks like:\n  " + "\n  ".join(failures[:20])
        )

    def test_the_liquidity_beta_never_moves_once_published(
        self, engine, crypto_observations, config
    ):
        """The specific trap this engine has.

        A rolling regression would re-estimate every historical month on every
        run and the chart would rewrite itself nightly. Comparing the published
        ``liquidity_beta`` rows across two cutoffs is the direct test of which
        window was used.
        """
        early_cutoff, late_cutoff = date(2021, 11, 9), date(2026, 8, 5)

        def betas(cutoff):
            engine._cache = None
            return {
                row.as_of: row.value
                for row in engine.outputs(
                    world_at(crypto_observations, cutoff, config=config, with_factors=False)
                )
                if row.metric == "liquidity_beta"
            }

        early, late = betas(early_cutoff), betas(late_cutoff)
        settled = early_cutoff - timedelta(days=SETTLEMENT_DAYS)
        shared = [k for k in set(early) & set(late) if k <= settled]

        assert len(shared) > 500, "the two cutoffs published too few overlapping dates"
        moved = [k for k in shared if abs(early[k] - late[k]) > 1e-9]
        assert not moved, (
            f"{len(moved)} historical beta value(s) changed when later months arrived; "
            "that is the signature of a rolling rather than an expanding window"
        )

    def test_the_supply_schedule_is_identical_at_every_cutoff(
        self, engine, crypto_observations, config
    ):
        """No market data enters it, so no cutoff can change it.

        A cheap test of an important property: the scarcity module is the one
        part of this engine that is a fact rather than an estimate, and a run in
        2017 and a run in 2026 must agree exactly about what supply was in 2016.
        """

        def supply(cutoff):
            engine._cache = None
            return {
                row.as_of: row.value
                for row in engine.outputs(
                    world_at(crypto_observations, cutoff, config=config, with_factors=False)
                )
                if row.metric == "issued_supply"
            }

        early, late = supply(date(2018, 12, 31)), supply(date(2026, 8, 5))
        shared = set(early) & set(late)
        assert len(shared) > 100
        assert all(early[k] == late[k] for k in shared)
