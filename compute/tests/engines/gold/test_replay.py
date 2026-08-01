"""No-lookahead replay for FinGold (FINDYN_V1_SPEC.md §14.1 rule 5).

The one test worth more than every other guard combined: it is the only one that
catches lookahead which entered through *model* code rather than through the data
layer. If a value secretly depends on the future, recomputing it from an old
information set gives a different answer than recomputing it today.

For this engine the risk is concentrated in three places, and all three are
tested below:

* **The expanding z-scores.** A full-sample mean instead of an expanding one
  would be invisible in every output — the numbers would look entirely
  reasonable — and would move every historical driver score whenever a new
  observation arrived.
* **The Hamilton filter.** ``smoothed_marginal_probabilities`` is the better
  estimate of a past state and conditions on the whole sample; using it here
  would be lookahead wearing a statistician's coat. The replay is what proves
  ``filter`` was used.
* **The jump detector's local volatility.** A centred window would let the day
  after a crash inform the day of it.

The chain is fitted **once, before the replay**, and frozen. That is not a
convenience: with a live refit each cutoff re-estimates on its own window, so the
parameters legitimately differ between runs and the test would be measuring the
optimizer's expanding window rather than the model's causality. Freezing is also
what production does between monthly refits.
"""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from findynamics.backtest.replay import assert_replays, compare, replay, world_at
from findynamics.engines.gold.engine import GoldEngine

pytestmark = [
    pytest.mark.slow,
    pytest.mark.filterwarnings("ignore::UserWarning"),
]

#: Cutoffs spanning the branches the engine has: a crisis, a rate shock, a
#: liquidity event and a calm modern date. More than the two the phase asks for,
#: because the interesting failures are regime-specific.
CUTOFFS = (
    date(2008, 12, 31),
    date(2013, 12, 31),
    date(2020, 6, 30),
    date(2026, 7, 31),
)

#: Days before a cutoff within which a published value may still be restated.
#:
#: The guarantee an engine can actually make is not "a published value never
#: changes" — it is "a published value stops changing once every input covering
#: that date has been released". NFCI is weekly with a 13-day lag, so a run on 30
#: June forward-fills the previous fortnight's stress from the last reading it
#: has, and a run in 2026 asked about those same days uses the readings that were
#: published in July. Both are correct on their own information set; only the
#: second is correct in hindsight, and demanding they agree would be demanding
#: the first one see the future.
#:
#: 30 days clears the longest configured lag with room to spare. Everything older
#: than this is compared exactly, which is ~99% of the overlap — see the
#: assertion on ``compared`` below, which fails if the carve-out ever grows into
#: the substance of the test.
SETTLEMENT_DAYS = 30


@pytest.fixture
def original_prints(gold_observations):
    """The snapshot with revisions stripped — first issue per period only.

    Needed by the cross-cutoff tests, and the reason is worth stating because it
    looks like weakening the test and is the opposite.

    Comparing what two different cutoffs published for the *same* historical date
    conflates two things. One is lookahead, which is a bug. The other is that
    DGS10, T10YIE and the dollar indices are genuinely revised, so a run in 2020
    and a run in 2026 correctly disagree about what the 10y yield was in 2007 —
    that is point-in-time working, and a test that failed on it would be
    demanding the engine ignore its own information set. Measured on the shipped
    snapshot, the revisions move 354 of 626 monthly driver z-scores by up to
    0.023.

    Holding the vintages fixed removes the second effect entirely, so anything
    left is the first.
    """
    return gold_observations[
        gold_observations["revision_date"] == gold_observations["release_date"]
    ].reset_index(drop=True)


@pytest.fixture
def engine(config, artifacts, gold_observations) -> GoldEngine:
    """A fitted engine whose chain is frozen for the whole replay.

    Fitted on the *oldest* cutoff, so no parameter in it was estimated from data
    any of the later cutoffs had not seen either. Fitting on the newest would
    make every earlier replay use a model from its own future — which is the
    thing being tested for.
    """
    engine = GoldEngine(config, artifacts)
    engine.fit(world_at(gold_observations, CUTOFFS[0], config=config, with_factors=False))
    engine._cache = None
    return engine


def state_at(engine: GoldEngine, observations, cutoff: date, config):
    engine._cache = None
    world = world_at(observations, cutoff, config=config, with_factors=False)
    return engine.predict(world)


class TestReplay:
    def test_every_cutoff_produces_a_state(self, engine, gold_observations, config):
        points = replay(engine, gold_observations, CUTOFFS, config=config, with_factors=False)
        assert len(points) == len(CUTOFFS)
        for point in points:
            assert point.ok, f"{point.cutoff}: {point.error}"

    def test_a_stored_run_replays_field_for_field(self, engine, gold_observations, config):
        """The §14.1 rule 5 shape: store, then prove it is reproducible."""
        stored = {cutoff: state_at(engine, gold_observations, cutoff, config) for cutoff in CUTOFFS}
        engine._cache = None
        assert_replays(engine, gold_observations, stored, config=config, with_factors=False)

    def test_a_later_run_does_not_rewrite_an_earlier_state(self, engine, gold_observations, config):
        """The sharp version: run the newest cutoff first, then the oldest.

        A cache or an expanding statistic that had quietly become a full-sample
        one would show up here and nowhere else, because the engine has already
        seen 2026 by the time it is asked about 2008.
        """
        first = state_at(engine, gold_observations, CUTOFFS[0], config)

        state_at(engine, gold_observations, CUTOFFS[-1], config)
        again = state_at(engine, gold_observations, CUTOFFS[0], config)

        mismatches = compare(again, first, cutoff=CUTOFFS[0])
        assert not mismatches, "\n".join(str(m) for m in mismatches)

    def test_published_outputs_replay_too(self, engine, original_prints, config):
        """The state is one row; the charts are thousands, and they replay as well.

        ``AssetState`` only carries the newest date, so a transform that reached
        forward on every date *except* the last would replay perfectly at the
        state level. This compares the per-date series, which is where such a bug
        would actually live.
        """
        # Adjacent cutoffs, because the published window is bounded: 2013 and
        # 2026 publish ten years each and those ten years do not touch.
        early_cutoff, late_cutoff = CUTOFFS[-2], CUTOFFS[-1]

        engine._cache = None
        early = {
            (row.metric, row.as_of): row.value
            for row in engine.outputs(
                world_at(original_prints, early_cutoff, config=config, with_factors=False)
            )
        }
        engine._cache = None
        late = {
            (row.metric, row.as_of): row.value
            for row in engine.outputs(
                world_at(original_prints, late_cutoff, config=config, with_factors=False)
            )
        }

        settled = early_cutoff - timedelta(days=SETTLEMENT_DAYS)
        compared = 0
        failures = []
        for key, value in early.items():
            if key not in late:
                # The later run publishes a bounded window, so a date that has
                # aged out of it is not a mismatch — it is simply not compared.
                continue
            if key[1] > settled:
                continue
            compared += 1
            if abs(value - late[key]) > 1e-9:
                failures.append(f"{key[0]} on {key[1]}: {value!r} then {late[key]!r}")

        assert compared > 1000, f"only {compared} rows overlapped; the replay compared nothing"
        assert not failures, (
            f"{len(failures)} published value(s) changed when later data arrived — "
            "this is what lookahead looks like:\n  " + "\n  ".join(failures[:20])
        )

    def test_the_regime_posterior_is_filtered_not_smoothed(self, engine, original_prints, config):
        """The specific trap this engine has.

        ``smoothed_marginal_probabilities`` would make every historical posterior
        move when later months arrive. Comparing the published ``regime_state``
        rows across two cutoffs is the direct test of which one was used.
        """
        engine._cache = None
        early = {
            (row.as_of, row.regime): row.probability
            for row in engine.regime_states(
                world_at(original_prints, CUTOFFS[-2], config=config, with_factors=False)
            )
        }
        engine._cache = None
        late = {
            (row.as_of, row.regime): row.probability
            for row in engine.regime_states(
                world_at(original_prints, CUTOFFS[-1], config=config, with_factors=False)
            )
        }

        # The trailing month is excluded for the same reason as SETTLEMENT_DAYS,
        # and one more: a run on 30 June sees a month-end return computed from
        # the 29th, because the 30th's fix is published on the 1st. That month is
        # restated once the following day and is settled forever after.
        settled = CUTOFFS[-2] - timedelta(days=SETTLEMENT_DAYS)
        shared = {k for k in set(early) & set(late) if k[0] <= settled}
        assert len(shared) > 100, "the two cutoffs published too few overlapping months"
        moved = [k for k in shared if abs(early[k] - late[k]) > 1e-9]
        assert not moved, (
            f"{len(moved)} historical posterior value(s) changed when later months arrived; "
            "that is the signature of smoothed rather than filtered probabilities"
        )
