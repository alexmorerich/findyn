"""No-lookahead replay for FinMoney (FINDYN_V1_SPEC.md §14.1 rule 5).

The one test worth more than every other guard combined, because it is the only
one that catches lookahead which entered through *model* code rather than through
the data layer. If a value secretly depends on the future, recomputing it from an
old information set gives a different answer than recomputing it today.

For this engine the risk is concentrated and specific: the wealth index is a
cumulative sum, and every rolling window in the liquidity classifier is a place a
centred window could hide. A cumulative series is also the case where a naive
replay test passes for the wrong reason — so the assertions below check that the
*shared prefix* of two runs agrees, not merely that each run terminates.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from findynamics.backtest.replay import assert_replays, compare, replay, world_at
from findynamics.engines.money.engine import MoneyEngine

#: Cutoffs chosen to span every branch the engine has: the pre-SOFR splice, both
#: stress episodes, and a calm modern date.
CUTOFFS = (
    date(2018, 3, 1),
    date(2019, 9, 19),
    date(2020, 3, 16),
    date(2020, 12, 31),
    date(2024, 10, 31),
)


@pytest.fixture
def engine(config) -> MoneyEngine:
    """A fresh engine; the analysis cache must not carry across cutoffs."""
    return MoneyEngine(config)


def state_at(engine: MoneyEngine, observations: pd.DataFrame, cutoff: date, config):
    engine._cache = None
    world = world_at(observations, cutoff, config=config, with_factors=False)
    return engine.predict(world)


class TestReplay:
    def test_every_cutoff_produces_a_state(self, engine, money_observations, config):
        points = replay(engine, money_observations, CUTOFFS, config=config, with_factors=False)
        assert len(points) == len(CUTOFFS)
        for point in points:
            assert point.ok, f"{point.cutoff}: {point.error}"

    def test_a_stored_run_replays_field_for_field(self, engine, money_observations, config):
        """The §14.1 rule 5 shape: store, then prove it is reproducible."""
        stored = {
            cutoff: state_at(engine, money_observations, cutoff, config) for cutoff in CUTOFFS
        }
        engine._cache = None
        assert_replays(engine, money_observations, stored, config=config, with_factors=False)

    def test_replay_is_deterministic(self, engine, money_observations, config):
        for cutoff in CUTOFFS:
            first = state_at(engine, money_observations, cutoff, config)
            second = state_at(engine, money_observations, cutoff, config)
            assert not compare(first, second, cutoff=cutoff)

    def test_a_state_never_sees_past_its_own_cutoff(self, engine, money_observations, config):
        for cutoff in CUTOFFS:
            state = state_at(engine, money_observations, cutoff, config)
            assert state.as_of <= cutoff

    def test_each_cutoff_reports_its_own_era(self, engine, money_observations, config):
        """A run in 2018 must not know 2020's rates — sanity on the whole exercise."""
        rates = {
            cutoff: state_at(engine, money_observations, cutoff, config).expected_return
            for cutoff in CUTOFFS
        }
        assert rates[date(2018, 3, 1)] == pytest.approx(0.0164, abs=0.002)
        assert rates[date(2020, 12, 31)] < 0.005
        assert rates[date(2024, 10, 31)] > 0.04


class TestCumulativeSeriesDoNotDrift:
    """Where lookahead would actually show up in this engine."""

    def test_the_wealth_index_prefix_is_identical_from_a_later_cutoff(
        self, engine, money_observations, config
    ):
        """A dollar's history cannot be revised by what happened afterwards.

        The strongest available statement: run the engine at an early cutoff and a
        late one, and every date they share must carry the same wealth index to
        floating-point tolerance. A cumulative sum that peeked forward would
        diverge across the whole overlap, not just at the end.
        """
        early = engine.outputs(
            world_at(money_observations, date(2019, 6, 3), config=config, with_factors=False)
        )
        engine._cache = None
        late = engine.outputs(
            world_at(money_observations, date(2020, 12, 31), config=config, with_factors=False)
        )

        def series(rows, metric):
            return {r.as_of: r.value for r in rows if r.metric == metric}

        a, b = series(early, "wealth_index"), series(late, "wealth_index")
        shared = set(a) & set(b)
        assert len(shared) > 300

        for day in sorted(shared):
            assert a[day] == pytest.approx(b[day], rel=1e-12), day

    def test_the_carry_history_prefix_is_identical_too(self, engine, money_observations, config):
        early = engine.outputs(
            world_at(money_observations, date(2019, 6, 3), config=config, with_factors=False)
        )
        engine._cache = None
        late = engine.outputs(
            world_at(money_observations, date(2020, 12, 31), config=config, with_factors=False)
        )

        def series(rows, metric):
            return {r.as_of: r.value for r in rows if r.metric == metric}

        a, b = series(early, "carry_3m"), series(late, "carry_3m")
        shared = set(a) & set(b)
        assert len(shared) > 200
        for day in sorted(shared):
            assert a[day] == pytest.approx(b[day], rel=1e-12), day

    def test_the_liquidity_history_prefix_does_not_get_relabelled(
        self, engine, money_observations, config
    ):
        """September 2019 must read the same from 2019 as it does from 2024."""
        early = engine.outputs(
            world_at(money_observations, date(2019, 10, 15), config=config, with_factors=False)
        )
        engine._cache = None
        late = engine.outputs(
            world_at(money_observations, date(2020, 12, 31), config=config, with_factors=False)
        )

        def labels(rows):
            return {
                r.as_of: (r.meta or {}).get("liquidity")
                for r in rows
                if r.metric == "liquidity_code"
            }

        a, b = labels(early), labels(late)
        shared = set(a) & set(b)
        assert len(shared) > 300
        mismatches = [(d, a[d], b[d]) for d in sorted(shared) if a[d] != b[d]]
        assert not mismatches, mismatches


class TestTheCurveReadBackObeysTheCutoff:
    """The published-output path must be under the same law as everything else."""

    def test_a_cutoff_before_the_curve_was_published_sees_no_curve(
        self, engine, money_observations, config
    ):
        """This is why the NS factors travel as observations rather than objects.

        Rows published on 2020-12-30 describing 2019 dates are invisible to a run
        standing in 2019 — it degrades to the flat short rate instead of
        discounting with a curve that did not exist yet.
        """
        from tests.engines.money.conftest import curve_rows

        described = [date(2019, 9, 16), date(2019, 9, 17), date(2019, 9, 18)]
        frame = pd.concat(
            [
                money_observations,
                pd.DataFrame(
                    curve_rows(
                        {"level": 1.5, "slope": 1.4, "curvature": -0.3, "lambda": 0.609},
                        described,
                        published_on=date(2020, 12, 30),
                    )
                ),
            ],
            ignore_index=True,
        )

        past = state_at(engine, frame, date(2019, 9, 19), config)
        assert "ns_lambda" not in (past.components or {})
        assert any(s.name == "curve_source_degraded" for s in past.signals)

        present = state_at(engine, frame, date(2020, 12, 31), config)
        assert (present.components or {})["ns_lambda"] == 0.609
