"""FINDYN_V1_SPEC.md §14.1 rule 5 — the replay test, for the feature path.

The most valuable no-lookahead guard in the system, and the only one that can
catch lookahead which entered through *model* code rather than through the data
layer. Every other rule is about how data is read; this one recomputes the answer
and checks it did not change.

Cutoffs are drawn at random from the usable span, with a fixed seed. Random so
the test is not quietly satisfied by the three dates someone happened to pick
while debugging; seeded so a failure is reproducible.
"""

from __future__ import annotations

import random
from datetime import date, timedelta

import pytest

from findynamics.backtest.replay import (
    assert_features_replay,
    feature_history,
    world_at,
)
from findynamics.engines.equity.engine import EquityEngine
from tests.engines.equity.conftest import SNAPSHOT_AS_OF, world_from

#: Fixed seed: reproducible cutoffs, but not cutoffs chosen to pass.
SEED = 20260731

#: The publication series starts 2016-08; the filter needs two years to identify
#: its variances, so replays begin comfortably after that.
EARLIEST_CUTOFF = date(2020, 1, 2)


def random_cutoffs(n: int, *, start: date = EARLIEST_CUTOFF, end: date = SNAPSHOT_AS_OF):
    rng = random.Random(SEED)
    span = (end - start).days
    days = sorted({rng.randrange(span) for _ in range(n * 2)})[:n]
    return [start + timedelta(days=d) for d in days]


@pytest.fixture
def frozen_engine(equity_engine: EquityEngine, equity_observations) -> EquityEngine:
    """The engine with its parameters frozen from the *earliest* cutoff.

    Frozen for the reason spelled out in ``assert_features_replay``: an
    expanding-window refit legitimately moves the Kalman variances and ``d``
    between runs, and a replay that let them move would be measuring the
    estimator rather than the transform.

    Frozen from the earliest cutoff rather than the latest, so the parameters
    every replay uses were themselves knowable at every replay date. Fitting at
    the end and replaying backwards would hold the transform fixed too, but by
    handing the early runs a parameter set estimated on their future — a control
    that works, and a strange thing to write into a lookahead test.
    """
    equity_engine.fit(world_at(equity_observations, EARLIEST_CUTOFF, with_factors=False))
    return equity_engine


def test_the_feature_state_replays_at_random_historical_cutoffs(frozen_engine, equity_observations):
    """Rule 5, in one assertion."""
    cutoffs = [*random_cutoffs(6), SNAPSHOT_AS_OF]
    compared = assert_features_replay(frozen_engine, equity_observations, cutoffs)
    # A green replay over nothing is the failure mode this floor exists for.
    assert compared > 2000, f"only {compared} rows were actually compared"


def test_the_replay_would_catch_a_planted_lookahead(frozen_engine, equity_observations):
    """A test that has never been seen to fail proves nothing.

    The plant is a **full-sample** z-score rather than an expanding one — the
    exact mistake §8.3 rules out, and the realistic one: it is a single call and
    the resulting series looks entirely reasonable. Its values move as the sample
    grows, so the same date scores differently at two cutoffs and the replay
    catches it.

    Worth being explicit about what this design does *not* catch: a feature that
    reads a fixed number of future observations (a ``shift(-1)``) agrees between
    two runs on every date except the last few, because tomorrow's close is
    already inside the earlier information set for everything but the edge. That
    class is caught by ``test_features_never_extend_past_the_information_set``
    and by the causality assertions in ``test_features.py``, not here.
    """
    from findynamics.core.contracts.state import DerivedFeature

    honest = frozen_engine.derived_features

    def peeking(world):
        rows = list(honest(world))
        closes = world.series.wide(["FRED:SP500"])["FRED:SP500"].dropna()
        # Normalized against the whole sample, not an expanding window.
        scored = (closes - closes.mean()) / closes.std()
        rows.extend(
            DerivedFeature(
                asset="equity",
                feature="full_sample_z",
                as_of=key.date(),
                value=float(value),
                model_version=rows[0].model_version if rows else "equity-test",
            )
            for key, value in scored.items()
        )
        return tuple(rows)

    frozen_engine.derived_features = peeking  # type: ignore[method-assign]
    with pytest.raises(AssertionError, match="lookahead"):
        assert_features_replay(
            frozen_engine,
            equity_observations,
            [date(2024, 6, 3), SNAPSHOT_AS_OF],
        )


def test_recomputing_at_the_same_cutoff_is_bit_for_bit_identical(
    frozen_engine, equity_observations
):
    """Reproducibility, separate from causality: no clock, no global state, no
    ordering dependence anywhere in the path."""
    first = feature_history(frozen_engine, equity_observations, date(2023, 3, 15))
    frozen_engine._cache = None  # defeat the per-run memoization
    second = feature_history(frozen_engine, equity_observations, date(2023, 3, 15))
    assert first == second


def test_an_unfrozen_engine_is_still_reproducible_at_its_own_cutoff(
    equity_engine, equity_observations
):
    """The other half of the claim.

    Without frozen parameters the values for a given date *do* move between runs
    — that is an expanding-window estimator working correctly, not lookahead. But
    a run at a given cutoff must still be reproducible from that cutoff alone,
    which is what a daily run's write-back depends on.
    """
    cutoff = date(2022, 9, 30)
    first = feature_history(equity_engine, equity_observations, cutoff)
    equity_engine._cache = None
    second = feature_history(equity_engine, equity_observations, cutoff)
    assert first == second
    assert first, "the engine produced no features to compare"


def test_features_never_extend_past_the_information_set(equity_engine, equity_observations):
    """The blunt version of the same rule: no feature may carry a date after the
    cutoff, whatever the transform did internally."""
    for cutoff in random_cutoffs(4):
        equity_engine._cache = None
        world = world_from(equity_observations, cutoff)
        rows = equity_engine.derived_features(world)
        assert rows, f"no features at {cutoff}"
        assert max(row.as_of for row in rows) <= cutoff
