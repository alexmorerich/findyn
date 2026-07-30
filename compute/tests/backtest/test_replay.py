"""Point-in-time replay — FINDYN_V1_SPEC.md §14.1 rule 5.

The test the whole no-lookahead architecture exists to pass: recompute a
historical state using only what had been released by that date, and demand it
matches the state the run published at the time.

It is the only guard that catches lookahead which entered through *model* code
rather than through the data layer. `pit_join` can be perfect and a feature that
secretly reads the full sample will still fail here, because the value computed
standing at the cutoff will differ from the value computed standing today.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import date

import pandas as pd
import pytest

from findynamics.backtest.replay import (
    ReplayPoint,
    assert_replays,
    compare,
    replay,
    world_at,
)
from findynamics.core.contracts.state import AssetState, Signal
from findynamics.engines.rates.engine import RatesEngine
from tests.engines.rates.conftest import synthetic_curve_frame, world_from

# Three cutoffs spread across the fixture's history, each with plenty of curve
# behind it and plenty ahead of it — a cutoff at the end proves nothing.
CUTOFFS = [date(2007, 6, 30), date(2015, 6, 30), date(2023, 6, 30)]


@pytest.fixture
def monthly_config(config):
    """The fixture is month-end, so the engine's windows are retuned to match."""
    from tests.engines.rates.conftest import _with_params, monthly_params

    return _with_params(config, monthly_params(config))


@pytest.fixture
def engine(monthly_config, artifacts):
    return RatesEngine(monthly_config, artifacts)


class TestWorldAt:
    def test_binds_the_cutoff_to_the_accessor(self, treasury_observations):
        world = world_at(treasury_observations, date(2010, 1, 4), with_factors=False)
        assert world.as_of == date(2010, 1, 4)
        assert world.series.as_of == date(2010, 1, 4)

    def test_nothing_released_after_the_cutoff_is_visible(self, treasury_observations):
        world = world_at(treasury_observations, date(2010, 1, 4), with_factors=False)
        history = world.series.history(["FRED:DGS10"])
        assert history["release_date"].max() <= pd.Timestamp("2010-01-04")


class TestReplayRatesEngine:
    """The §14.1 rule-5 pattern, applied to FinRates."""

    def test_a_state_recomputed_at_its_cutoff_matches_the_stored_run(
        self, engine, treasury_observations
    ):
        # "Stored" run: what the engine published standing at each cutoff.
        stored = {
            cutoff: engine.predict(world_from(treasury_observations, cutoff)) for cutoff in CUTOFFS
        }

        # Replay from scratch, each on its own information set. A fresh engine
        # so nothing is carried over in the analysis cache.
        assert_replays(
            RatesEngine(engine._config, engine._artifacts),
            treasury_observations,
            stored,
            with_factors=False,
        )

    def test_replaying_with_the_future_appended_changes_nothing(
        self, engine, treasury_observations
    ):
        """The real lookahead test: adding data *after* the cutoff must not move it.

        If any part of the pipeline reached past its information set — a
        full-sample percentile, a centred window, a normalization over the whole
        frame — this is where it shows up.
        """
        cutoff = date(2015, 6, 30)

        truncated = treasury_observations[
            treasury_observations["release_date"] <= pd.Timestamp(cutoff)
        ]
        from_truncated = engine.predict(world_from(truncated, cutoff))
        from_full = RatesEngine(engine._config, engine._artifacts).predict(
            world_from(treasury_observations, cutoff)
        )

        assert compare(from_truncated, from_full, cutoff=cutoff) == []

    def test_each_cutoff_sees_only_its_own_history(self, engine, treasury_observations):
        points = replay(engine, treasury_observations, CUTOFFS, with_factors=False)

        assert all(p.ok for p in points)
        for point in points:
            assert point.state is not None
            assert point.state.as_of <= point.cutoff

    def test_states_differ_across_cutoffs(self, engine, treasury_observations):
        """A replay that returns the same answer everywhere is testing nothing."""
        points = replay(engine, treasury_observations, CUTOFFS, with_factors=False)
        levels = {p.state.components["ns_level"] for p in points if p.state and p.state.components}
        assert len(levels) == len(CUTOFFS)

    def test_a_cutoff_with_no_data_yet_is_recorded_not_raised(self, engine, treasury_observations):
        """The early years of any series legitimately have nothing to say."""
        points = replay(engine, treasury_observations, [date(1950, 1, 1)], with_factors=False)

        assert len(points) == 1
        assert not points[0].ok
        assert points[0].error is not None

    def test_a_divergence_is_reported_rather_than_passing_quietly(
        self, engine, treasury_observations
    ):
        actual = engine.predict(world_from(treasury_observations, CUTOFFS[0]))
        # Both fields are perturbed away from whatever the engine actually said,
        # so the test does not depend on what that happened to be.
        stored = {
            CUTOFFS[0]: replace(
                actual,
                regime="flat" if actual.regime != "flat" else "inverted",
                risk_score=(actual.risk_score + 50.0) % 100.0,
            )
        }

        with pytest.raises(AssertionError) as excinfo:
            assert_replays(engine, treasury_observations, stored, with_factors=False)

        message = str(excinfo.value)
        assert "regime" in message
        assert "risk_score" in message


class TestReplayOnSyntheticData:
    def test_detects_a_state_that_depends_on_the_future(self, engine, config, artifacts):
        """A deliberately contaminated engine must fail the replay.

        Without this, a green replay only proves the harness runs.
        """
        betas = [(3.0 + i * 0.01, -1.0, 0.5) for i in range(400)]
        observations = synthetic_curve_frame(betas, start=date(2020, 1, 1))
        cutoff = date(2020, 6, 1)

        class LookaheadEngine(RatesEngine):
            """Reads the last row of the frame instead of the last knowable one."""

            def predict(self, world):
                state = super().predict(world)
                # The contamination: a number from the end of the *table*, which
                # at an early cutoff has not happened yet.
                contaminated = float(observations["value"].iloc[-1])
                return replace(state, risk_score=min(abs(contaminated), 100.0))

        honest = RatesEngine(config, artifacts)
        stored = {cutoff: honest.predict(world_from(observations, cutoff))}

        with pytest.raises(AssertionError, match="risk_score"):
            assert_replays(
                LookaheadEngine(config, artifacts), observations, stored, with_factors=False
            )


class TestCompare:
    def _state(self, **overrides) -> AssetState:
        base = {
            "asset": "rates",
            "as_of": date(2026, 7, 28),
            "regime": "flat",
            "expected_return": 0.04,
            "risk_score": 30.0,
            "confidence": 0.8,
            "signals": (Signal(name="curve_inversion", value=0.1, direction=0),),
            "model_version": "rates-1.0.0",
            "components": {"ns_level": 4.0},
        }
        return AssetState(**{**base, **overrides})

    def test_identical_states_have_no_mismatches(self):
        assert compare(self._state(), self._state(), cutoff=date(2026, 7, 28)) == []

    def test_float_noise_within_tolerance_is_not_a_divergence(self):
        """The same sum in a different order can differ in the last bit."""
        left = self._state(risk_score=30.0)
        right = self._state(risk_score=30.0 + 1e-12)
        assert compare(left, right, cutoff=date(2026, 7, 28)) == []

    def test_a_real_difference_is_reported(self):
        mismatches = compare(
            self._state(risk_score=30.0), self._state(risk_score=31.0), cutoff=date(2026, 7, 28)
        )
        assert [m.field for m in mismatches] == ["risk_score"]

    def test_model_version_is_compared(self):
        """Matching numbers from a different model is not the same state."""
        mismatches = compare(
            self._state(), self._state(model_version="rates-2.0.0"), cutoff=date(2026, 7, 28)
        )
        assert [m.field for m in mismatches] == ["model_version"]

    def test_components_are_compared_by_key(self):
        mismatches = compare(
            self._state(components={"ns_level": 4.0}),
            self._state(components={"ns_level": 4.0, "ns_slope": 1.0}),
            cutoff=date(2026, 7, 28),
        )
        assert [m.field for m in mismatches] == ["components[ns_slope]"]

    def test_a_mismatch_renders_readably(self):
        mismatch = compare(
            self._state(regime="flat"), self._state(regime="inverted"), cutoff=date(2026, 7, 28)
        )[0]
        assert "regime" in str(mismatch)
        assert "inverted" in str(mismatch)


class TestReplayPoint:
    def test_ok_reflects_whether_a_state_was_produced(self):
        assert ReplayPoint(cutoff=date(2020, 1, 1), state=None, error="nope").ok is False
