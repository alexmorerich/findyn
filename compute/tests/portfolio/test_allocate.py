"""The allocator: tilts, distributions, caps, and the degraded fallback."""

from __future__ import annotations

from datetime import date

import pytest

from findynamics.portfolio import compute_allocations
from findynamics.portfolio.allocate import weights_blob
from tests.portfolio.conftest import full_panel_states, load_config, make_state, make_world

AS_OF = date(2024, 6, 28)


def _balanced(states):
    return compute_allocations(states, make_world(AS_OF), profiles=["balanced"])["balanced"]


class TestDistribution:
    def test_every_profile_sums_to_one_and_is_a_distribution(self):
        allocs = compute_allocations(full_panel_states(AS_OF), make_world(AS_OF))
        for alloc in allocs.values():
            assert pytest.approx(sum(b.mean for b in alloc.weights.values()), abs=1e-9) == 1.0
            for band in alloc.weights.values():
                # A distribution, never a single number: the quantiles bracket the mean.
                q = band.quantiles
                assert q["q05"] <= q["q50"] <= q["q95"]

    def test_lower_confidence_widens_the_band(self):
        """The whole method: an unconfident engine gets a wider weight fan."""
        confident = dict(full_panel_states(AS_OF))
        unconfident = dict(full_panel_states(AS_OF))
        unconfident["equity"] = make_state(
            "equity", AS_OF, expected_return=0.085, risk_score=55.0,
            confidence=0.15, regime="normal_expansion",
        )

        eq_confident = _balanced(confident).weights["equity"].quantiles
        eq_unconfident = _balanced(unconfident).weights["equity"].quantiles
        width_c = eq_confident["q95"] - eq_confident["q05"]
        width_u = eq_unconfident["q95"] - eq_unconfident["q05"]
        assert width_u > width_c

    def test_a_high_excess_return_tilts_the_asset_overweight(self):
        states = dict(full_panel_states(AS_OF))
        # Make equity dramatically more attractive; it should lean above neutral.
        states["equity"] = make_state(
            "equity", AS_OF, expected_return=0.20, risk_score=45.0, confidence=0.8,
            regime="bull_expansion",
        )
        alloc = _balanced(states)
        assert alloc.weights["equity"].mean > alloc.weights["equity"].neutral

    def test_an_equal_score_panel_stays_at_neutral(self):
        """When every risk-adjusted excess return is equal, the tilt vanishes."""
        # Give all three risky assets the same sharpe by matching excess/sigma.
        # Simplest: equal expected_return over risk_free and equal risk_score.
        rf = 0.04
        states = {
            "money": make_state("money", AS_OF, expected_return=rf, risk_score=2.0,
                                confidence=0.8, regime="normal"),
            "rates": make_state("rates", AS_OF, expected_return=rf + 0.03, risk_score=40.0,
                                confidence=0.6, regime="flat"),
            "equity": make_state("equity", AS_OF, expected_return=rf + 0.03, risk_score=40.0,
                                 confidence=0.6, regime="normal_expansion"),
            "gold": make_state("gold", AS_OF, expected_return=rf + 0.03, risk_score=40.0,
                               confidence=0.6, regime="hedge_bid"),
        }
        alloc = _balanced(states)
        for band in alloc.weights.values():
            assert band.mean == pytest.approx(band.neutral, abs=2e-3)

    def test_caps_are_respected_in_every_band(self):
        config = load_config()
        balanced = config.profile("balanced")
        states = dict(full_panel_states(AS_OF))
        states["equity"] = make_state(
            "equity", AS_OF, expected_return=0.5, risk_score=30.0, confidence=0.9,
            regime="bull_expansion",
        )
        alloc = _balanced(states)
        # Even the 95th-percentile weight cannot exceed the cap.
        assert alloc.weights["equity"].quantiles["q95"] <= balanced.cap("equity") + 1e-9
        assert alloc.weights["equity"].mean <= balanced.cap("equity") + 1e-9


class TestReproducibility:
    def test_the_same_states_produce_the_same_allocation(self):
        """Deterministic given the seed — the PIT replay guarantee rests on this."""
        a = _balanced(full_panel_states(AS_OF))
        b = _balanced(full_panel_states(AS_OF))
        for asset in a.weights:
            assert a.weights[asset].mean == b.weights[asset].mean
            assert a.weights[asset].quantiles == b.weights[asset].quantiles

    def test_profiles_draw_independently(self):
        allocs = compute_allocations(full_panel_states(AS_OF), make_world(AS_OF))
        # Growth leans harder into equity than conservative — different tilt and
        # different neutral, so the two are not the same allocation.
        assert allocs["growth"].weights["equity"].mean > allocs["conservative"].weights["equity"].mean


class TestSerialization:
    def test_the_blob_carries_the_distribution_and_the_inputs(self):
        alloc = _balanced(full_panel_states(AS_OF))
        blob = weights_blob(alloc)
        assert blob["risk_free"] == pytest.approx(0.045)
        assert {a["asset"] for a in blob["assets"]} == {"equity", "rates", "gold"}
        # Input AssetState references travel with the allocation for the "why" panel.
        assert {i["asset"] for i in blob["inputs"]} == {"equity", "rates", "gold"}
        for asset in blob["assets"]:
            assert set(asset["quantiles"]) == {"q05", "q25", "q50", "q75", "q95"}
