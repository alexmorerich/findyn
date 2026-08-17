"""Chaos: kill one engine's data and the allocation degrades gracefully.

The acceptance is end-to-end (API 200, neutral fallback, degraded badge); this is
the compute half — that the allocator, given a missing or frozen engine, holds
that asset at neutral, keeps summing to 1, and flags ``degraded`` with a reason.
The serving half lives in ``serving/test/p6-portfolio.spec.ts``.
"""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from findynamics.portfolio import compute_allocations
from findynamics.portfolio.allocate import weights_blob
from tests.portfolio.conftest import full_panel_states, make_state, make_world

AS_OF = date(2024, 6, 28)


def _balanced(states):
    return compute_allocations(states, make_world(AS_OF), profiles=["balanced"])["balanced"]


def test_a_missing_engine_falls_back_to_neutral_and_flags_degraded():
    states = dict(full_panel_states(AS_OF))
    del states["equity"]  # equity engine's data is gone

    alloc = _balanced(states)

    assert alloc.degraded is True
    assert alloc.weights["equity"].degraded is True
    # Pinned exactly at neutral, not tilted.
    assert alloc.weights["equity"].mean == pytest.approx(alloc.weights["equity"].neutral)
    # And its band is a point mass — no fabricated uncertainty on a missing input.
    q = alloc.weights["equity"].quantiles
    assert q["q05"] == pytest.approx(q["q95"])
    # The rest of the allocation still sums to 1.
    assert pytest.approx(sum(b.mean for b in alloc.weights.values()), abs=1e-9) == 1.0
    assert any("equity" in reason for reason in alloc.degraded_reason)


def test_a_frozen_engine_reads_as_stale():
    """Simulate staleness the way the chaos test does: freeze one engine's as_of."""
    states = dict(full_panel_states(AS_OF))
    states["gold"] = make_state(
        "gold", AS_OF - timedelta(days=14), expected_return=0.03, risk_score=40.0,
        confidence=0.5, regime="hedge_bid",
    )
    alloc = _balanced(states)
    assert alloc.weights["gold"].degraded is True
    assert alloc.weights["gold"].mean == pytest.approx(alloc.weights["gold"].neutral)


def test_the_degraded_flag_and_reason_survive_into_the_blob():
    """What the API serves must carry the degradation, not just the compute object."""
    states = dict(full_panel_states(AS_OF))
    del states["rates"]
    alloc = _balanced(states)
    blob = weights_blob(alloc)

    rates = next(a for a in blob["assets"] if a["asset"] == "rates")
    assert rates["degraded"] is True
    assert rates["mean"] == pytest.approx(rates["neutral"])
    assert blob["degraded_reason"]  # non-empty


def test_all_engines_down_returns_the_static_neutral_mix():
    """The worst case is still a valid, honest allocation: the strategic mix."""
    alloc = _balanced({})
    assert alloc.degraded is True
    for band in alloc.weights.values():
        assert band.mean == pytest.approx(band.neutral)
    assert pytest.approx(sum(b.mean for b in alloc.weights.values()), abs=1e-9) == 1.0
    # The implication says the panel was partial.
    assert "degraded" in alloc.implication.lower()
