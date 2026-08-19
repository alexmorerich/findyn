"""PIT replay of the portfolio state (acceptance §Acceptance).

The portfolio layer reads no series directly — it consumes ``AssetState`` objects
and ``WorldState.as_of`` — so its only lookahead surface is whether an allocation
computed at cutoff *t* depends on anything from a later run. This asserts the
same property ``backtest/replay.py`` asserts for engines: a portfolio state is a
function of its information set alone, and recomputing it from that information
set reproduces it exactly, at more than two cutoffs.
"""

from __future__ import annotations

from datetime import date

from findynamics.portfolio import compute_allocations
from findynamics.portfolio.allocate import weights_blob
from tests.portfolio.conftest import make_state, make_world

# Three month-end cutoffs, each with its own information set. The expected
# returns drift so the allocations genuinely differ between cutoffs — a replay
# that produced the same weights everywhere would prove nothing.
CUTOFFS = (date(2020, 3, 31), date(2020, 9, 30), date(2021, 6, 30))

_EQUITY_ER = {CUTOFFS[0]: -0.02, CUTOFFS[1]: 0.09, CUTOFFS[2]: 0.11}
_EQUITY_CONF = {CUTOFFS[0]: 0.35, CUTOFFS[1]: 0.6, CUTOFFS[2]: 0.7}


def _states_at(cutoff: date) -> dict:
    return {
        "money": make_state(
            "money", cutoff, expected_return=0.01, risk_score=2.0, confidence=0.8, regime="normal"
        ),
        "rates": make_state(
            "rates",
            cutoff,
            expected_return=0.03,
            risk_score=35.0,
            confidence=0.6,
            regime="steep_easing",
        ),
        "equity": make_state(
            "equity",
            cutoff,
            expected_return=_EQUITY_ER[cutoff],
            risk_score=60.0,
            confidence=_EQUITY_CONF[cutoff],
            regime="bear" if cutoff == CUTOFFS[0] else "normal_expansion",
        ),
        "gold": make_state(
            "gold",
            cutoff,
            expected_return=0.05,
            risk_score=42.0,
            confidence=0.5,
            regime="crisis_bid",
        ),
    }


def _alloc_at(cutoff: date):
    return compute_allocations(_states_at(cutoff), make_world(cutoff), profiles=["balanced"])[
        "balanced"
    ]


def test_the_allocation_reproduces_from_its_own_information_set():
    """Recomputing each cutoff yields byte-identical weights — the replay guarantee."""
    for cutoff in CUTOFFS:
        stored = weights_blob(_alloc_at(cutoff))
        replayed = weights_blob(_alloc_at(cutoff))
        assert stored == replayed


def test_a_later_run_does_not_change_an_earlier_cutoff():
    """No state leaks between runs: computing 2021 must not alter 2020's answer."""
    first = weights_blob(_alloc_at(CUTOFFS[0]))
    # Compute the later cutoffs in between, then recompute the first.
    _alloc_at(CUTOFFS[2])
    _alloc_at(CUTOFFS[1])
    again = weights_blob(_alloc_at(CUTOFFS[0]))
    assert first == again


def test_the_cutoffs_genuinely_differ():
    """Guard the guard: if every cutoff allocated identically, the replay is vacuous."""
    equity = [_alloc_at(c).weights["equity"].mean for c in CUTOFFS]
    # The March-2020 bear cutoff should hold less equity than the 2021 one.
    assert equity[0] < equity[2]
    assert len(set(round(w, 6) for w in equity)) == len(CUTOFFS)


def test_the_state_carries_the_cutoff_it_describes():
    for cutoff in CUTOFFS:
        assert _alloc_at(cutoff).as_of == cutoff
