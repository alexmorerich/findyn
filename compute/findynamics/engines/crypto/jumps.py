"""Crypto's use of the shared jump detector.

The detector — Lee-Mykland with bipower local volatility and a Gumbel
threshold — lives in :mod:`findynamics.backtest.jumps`. It is the same estimator
FinGold uses, reached through the shared module rather than by importing that
engine: engines may not import each other (``01-target-architecture.md`` §3 rule
2, enforced by ``lint-imports``), and a second copy would be a second detector
that drifts from the first.

What is different here is the calendar and only the calendar. Bitcoin trades
every day of the year, so the intensity annualizes on 365 rather than 252. Left
at the gold default the reported jumps-per-year would be understated by 31% —
silently, in the number the risk score scales against a configured reference.
That is why ``periods_per_year`` is a rule rather than a constant.

What is *not* different is the interpretation, and that deserves saying: a jump
in gold is a crisis signature, because gold spends most of its life not doing
very much. Bitcoin's ordinary week contains moves that would be gold's decade.
The detector handles this correctly by construction — the threshold is scaled by
each date's own trailing bipower volatility, so it asks "large for bitcoin
lately", not "large in absolute terms" — but it means the intensity here is a
measure of how *disorderly* the price process has become relative to its own
recent behaviour, not of how dangerous the asset is. The asset is dangerous
throughout.
"""

from __future__ import annotations

from findynamics.backtest.jumps import (
    C1,
    CALENDAR_DAYS,
    JumpResult,
    JumpRules,
    bipower_sigma,
    detect,
    gumbel_constants,
    gumbel_threshold,
)

__all__ = [
    "C1",
    "CALENDAR_DAYS",
    "JumpResult",
    "JumpRules",
    "bipower_sigma",
    "detect",
    "gumbel_constants",
    "gumbel_threshold",
]
