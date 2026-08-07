"""Gold's use of the shared jump detector, plus the crisis premium it feeds.

A crisis bid does not arrive smoothly. Gold's response to stress is a small
number of very large days, and the monthly regime model cannot see them: March
2020 fell 12% and recovered inside one month, and the month-end return was
+0.6%. The jump intensity is how that violence reaches the state.

The detector itself — Lee-Mykland with bipower local volatility and a Gumbel
threshold — lives in :mod:`findynamics.backtest.jumps`, where a second engine
can use it without importing this one (``01-target-architecture.md`` §3 rule 2).
Read that module's docstring for the mathematics and for why the bipower
estimator is not an implementation detail. What is left here is the part that is
about *gold*: turning an intensity into a crisis premium.

The names are re-exported so gold's own modules and tests keep one import path
for the whole detector.
"""

from __future__ import annotations

import logging

import pandas as pd

from findynamics.backtest.jumps import (
    C1,
    TRADING_DAYS,
    JumpResult,
    JumpRules,
    bipower_sigma,
    detect,
    gumbel_constants,
    gumbel_threshold,
)

log = logging.getLogger("findynamics.engines.gold.jumps")


def crisis_premium(
    intensity: pd.Series,
    stress_z: pd.Series,
    *,
    intensity_reference: float,
    stress_weight: float,
) -> pd.Series:
    """0-1 crisis premium: jump intensity, lifted by financial stress.

    Two inputs because either alone is ambiguous. Jumps without stress are a
    positioning washout — the April 2013 two-day collapse was gold-specific and
    financial conditions never moved. Stress without jumps is a slow tightening,
    which is a headwind rather than a crisis. The premium is the intensity
    scaled against a reference rate, multiplied up when conditions are also
    tight::

        premium = clip(intensity / reference, 0, 1) * (1 + stress_weight * clip(z_stress, 0, ...))

    then clipped back into 0-1. Negative stress does not reduce the premium below
    what the jumps themselves say: calm conditions are not evidence that the
    jumps did not happen.
    """
    if intensity.empty:
        return pd.Series(dtype=float)
    base = (intensity / max(intensity_reference, 1e-9)).clip(0.0, 1.0)
    lift = 1.0 + stress_weight * stress_z.reindex(intensity.index).fillna(0.0).clip(lower=0.0)
    return (base * lift).clip(0.0, 1.0)


__all__ = [
    "C1",
    "TRADING_DAYS",
    "JumpResult",
    "JumpRules",
    "bipower_sigma",
    "crisis_premium",
    "detect",
    "gumbel_constants",
    "gumbel_threshold",
]
