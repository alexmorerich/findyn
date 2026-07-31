"""Velocity, acceleration and the jerk indicator (FINDYN_V1_SPEC.md §2.1, §3.1, §8.3).

Derivatives are taken off the *filtered state*, never off raw price. Differencing
a noisy series amplifies its noise by roughly the differencing order, so a third
derivative of a raw index is almost entirely microstructure. The Kalman slope is
already an estimate of the trend, so its first difference is an estimate of how
that trend is changing rather than of how the last two closes happened to land.

The demotion in §3.1 is respected literally:

* **Velocity** and **acceleration** are published as quantities.
* **Jerk** is published only as an expanding-window z-score with a threshold
  reading — a *trend instability indicator*, not a physical third derivative.
  Nothing consumes the raw ``Δacceleration``; the RII and the regime features
  take the z-score.
* **Snap does not exist.** It is replaced entirely by the Regime Instability
  Index (sub-milestone C), which is a composite, not a fourth derivative.

Units. The Kalman slope is a log change *per observation*. Annualizing it here,
where the series' own ``periods_per_year`` is known, is what lets a daily
velocity and a monthly one be plotted on the same axis and mean the same thing.
Acceleration is therefore per year squared.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

log = logging.getLogger("findynamics.engines.equity.features.kinematics")

#: §3.1 — |z| above this is "elevated" trend instability.
JERK_ELEVATED_Z = 2.0

#: The second lamp position. Not in the spec, which stops at elevated; a single
#: threshold gives a dashboard lamp only two states, and the difference between
#: a 2-sigma and a 5-sigma move is exactly what the panel is there to show.
JERK_EXTREME_Z = 3.0

#: §8.3 — z-normalization baseline, expanding, minimum ten years.
DEFAULT_Z_MIN_YEARS = 10.0

#: Floor for the baseline when the series is shorter than ten years. Two years
#: of observations is not a decade, and a z-score built on it is noisier — which
#: is why the effective baseline is reported rather than assumed.
MIN_Z_YEARS = 2.0

#: Momentum windows in months (§2.1: 21/63/252 trading days). Expressed in
#: months rather than observations so that the same numbers mean the same spans
#: on the daily paths and on the monthly deep history.
DEFAULT_MOMENTUM_MONTHS: tuple[int, ...] = (1, 3, 12)


@dataclass(frozen=True)
class Kinematics:
    """The kinematic block for one series. All series share its index."""

    #: Annualized log drift — the Kalman slope times ``periods_per_year``.
    velocity: pd.Series
    #: Change in velocity, per year squared.
    acceleration: pd.Series
    #: Expanding z-score of Δacceleration. The only published form of jerk.
    jerk_z: pd.Series
    #: Trailing sums of the FFD series, keyed ``momentum_1m`` etc.
    momentum: dict[str, pd.Series]
    #: Observations the z-score baseline required before producing a value.
    baseline_periods: int
    #: True when the baseline had to fall below ten years for lack of history.
    baseline_is_short: bool


def baseline_window(
    observations: int,
    periods_per_year: float,
    *,
    min_years: float = DEFAULT_Z_MIN_YEARS,
) -> tuple[int, bool]:
    """Minimum periods the expanding z-score waits for, and whether it is short.

    Ten years of the publication series is the whole publication series —
    ``FRED:SP500`` is licence-capped at roughly that — so demanding a full decade
    would produce a column of NaN and a dashboard lamp that never lights. The
    baseline therefore degrades to half the available history, floored at
    :data:`MIN_Z_YEARS`, and says that it did so.
    """
    wanted = int(round(min_years * periods_per_year))
    if observations >= wanted:
        return wanted, False
    floor = int(round(MIN_Z_YEARS * periods_per_year))
    return max(floor, observations // 2), True


def expanding_z(series: pd.Series, min_periods: int) -> pd.Series:
    """Z-score against the expanding history of the series itself.

    Expanding and inclusive of *t*: the window is ``[0, t]``, which uses no
    information the date does not have. A centred or trailing-symmetric window
    would (§14.1 rule 3), which is the whole reason this is not a ``rolling``.
    """
    expanding = series.expanding(min_periods=max(min_periods, 2))
    mean = expanding.mean()
    std = expanding.std()
    # A zero standard deviation means a flat stretch, not a huge z-score.
    scaled = (series - mean) / std.replace(0.0, np.nan)
    return scaled.replace([np.inf, -np.inf], np.nan)


def jerk_lamp(z: float | None) -> str:
    """``calm`` | ``elevated`` | ``extreme`` — the thresholded reading of §3.1."""
    if z is None or not np.isfinite(z):
        return "unknown"
    magnitude = abs(float(z))
    if magnitude >= JERK_EXTREME_Z:
        return "extreme"
    if magnitude >= JERK_ELEVATED_Z:
        return "elevated"
    return "calm"


#: Lamp label -> the integer ``engine_output`` carries, since that table stores
#: REALs. Ordered so a chart of it reads upwards as instability rises.
JERK_LAMP_CODES: dict[str, int] = {"unknown": -1, "calm": 0, "elevated": 1, "extreme": 2}


def kinematics(
    slope: pd.Series,
    *,
    periods_per_year: float,
    ffd: pd.Series | None = None,
    momentum_months: tuple[int, ...] = DEFAULT_MOMENTUM_MONTHS,
    z_min_years: float = DEFAULT_Z_MIN_YEARS,
) -> Kinematics:
    """Derive the kinematic block from a filtered slope path.

    ``ffd`` is the fractionally differenced price the momentum windows are summed
    over — momentum on the raw level would be a level, not a momentum. Absent it,
    the momentum block comes back empty rather than being faked off the slope.
    """
    velocity = (slope * periods_per_year).rename("velocity")
    # diff() of an annualized rate is per period; scaling again gives per year
    # squared, so acceleration and velocity are on consistent time units.
    acceleration = (velocity.diff() * periods_per_year).rename("acceleration")
    jerk_raw = acceleration.diff()

    min_periods, is_short = baseline_window(
        len(slope.dropna()), periods_per_year, min_years=z_min_years
    )
    if is_short:
        log.info(
            "kinematics: %d observations is under %.0f years; the jerk baseline "
            "uses %d periods instead",
            len(slope.dropna()),
            z_min_years,
            min_periods,
        )

    jerk_z = expanding_z(jerk_raw, min_periods).rename("jerk_z")

    momentum: dict[str, pd.Series] = {}
    if ffd is not None:
        for months in momentum_months:
            window = max(int(round(months * periods_per_year / 12.0)), 1)
            name = f"momentum_{months}m"
            momentum[name] = ffd.rolling(window=window, min_periods=window).sum().rename(name)

    return Kinematics(
        velocity=velocity,
        acceleration=acceleration,
        jerk_z=jerk_z,
        momentum=momentum,
        baseline_periods=min_periods,
        baseline_is_short=is_short,
    )


__all__ = [
    "DEFAULT_MOMENTUM_MONTHS",
    "DEFAULT_Z_MIN_YEARS",
    "JERK_ELEVATED_Z",
    "JERK_EXTREME_Z",
    "JERK_LAMP_CODES",
    "MIN_Z_YEARS",
    "Kinematics",
    "baseline_window",
    "expanding_z",
    "jerk_lamp",
    "kinematics",
]
