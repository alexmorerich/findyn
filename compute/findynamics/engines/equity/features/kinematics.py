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

#: How much of the start of a filtered path is discarded before its derivatives
#: are published, in years.
#:
#: The local linear trend is initialized diffusely: the filter starts with a
#: slope standard error of 1e3 and shrinks it as observations arrive. Measured on
#: the 1927+ S&P path it is 6.9x its converged value after 20 observations, 1.9x
#: after 252 and 1.2x after 656. The velocities that come out of that stretch are
#: a statement about the prior, not about the market — the first published value
#: on the century path was **+142% annualized**, on 1928-01-03, and the same
#: artifact was visible on the ten-year path as a -177% spike in August 2016.
#:
#: One year, because that is where the slope's own uncertainty comes inside a
#: factor of two of where it ends up, and because expressing it in years means
#: the same number is 252 observations on a daily path and 12 on the monthly
#: deep history rather than one span meaning two things.
#:
#: It is not cosmetic. ``jerk_z`` is scored against an **expanding** baseline, so
#: the start-up transient inflates the denominator for every later date and never
#: washes out: across the century the untrimmed baseline is 3.2x too wide, which
#: leaves 4 elevated readings in 22,241 sessions and — the tell — rates Black
#: Monday 1987 at |z| = 2.95, just under the threshold at which the lamp would
#: have lit.
DEFAULT_BURN_IN_YEARS = 1.0

#: Never discard more than this share of a path. A series too short to give up a
#: year to the filter's start-up should publish a noisier estimate and say so,
#: not publish a column of NaN.
MAX_BURN_IN_FRACTION = 0.5

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
    #: Leading observations whose derivatives were discarded as filter start-up.
    burn_in_periods: int = 0
    #: True when the path was too short to give up a full year to start-up.
    burn_in_is_short: bool = False


def baseline_window(
    observations: int,
    periods_per_year: float,
    *,
    min_years: float = DEFAULT_Z_MIN_YEARS,
) -> tuple[int, bool]:
    """Minimum periods the expanding z-score waits for, and whether it is short.

    §8.3's full decade is available on the shipped configuration — the
    publication path is the daily record spliced back to 1927, so this returns
    ``(2520, False)`` and the lamp is scored against ten years as specified.

    The degradation is kept for the configuration that loses the backfill, where
    the publication path returns to the ten years ``FRED:SP500`` licences in
    total. Demanding a full decade of *that* would produce a column of NaN and a
    dashboard lamp that never lights, so the baseline falls to half the available
    history, floored at :data:`MIN_Z_YEARS`, and says that it did so.
    """
    wanted = int(round(min_years * periods_per_year))
    if observations >= wanted:
        return wanted, False
    floor = int(round(MIN_Z_YEARS * periods_per_year))
    return max(floor, observations // 2), True


def burn_in_window(
    observations: int,
    periods_per_year: float,
    *,
    years: float = DEFAULT_BURN_IN_YEARS,
) -> tuple[int, bool]:
    """Leading observations to discard as filter start-up, and whether it is short.

    Capped at :data:`MAX_BURN_IN_FRACTION` of the path for the same reason
    :func:`baseline_window` has a floor: a transform that returns nothing on a
    short series is worse than one that returns something noisier and reports
    that it did.
    """
    wanted = max(int(round(years * periods_per_year)), 0)
    ceiling = int(observations * MAX_BURN_IN_FRACTION)
    if wanted <= ceiling:
        return wanted, False
    return max(ceiling, 0), True


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
    burn_in_years: float = DEFAULT_BURN_IN_YEARS,
) -> Kinematics:
    """Derive the kinematic block from a filtered slope path.

    ``ffd`` is the fractionally differenced price the momentum windows are summed
    over — momentum on the raw level would be a level, not a momentum. Absent it,
    the momentum block comes back empty rather than being faked off the slope.

    The start of the slope path is discarded rather than published: see
    :data:`DEFAULT_BURN_IN_YEARS`. Discarded here rather than in the filter so
    that ``price_filtered`` keeps its full span — the *level* is pinned by the
    first observation, and only the *slope* has to be estimated out of a diffuse
    prior.
    """
    burn_in, burn_in_is_short = burn_in_window(len(slope), periods_per_year, years=burn_in_years)
    settled = slope.copy()
    if burn_in:
        settled.iloc[:burn_in] = np.nan
    if burn_in_is_short:
        log.info(
            "kinematics: %d observations cannot give up %.0f year(s) to the filter's "
            "start-up; discarding %d instead, so the earliest derivatives carry more "
            "of the diffuse prior than usual",
            len(slope),
            burn_in_years,
            burn_in,
        )

    velocity = (settled * periods_per_year).rename("velocity")
    # diff() of an annualized rate is per period; scaling again gives per year
    # squared, so acceleration and velocity are on consistent time units.
    acceleration = (velocity.diff() * periods_per_year).rename("acceleration")
    jerk_raw = acceleration.diff()

    min_periods, is_short = baseline_window(
        len(settled.dropna()), periods_per_year, min_years=z_min_years
    )
    if is_short:
        log.info(
            "kinematics: %d observations is under %.0f years; the jerk baseline "
            "uses %d periods instead",
            len(settled.dropna()),
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
        burn_in_periods=burn_in,
        burn_in_is_short=burn_in_is_short,
    )


__all__ = [
    "DEFAULT_BURN_IN_YEARS",
    "DEFAULT_MOMENTUM_MONTHS",
    "DEFAULT_Z_MIN_YEARS",
    "MAX_BURN_IN_FRACTION",
    "JERK_ELEVATED_Z",
    "JERK_EXTREME_Z",
    "JERK_LAMP_CODES",
    "MIN_Z_YEARS",
    "Kinematics",
    "baseline_window",
    "burn_in_window",
    "expanding_z",
    "jerk_lamp",
    "kinematics",
]
