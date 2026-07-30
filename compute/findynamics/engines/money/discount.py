"""Discount factors — what a dollar at horizon ``h`` is worth today.

``D(t, h) = exp(-z(h) · h)``, continuously compounded, ``h`` in years on the
standard grid in ``core/contracts/vocab.py``. Continuous compounding is chosen so
that composing horizons is addition in the exponent and nothing depends on an
arbitrary coupon frequency; any consumer that wants a semi-annual bond-equivalent
rate can convert, and it will know it converted.

Two sources of ``z(h)``, split at a year:

* **``h ≤ 1y`` — the observed short rate, flat.** The money-market engine's own
  input. Extrapolating the overnight rate flat across a year is an assumption and
  is labelled as one; it is not a forecast that rates will hold, it is the
  statement "discount the near term at the rate cash is actually earning". Inside
  a year that is within a few basis points of the bill curve except during a
  policy turn.
* **``h > 1y`` — FinRates' fitted Nelson-Siegel curve.** Read out of
  ``engine_output`` as a series (``ENGINE:rates.ns_level`` and friends) through
  ``WorldState.series``, never by importing ``engines.rates``. The engines are
  independent by contract and ``lint-imports`` proves it; what crosses between
  them is published data, subject to the same release-date filter as any
  observation.

When the curve is not in the information set — the first run of the system, an
unreachable serving plane, a replay at a cutoff before FinRates ever
published — long horizons fall back to the same flat short rate, and every
affected factor is stamped ``curve_source="short_rate"``. Degrading loudly beats
withholding the numeraire, but a flat 30-year discount factor is a much weaker
statement than a fitted one and must not be mistaken for it.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Literal

import pandas as pd

from findynamics.core.contracts.pit import PITAccessor
from findynamics.core.contracts.vocab import DISCOUNT_HORIZON_YEARS, DISCOUNT_HORIZONS

log = logging.getLogger("findynamics.engines.money.discount")

#: Below this the short rate is used even when a curve is available; above it the
#: curve is used when there is one. One year, because that is where the
#: money-market instruments this engine reads actually stop.
SHORT_RATE_MAX_YEARS: float = 1.0

CurveSource = Literal["ns", "short_rate"]


@dataclass(frozen=True)
class NelsonSiegelFactors:
    """One date's published curve factors, as read back from ``engine_output``.

    A deliberate re-declaration of a shape ``engines.rates`` also knows: sharing
    the dataclass would mean importing the producer, which is the coupling the
    independence rule forbids. What is shared is the *wire format* — four named
    metrics in ``engine_output`` — and this is the reader for it.
    """

    level: float
    #: As published: ``-b1``. Negated on the way into the loading, below.
    slope: float
    curvature: float
    lambda_: float

    def yield_at(self, maturity: float) -> float:
        """Fitted par yield at ``maturity`` years, in percent.

        ``y(m) = b0 + b1·decay + b2·(decay - exp(-λm))``, with
        ``decay = (1 - exp(-λm)) / (λm)`` and ``b1 = -slope``. Identical in form
        to ``engines/rates/nelson_siegel.py`` and to the browser copy in
        ``dashboard/src/scripts/rates.ts``; if the three ever disagree, two of
        them have stopped describing the model.
        """
        lt = self.lambda_ * maturity
        if lt <= 1e-12:
            # The limit of the loading as m -> 0: decay -> 1, exp(-lt) -> 1, so
            # the curvature term vanishes and only the instantaneous rate remains.
            return self.level - self.slope
        decay = (1.0 - math.exp(-lt)) / lt
        return self.level - self.slope * decay + self.curvature * (decay - math.exp(-lt))


@dataclass(frozen=True)
class DiscountCurve:
    """Discount factors on the standard horizon grid, with their provenance."""

    #: Horizon label -> discount factor. ``D`` is bounded to (0, 1] for a
    #: non-negative rate and may exceed 1 where the rate is negative, which is a
    #: real thing that has happened and is not clamped away.
    factors: dict[str, float]
    #: Horizon label -> which source produced it.
    sources: dict[str, CurveSource]
    #: Annualized short rate the near end was built from, in percent.
    short_rate_pct: float
    curve: NelsonSiegelFactors | None

    @property
    def has_curve(self) -> bool:
        return self.curve is not None

    def factor(self, horizon: str) -> float | None:
        return self.factors.get(horizon)


def discount_factor(zero_rate_pct: float, years: float) -> float:
    """``exp(-z·h)`` with ``z`` in percent and ``h`` in years.

    ``D(t, 0) = 1`` exactly, at any rate: a dollar today is a dollar. That is not
    a convention this function chooses, it is what the expression evaluates to,
    and the test asserting it is checking the caller passes 0 rather than that
    ``exp`` works.
    """
    if years <= 0.0:
        return 1.0
    return math.exp(-(zero_rate_pct / 100.0) * years)


def read_curve_factors(
    series: PITAccessor,
    ids: dict[str, str],
) -> NelsonSiegelFactors | None:
    """Latest published NS factors knowable to ``series``, or ``None``.

    ``ids`` maps the four roles (``level``, ``slope``, ``curvature``, ``lambda``)
    to their series ids. All four are required: three of four plus a guessed
    lambda would produce a curve that looks fitted and is not, and a wrong lambda
    misplaces the whole belly of it.
    """
    needed = ("level", "slope", "curvature", "lambda")
    missing = [role for role in needed if role not in ids]
    if missing:
        raise ValueError(f"read_curve_factors needs ids for {needed}; missing {missing}")

    values: dict[str, float] = {}
    for role in needed:
        value = series.value(ids[role])
        if value is None:
            log.info(
                "money: no published %s (%s) in the information set at %s; "
                "long horizons will discount at the flat short rate",
                role,
                ids[role],
                series.as_of,
            )
            return None
        values[role] = float(value)

    if values["lambda"] <= 0.0:
        log.warning("money: published ns_lambda is %s; ignoring the curve", values["lambda"])
        return None

    return NelsonSiegelFactors(
        level=values["level"],
        slope=values["slope"],
        curvature=values["curvature"],
        lambda_=values["lambda"],
    )


def curve_factor_history(
    series: PITAccessor,
    ids: dict[str, str],
    *,
    start: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Per-date published NS factors, for the discount-factor histories.

    Only dates carrying all four metrics survive, for the reason in
    :func:`read_curve_factors`. An empty frame is a normal answer.
    """
    wanted = [ids[role] for role in ("level", "slope", "curvature", "lambda") if role in ids]
    if len(wanted) < 4:
        return pd.DataFrame()

    frame = series.wide(wanted, start=start)
    if frame.empty:
        return pd.DataFrame()

    present = [c for c in wanted if c in frame.columns]
    if len(present) < 4:
        return pd.DataFrame()

    named = frame[present].rename(
        columns={
            ids["level"]: "level",
            ids["slope"]: "slope",
            ids["curvature"]: "curvature",
            ids["lambda"]: "lambda_",
        }
    )
    usable = named.dropna(how="any")
    return usable[usable["lambda_"] > 0.0]


def build_curve(
    short_rate_pct: float,
    curve: NelsonSiegelFactors | None,
    *,
    horizons: tuple[str, ...] = DISCOUNT_HORIZONS,
    short_rate_max_years: float = SHORT_RATE_MAX_YEARS,
) -> DiscountCurve:
    """Discount factors across ``horizons`` from the short rate and the NS curve.

    Monotone non-increasing in the horizon whenever every zero rate involved is
    non-negative — the property a present value depends on. Two documented
    exceptions, neither of which is corrected:

    * **A negative zero rate** puts a factor *above* 1 and makes the near end
      increase with horizon. That is arithmetic, and negative money-market rates
      have happened.
    * **A deeply inverted curve, at the stitch.** Ordinary inversion does *not*
      break monotonicity: ``D`` compares ``z(h)·h`` across horizons, so doubling
      the horizon outweighs a yield a point or two lower. It takes
      ``z(2y) < z(1y)/2`` — the overnight rate pinned high while the curve prices
      deep cuts, as in early 2020 — before ``D(2y)`` rises above ``D(1y)``.

    Smoothing either away would be inventing a curve the market did not quote.
    """
    factors: dict[str, float] = {}
    sources: dict[str, CurveSource] = {}

    for horizon in horizons:
        years = DISCOUNT_HORIZON_YEARS.get(horizon)
        if years is None:
            raise ValueError(f"no year count configured for discount horizon {horizon!r}")

        if years <= short_rate_max_years or curve is None:
            zero = short_rate_pct
            source: CurveSource = "short_rate"
        else:
            zero = curve.yield_at(years)
            source = "ns"

        if not math.isfinite(zero):
            continue
        factors[horizon] = discount_factor(zero, years)
        sources[horizon] = source

    return DiscountCurve(
        factors=factors,
        sources=sources,
        short_rate_pct=short_rate_pct,
        curve=curve,
    )


__all__ = [
    "SHORT_RATE_MAX_YEARS",
    "DiscountCurve",
    "NelsonSiegelFactors",
    "build_curve",
    "curve_factor_history",
    "discount_factor",
    "read_curve_factors",
]
