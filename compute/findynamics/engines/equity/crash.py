"""Crash risk as three factors, never as one number (FINDYN_V1_SPEC.md §4).

    P(crash, h) = P(regime transition, h) × P(shock | fragile) × P(transmission)

§4 is emphatic that this must not be collapsed: "Publish ``crash_risk ∈ [0,100]``
**and** the three factors separately — the decomposition is part of the
explainability contract." A single composite is unfalsifiable in practice. Three
factors can each be wrong in a way somebody can point at, which is the only kind
of wrong that ever gets fixed.

The three, and where each comes from:

``p_transition``
    Derived from the **fitted HMM's own transition matrix and today's
    posterior** — not from the L3 classifier. The classifier has no
    out-of-sample skill (``equity-open-issues.md`` §2 and §3c), so feeding it
    into a published crash probability would launder a number with no
    demonstrated predictive content into one that looks structural.

    The transition matrix is a different kind of object: it is a *description*
    of how the fitted states follow one another, estimated over the whole record
    and not asked to forecast anything it has not already seen. Propagating
    today's posterior through it and asking for the first passage into an
    adverse state is arithmetic on the model, not a second model. The
    classifier's probabilities are still published as descriptive signals, and
    they stop there.

``p_shock``
    Extreme-value theory. A Generalized Pareto fitted to the tail of drawdown
    magnitudes over the 1871+ record, with the exceedance hazard scaled by the
    RII decile as §4 specifies. This is the only factor with real statistical
    machinery behind it, and it is the one whose units are easiest to get wrong —
    see the frequency conversion below.

``p_transmission``
    A fragility score: does a shock propagate, or get absorbed? High-yield
    spread level and velocity, liquidity conditions, and the curve state.

The frequency conversion, which is the subtle part
--------------------------------------------------

The 1871+ record is **monthly** — ``SHILLER:NOMINAL_PRICE`` is the only series
that reaches back that far, and it is month-end closes. So the GPD is fitted to
*monthly* drawdown magnitudes, and the exceedance rate that comes out is "per
month".

Reading that rate as if it were daily is the mistake this docstring exists to
prevent. It is not a small one: the same threshold is crossed roughly 21 times
less often per day than per month, and a magnitude that is a 1-in-100 month is a
far rarer day. Under a diffusion scaling the severity converts by ``√21`` — the
prompt's "understates crash frequency by roughly the square root of 21".

So the conversion is explicit and in one place, :func:`horizon_probability`:
the exceedance rate is scaled linearly by horizon length in *months*, and any
magnitude compared across frequencies is scaled by ``√(periods ratio)``. Neither
is left implicit in a call site.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

log = logging.getLogger("findynamics.engines.equity.crash")

#: Trading days in a month, and the ratio the monthly→daily conversion uses.
DAYS_PER_MONTH = 21.0

#: The states a first passage *into* is what "transition" means here.
ADVERSE: frozenset[str] = frozenset({"bear", "crisis"})

#: Drawdown depth above which an observation counts as a tail exceedance.
#: 10% is deep enough to be a genuine event and shallow enough that a century of
#: monthly data holds enough of them to fit two GPD parameters.
DEFAULT_THRESHOLD = 0.10

#: The drawdown depth ``p_shock`` is quoted for — "a shock of at least this".
DEFAULT_SEVERITY = 0.20

#: §4 — the hazard is "scaled by RII decile". The multiplier runs from calm to
#: maximally unstable; at the midpoint it is 1, so a neutral RII neither inflates
#: nor discounts the unconditional rate.
DEFAULT_RII_HAZARD_RANGE = (0.5, 2.0)

#: A shock never fails to propagate entirely. Without this the fragility score
#: reaches 0.012 in a benign environment — three of four sub-scores clip to zero
#: — and a multiplicative decomposition with one factor at zero publishes a
#: crash risk of zero however unstable the other two factors are.
TRANSMISSION_FLOOR = 0.10

#: Below this many *episodes* a GPD fit is two parameters from a handful of
#: points, and its tail is an artefact of which handful.
MIN_EXCEEDANCES = 25


class TailFitError(ValueError):
    """Raised when the extreme-value fit cannot be produced."""


@dataclass(frozen=True)
class TailFit:
    """A fitted GPD over drawdown exceedances, with its own provenance."""

    #: Shape (ξ) and scale (β) of the Generalized Pareto.
    shape: float
    scale: float
    threshold: float
    #: Exceedances observed, and the total observations they came from.
    exceedances: int
    observations: int
    #: Observations per year of the series this was fitted on. **The units of
    #: the exceedance rate**, and the reason the conversion cannot be implicit.
    periods_per_year: float
    series_id: str

    @property
    def exceedance_rate(self) -> float:
        """Drawdown *episodes* per observation — per *month* on the 1871+ record.

        Episodes, not months-below-peak: see :func:`decluster` for why the
        distinction is the difference between a tail estimate and a nonsense one.
        """
        return self.exceedances / max(self.observations, 1)

    def as_dict(self) -> dict[str, Any]:
        return {
            "shape": self.shape,
            "scale": self.scale,
            "threshold": self.threshold,
            "exceedances": self.exceedances,
            "observations": self.observations,
            "periods_per_year": self.periods_per_year,
            "series_id": self.series_id,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> TailFit:
        return cls(
            shape=float(raw["shape"]),
            scale=float(raw["scale"]),
            threshold=float(raw["threshold"]),
            exceedances=int(raw["exceedances"]),
            observations=int(raw["observations"]),
            periods_per_year=float(raw["periods_per_year"]),
            series_id=str(raw["series_id"]),
        )

    def survival(self, severity: float) -> float:
        """P(a drawdown exceeds ``severity`` | it exceeded the threshold)."""
        if severity <= self.threshold:
            return 1.0
        return float(stats.genpareto.sf(severity - self.threshold, self.shape, scale=self.scale))


def drawdown_magnitudes(log_price: pd.Series) -> pd.Series:
    """Fractional drawdown from the running maximum, as positive numbers.

    Running maximum, so every value is knowable on its own date — a peak-to-
    trough measured against a peak that has not happened yet is a number from the
    future, and this series is fitted on.
    """
    return (1.0 - np.exp(log_price - log_price.cummax())).rename("drawdown")


def decluster(drawdowns: pd.Series, threshold: float) -> pd.Series:
    """One observation per drawdown *episode*: its deepest point.

    Peaks-over-threshold assumes exceedances are independent, and a drawdown
    series violates that as badly as any series can — it is a running distance
    below a peak, so a single bear market contributes an exceedance *every
    month it lasts*.

    Skipping this step is not a subtle error. Undeclustered, the 1871+ record
    produced 1,039 exceedances from 1,845 months: 56% of all months "exceeded" a
    10% drawdown, and the resulting P(≥20% drawdown within three months) came
    out at 53-90%. That is not a tail estimate, it is a measurement of how much
    of history is spent below a previous high.

    An episode runs from the first crossing of the threshold to the recovery to a
    new high; its deepest point is the observation the GPD is fitted on. Roughly
    one per two to four years on the real record, which is what a >10% drawdown
    actually is.
    """
    below = drawdowns > threshold
    if not below.any():
        return pd.Series(dtype=float, name="episode_depth")

    # A new episode starts on each transition from at-peak to below-threshold.
    episode = (below & ~below.shift(1, fill_value=False)).cumsum()
    depths = drawdowns[below].groupby(episode[below]).max()
    return pd.Series(depths.to_numpy(), name="episode_depth")


def adverse_first_passage(
    posterior: np.ndarray,
    transmat: np.ndarray,
    adverse: list[int],
    steps: int,
) -> float:
    """P(the chain enters an adverse state within ``steps``), from today's posterior.

    Computed by making the adverse states absorbing and propagating the
    posterior: the mass that has been absorbed after ``steps`` transitions is
    exactly the probability of having entered one by then. That is the *first
    passage* probability, which is the question — "does it get bad at any point
    in the next year", not "is it bad in exactly a year".

    Mass already sitting on an adverse state today is excluded from the start
    vector and reported separately by the caller, because "already there" and
    "about to go there" are different statements and averaging them is how a
    market in a crisis ends up with a low crash probability.
    """
    if steps <= 0:
        return 0.0

    absorbing = transmat.copy()
    for state in adverse:
        absorbing[state, :] = 0.0
        absorbing[state, state] = 1.0

    start = np.array(posterior, dtype=float).copy()
    start[adverse] = 0.0
    total = start.sum()
    if total <= 0.0:
        # Every bit of today's belief is already on an adverse state, so there
        # is no benign mass left that could transition into one.
        return 0.0
    start /= total

    # Repeated squaring: `steps` is 252 for a one-year daily horizon, and a
    # 5x5 matrix power is cheaper than 252 vector-matrix products.
    power = np.linalg.matrix_power(absorbing, int(steps))
    return float((start @ power)[adverse].sum())


def fit_tail(
    log_price: pd.Series,
    *,
    periods_per_year: float,
    series_id: str,
    threshold: float = DEFAULT_THRESHOLD,
) -> TailFit:
    """Peaks-over-threshold GPD fit on drawdown magnitudes.

    ``periods_per_year`` is stored on the result rather than assumed anywhere
    downstream: the fit is almost always monthly, every consumer wants a daily
    horizon, and the conversion has to be able to see both numbers.
    """
    drawdowns = drawdown_magnitudes(log_price).dropna()
    episodes = decluster(drawdowns, threshold)
    exceedances = episodes - threshold

    if len(exceedances) < MIN_EXCEEDANCES:
        raise TailFitError(
            f"{series_id}: {len(exceedances)} drawdown episodes exceed {threshold:.0%}, "
            f"below the {MIN_EXCEEDANCES} needed to fit a tail"
        )

    # floc=0: a peaks-over-threshold GPD is anchored at the threshold by
    # construction. Letting the location float would fit a different model and
    # quietly move the threshold the rate was counted against.
    shape, _, scale = stats.genpareto.fit(exceedances.to_numpy(), floc=0.0)

    fit = TailFit(
        shape=float(shape),
        scale=float(scale),
        threshold=threshold,
        exceedances=int(len(exceedances)),
        observations=int(len(drawdowns)),
        periods_per_year=periods_per_year,
        series_id=series_id,
    )
    log.info(
        "tail: %s ξ=%.3f β=%.4f over %d exceedances of %d observations "
        "(%.1f%% of periods, %.1f periods/year)",
        series_id,
        fit.shape,
        fit.scale,
        fit.exceedances,
        fit.observations,
        fit.exceedance_rate * 100,
        periods_per_year,
    )
    return fit


def horizon_probability(
    fit: TailFit,
    *,
    horizon_months: float,
    severity: float = DEFAULT_SEVERITY,
    hazard_multiplier: float = 1.0,
) -> float:
    """P(a drawdown of at least ``severity`` begins within the horizon).

    **The frequency conversion lives here and nowhere else.** ``fit`` carries the
    frequency it was estimated at; the horizon is quoted in months; the
    exceedance rate is converted between the two explicitly. A caller cannot get
    this wrong without editing this function, which is the point.

    The rate is per *observation* of the fitted series. Expected exceedances over
    the horizon are therefore ``rate × observations-in-the-horizon``, and the
    probability of at least one comes from a Poisson tail. Treating the monthly
    rate as daily would overstate the count by a factor of about 21.
    """
    observations = horizon_months * fit.periods_per_year / 12.0
    expected = fit.exceedance_rate * observations * fit.survival(severity)
    expected *= max(hazard_multiplier, 0.0)
    return float(1.0 - math.exp(-max(expected, 0.0)))


def severity_at_frequency(
    magnitude: float,
    *,
    from_periods_per_year: float,
    to_periods_per_year: float,
) -> float:
    """Rescale a drawdown magnitude between observation frequencies.

    Under a diffusion, dispersion scales with the square root of time, so a
    magnitude estimated on monthly data corresponds to a smaller one daily by
    ``√(21)``. Exposed as a named function because the alternative — a bare
    ``/ np.sqrt(21)`` at a call site — is exactly the line that gets copied
    somewhere it does not belong.
    """
    if from_periods_per_year <= 0 or to_periods_per_year <= 0:
        raise ValueError("frequencies must be positive")
    return magnitude * math.sqrt(from_periods_per_year / to_periods_per_year)


def rii_hazard_multiplier(
    rii: float | None,
    *,
    hazard_range: tuple[float, float] = DEFAULT_RII_HAZARD_RANGE,
) -> float:
    """§4's "hazard rate scaled by RII decile", as a bounded multiplier.

    Linear in the RII between the two ends of ``hazard_range``, so a mid RII
    leaves the unconditional rate alone. Bounded on purpose: an unbounded scaling
    would let one very unstable day publish a crash probability driven entirely
    by a term whose own components are mostly percentiles.
    """
    low, high = hazard_range
    if rii is None or not math.isfinite(rii):
        return 1.0
    # Decile rather than the raw score, as the spec words it — coarse enough that
    # a one-point move in a composite does not move the published hazard.
    decile = min(max(math.floor(rii / 10.0), 0), 9) / 9.0
    return low + (high - low) * decile


@dataclass(frozen=True)
class CrashFactors:
    """The three factors, and the composite that must never travel alone."""

    p_transition: float
    p_shock: float
    p_transmission: float
    horizon_months: float
    #: Diagnostics: what fed the fragility score, and how the hazard was scaled.
    detail: dict[str, float]

    @property
    def crash_risk(self) -> float:
        """0-100. §4's product, published *beside* its factors, never instead."""
        return round(100.0 * self.p_transition * self.p_shock * self.p_transmission, 6)

    def as_components(self) -> dict[str, float]:
        return {
            f"p_transition_{self.horizon_months:g}m": round(self.p_transition, 6),
            "p_shock": round(self.p_shock, 6),
            "p_transmission": round(self.p_transmission, 6),
            "crash_risk": self.crash_risk,
            **{k: round(v, 6) for k, v in self.detail.items()},
        }


def transmission_score(
    *,
    credit_spread: float | None,
    credit_velocity: float | None,
    liquidity: float | None,
    curve_slope: float | None,
) -> tuple[float, dict[str, float]]:
    """Fragility: would a shock propagate, or be absorbed?

    §4 lists margin debt, HY OAS level and velocity, NFCI and the curve state.
    FINRA margin debt has no adapter, so it is absent rather than approximated;
    the rest are averaged as 0-1 sub-scores and the average is over what exists.

    Each sub-score is a bounded transform of a level with a stated reference
    point, not a percentile. That is deliberate here and different from the RII:
    fragility should mean the same thing in 1935 and 2026, and an expanding
    percentile would define "wide spreads" relative to whatever the last decade
    happened to contain.
    """
    parts: dict[str, float] = {}

    if credit_spread is not None and math.isfinite(credit_spread):
        # HY OAS: ~3% is benign, ~10% is a credit crisis.
        parts["credit_level"] = min(max((credit_spread - 3.0) / 7.0, 0.0), 1.0)
    if credit_velocity is not None and math.isfinite(credit_velocity):
        # A 2pp move in a month is a violent repricing of credit.
        parts["credit_velocity"] = min(max(credit_velocity / 2.0, 0.0), 1.0)
    if liquidity is not None and math.isfinite(liquidity):
        # NFCI is already standardized: 0 is average, +1 is tight.
        parts["liquidity"] = min(max((liquidity + 0.5) / 2.0, 0.0), 1.0)
    if curve_slope is not None and math.isfinite(curve_slope):
        # An inverted curve is the classic transmission channel; the deeper the
        # inversion the more of the system is repricing at once.
        parts["curve"] = min(max(-curve_slope / 1.5, 0.0), 1.0)

    if not parts:
        # No fragility inputs at all: assume a fully transmitting system rather
        # than a safe one. The factor is a *multiplier* on crash risk, and
        # guessing 0 here would silently zero the published number.
        return 1.0, {"transmission_inputs": 0.0}

    # Floored, because every sub-score clips to zero in a benign environment and
    # three zeros out of four took the published transmission to 0.012 — which
    # zeroes the whole product regardless of what the other two factors say. A
    # shock in calm conditions transmits less; it does not fail to transmit.
    score = max(float(np.mean(list(parts.values()))), TRANSMISSION_FLOOR)
    detail = {f"fragility_{k}": v for k, v in parts.items()}
    detail["transmission_inputs"] = float(len(parts))
    return score, detail


def crash_factors(
    *,
    posterior: np.ndarray,
    transmat: np.ndarray,
    adverse_states: list[int],
    periods_per_year: float,
    tail: TailFit | None,
    rii: float | None,
    horizon_months: float,
    severity: float = DEFAULT_SEVERITY,
    credit_spread: float | None = None,
    credit_velocity: float | None = None,
    liquidity: float | None = None,
    curve_slope: float | None = None,
) -> CrashFactors:
    """Assemble §4's decomposition for one horizon.

    ``p_transition`` is computed here from the posterior and the transition
    matrix rather than accepted as an argument. That is deliberate: taking it as
    a parameter would let a caller pass the L3 classifier's output, and the
    classifier has no demonstrated out-of-sample skill. Making the function
    derive it means the published crash probability cannot be built on a number
    that has not earned it, no matter what a call site would like to hand over.
    """
    steps = int(round(horizon_months * periods_per_year / 12.0))
    p_transition = adverse_first_passage(posterior, transmat, adverse_states, steps)

    multiplier = rii_hazard_multiplier(rii)

    if tail is None:
        # No deep history means no tail estimate. Publishing 0 would say "a shock
        # is impossible"; publishing 1 would say "certain". The unconditional
        # long-run base rate of a 20% drawdown starting in a given year is the
        # honest stand-in, and it is flagged.
        p_shock = 1.0 - math.exp(-0.25 * horizon_months / 12.0)
        detail_tail = {"tail_fitted": 0.0}
    else:
        p_shock = horizon_probability(
            tail,
            horizon_months=horizon_months,
            severity=severity,
            hazard_multiplier=multiplier,
        )
        detail_tail = {
            "tail_fitted": 1.0,
            "tail_shape": round(tail.shape, 6),
            "tail_scale": round(tail.scale, 6),
            "tail_exceedances": float(tail.exceedances),
            "tail_periods_per_year": tail.periods_per_year,
        }

    transmission, fragility = transmission_score(
        credit_spread=credit_spread,
        credit_velocity=credit_velocity,
        liquidity=liquidity,
        curve_slope=curve_slope,
    )

    return CrashFactors(
        p_transition=float(min(max(p_transition, 0.0), 1.0)),
        p_shock=float(min(max(p_shock, 0.0), 1.0)),
        p_transmission=float(min(max(transmission, 0.0), 1.0)),
        horizon_months=horizon_months,
        detail={
            "transition_steps": float(steps),
            "p_adverse_now": round(float(np.asarray(posterior)[adverse_states].sum()), 6),
            "hazard_multiplier": round(multiplier, 6),
            "shock_severity": severity,
            **detail_tail,
            **fragility,
        },
    )


__all__ = [
    "ADVERSE",
    "TRANSMISSION_FLOOR",
    "DAYS_PER_MONTH",
    "DEFAULT_RII_HAZARD_RANGE",
    "DEFAULT_SEVERITY",
    "DEFAULT_THRESHOLD",
    "MIN_EXCEEDANCES",
    "CrashFactors",
    "TailFit",
    "TailFitError",
    "adverse_first_passage",
    "crash_factors",
    "crash_history",
    "decluster",
    "drawdown_magnitudes",
    "fit_tail",
    "horizon_probability",
    "rii_hazard_multiplier",
    "severity_at_frequency",
    "transmission_score",
]


def crash_history(
    posteriors: pd.DataFrame,
    *,
    transmat: np.ndarray,
    adverse_states: list[int],
    periods_per_year: float,
    horizon_months: float,
    tail: TailFit | None,
    rii: pd.Series,
    credit_spread: pd.Series | None = None,
    credit_velocity: pd.Series | None = None,
    liquidity: pd.Series | None = None,
    curve_slope: pd.Series | None = None,
    severity: float = DEFAULT_SEVERITY,
) -> pd.DataFrame:
    """The three factors per date, for the ``/instability`` history.

    The expensive part of :func:`adverse_first_passage` is the matrix power, and
    it does not depend on the date — only the start vector does. So it is
    computed once here and every date is a single vector-matrix product, which
    turns a five-second history into a fifty-millisecond one.
    """
    steps = int(round(horizon_months * periods_per_year / 12.0))

    absorbing = np.asarray(transmat, dtype=float).copy()
    for state in adverse_states:
        absorbing[state, :] = 0.0
        absorbing[state, state] = 1.0
    power = np.linalg.matrix_power(absorbing, max(steps, 1))

    values = posteriors.to_numpy(dtype=float)
    benign = values.copy()
    benign[:, adverse_states] = 0.0
    totals = benign.sum(axis=1)
    # Rows whose belief is entirely on an adverse state have no benign mass that
    # could transition into one; their first-passage probability is zero.
    safe = np.divide(benign, totals[:, None], out=np.zeros_like(benign), where=totals[:, None] > 0)
    p_transition = (safe @ power)[:, adverse_states].sum(axis=1)

    aligned_rii = rii.reindex(posteriors.index)
    multipliers = aligned_rii.map(rii_hazard_multiplier).to_numpy(dtype=float)

    if tail is None:
        base = 1.0 - np.exp(-0.25 * horizon_months / 12.0)
        p_shock = np.full(len(posteriors), base)
    else:
        observations = horizon_months * tail.periods_per_year / 12.0
        expected = tail.exceedance_rate * observations * tail.survival(severity)
        p_shock = 1.0 - np.exp(-np.clip(expected * multipliers, 0.0, None))

    def column(series: pd.Series | None) -> np.ndarray:
        if series is None:
            return np.full(len(posteriors), np.nan)
        return series.reindex(posteriors.index).ffill().to_numpy(dtype=float)

    credit = column(credit_spread)
    velocity = column(credit_velocity)
    liquid = column(liquidity)
    curve = column(curve_slope)

    transmission = np.array(
        [
            transmission_score(
                credit_spread=credit[i],
                credit_velocity=velocity[i],
                liquidity=liquid[i],
                curve_slope=curve[i],
            )[0]
            for i in range(len(posteriors))
        ]
    )

    return pd.DataFrame(
        {
            "p_transition": p_transition,
            "p_shock": p_shock,
            "p_transmission": transmission,
            "crash_risk": 100.0 * p_transition * p_shock * transmission,
        },
        index=posteriors.index,
    )
