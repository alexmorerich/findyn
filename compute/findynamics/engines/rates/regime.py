"""Rule-based rate regimes (Phase 1).

Deliberately not a model. The Phase-2 plan is an HMM/Kalman state space
(``01-target-architecture.md`` §5), and shipping a latent-state model before
there is a working curve pipeline to feed it would mean debugging two things at
once. These rules are transparent, cheap to replay over 25 years, and produce
the labelled history a state-space model will later need for sanity checks.

Two transforms drive every rule, both read off the *fitted* curve:

``spread``
    ``y(10y) - y(3m)``. The fitted value rather than the raw quotes, so one
    missing tenor cannot flip a regime. This is the spread the inversion
    literature is written about; the published ``ns_slope`` factor (``-b1``) is
    a 30y-versus-instantaneous read and is a different number.
``level trend``
    Change in the NS level factor over the trailing window. Trailing, never
    centred (§14.1 rule 3).

Thresholds come from ``config/engines/rates.yaml``. Anything this module decides
must be recalibratable without a deploy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from findynamics.engines.rates.domain import RATE_REGIMES


@dataclass(frozen=True)
class RegimeRules:
    """Thresholds for the classifier, loaded from engine config."""

    spread_short_years: float = 0.25
    spread_long_years: float = 10.0
    inverted_below: float = 0.0
    flat_below: float = 0.5
    re_steepening_lookback_days: int = 365
    re_steepening_min_delta: float = 0.25
    trend_days: int = 252

    @classmethod
    def from_params(cls, params: dict[str, Any]) -> RegimeRules:
        raw = params.get("regime") or {}
        if not isinstance(raw, dict):
            raise ValueError("engines/rates.yaml: 'regime' must be a mapping")
        defaults = cls()
        return cls(
            spread_short_years=float(raw.get("spread_short_years", defaults.spread_short_years)),
            spread_long_years=float(raw.get("spread_long_years", defaults.spread_long_years)),
            inverted_below=float(raw.get("inverted_below", defaults.inverted_below)),
            flat_below=float(raw.get("flat_below", defaults.flat_below)),
            re_steepening_lookback_days=int(
                raw.get("re_steepening_lookback_days", defaults.re_steepening_lookback_days)
            ),
            re_steepening_min_delta=float(
                raw.get("re_steepening_min_delta", defaults.re_steepening_min_delta)
            ),
            trend_days=int(raw.get("trend_days", defaults.trend_days)),
        )


@dataclass(frozen=True)
class RegimeInputs:
    """The transforms one classification consumes, all trailing."""

    spread: float
    #: Change in ``spread`` over the trend window. ``None`` when history is short.
    spread_delta: float | None
    #: Change in the NS level factor over the trend window.
    level_delta: float | None
    #: Was the curve inverted at any point inside the re-steepening lookback?
    recently_inverted: bool


def classify(inputs: RegimeInputs, rules: RegimeRules) -> str:
    """One of :data:`RATE_REGIMES`.

    Ordered, and the order carries meaning: inversion outranks everything
    because a curve can be both inverted and flat by the numbers, and inverted
    is the more informative statement. Re-steepening is tested before flat for
    the same reason — the exit from inversion is the signal, and it happens to
    pass through the flat band on its way out.
    """
    if inputs.spread < rules.inverted_below:
        return "inverted"

    if (
        inputs.recently_inverted
        and inputs.spread_delta is not None
        and inputs.spread_delta > rules.re_steepening_min_delta
    ):
        return "re_steepening"

    if abs(inputs.spread) < rules.flat_below:
        return "flat"

    # Positively sloped. Which of the two steep regimes it is depends on where
    # the whole curve is going, not on the slope: a steep curve with the level
    # falling is accommodation, a steep curve with the level rising is not.
    # A level that has not moved reads as tightening only if it rose; ties go to
    # easing, the less alarming call.
    if inputs.level_delta is not None and inputs.level_delta > 0:
        return "steep_tightening"
    return "steep_easing"


def build_inputs(
    factors: pd.DataFrame,
    as_of: pd.Timestamp,
    rules: RegimeRules,
) -> RegimeInputs | None:
    """Assemble :class:`RegimeInputs` for one date from a factor history.

    ``factors`` is indexed by date and carries at least ``spread`` and
    ``ns_level`` columns, as produced by
    :func:`findynamics.engines.rates.engine.factor_history`. Only rows at or
    before ``as_of`` are read — the slice is the information set.
    """
    window = factors.loc[:as_of]
    if window.empty or "spread" not in window.columns:
        return None

    spread = float(window["spread"].iloc[-1])
    if not np.isfinite(spread):
        return None

    spread_delta = _trailing_delta(window["spread"], rules.trend_days)
    level_delta = (
        _trailing_delta(window["ns_level"], rules.trend_days)
        if "ns_level" in window.columns
        else None
    )

    lookback_start = as_of - pd.Timedelta(days=rules.re_steepening_lookback_days)
    recent = window.loc[lookback_start:, "spread"]
    # Exclude today: a curve that is inverted right now is classified inverted,
    # and "recently inverted" is about where it has just come from.
    recently_inverted = bool((recent.iloc[:-1] < rules.inverted_below).any())

    return RegimeInputs(
        spread=spread,
        spread_delta=spread_delta,
        level_delta=level_delta,
        recently_inverted=recently_inverted,
    )


def _trailing_delta(series: pd.Series, window: int) -> float | None:
    """Change over the last ``window`` observations, or ``None`` if too short."""
    clean = series.dropna()
    if len(clean) <= window:
        return None
    delta = float(clean.iloc[-1] - clean.iloc[-1 - window])
    return delta if np.isfinite(delta) else None


def classify_history(factors: pd.DataFrame, rules: RegimeRules) -> pd.Series:
    """Regime label for every date in ``factors``, computed causally.

    Each date is classified from its own trailing window, so the series is what
    a daily run would have produced on each of those dates — not a relabelling
    of history with today's knowledge.
    """
    if factors.empty or "spread" not in factors.columns:
        return pd.Series(dtype=object)

    spread = factors["spread"]
    level = factors["ns_level"] if "ns_level" in factors.columns else pd.Series(index=factors.index)

    # Vectorized trailing transforms: shift() looks backwards only.
    spread_delta = spread - spread.shift(rules.trend_days)
    level_delta = level - level.shift(rules.trend_days)
    inverted = spread < rules.inverted_below
    # `closed='left'` excludes the current row, matching build_inputs.
    recently_inverted = (
        inverted.rolling(f"{rules.re_steepening_lookback_days}D", closed="left").max().fillna(0.0)
        > 0
    )

    labels = []
    for i in range(len(factors)):
        labels.append(
            classify(
                RegimeInputs(
                    spread=float(spread.iloc[i]),
                    spread_delta=_finite_or_none(spread_delta.iloc[i]),
                    level_delta=_finite_or_none(level_delta.iloc[i]),
                    recently_inverted=bool(recently_inverted.iloc[i]),
                ),
                rules,
            )
            if np.isfinite(spread.iloc[i])
            else None
        )

    result = pd.Series(labels, index=factors.index, dtype=object)
    unknown = set(result.dropna()) - set(RATE_REGIMES)
    if unknown:
        raise ValueError(f"classifier produced labels outside the vocabulary: {sorted(unknown)}")
    return result


def _finite_or_none(value: float) -> float | None:
    return float(value) if value is not None and np.isfinite(value) else None
