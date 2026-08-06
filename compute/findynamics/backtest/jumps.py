"""Lee-Mykland jump detection — shared by every engine that needs one.

Extracted from ``engines/gold/jumps.py`` in P5, when a second engine needed the
same estimator. It lives here rather than in either engine because engines may
not import each other (``01-target-architecture.md`` §3 rule 2) and because a
copy-pasted detector is a detector that drifts: two engines silently disagreeing
about what counts as a jump is worse than either answer.

Why ``backtest/`` and not ``factors/``: ``factors/`` is Layer 0, the shared risk
factors scored 0-100 and handed to every engine through ``WorldState``. A jump
detector is none of those things — it is an engine-agnostic *estimator* that an
engine runs on its own series, which is exactly what this package already holds
in ``replay.py``.

The test (Lee & Mykland 2008)
-----------------------------

For each day *t*, standardize the return by a **bipower** estimate of local
volatility::

    sigma_t^2 = 1 / (K - 2) * sum_{i=t-K+2}^{t} |r_{i-1}| * |r_i|

Bipower rather than realized variance is the whole point: a product of two
*adjacent* returns is barely moved by one large day, because a jump enters only
the two terms it appears in and is multiplied by an ordinary return in both. A
sum of squares, by contrast, is dominated by the very observation the test is
trying to detect — so the day of the crash raises its own threshold and hides
itself. That failure is silent and it is why this estimator exists.

The statistic is then ``L_t = r_t / (c1 * sigma_t)`` with ``c1 = sqrt(2/pi)``,
the expectation of a half-normal, which is what makes the bipower sum an
unbiased variance estimate. Under the null of no jump, the maximum of ``|L|``
over *n* observations follows a Gumbel law, so the threshold is::

    (|L_t| - C_n) / S_n  >  -log(-log(1 - alpha))

    C_n = (2 log n)^(1/2) / c1 - (log(pi) + log(log n)) / (2 * c1 * (2 log n)^(1/2))
    S_n = 1 / (c1 * (2 log n)^(1/2))

Using the *maximum's* distribution rather than a fixed z-cut is what keeps the
false-positive rate at alpha across the whole sample instead of per day: at a
naive 3-sigma cut, fourteen thousand days of gold produce about forty "jumps" by
arithmetic alone, and a detector that fires forty times on noise cannot be used
to say a crisis is happening.

Calendars
---------

``periods_per_year`` annualizes the intensity and is a property of the *market*,
not of the test. Gold fixes on 252 days a year; bitcoin trades on 365, weekends
and Christmas included. Getting it wrong does not change which days are jumps —
it changes the reported rate by 45%, silently, in a number two engines compare
against thresholds calibrated in jumps-per-year.

Causality
---------

The local volatility window is strictly trailing — ``sigma_t`` uses returns up
to and including *t*, never after — so a date's classification is what a run on
that date would have produced. ``rolling(...)`` with the default right-closed
window is the mechanism; there is no ``center=True`` anywhere in this module and
adding one would be lookahead, not smoothing.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

log = logging.getLogger("findynamics.backtest.jumps")

#: E|Z| for a standard normal — the scaling that makes the bipower sum unbiased.
C1 = math.sqrt(2.0 / math.pi)

#: Observations per year on an exchange-traded calendar. The default because it
#: is what the first caller (gold) uses; crypto overrides it with 365.
TRADING_DAYS = 252

#: Observations per year on a market that never closes.
CALENDAR_DAYS = 365


@dataclass(frozen=True)
class JumpRules:
    """Detector settings, from an engine's ``jumps:`` config block."""

    #: Observations in the local bipower volatility window. Lee-Mykland suggest
    #: roughly sqrt(observations per year) x a constant; for daily data the
    #: literature's practical choice is a quarter to a half year. 63 is a
    #: quarter: long enough for the bipower sum to be stable, short enough that
    #: it tracks a volatility regime rather than averaging over two of them.
    window: int = 63
    #: Gumbel test size. 1% rather than 5%: this feeds a *crisis premium*, and a
    #: detector that fires on one ordinary day in twenty would make the premium a
    #: measure of how many days there are.
    alpha: float = 0.01
    #: Trailing window, in observations, over which the intensity is counted.
    intensity_window: int = 252
    #: Minimum returns before the detector will speak at all.
    min_observations: int = 126
    #: Observations per year, for annualizing the intensity. See the module
    #: docstring: this is a fact about the market's calendar, not about the test.
    periods_per_year: int = TRADING_DAYS

    @classmethod
    def from_params(cls, params: dict[str, Any], *, where: str = "engine config") -> JumpRules:
        """Read the ``jumps:`` block out of an engine's params."""
        raw = params.get("jumps") or {}
        if not isinstance(raw, dict):
            raise ValueError(f"{where}: 'jumps' must be a mapping")
        defaults = cls()
        return cls(
            window=int(raw.get("window", defaults.window)),
            alpha=float(raw.get("alpha", defaults.alpha)),
            intensity_window=int(raw.get("intensity_window", defaults.intensity_window)),
            min_observations=int(raw.get("min_observations", defaults.min_observations)),
            periods_per_year=int(raw.get("periods_per_year", defaults.periods_per_year)),
        )


@dataclass(frozen=True)
class JumpResult:
    """Per-date detector output."""

    #: Lee-Mykland statistic, signed — the sign is the jump's direction.
    statistic: pd.Series
    #: Boolean flag per date.
    is_jump: pd.Series
    #: Jumps in the trailing window, annualized: expected jumps per year.
    intensity: pd.Series
    #: The Gumbel threshold each date was tested against.
    threshold: pd.Series

    @property
    def empty(self) -> bool:
        return self.statistic.empty

    def dates(self) -> list[pd.Timestamp]:
        """Every detected jump date, oldest first."""
        if self.is_jump.empty:
            return []
        return list(self.is_jump[self.is_jump].index)

    def latest_intensity(self) -> float | None:
        if self.intensity.empty:
            return None
        value = self.intensity.dropna()
        return None if value.empty else float(value.iloc[-1])


def gumbel_constants(n: int) -> tuple[float, float]:
    """``(C_n, S_n)`` — the centring and scaling of the maximum's Gumbel law."""
    if n < 3:
        raise ValueError(f"the Gumbel approximation needs at least 3 observations, got {n}")
    root = math.sqrt(2.0 * math.log(n))
    c_n = root / C1 - (math.log(math.pi) + math.log(math.log(n))) / (2.0 * C1 * root)
    s_n = 1.0 / (C1 * root)
    return c_n, s_n


def gumbel_threshold(n: int, alpha: float) -> float:
    """Critical value for ``|L|`` at test size ``alpha`` over ``n`` observations."""
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    c_n, s_n = gumbel_constants(n)
    beta = -math.log(-math.log(1.0 - alpha))
    return c_n + s_n * beta


def bipower_sigma(returns: pd.Series, window: int) -> pd.Series:
    """Trailing bipower local volatility.

    ``|r_{t-1}| * |r_t|`` averaged over the window and square-rooted. The
    ``rolling`` window is right-closed, so date *t*'s estimate is a function of
    dates <= *t* only.
    """
    if window < 3:
        raise ValueError(f"the bipower window needs at least 3 observations, got {window}")
    absolute = returns.abs()
    products = absolute * absolute.shift(1)
    # K - 2 rather than K - 1: one term is lost to the lag and one to the fact
    # that the first product in the window has no predecessor inside it.
    variance = products.rolling(window, min_periods=window - 2).sum() / (window - 2)
    return np.sqrt(variance)


def detect(returns: pd.Series, rules: JumpRules, *, label: str = "jumps") -> JumpResult:
    """Run the detector over a daily return series.

    ``returns`` are log returns, oldest first. The series is used as given: any
    resampling or winsorizing belongs upstream, because a detector whose input
    has already been smoothed is measuring the smoother.

    ``label`` only names the caller in the log line.
    """
    clean = returns.dropna().astype(float)
    if len(clean) < rules.min_observations:
        empty = pd.Series(dtype=float)
        log.info(
            "%s: %d returns is under the %d-observation floor; no detection",
            label,
            len(clean),
            rules.min_observations,
        )
        return JumpResult(
            statistic=empty,
            is_jump=pd.Series(dtype=bool),
            intensity=empty,
            threshold=empty,
        )

    sigma = bipower_sigma(clean, rules.window)
    statistic = clean / (C1 * sigma.replace(0.0, np.nan))

    # The threshold grows with the sample the maximum is taken over, and the
    # sample a given date belongs to is the one that existed by then — so n is
    # the count of testable observations up to that date, not the whole series.
    # Using the final count everywhere would test the 1970s against a threshold
    # calibrated on fifty years that had not happened.
    testable = statistic.notna().cumsum()
    threshold = pd.Series(
        [gumbel_threshold(int(n), rules.alpha) if n >= 3 else np.nan for n in testable.to_numpy()],
        index=statistic.index,
    )

    is_jump = (statistic.abs() > threshold) & statistic.notna() & threshold.notna()
    intensity = (
        is_jump.rolling(rules.intensity_window, min_periods=rules.window).mean()
        * rules.periods_per_year
    )

    log.info(
        "%s: %d detected over %d returns (%.1f/yr at the end of the sample)",
        label,
        int(is_jump.sum()),
        len(clean),
        float(intensity.dropna().iloc[-1]) if intensity.notna().any() else float("nan"),
    )
    return JumpResult(
        statistic=statistic,
        is_jump=is_jump,
        intensity=intensity,
        threshold=threshold,
    )


__all__ = [
    "C1",
    "CALENDAR_DAYS",
    "TRADING_DAYS",
    "JumpResult",
    "JumpRules",
    "bipower_sigma",
    "detect",
    "gumbel_constants",
    "gumbel_threshold",
]
