"""The vol/drawdown regime: ``frenzy | normal | winter``.

Deliberately the simplest model in the system, and the choice is the point.

FinGold fits a three-state Markov chain because gold has six hundred months of
monthly returns and three regimes that are genuinely distinguishable in the
likelihood. Bitcoin has about a hundred and eighty months, three of which are
2017, and a switching model on that sample would estimate nine transition
probabilities and six state parameters from four observed cycles. It would
converge, it would produce a posterior, the posterior would look authoritative
on a dashboard, and it would be fitting the four cycles it was given. The
honest instrument for a four-cycle sample is a pair of thresholds you can state
in a sentence.

The rules
---------

Evaluated in this order, first match wins::

    winter   drawdown from the trailing-year peak <= -drawdown_threshold
    frenzy   12-month log return >= frenzy_return AND realized vol >= frenzy_vol
    normal   otherwise

**Winter is checked first** because the two are not mutually exclusive and the
drawdown is the more consequential read. November 2021 into January 2022 was
both — up 60% year on year and 45% off the peak — and calling that a frenzy
because the trailing twelve months were strong would be describing the top of a
market as the middle of a boom.

**Frenzy requires both conditions.** Either alone is unremarkable for this asset:
bitcoin has had +100% years without a blowoff (2019, 2023) and volatile years
without a trend (2015, 2018 H2). It is the conjunction that has preceded every
drawdown of the kind winter names.

Winter is a statement about **depth, not duration**. March 2020 spends a
fortnight in it on a 50% crash and leaves. That is the correct reading of a
fortnight in which the asset had halved, and smoothing it away would be
smoothing away the thing worth reporting.

Causality
---------

Trailing peak, trailing return, trailing volatility. No centred window, no
``bfill``, no hysteresis that peeks at the following day to decide the current
one. A date's regime is what a run on that date would have published.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from findynamics.engines.crypto.domain import CRYPTO_REGIMES

log = logging.getLogger("findynamics.engines.crypto.regime")

#: Bitcoin has no exchange calendar; a year is 365 observations.
CALENDAR_DAYS = 365


@dataclass(frozen=True)
class RegimeRules:
    """Thresholds, from ``config/engines/crypto.yaml``.

    Every number the regime branches on is here. A rule that only exists in
    Python is a rule nobody can recalibrate without a deploy.
    """

    #: Observations in the trailing peak, return and volatility windows.
    window: int = CALENDAR_DAYS
    #: Fraction below the trailing-year peak that counts as winter. 0.45 rather
    #: than 0.20: a 20% drawdown is a Tuesday for this asset and would put half
    #: the record in winter, which would make the label carry no information.
    drawdown_threshold: float = 0.45
    #: Trailing 12-month LOG return above which the trend leg of frenzy fires.
    #: 0.69 is a double. Chosen because it is the round number nearest the
    #: median 12-month return of the periods anyone would name as manias, and
    #: because a threshold below it fires in ordinary recovery years.
    frenzy_return: float = 0.69
    #: Annualized realized volatility (percent) above which the vol leg fires.
    #: 60% is roughly bitcoin's own median — the leg is asking "unusually
    #: volatile for bitcoin", not "volatile".
    frenzy_vol: float = 60.0
    #: Observations required before any regime is published.
    min_observations: int = 400

    @classmethod
    def from_params(cls, params: dict[str, Any]) -> RegimeRules:
        raw = params.get("regime") or {}
        if not isinstance(raw, dict):
            raise ValueError("engines/crypto.yaml: 'regime' must be a mapping")
        defaults = cls()
        return cls(
            window=int(raw.get("window", defaults.window)),
            drawdown_threshold=float(raw.get("drawdown_threshold", defaults.drawdown_threshold)),
            frenzy_return=float(raw.get("frenzy_return", defaults.frenzy_return)),
            frenzy_vol=float(raw.get("frenzy_vol", defaults.frenzy_vol)),
            min_observations=int(raw.get("min_observations", defaults.min_observations)),
        )


@dataclass(frozen=True)
class RegimeView:
    """Per-date regime and the three quantities it was decided from."""

    #: Regime label per date, from :data:`CRYPTO_REGIMES`.
    label: pd.Series
    #: Fraction below the trailing-year peak, negative or zero.
    drawdown: pd.Series
    #: Trailing 12-month log return.
    return_12m: pd.Series
    #: Annualized realized volatility, percent.
    realized_vol: pd.Series

    @property
    def empty(self) -> bool:
        return self.label.empty or not self.label.notna().any()

    def latest(self) -> str | None:
        """Newest published label, or ``None`` when there is none."""
        if self.label.empty:
            return None
        clean = self.label.dropna()
        return None if clean.empty else str(clean.iloc[-1])

    def latest_values(self) -> dict[str, float]:
        """The three deciding quantities on the newest date they all exist."""
        out: dict[str, float] = {}
        for name, series in (
            ("drawdown", self.drawdown),
            ("return_12m", self.return_12m),
            ("realized_vol", self.realized_vol),
        ):
            clean = series.dropna()
            if not clean.empty:
                out[name] = float(clean.iloc[-1])
        return out


def realized_volatility(log_returns: pd.Series, window: int) -> pd.Series:
    """Annualized trailing standard deviation of daily log returns, in percent."""
    return log_returns.rolling(window, min_periods=window // 4).std() * (
        math.sqrt(CALENDAR_DAYS) * 100.0
    )


def drawdown_from_peak(price: pd.Series, window: int) -> pd.Series:
    """Fraction below the trailing peak. ``rolling`` is right-closed, so trailing."""
    peak = price.rolling(window, min_periods=1).max()
    return (price / peak.replace(0.0, np.nan)) - 1.0


def classify(price: pd.Series, rules: RegimeRules) -> RegimeView:
    """Label every date from the trailing price path alone.

    ``price`` is the daily close, oldest first, positive.
    """
    clean = price.dropna()
    clean = clean[clean > 0]
    if clean.empty:
        empty = pd.Series(dtype=float)
        return RegimeView(
            label=pd.Series(dtype=object),
            drawdown=empty,
            return_12m=empty,
            realized_vol=empty,
        )

    logged = np.log(clean)
    log_returns = logged.diff()

    drawdown = drawdown_from_peak(clean, rules.window)
    return_12m = logged - logged.shift(rules.window)
    vol = realized_volatility(log_returns, rules.window)

    winter = drawdown <= -rules.drawdown_threshold
    frenzy = (return_12m >= rules.frenzy_return) & (vol >= rules.frenzy_vol)

    # np.select rather than a chain of .where: the order below IS the rule, and
    # writing it as an ordered list is the form that cannot be reordered by
    # accident during a refactor.
    label = pd.Series(
        np.select(
            [winter.fillna(False), frenzy.fillna(False)],
            ["winter", "frenzy"],
            default="normal",
        ),
        index=clean.index,
        dtype=object,
    )

    # Before the windows are full, the inputs are not knowable and neither is the
    # regime. Blanked rather than defaulted to `normal`, because "not enough
    # history" and "nothing is happening" are different statements and only one
    # of them is true on day 30.
    knowable = drawdown.notna() & return_12m.notna() & vol.notna()
    ready = pd.Series(np.arange(len(clean)) >= rules.min_observations, index=clean.index)
    label = label.where(knowable & ready)

    counts = label.value_counts().to_dict()
    log.info(
        "crypto regime: %s over %d observations",
        ", ".join(f"{name} {counts.get(name, 0)}" for name in CRYPTO_REGIMES),
        len(clean),
    )

    return RegimeView(label=label, drawdown=drawdown, return_12m=return_12m, realized_vol=vol)


__all__ = [
    "CALENDAR_DAYS",
    "RegimeRules",
    "RegimeView",
    "classify",
    "drawdown_from_peak",
    "realized_volatility",
]
