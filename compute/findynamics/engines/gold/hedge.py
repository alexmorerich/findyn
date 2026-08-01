"""The hedge score — how well gold is currently protecting an equity portfolio.

One number, 0-100, answering a conditional question: **on the days equities are
falling, what does gold do?**

The formula
-----------

Three steps, each of which is published separately in ``components`` so the score
can be taken apart::

    1.  drawdown_t   = equity close is below (1 - theta) x its trailing 252-day peak
    2.  rho_t        = corr(gold return, equity return) over the last W trading
                       days, RESTRICTED to days where drawdown is true
    3.  diversification_t = (1 - rho_t) / 2                       -> 0..1
    4.  regime_support_t   = P(hedge_bid) + P(crisis_bid)          -> 0..1
    5.  hedge_score_t = 100 x (w_c * diversification_t + w_r * regime_support_t)

with ``w_c + w_r = 1`` (defaults 0.6 / 0.4, both in ``config/engines/gold.yaml``).

Why conditional correlation and not correlation
-----------------------------------------------

Unconditional gold-equity correlation is close to zero and has been for decades,
which sounds like a perfect hedge and means almost nothing. A hedge is not an
asset that is uncorrelated on average; it is one that is *negatively* correlated
when it matters. Those are different properties and they come apart exactly when
the difference is expensive: gold's correlation with equities is mildly positive
in calm markets, goes sharply negative in the middle of a drawdown, and turns
positive again in the first days of a liquidity crisis when everything is sold
for cash. Averaging those three regimes together produces zero and describes none
of them.

So the correlation is computed only over days that are *inside an equity
drawdown* — the market below a threshold off its own trailing peak, which is a
trailing definition and knowable on the day. That is the state a hedge exists
for, and it is the state the score is about.

Why the regime term is there
----------------------------

The correlation is backward-looking by construction: it needs a drawdown to have
happened before it can say anything, so on the day a crisis starts it is still
describing the last one. The regime posterior is the forward-looking half — a
crisis bid or a hedge bid says gold is *currently* being held for protection,
and a carry headwind says the opposite. Blending the two means the score neither
waits for a drawdown to notice a regime change nor forgets what gold actually did
the last time equities fell.

The blend is convex and both weights are configuration, so a deployment that
trusts only the realized correlation can set ``regime_weight: 0.0`` and get
exactly step 3.

Degradation
-----------

No equity series, or fewer than ``min_drawdown_days`` drawdown days inside the
window: the correlation term is unavailable and the score falls back to the
regime term alone, scaled to the full range. The state says so through a
``hedge_score_degraded`` signal rather than publishing a confident number built
from one input. The engine never fabricates a correlation from too few days —
a correlation over eleven observations is noise with a decimal point.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

log = logging.getLogger("findynamics.engines.gold.hedge")


@dataclass(frozen=True)
class HedgeRules:
    """Windows, thresholds and blend weights, from ``config/engines/gold.yaml``."""

    #: Trailing window for the peak an equity drawdown is measured from.
    peak_window: int = 252
    #: How far below that peak counts as a drawdown. 5% rather than 10%: at 10%
    #: there are stretches of years with no qualifying day at all, and a hedge
    #: score that goes blank through every calm market is not a hedge score.
    drawdown_threshold: float = 0.05
    #: Trailing window the conditional correlation is measured over.
    correlation_window: int = 504
    #: Drawdown days required inside that window before a correlation is quoted.
    min_drawdown_days: int = 40
    #: Convex blend. correlation_weight + regime_weight must equal 1.
    correlation_weight: float = 0.6
    regime_weight: float = 0.4

    @classmethod
    def from_params(cls, params: dict[str, Any]) -> HedgeRules:
        raw = params.get("hedge") or {}
        if not isinstance(raw, dict):
            raise ValueError("engines/gold.yaml: 'hedge' must be a mapping")
        defaults = cls()
        rules = cls(
            peak_window=int(raw.get("peak_window", defaults.peak_window)),
            drawdown_threshold=float(raw.get("drawdown_threshold", defaults.drawdown_threshold)),
            correlation_window=int(raw.get("correlation_window", defaults.correlation_window)),
            min_drawdown_days=int(raw.get("min_drawdown_days", defaults.min_drawdown_days)),
            correlation_weight=float(raw.get("correlation_weight", defaults.correlation_weight)),
            regime_weight=float(raw.get("regime_weight", defaults.regime_weight)),
        )
        total = rules.correlation_weight + rules.regime_weight
        if not np.isclose(total, 1.0):
            raise ValueError(
                "engines/gold.yaml hedge.correlation_weight + hedge.regime_weight must be 1.0, "
                f"got {total:.4f} — the score is a convex blend and would leave 0-100 otherwise"
            )
        return rules


@dataclass(frozen=True)
class HedgeResult:
    """Per-date hedge score and the terms it was built from."""

    score: pd.Series
    #: (1 - conditional correlation) / 2, or NaN where too few drawdown days.
    diversification: pd.Series
    #: The conditional correlation itself, for the page.
    conditional_correlation: pd.Series
    #: P(hedge_bid) + P(crisis_bid) per date.
    regime_support: pd.Series
    #: Drawdown days inside each date's correlation window.
    drawdown_days: pd.Series

    @property
    def empty(self) -> bool:
        return self.score.dropna().empty

    def latest(self) -> float | None:
        clean = self.score.dropna()
        return None if clean.empty else float(clean.iloc[-1])

    def latest_correlation(self) -> float | None:
        clean = self.conditional_correlation.dropna()
        return None if clean.empty else float(clean.iloc[-1])

    @property
    def correlation_available(self) -> bool:
        return not self.conditional_correlation.dropna().empty


def equity_drawdown(equity: pd.Series, rules: HedgeRules) -> pd.Series:
    """Boolean: is the index below ``threshold`` off its trailing peak?

    The peak is a trailing rolling maximum including today, so the flag is
    knowable on its own date. A peak taken over the whole sample would mark the
    2000s as a drawdown because of what happened in 2008.
    """
    clean = equity.dropna()
    if clean.empty:
        return pd.Series(dtype=bool)
    peak = clean.rolling(rules.peak_window, min_periods=max(rules.peak_window // 4, 20)).max()
    return (clean < peak * (1.0 - rules.drawdown_threshold)).reindex(equity.index).fillna(False)


def conditional_correlation(
    gold_returns: pd.Series,
    equity_returns: pd.Series,
    drawdown: pd.Series,
    rules: HedgeRules,
) -> tuple[pd.Series, pd.Series]:
    """``(correlation, drawdown_day_count)`` over a trailing window.

    Computed by masking the two return series to drawdown days and rolling a
    Pearson correlation over what is left. The mask is applied *before* the roll,
    so the window is "the last W calendar-ordered trading days" and the
    correlation inside it uses only the drawdown days among them — which is the
    conditional quantity, not a correlation of a resampled series.
    """
    frame = pd.concat(
        {"gold": gold_returns, "equity": equity_returns, "draw": drawdown},
        axis=1,
    ).dropna(subset=["gold", "equity"])
    if frame.empty:
        return pd.Series(dtype=float), pd.Series(dtype=float)

    mask = frame["draw"].fillna(False).astype(bool)
    gold = frame["gold"].where(mask)
    equity = frame["equity"].where(mask)

    window = rules.correlation_window
    count = mask.rolling(window, min_periods=1).sum()
    # min_periods on the correlation itself is the same floor as the day count,
    # so a window that has not yet accumulated enough drawdown days yields NaN
    # rather than a correlation over a handful of points.
    correlation = gold.rolling(window, min_periods=rules.min_drawdown_days).corr(equity)
    correlation = correlation.where(count >= rules.min_drawdown_days)
    return correlation, count


def compute(
    gold_returns: pd.Series,
    equity: pd.Series | None,
    regime_support: pd.Series,
    rules: HedgeRules,
) -> HedgeResult:
    """The 0-100 hedge score per date. See the module docstring for the formula.

    ``regime_support`` is ``P(hedge_bid) + P(crisis_bid)`` on the same index as
    ``gold_returns``; where it is missing the regime term is treated as neutral
    (0.5) rather than as zero, because "no regime yet" is not evidence that gold
    has stopped hedging.
    """
    index = gold_returns.index
    support = regime_support.reindex(index).fillna(0.5).clip(0.0, 1.0)

    if equity is None or equity.dropna().empty:
        log.info("gold hedge: no equity series in the information set; regime term only")
        empty = pd.Series(np.nan, index=index)
        score = (100.0 * support).clip(0.0, 100.0)
        return HedgeResult(
            score=score,
            diversification=empty,
            conditional_correlation=empty,
            regime_support=support,
            drawdown_days=pd.Series(0.0, index=index),
        )

    equity_aligned = equity.reindex(index.union(equity.index)).ffill().reindex(index)
    equity_returns = np.log(equity_aligned.where(equity_aligned > 0)).diff()
    drawdown = equity_drawdown(equity_aligned, rules).reindex(index).fillna(False)

    correlation, days = conditional_correlation(gold_returns, equity_returns, drawdown, rules)
    correlation = correlation.reindex(index)
    days = days.reindex(index).fillna(0.0)

    diversification = ((1.0 - correlation) / 2.0).clip(0.0, 1.0)

    blended = rules.correlation_weight * diversification + rules.regime_weight * support
    # Where the correlation is unavailable the blend would be NaN, which would
    # blank the chart. Fall back to the regime term across the full range and let
    # the state's signal say the score is running on one input.
    score = (100.0 * blended.fillna(support)).clip(0.0, 100.0)

    log.info(
        "gold hedge: score %.1f (correlation %s over %d drawdown days)",
        float(score.dropna().iloc[-1]) if score.notna().any() else float("nan"),
        f"{correlation.dropna().iloc[-1]:+.2f}" if correlation.notna().any() else "unavailable",
        int(days.iloc[-1]) if not days.empty else 0,
    )
    return HedgeResult(
        score=score,
        diversification=diversification,
        conditional_correlation=correlation,
        regime_support=support,
        drawdown_days=days,
    )


__all__ = [
    "HedgeResult",
    "HedgeRules",
    "compute",
    "conditional_correlation",
    "equity_drawdown",
]
