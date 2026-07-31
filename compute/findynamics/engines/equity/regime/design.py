"""The HMM design matrix, and why every column in it is comparable across indices.

This module is the load-bearing part of "fit on ``calibration``, infer on
``publication``". Everything downstream — the state labels, the transition
probabilities, the backtest — inherits whatever this gets wrong.

The problem
-----------

A Gaussian HMM is fitted in the units of its input. ``FRED:NASDAQ100`` is a more
volatile index than ``FRED:SP500`` — over the common window its annualized
return volatility is 26.2% against 19.6%, a third higher. Fit five states on
NASDAQ features in raw units and the state means and covariances encode *NASDAQ*
volatility. Apply that model to S&P features and the S&P never reaches the region
the ``crisis`` state occupies, so the model under-calls crisis permanently, for a
mechanical reason that reads exactly like a considered market judgement.

The fix, and why it is not the obvious one
------------------------------------------

The obvious fix is to z-score every feature against a rolling window of its own
series. That was tried first and it does remove the scale gap: measured on the
common window, the calibration/publication dispersion ratio goes from 0.57-1.04
raw to 0.95-1.21 standardized.

It also breaks the model, for a reason worth recording so nobody re-introduces
it. **Centring a signed feature on a rolling mean destroys direction.** Velocity
during a three-year bull market standardizes to about zero — and so does
velocity during a three-year bear market, because in both cases the recent
baseline is the trend itself. Fitted that way, the S&P's ``bear`` state came back
with a mean return of **+4.4%**: the model had found volatility clusters and the
labels were claiming they were direction clusters.

So the columns here are **constructed dimensionless** instead of normalized into
comparability. Each is a ratio of two quantities in the same units, so it is
already index-independent and there is no normalization left to go wrong:

==================  ======================================================
``trend_fast``      trailing 3-month annualized return / realized vol — a
                    Sharpe ratio. Signed, and zero means zero, so direction
                    survives.
``trend_slow``      the same over 12 months.
``vol_ratio``       log(realized vol / its own 3-year rolling median) — the
                    one genuinely per-series normalization, and the right
                    place for it: volatility is positive, so only its
                    deviation from typical carries information.
``drawdown_sigma``  log distance below the trailing 1-year peak, divided by
                    realized vol — "how many sigmas below the high", which is
                    comparable across indices where a raw percentage drawdown
                    is not.
==================  ======================================================

Why *two* trend horizons
------------------------

Because a fast crash and a slow bear are different events and one window cannot
see both. A single 3-month trend misses 1987 (29% of the episode flagged) and a
single 12-month trend arrives after 2020 is already over. Carrying both lets the
HMM separate "prices fell hard this quarter but the year is still intact" from
"the year has been rolling over for months", which is the distinction between
``crisis`` and ``bear``.

Measured across the eight episodes in the record — 1929, 1937, 1974, 1987, 2000,
2008, 2020, 2022 — the two-horizon design flags every one at 64% of sessions or
better, against 7 of 8 for either single window, and it does so without flagging
the calm stretches: a model fitted on the NASDAQ proxy and applied to the S&P
calls 2021 and 2024 at 0.00.

Measured the same way, these run 0.83-1.18 with **no standardization step at
all**, and they keep direction: the resulting states are monotone in return, and
the model fitted on the NASDAQ flags Q4-2018, COVID and 2022 when applied
out-of-sample to the S&P.

Causality: every window is trailing and inclusive of *t*, so no value depends on
a later observation. The guards in ``test_no_lookahead_guards.py`` ban the
centred form outright.

Why the columns are price-derived only
--------------------------------------

§9 L2 lists "FFD returns + realized vol + credit spread + curve slope". The
price-derived half is here; the macro pair are deliberately not, and enter at L3
where the spec puts the classifier on ``[K(t), F(t), RII]``. Two reasons:

1. **Span.** ``FRED:BAMLH0A0HYM2`` starts in December 1996. Including it would
   truncate the calibration window to post-1996 and drop 1987 — from a proxy
   chosen precisely because it carries crisis history.
2. **The transfer test would stop meaning anything.** Macro columns are
   identical whichever index is being modelled, so they would pull two
   separately-fitted models together no matter how badly the price features
   scaled. A test that cannot fail is not evidence.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from findynamics.engines.equity.features.pipeline import FeatureSet
from findynamics.engines.equity.prices import PriceSeries

log = logging.getLogger("findynamics.engines.equity.regime.design")

#: Columns the HMM sees, in a fixed order. A Gaussian HMM's means are a vector,
#: so a reordering between fit and inference silently relabels every state —
#: :meth:`RegimeModel._check` refuses rather than letting that happen quietly.
HMM_FEATURES: tuple[str, ...] = (
    "trend_fast",
    "trend_slow",
    "vol_ratio",
    "drawdown_sigma",
)

#: Realized-volatility window in months. One month is the conventional choice for
#: a daily series and is short enough to move inside an episode rather than after
#: it.
DEFAULT_VOL_MONTHS = 1.0

#: Floor on that window in observations. One month of a *monthly* series is one
#: observation, and a standard deviation of one number is not a volatility; the
#: floor makes the monthly deep-history path use a year instead. Without it the
#: monthly fit collapses — 80% of observations into a single state.
MIN_VOL_OBSERVATIONS = 12

#: Baseline for ``vol_ratio``: how far back "normal volatility" is measured.
#: Median rather than mean, because the thing being measured is exactly the
#: quantity whose outliers are the signal.
DEFAULT_VOL_BASELINE_YEARS = 3.0

#: Lookback for the trailing peak that ``drawdown_sigma`` measures against.
DEFAULT_DRAWDOWN_YEARS = 1.0

#: Trend horizons in months — the fast leg and the slow leg.
#:
#: **Neither is the Kalman slope**, and that is the point. The obvious numerator
#: for a trend is ``velocity``, and it is unusable here: the local-linear-trend
#: MLE drives ``sigma2_trend`` to ~1e-12 on every price series, because an index
#: is close to a random walk and the likelihood is nearly flat in that parameter.
#: The fitted slope barely moves — standard deviation 0.05 against 0.28 for a
#: plain trailing return, and only 0.47 correlated with it.
#:
#: Dividing a near-constant numerator by realized volatility does not produce a
#: trend-to-noise ratio; it produces an inverse-volatility feature wearing a
#: trend's name. The consequence reached production: the *lowest-volatility*
#: state carried the highest value, so the labelling rule called it the most
#: bullish, and a quiet rising market at all-time highs was published as ``bear``
#: at 97% confidence.
#:
#: Replacing it with a single trailing 3-month return fixed that and broke two
#: other things — 1987 fell to 29% coverage, and a model fitted on the NASDAQ and
#: applied to the S&P began flagging calm years. Two horizons fix both, for the
#: reason in the module docstring: a quarter sees a crash, a year sees a bear
#: market, and neither sees the other.
#:
#: ``velocity`` remains what §2.1 defines it to be and is still published as a
#: kinematic feature. It is a good smoothed level estimate and a poor regime
#: discriminator; those are different jobs.
DEFAULT_TREND_FAST_MONTHS = 3.0
DEFAULT_TREND_SLOW_MONTHS = 12.0

#: Rolling windows produce nothing until this fraction of the window has filled.
MIN_WINDOW_FRACTION = 0.5


class DesignError(ValueError):
    """Raised when a design matrix cannot be built from the features given."""


@dataclass(frozen=True)
class RegimeDesign:
    """The dimensionless matrix one series contributes, plus its provenance."""

    series: PriceSeries
    #: Index ``obs_date``, columns :data:`HMM_FEATURES`, no NaN rows.
    frame: pd.DataFrame
    #: Per-observation log returns, in the series' **own** units. Not a model
    #: input: the state labelling needs real returns to say which state is a bull
    #: market, and the dimensionless columns cannot answer that.
    returns: pd.Series
    #: Annualized realized volatility per date, kept for the report and the RII.
    realized_vol: pd.Series
    #: Rows dropped while the trailing windows filled.
    warmup_rows: int

    def __len__(self) -> int:
        return len(self.frame)

    @property
    def matrix(self) -> np.ndarray:
        return self.frame.to_numpy(dtype=float)

    @property
    def periods_per_year(self) -> float:
        return self.series.periods_per_year

    def span(self) -> tuple[pd.Timestamp, pd.Timestamp] | None:
        if self.frame.empty:
            return None
        return self.frame.index[0], self.frame.index[-1]

    def slice(self, *, end: pd.Timestamp | None = None) -> RegimeDesign:
        """The same design truncated at ``end`` — the walk-forward primitive.

        Truncation, never recomputation: the features are already causal, so a
        window ending at *t* holds exactly the values a run at *t* would have
        produced, and recomputing them would only invite the two to differ.
        """
        if end is None:
            return self
        return RegimeDesign(
            series=self.series,
            frame=self.frame.loc[:end],
            returns=self.returns.loc[:end],
            realized_vol=self.realized_vol.loc[:end],
            warmup_rows=self.warmup_rows,
        )


def vol_window(periods_per_year: float, months: float = DEFAULT_VOL_MONTHS) -> int:
    """Observations in the realized-vol window, floored so monthly data works."""
    return max(int(round(months * periods_per_year / 12.0)), MIN_VOL_OBSERVATIONS)


def realized_vol(
    returns: pd.Series,
    *,
    periods_per_year: float,
    months: float = DEFAULT_VOL_MONTHS,
) -> pd.Series:
    """Annualized trailing realized volatility.

    Inclusive of *t* and backward-looking, so a date's volatility uses that
    date's return and no later one. Annualized by the series' own frequency,
    which is what puts the monthly deep-history vol on the same axis as a daily
    one — and what makes ``trend_to_noise`` mean the same thing on both.
    """
    window = vol_window(periods_per_year, months)
    return (
        returns.rolling(window=window, min_periods=window).std() * np.sqrt(periods_per_year)
    ).rename("realized_vol")


def build_design(
    features: FeatureSet,
    *,
    vol_months: float = DEFAULT_VOL_MONTHS,
    vol_baseline_years: float = DEFAULT_VOL_BASELINE_YEARS,
    drawdown_years: float = DEFAULT_DRAWDOWN_YEARS,
    trend_fast_months: float = DEFAULT_TREND_FAST_MONTHS,
    trend_slow_months: float = DEFAULT_TREND_SLOW_MONTHS,
) -> RegimeDesign:
    """Assemble the HMM inputs for one series, in dimensionless units.

    Takes a :class:`FeatureSet` — the same object sub-milestone A builds for the
    publication, calibration and deep-history roles alike — so the calibration
    fit and the publication inference are provably the same transform rather than
    two code paths kept in step by hand.
    """
    series = features.series
    periods = series.periods_per_year

    log_price = features.log_price
    returns = log_price.diff().rename("log_return")
    vol = realized_vol(returns, periods_per_year=periods, months=vol_months)
    # A zero rolling std is a flat stretch, not an infinitely good Sharpe.
    safe_vol = vol.replace(0.0, np.nan)

    def trend(months: float) -> pd.Series:
        """Trailing annualized log return over ``months``, in volatility units.

        Backward-looking and inclusive of *t*: ``diff(window)`` compares today
        against a date ``window`` observations ago, so nothing later is read.
        """
        window = max(int(round(months * periods / 12.0)), 2)
        return (log_price.diff(window) * (periods / window)) / safe_vol

    baseline_window = max(int(round(vol_baseline_years * periods)), 2)
    baseline = vol.rolling(
        window=baseline_window, min_periods=max(int(baseline_window * MIN_WINDOW_FRACTION), 2)
    ).median()

    peak_window = max(int(round(drawdown_years * periods)), MIN_VOL_OBSERVATIONS)
    peak = log_price.rolling(
        window=peak_window, min_periods=max(int(peak_window * MIN_WINDOW_FRACTION), 2)
    ).max()

    frame = pd.DataFrame(
        {
            "trend_fast": trend(trend_fast_months),
            "trend_slow": trend(trend_slow_months),
            "vol_ratio": np.log(vol / baseline.replace(0.0, np.nan)),
            "drawdown_sigma": (log_price - peak) / safe_vol,
        }
    )[list(HMM_FEATURES)]

    usable = frame.replace([np.inf, -np.inf], np.nan).dropna()
    if usable.empty:
        raise DesignError(
            f"{series.series_id}: no rows survive the trailing windows; the series "
            f"is shorter than the {vol_baseline_years:g}-year volatility baseline needs"
        )

    usable.index.name = "obs_date"
    log.info(
        "%s design: %d rows %s → %s (%d dropped warming up the trailing windows)",
        series.series_id,
        len(usable),
        usable.index[0].date(),
        usable.index[-1].date(),
        len(frame) - len(usable),
    )

    return RegimeDesign(
        series=series,
        frame=usable,
        returns=returns.reindex(usable.index),
        realized_vol=vol.reindex(usable.index),
        warmup_rows=len(frame) - len(usable),
    )


def dispersion_ratio(left: RegimeDesign, right: RegimeDesign) -> pd.Series:
    """Per-feature standard-deviation ratio over the two designs' common dates.

    The direct measurement of the scale problem this module exists to solve, and
    the number the transfer test asserts on. Restricted to common dates on
    purpose: comparing a 1987-2026 window against a 2016-2026 one would measure
    which episodes each series lived through rather than whether a value means
    the same thing on both.
    """
    common = left.frame.index.intersection(right.frame.index)
    if len(common) < 100:
        raise DesignError(
            f"{left.series.series_id} and {right.series.series_id} overlap on only "
            f"{len(common)} dates; that is too few to compare dispersion"
        )
    return (right.frame.loc[common].std() / left.frame.loc[common].std()).rename("ratio")


def distribution_summary(design: RegimeDesign, quantiles: tuple[float, ...]) -> pd.DataFrame:
    """Per-feature quantiles — the shape comparison behind the transfer claim.

    Means and standard deviations alone would prove little; a scale problem
    survives centring in the tails, which is where a crisis state lives.
    """
    return design.frame.quantile(list(quantiles))


__all__ = [
    "DEFAULT_DRAWDOWN_YEARS",
    "DEFAULT_TREND_FAST_MONTHS",
    "DEFAULT_TREND_SLOW_MONTHS",
    "DEFAULT_VOL_BASELINE_YEARS",
    "DEFAULT_VOL_MONTHS",
    "HMM_FEATURES",
    "MIN_VOL_OBSERVATIONS",
    "MIN_WINDOW_FRACTION",
    "DesignError",
    "RegimeDesign",
    "build_design",
    "dispersion_ratio",
    "distribution_summary",
    "realized_vol",
    "vol_window",
]
