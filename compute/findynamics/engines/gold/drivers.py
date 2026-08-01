"""The driver panel — what moves gold, since nothing discounts it.

Gold has no cash flow. There is no earnings stream to grow, no coupon to
discount and no book value to compare a price against, so every valuation
technique the other engines use is not merely inaccurate here, it is undefined.
What can be modelled is *demand for a monetary asset with no issuer*, and that
demand has four observable drivers:

``real_rate`` (level and 12-month change)
    The one genuine cost of holding gold. A non-yielding asset competes with a
    real yield; when the real yield rises, the competition is arithmetic. Both
    the level and the change are published because they say different things —
    a real rate of 2% that has been 2% for a decade is a settled headwind, and a
    real rate of 0% that was -2% a year ago is an active one. The regime model
    reads the change; the level is published for the page and for signals.
``usd_trend``
    Gold is quoted in dollars, so a stronger dollar is a mechanical headwind
    before any behavioural story about it. Read as a 12-month log change rather
    than a level: dollar index levels are arbitrary (each index is 100 at its own
    base date) and only their changes are comparable.
``stress``
    The Chicago Fed's NFCI — financial conditions in one number, positive being
    tight. This is the crisis channel.
``equity_rii``
    FinEquity's instability index, read back through the point-in-time gateway as
    an ordinary series (``ENGINE:equity.rii``). Optional by construction: the
    first run of a system has published nothing, and a driver panel that cannot
    be built without another engine's output would make the two engines an
    ordering dependency in all but name.

Two splices, and both are honest about being splices
----------------------------------------------------

**The real rate.** ``DGS10 - T10YIE`` is an ex-ante real rate and it is the right
number — but TIPS breakevens start in 2003, and a driver that starts in 2003 puts
1980 out of reach. 1980 is not optional history here: the Volcker rate shock is
the archetype of ``carry_headwind`` and the only unambiguous one in the record.
So before breakevens exist the panel falls back to an **ex-post** real rate,
``DGS10 - trailing 12m CPI inflation``. That is a different quantity — realized
rather than expected inflation — so it is not blended: each date's rate carries
``real_rate_is_ex_post`` saying which definition produced it.

**The dollar.** The broad index (DTWEXBGS) starts in 2006; the major-currencies
index (DTWEXM) runs 1973-2019. They are different baskets at different base
levels, so splicing the *levels* would print a step change of forty index points
in 2006 and hand the model a fictional dollar collapse. What is spliced is the
12-month log change — the only thing the driver reads, and the only thing the two
indices agree about.

Causality
---------

Every transform here is trailing: ``diff``, ``expanding``, ``ffill``. No
centred window, no ``shift(-n)``, no ``bfill``. Standardization is an expanding
z-score (§14.1 rule 3), so a date's score uses only what was knowable on it, and
the fixed clip keeps one 1980 print from compressing forty years of the scale.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

log = logging.getLogger("findynamics.engines.gold.drivers")

#: Roles resolved against ``engines.gold.series`` in series.yaml. Required roles
#: are the ones without which there is no panel at all.
REQUIRED_ROLES = ("price", "nominal_10y")
OPTIONAL_ROLES = (
    "breakeven_10y",
    "cpi",
    "usd_index",
    "usd_index_legacy",
    "liquidity_stress",
    "equity_proxy",
    "equity_rii",
)

#: Columns the panel always carries, whether or not their inputs arrived.
DRIVER_COLUMNS = (
    "real_rate",
    "real_rate_change_12m",
    "usd_trend",
    "stress",
    "equity_rii",
)

#: Standardized companions, one per driver. The regime model reads these.
Z_COLUMNS = tuple(f"z_{name}" for name in DRIVER_COLUMNS)


@dataclass(frozen=True)
class DriverRules:
    """Windows and standardization settings, from ``config/engines/gold.yaml``."""

    #: Trading days in the dollar-trend and real-rate-change lookbacks. Calendar
    #: months would be the natural unit, but the panel is built on a daily index
    #: whose spacing is the market's, so the windows are stated in observations.
    trend_days: int = 252
    #: Minimum observations before an expanding z-score means anything. Three
    #: years: below that the standard deviation is estimated from too few
    #: independent moves and the score swings on its own denominator.
    z_min_observations: int = 756
    #: Expanding z-scores are clipped here. A 1980-sized print is real and must
    #: not be winsorized away, but it must also not set the scale for the next
    #: forty years.
    z_clip: float = 4.0
    #: Months of trailing CPI used for the ex-post real rate fallback.
    inflation_months: int = 12

    @classmethod
    def from_params(cls, params: dict[str, Any]) -> DriverRules:
        raw = params.get("drivers") or {}
        if not isinstance(raw, dict):
            raise ValueError("engines/gold.yaml: 'drivers' must be a mapping")
        defaults = cls()
        return cls(
            trend_days=int(raw.get("trend_days", defaults.trend_days)),
            z_min_observations=int(raw.get("z_min_observations", defaults.z_min_observations)),
            z_clip=float(raw.get("z_clip", defaults.z_clip)),
            inflation_months=int(raw.get("inflation_months", defaults.inflation_months)),
        )


@dataclass(frozen=True)
class DriverPanel:
    """Per-date drivers, daily and monthly, with a record of what was available."""

    #: Daily panel: the driver columns, their z-scores, and ``log_price``.
    daily: pd.DataFrame
    #: Month-end panel with ``ret`` — the monthly log return the regime model reads.
    monthly: pd.DataFrame
    #: Driver name -> whether its inputs arrived at all.
    available: dict[str, bool] = field(default_factory=dict)
    #: True where the real rate is the ex-post CPI fallback rather than a breakeven.
    ex_post_share: float = 0.0

    @property
    def empty(self) -> bool:
        return self.daily.empty

    def latest(self) -> pd.Series | None:
        """The newest fully-formed daily row, or ``None`` when there is none."""
        if self.daily.empty:
            return None
        return self.daily.iloc[-1]

    def explain(self) -> dict[str, float]:
        """Driver values on the newest date, for ``AssetState.components``."""
        row = self.latest()
        if row is None:
            return {}
        out: dict[str, float] = {}
        for name in (*DRIVER_COLUMNS, *Z_COLUMNS):
            value = row.get(name)
            if value is not None and np.isfinite(value):
                out[name] = round(float(value), 6)
        return out


def _expanding_z(series: pd.Series, rules: DriverRules) -> pd.Series:
    """Expanding-window z-score, clipped. Trailing by construction (§14.1 rule 3).

    ``expanding`` includes the current observation, which is correct: the value
    being scored was knowable on its own date. What must never appear here is a
    centred window or a full-sample mean, either of which would let a date be
    scored against a distribution that had not happened yet.
    """
    clean = series.astype(float)
    mean = clean.expanding(rules.z_min_observations).mean()
    std = clean.expanding(rules.z_min_observations).std()
    z = (clean - mean) / std.replace(0.0, np.nan)
    return z.clip(-rules.z_clip, rules.z_clip)


def _trailing_change(series: pd.Series, window: int) -> pd.Series:
    """Change over ``window`` observations. ``shift`` looks backwards only."""
    clean = series.dropna()
    if clean.empty:
        return pd.Series(dtype=float)
    return (clean - clean.shift(window)).reindex(series.index)


def _trailing_log_change(series: pd.Series, window: int) -> pd.Series:
    clean = series.dropna()
    clean = clean[clean > 0]
    if clean.empty:
        return pd.Series(dtype=float)
    logged = np.log(clean)
    return (logged - logged.shift(window)).reindex(series.index)


def real_rate(
    frame: pd.DataFrame,
    ids: dict[str, str],
    rules: DriverRules,
) -> tuple[pd.Series, pd.Series]:
    """The 10y real rate and a flag for which definition produced each date.

    Returns ``(rate, is_ex_post)``. The breakeven-implied rate wins wherever it
    exists; the CPI fallback carries the years before TIPS. They are not
    averaged anywhere they overlap — one of them is the answer for a given date
    and the flag says which.
    """
    nominal = frame.get(ids["nominal_10y"])
    if nominal is None:
        return pd.Series(dtype=float), pd.Series(dtype=float)
    nominal = nominal.ffill()

    ex_ante = pd.Series(np.nan, index=frame.index)
    breakeven_id = ids.get("breakeven_10y")
    if breakeven_id and breakeven_id in frame:
        breakeven = frame[breakeven_id].ffill()
        ex_ante = (nominal - breakeven).where(breakeven.notna())

    ex_post = pd.Series(np.nan, index=frame.index)
    cpi_id = ids.get("cpi")
    if cpi_id and cpi_id in frame:
        cpi = frame[cpi_id].ffill()
        # Month-on-month is far too noisy to subtract from a 10y yield; the
        # trailing year is the inflation a holder actually experienced.
        monthly = cpi.resample("ME").last().dropna()
        yoy = (monthly / monthly.shift(rules.inflation_months) - 1.0) * 100.0
        ex_post = nominal - yoy.reindex(frame.index, method="ffill")

    rate = ex_ante.combine_first(ex_post)
    is_ex_post = rate.notna() & ex_ante.isna()
    return rate, is_ex_post


def usd_trend(frame: pd.DataFrame, ids: dict[str, str], rules: DriverRules) -> pd.Series:
    """12-month log change of the dollar, spliced across the two index families.

    The splice is on the CHANGE, never on the level: DTWEXBGS and DTWEXM are
    different baskets with different base dates, and a level splice would print a
    step change that no dollar move produced.
    """
    primary_id, legacy_id = ids.get("usd_index"), ids.get("usd_index_legacy")
    primary = (
        _trailing_log_change(frame[primary_id].ffill(), rules.trend_days)
        if primary_id and primary_id in frame
        else pd.Series(dtype=float)
    )
    legacy = (
        _trailing_log_change(frame[legacy_id].ffill(), rules.trend_days)
        if legacy_id and legacy_id in frame
        else pd.Series(dtype=float)
    )
    if primary.empty:
        return legacy.reindex(frame.index)
    if legacy.empty:
        return primary.reindex(frame.index)
    return primary.reindex(frame.index).combine_first(legacy.reindex(frame.index))


def build_panel(
    frame: pd.DataFrame,
    ids: dict[str, str],
    rules: DriverRules,
) -> DriverPanel:
    """Assemble the daily and monthly driver panels from a wide PIT frame.

    ``frame`` is ``world.series.wide(...)`` — one column per series id, indexed by
    observation date, missing values left as NaN. Everything below is a trailing
    transform of that frame and of nothing else.
    """
    if frame.empty or ids["price"] not in frame:
        return DriverPanel(daily=pd.DataFrame(), monthly=pd.DataFrame())

    price = frame[ids["price"]].dropna()
    price = price[price > 0]
    if price.empty:
        return DriverPanel(daily=pd.DataFrame(), monthly=pd.DataFrame())

    # The gold price is the spine: the panel speaks about dates gold traded on.
    index = price.index
    aligned = frame.reindex(index).ffill()

    rate, is_ex_post = real_rate(frame, ids, rules)
    panel = pd.DataFrame(index=index)
    panel["price"] = price
    panel["log_price"] = np.log(price)
    panel["real_rate"] = rate.reindex(index)
    panel["real_rate_change_12m"] = _trailing_change(panel["real_rate"], rules.trend_days)
    panel["usd_trend"] = usd_trend(frame, ids, rules).reindex(index)

    stress_id = ids.get("liquidity_stress")
    panel["stress"] = (
        aligned[stress_id] if stress_id and stress_id in aligned else pd.Series(np.nan, index=index)
    )

    rii_id = ids.get("equity_rii")
    panel["equity_rii"] = (
        aligned[rii_id] if rii_id and rii_id in aligned else pd.Series(np.nan, index=index)
    )

    for name in DRIVER_COLUMNS:
        panel[f"z_{name}"] = _expanding_z(panel[name], rules)

    available = {name: bool(panel[name].notna().any()) for name in DRIVER_COLUMNS}
    missing = [name for name, present in available.items() if not present]
    if missing:
        log.info("gold drivers unavailable in this information set: %s", ", ".join(missing))

    ex_post_flags = is_ex_post.reindex(index).fillna(False)
    known = panel["real_rate"].notna()
    ex_post_share = float(ex_post_flags[known].mean()) if known.any() else 0.0

    monthly = panel.resample("ME").last()
    # The monthly return is the change in the month-end log price, so it is a
    # function of two observations the engine has already seen. Dropping the
    # first (which has no predecessor) rather than filling it keeps the series
    # free of a zero that never happened.
    monthly["ret"] = monthly["log_price"].diff() * 100.0
    monthly = monthly.dropna(subset=["ret"])

    return DriverPanel(
        daily=panel,
        monthly=monthly,
        available=available,
        ex_post_share=round(ex_post_share, 6),
    )


__all__ = [
    "DRIVER_COLUMNS",
    "OPTIONAL_ROLES",
    "REQUIRED_ROLES",
    "Z_COLUMNS",
    "DriverPanel",
    "DriverRules",
    "build_panel",
    "real_rate",
    "usd_trend",
]
