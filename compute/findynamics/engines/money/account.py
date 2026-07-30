"""The money-market account — the numeraire every other engine discounts against.

``M(t) = M(0) · exp(Σ r(tᵢ) · Δtᵢ)``

One dollar rolled overnight, forever. No model, no parameters, no fitting: the
only inputs are the observed short-rate path and a day-count convention, and the
only output is what that dollar is worth. Everything interesting about this
module is in the conventions, so they are spelled out rather than implied.

**Day count: ACT/360.** ``Δtᵢ`` is the actual number of calendar days between
consecutive observations divided by 360. That is the convention SOFR and US
Treasury bills are quoted on, and using the quote's own basis is what makes the
integral reproduce a real cash balance rather than an approximation of one. The
360 is not a typo for 365: a year of a constant 5% ACT/360 rate compounds to
``exp(0.05 · 365/360) - 1 = 5.20%``, and that extra 20bp is money that was
actually earned.

**Accrual is left-endpoint.** The rate observed for date ``d`` earns over
``[d, d_next)``. This is both the market convention — an overnight rate stamped
with date ``d`` is the rate for the period beginning ``d`` — and the only choice
compatible with the no-lookahead law: an interval may only be accrued at a rate
already known when it started. Right-endpoint accrual would pay today's balance
at a rate published tomorrow.

**Weekends and holidays accrue.** A Friday observation carries three days of
``Δt``. Cash does not stop earning because the desk is shut; it earns at
Friday's rate until Monday resets it. Nothing is interpolated and no observation
is invented — a gap is simply a longer accrual period at the rate that was in
force through it.

**The splice.** SOFR begins 2018-04-03. Before that the path is the 3-month
bill (``DTB3``), which FRED quotes on a **discount** basis, so it is converted
to an investment yield before it accrues (:func:`investment_yield_from_discount`)
rather than being used as if the two were the same number. Untreated the error
is only a handful of basis points a year, but the pre-2018 leg is sixty years
long and the error compounds in one direction the whole way.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

log = logging.getLogger("findynamics.engines.money.account")

#: Days in a year for the accrual. ACT/360 — see the module docstring.
DAY_COUNT_BASIS: float = 360.0

#: Nominal term of the fallback bill, for the discount-to-yield conversion.
BILL_TERM_DAYS: int = 91

#: An accrual gap longer than this is a data hole rather than a holiday, and is
#: not bridged: 40 days at one stale overnight rate is a fabricated balance, not
#: a conservative one. The path is cut and restarted instead.
MAX_ACCRUAL_GAP_DAYS: int = 40


class InsufficientRatePathError(RuntimeError):
    """The information set contains no short-rate path to compound."""


def investment_yield_from_discount(discount_pct: float, *, days: int = BILL_TERM_DAYS) -> float:
    """Convert a bank-discount quote to a money-market investment yield, in percent.

    A bill quoted at a discount rate ``d`` for ``n`` days costs
    ``1 - d·n/360`` per unit of face, so what the *investor* earns on the money
    actually put up is ``d / (1 - d·n/360)`` — always slightly more than ``d``.
    ``DTB3`` is quoted the first way; an accrual needs the second.

    Returns the input unchanged when the conversion is undefined (a price of zero
    or less, which no bill has ever traded at) rather than producing an infinity
    that would silently poison the whole wealth index.
    """
    if not math.isfinite(discount_pct):
        return discount_pct
    price = 1.0 - (discount_pct / 100.0) * (days / DAY_COUNT_BASIS)
    if price <= 1e-9:
        return discount_pct
    return discount_pct / price


@dataclass(frozen=True)
class RatePath:
    """The spliced short-rate path one run compounds.

    ``rates`` is in **percent** (4.31 means 4.31%), indexed by observation date,
    ascending and free of duplicates. ``sources`` labels each observation with
    the series it came from, which is what makes the splice auditable.
    """

    rates: pd.Series
    sources: pd.Series

    def __len__(self) -> int:
        return len(self.rates)

    @property
    def start(self) -> date:
        return pd.Timestamp(self.rates.index[0]).date()

    @property
    def end(self) -> date:
        return pd.Timestamp(self.rates.index[-1]).date()

    @property
    def latest(self) -> float:
        """Newest observed annualized short rate, in percent."""
        return float(self.rates.iloc[-1])

    @property
    def latest_source(self) -> str:
        return str(self.sources.iloc[-1])

    def share_from(self, series_id: str) -> float:
        """Fraction of the path taken from ``series_id`` — 0.0 to 1.0."""
        if len(self.sources) == 0:
            return 0.0
        return float((self.sources == series_id).sum()) / float(len(self.sources))


def splice_rate_path(
    frame: pd.DataFrame,
    *,
    primary: str,
    fallback: str | None = None,
    fallback_is_discount: bool = True,
    bill_term_days: int = BILL_TERM_DAYS,
) -> RatePath | None:
    """Build the short-rate path, preferring ``primary`` wherever it exists.

    Per date, not per era: the boundary is wherever the primary series actually
    starts, so nothing here needs to know that SOFR began in April 2018. On a day
    the primary is missing but the fallback quoted — a publication gap as much as
    the pre-2018 history — the fallback fills that single day, which is the same
    rule applied consistently rather than a special case.

    ``frame`` is the wide PIT frame: index ``obs_date``, one column per series id.
    """
    if frame.empty:
        return None

    primary_values = _column(frame, primary)
    fallback_values = _column(frame, fallback)
    if fallback_values is not None and fallback_is_discount:
        fallback_values = fallback_values.map(
            lambda v: investment_yield_from_discount(v, days=bill_term_days)
        )

    if primary_values is None and fallback_values is None:
        return None

    if primary_values is None:
        rates, sources = fallback_values, _labels(fallback_values, str(fallback))
    elif fallback_values is None:
        rates, sources = primary_values, _labels(primary_values, primary)
    else:
        rates = primary_values.combine_first(fallback_values)
        sources = pd.Series(
            np.where(primary_values.reindex(rates.index).notna(), primary, str(fallback)),
            index=rates.index,
        )

    usable = rates.replace([np.inf, -np.inf], np.nan).dropna().sort_index()
    if usable.empty:
        return None
    return RatePath(rates=usable, sources=sources.reindex(usable.index))


def _column(frame: pd.DataFrame, series_id: str | None) -> pd.Series | None:
    if not series_id or series_id not in frame.columns:
        return None
    values = pd.to_numeric(frame[series_id], errors="coerce").dropna()
    return None if values.empty else values.sort_index()


def _labels(values: pd.Series, series_id: str) -> pd.Series:
    return pd.Series(series_id, index=values.index)


def wealth_index(
    path: RatePath,
    *,
    basis: float = DAY_COUNT_BASIS,
    max_gap_days: int = MAX_ACCRUAL_GAP_DAYS,
) -> pd.Series:
    """Cumulative value of one unit invested at ``path.start``.

    ``M(t) = exp(Σ r(tᵢ)·Δtᵢ)``, left-endpoint accrual, ACT/``basis``. The first
    entry is exactly 1.0 by construction, which is also what makes the series a
    ratio anyone can rebase: the carry over any window is a quotient of two
    points and the base cancels.

    A gap wider than ``max_gap_days`` restarts the accrual from 1.0 rather than
    paying one stale rate across it. That is visible in the output — the index
    steps back to 1.0 — which is the point: a hole in the data should look like a
    hole, not like a plausible balance.
    """
    if len(path) == 0:
        return pd.Series(dtype=float)

    index = pd.DatetimeIndex(path.rates.index)
    rates = path.rates.to_numpy(dtype=float) / 100.0

    # Divide timedeltas rather than scaling the index's raw integers. pandas 2
    # preserves whatever resolution a frame was built at, so the same dates can
    # arrive as datetime64[ns] from one code path and datetime64[us] from
    # another; a hard-coded nanosecond divisor silently understates every accrual
    # by a factor of a thousand on the second one.
    gaps = np.diff(index.to_numpy()) / np.timedelta64(1, "D")

    # Rate at the START of each interval earns over it: rates[:-1], not rates[1:].
    increments = np.where(gaps <= max_gap_days, rates[:-1] * gaps / basis, np.nan)

    log_index = np.zeros(len(index), dtype=float)
    for i, increment in enumerate(increments, start=1):
        # A gap resets to log 1.0 = 0.0 instead of carrying a fabricated accrual.
        log_index[i] = 0.0 if math.isnan(increment) else log_index[i - 1] + increment

    dropped = int(np.isnan(increments).sum())
    if dropped:
        log.warning(
            "money: %d accrual gap(s) wider than %d days; the wealth index restarts at each",
            dropped,
            max_gap_days,
        )
    return pd.Series(np.exp(log_index), index=index, name="wealth_index")


def accrual_base(index: pd.Series) -> date | None:
    """Date the current accrual segment started — where the index is exactly 1.0.

    A wealth index is meaningless without the date the dollar was invested, and
    after a gap reset that date is no longer the start of the path. Both the first
    observation and every reset are written as ``exp(0.0)``, which is exactly
    ``1.0``; an ordinary accrual would have to sum to precisely zero in floating
    point to collide with that, which a positive rate never does and a sign-changing
    path effectively never does.
    """
    if index.empty:
        return None
    exact = index.index[index.to_numpy() == 1.0]
    key = exact[-1] if len(exact) else index.index[0]
    return pd.Timestamp(key).date()


def realized_carry(
    index: pd.Series,
    window_days: int,
    *,
    asof: pd.Timestamp | None = None,
    basis: float = DAY_COUNT_BASIS,
    tolerance_days: int = 10,
) -> float | None:
    """Annualized carry actually earned over the trailing ``window_days``.

    Returned as the **continuously-compounded ACT/``basis`` rate that would have
    produced the observed growth**:

        ``carry = ln(M(t) / M(t-h)) · basis / Δdays``

    which is the exact inverse of the accrual above — over a stretch where the
    short rate never moved, this returns that short rate to the last decimal. A
    simple-interest annualization would return something a few basis points off
    it and invite the reader to conclude the engine has a view. It does not; the
    number is arithmetic on the path.

    A decimal fraction, not percent: 0.0431 is 4.31%, matching
    ``AssetState.expected_return``.

    ``None`` when the path does not reach back far enough, or reaches back only
    across an accrual reset. Guessing from a shorter window would report a
    9-month carry as a 12-month one.
    """
    if index.empty or window_days <= 0:
        return None

    end = pd.Timestamp(asof) if asof is not None else pd.Timestamp(index.index[-1])
    target = end - pd.Timedelta(days=window_days)

    within = index.loc[:end]
    if within.empty:
        return None
    end_key = within.index[-1]

    earlier = within.loc[:target]
    if earlier.empty:
        return None
    start_key = earlier.index[-1]

    span_days = (end_key - start_key).days
    if span_days <= 0:
        return None
    # The window must land near where it was asked to. Beyond the tolerance the
    # nearest observation is answering a different question than the caller asked.
    if abs(span_days - window_days) > max(tolerance_days, window_days * 0.1):
        return None

    start_value = float(within.loc[start_key])
    end_value = float(within.loc[end_key])
    if start_value <= 0.0 or end_value <= 0.0:
        return None
    # An accrual reset inside the window makes the quotient meaningless: the
    # index went back to 1.0 partway, so the ratio is not a growth factor.
    if float(within.loc[start_key:end_key].min()) < start_value - 1e-12:
        return None

    growth = math.log(end_value / start_value)
    carry = growth * basis / span_days
    return carry if math.isfinite(carry) else None


__all__ = [
    "BILL_TERM_DAYS",
    "DAY_COUNT_BASIS",
    "MAX_ACCRUAL_GAP_DAYS",
    "InsufficientRatePathError",
    "RatePath",
    "accrual_base",
    "investment_yield_from_discount",
    "realized_carry",
    "splice_rate_path",
    "wealth_index",
]
