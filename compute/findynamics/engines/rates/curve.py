"""Yield-curve assembly from point-in-time series.

Turns the eleven constant-maturity series into a cross-section per date: a list
of (maturity, yield) pairs sorted by maturity. Two facts drive the design:

* **The curve is ragged.** DGS1MO only starts in 2001, DGS20 has a gap from 1987
  to 1993, DGS30 from 2002 to 2006, and every tenor is missing on holidays.
  Requiring the full set would throw away most of the history, so a date is kept
  whenever enough tenors are present to identify the three betas.
* **A yield of exactly zero is a real quote.** Bills traded at 0.00% for years
  after 2008. Missing data is therefore NaN and only NaN — nothing is
  forward-filled across dates, because carrying Friday's 10y into Monday would
  manufacture a quote nobody made.

Everything here reads through :class:`PITAccessor`, so what a date's curve looks
like depends on the information set the accessor was built with, not on when the
code runs (§14.1 rule 2).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from findynamics.core.contracts.pit import PITAccessor


@dataclass(frozen=True)
class YieldCurve:
    """One date's observable curve."""

    as_of: date
    maturities: tuple[float, ...]
    yields: tuple[float, ...]

    def __len__(self) -> int:
        return len(self.maturities)

    def yield_at(self, maturity: float) -> float | None:
        """Observed yield at exactly ``maturity``, if that tenor quoted today."""
        for m, y in zip(self.maturities, self.yields, strict=True):
            if abs(m - maturity) < 1e-9:
                return y
        return None

    def as_points(self) -> tuple[tuple[float, float], ...]:
        return tuple(zip(self.maturities, self.yields, strict=True))


def tenor_map(params: dict[str, Any]) -> dict[str, float]:
    """``series_id -> maturity in years`` from ``config/engines/rates.yaml``.

    Configuration, not code: adding the 4-month bill is a yaml edit.
    """
    raw = params.get("tenors") or {}
    if not isinstance(raw, dict) or not raw:
        raise ValueError("engines/rates.yaml: 'tenors' must be a non-empty mapping")

    tenors: dict[str, float] = {}
    for series_id, maturity in raw.items():
        value = float(maturity)
        if value <= 0:
            raise ValueError(f"engines/rates.yaml: tenor {series_id} must be positive, got {value}")
        tenors[str(series_id)] = value
    return tenors


def curve_frame(
    series: PITAccessor,
    tenors: dict[str, float],
    *,
    start: date | None = None,
) -> pd.DataFrame:
    """Wide frame of yields: index ``obs_date``, one column per maturity.

    Columns are maturities in years, ascending — the curve as a matrix, which is
    what both the per-date fit and the replay harness want. Dates on which no
    tenor quoted at all are dropped; partial dates are kept with NaNs.
    """
    frame = series.wide(sorted(tenors), start=start)
    if frame.empty:
        return pd.DataFrame(index=pd.DatetimeIndex([], name="obs_date"))

    present = {col: tenors[col] for col in frame.columns if col in tenors}
    if not present:
        return pd.DataFrame(index=pd.DatetimeIndex([], name="obs_date"))

    wide = frame[list(present)].rename(columns=present)
    wide = wide.reindex(sorted(wide.columns), axis=1)
    return wide.dropna(how="all")


def curve_on(frame: pd.DataFrame, as_of: date, *, min_tenors: int = 5) -> YieldCurve | None:
    """The curve on one date, or ``None`` if that date is too thin or absent.

    ``as_of`` must be an exact observation date. Nothing is interpolated to a
    non-trading day: a Sunday has no curve, and pretending otherwise would put a
    fabricated cross-section into a replay.
    """
    key = pd.Timestamp(as_of).normalize()
    if key not in frame.index:
        return None
    return _row_to_curve(frame.loc[key], key, min_tenors=min_tenors)


def latest_curve(frame: pd.DataFrame, *, min_tenors: int = 5) -> YieldCurve | None:
    """Newest date in ``frame`` that yields a usable curve.

    Walks backwards rather than taking the last row outright: the newest date
    might be a half-published day with two tenors on it, and the right answer
    then is yesterday's full curve, not no curve at all.
    """
    for key in reversed(frame.index):
        curve = _row_to_curve(frame.loc[key], key, min_tenors=min_tenors)
        if curve is not None:
            return curve
    return None


def iter_curves(frame: pd.DataFrame, *, min_tenors: int = 5):
    """Every usable curve in ``frame``, oldest first."""
    for key in frame.index:
        curve = _row_to_curve(frame.loc[key], key, min_tenors=min_tenors)
        if curve is not None:
            yield curve


def _row_to_curve(row: pd.Series, key: pd.Timestamp, *, min_tenors: int) -> YieldCurve | None:
    usable = row.dropna()
    if len(usable) < min_tenors:
        return None

    maturities = np.asarray(usable.index, dtype=float)
    yields = usable.to_numpy(dtype=float)
    finite = np.isfinite(maturities) & np.isfinite(yields)
    if int(finite.sum()) < min_tenors:
        return None

    order = np.argsort(maturities[finite])
    return YieldCurve(
        as_of=key.date(),
        maturities=tuple(float(m) for m in maturities[finite][order]),
        yields=tuple(float(y) for y in yields[finite][order]),
    )
