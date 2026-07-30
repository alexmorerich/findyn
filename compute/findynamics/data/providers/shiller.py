"""Robert Shiller's long-history dataset (`ie_data.xls`, monthly, 1871–).

The deep-history backbone: CAPE, real price, real earnings and dividend yield
back to 1871, with no API key. One workbook download serves every series, so the
file is fetched once through the transport cache and sliced in memory.

Two shapes in this file will bite a naive parser:

* Dates are floats where the fraction is the month — ``1871.1`` is October 1871,
  not "January-ish". Read as a decimal it silently mis-dates a quarter of history.
* The last row is a prose footnote ("Sept price is Sept 4th close"), not data.

FINDYN_V1_SPEC.md §5.1 source 2.
"""

from __future__ import annotations

import logging
from datetime import date
from io import BytesIO

import pandas as pd

from findynamics.data.providers.base import (
    Observation,
    ParseError,
    Provider,
    SeriesMetadata,
    synthesize_release_date,
)
from findynamics.data.providers.resilience import Transport

log = logging.getLogger("findynamics.data.providers.shiller")

DEFAULT_URL = (
    "https://img1.wsimg.com/blobby/go/e5e77e0b-59d1-44d9-ab25-4763ac982e53/downloads/ie_data.xls"
)

#: Header occupies rows 0-7; observations start at row 8.
FIRST_DATA_ROW = 8

#: Shiller publishes monthly with roughly a month's delay; no vintage is exposed,
#: so release dates are synthesized from the end of the period (§5.2).
PUBLICATION_LAG_DAYS = 30


class _Col:
    """Positional columns in the `Data` sheet. The sheet has no usable header row."""

    DATE = 0
    PRICE = 1
    DIVIDEND = 2
    EARNINGS = 3
    CPI = 4
    GS10 = 6
    REAL_PRICE = 7
    REAL_DIVIDEND = 8
    REAL_EARNINGS = 10
    CAPE = 12
    EXCESS_CAPE_YIELD = 16


_SERIES: dict[str, tuple[str, str, str]] = {
    # series_id: (title, unit, derivation key)
    "SHILLER:CAPE": ("Cyclically Adjusted P/E Ratio (CAPE)", "ratio", "cape"),
    "SHILLER:REAL_PRICE": ("S&P Composite Real Price", "index", "real_price"),
    "SHILLER:REAL_EPS": ("S&P Composite Real Earnings", "index", "real_earnings"),
    "SHILLER:DIVIDEND_YIELD": ("S&P Composite Dividend Yield", "percent", "dividend_yield"),
    "SHILLER:NOMINAL_PRICE": ("S&P Composite Nominal Price", "index", "price"),
    "SHILLER:CPI": ("Consumer Price Index (Shiller series)", "index", "cpi"),
    "SHILLER:GS10": ("Long Interest Rate (10Y)", "percent", "gs10"),
    "SHILLER:EXCESS_CAPE_YIELD": ("Excess CAPE Yield", "percent", "excess_cape_yield"),
}


def parse_shiller_date(raw: float) -> date | None:
    """Convert Shiller's ``YYYY.MM`` float to the first day of that month.

    ``1871.1`` is October: the fraction is a two-digit month, so it is scaled by
    100 and rounded rather than read as a decimal fraction of a year.
    """
    if raw is None or pd.isna(raw):
        return None
    year = int(raw)
    month = int(round((float(raw) - year) * 100))
    if not 1 <= month <= 12:
        return None
    return date(year, month, 1)


class ShillerProvider(Provider):
    id = "shiller"
    requires_api_key = False

    def __init__(
        self,
        transport: Transport,
        *,
        url: str = DEFAULT_URL,
        publication_lag_days: int = PUBLICATION_LAG_DAYS,
    ) -> None:
        self.transport = transport
        self.url = url
        self.publication_lag_days = publication_lag_days
        self._frame: pd.DataFrame | None = None

    def available_series(self) -> list[str]:
        return sorted(_SERIES)

    # -- workbook -------------------------------------------------------

    def _load(self) -> pd.DataFrame:
        """Download and normalize the workbook once per provider instance."""
        if self._frame is not None:
            return self._frame

        # A full day: the source updates monthly, so re-fetching 1.6MB per run
        # is pure waste.
        response = self.transport.get(self.url, cache_ttl=86_400)
        try:
            raw = pd.read_excel(BytesIO(response.content), sheet_name="Data", header=None)
        except Exception as err:
            raise ParseError(self.id, f"could not read ie_data.xls: {err}") from err

        if raw.shape[0] <= FIRST_DATA_ROW:
            raise ParseError(self.id, f"workbook has only {raw.shape[0]} rows")

        body = raw.iloc[FIRST_DATA_ROW:].copy()
        body["obs_date"] = body[_Col.DATE].map(parse_shiller_date)
        # Drops the trailing footnote row, whose date cell is prose or blank.
        body = body[body["obs_date"].notna()]
        if body.empty:
            raise ParseError(self.id, "no parseable observation rows")

        numeric = {
            "price": _Col.PRICE,
            "dividend": _Col.DIVIDEND,
            "earnings": _Col.EARNINGS,
            "cpi": _Col.CPI,
            "gs10": _Col.GS10,
            "real_price": _Col.REAL_PRICE,
            "real_dividend": _Col.REAL_DIVIDEND,
            "real_earnings": _Col.REAL_EARNINGS,
            "cape": _Col.CAPE,
            "excess_cape_yield": _Col.EXCESS_CAPE_YIELD,
        }
        frame = pd.DataFrame({"obs_date": body["obs_date"].to_numpy()})
        for name, col in numeric.items():
            frame[name] = pd.to_numeric(body[col], errors="coerce").to_numpy()

        # Yield is not published directly; D/P is the definition. Guard the
        # division so a zero or missing price yields NaN rather than an inf that
        # would later read as a real observation.
        price = frame["price"].where(frame["price"] > 0)
        frame["dividend_yield"] = (frame["dividend"] / price) * 100.0

        frame = frame.sort_values("obs_date").reset_index(drop=True)
        self._frame = frame
        log.info(
            "shiller: %d monthly rows, %s..%s",
            len(frame),
            frame["obs_date"].iloc[0],
            frame["obs_date"].iloc[-1],
        )
        return frame

    # -- Provider -------------------------------------------------------

    def fetch_metadata(self, series_id: str) -> SeriesMetadata:
        self._require_known_series(series_id)
        title, unit, key = _SERIES[series_id]
        frame = self._load()
        present = frame[frame[key].notna()]
        return SeriesMetadata(
            series_id=series_id,
            provider=self.id,
            title=title,
            frequency="monthly",
            unit=unit,
            seasonal_adjustment="not_seasonally_adjusted",
            first_observation=present["obs_date"].iloc[0] if not present.empty else None,
            last_observation=present["obs_date"].iloc[-1] if not present.empty else None,
            notes=(
                "Robert J. Shiller, Irrational Exuberance dataset (ie_data.xls). "
                "No vintage information is published; release dates are synthesized "
                f"as period end + {self.publication_lag_days} days."
            ),
        )

    def fetch_observations(
        self,
        series_id: str,
        *,
        start: date | None = None,
        end: date | None = None,
    ) -> list[Observation]:
        self._require_known_series(series_id)
        _, unit, key = _SERIES[series_id]
        frame = self._load()

        selection = frame[["obs_date", key]].dropna()
        if start is not None:
            selection = selection[selection["obs_date"] >= start]
        if end is not None:
            selection = selection[selection["obs_date"] <= end]

        observations: list[Observation] = []
        for obs_date, value in selection.itertuples(index=False):
            release = synthesize_release_date(obs_date, "monthly", self.publication_lag_days)
            observations.append(
                Observation(
                    series_id=series_id,
                    provider=self.id,
                    frequency="monthly",
                    unit=unit,
                    observation_date=obs_date,
                    release_date=release,
                    revision_date=release,
                    value=float(value),
                )
            )
        return observations
