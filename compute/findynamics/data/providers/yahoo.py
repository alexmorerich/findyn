"""Yahoo Finance daily closes (unofficial chart API).

FINDYN_V1_SPEC.md §5.1 source 7 named Yahoo a **fallback only**, to be isolated
behind one file so it can be deleted without touching anything else. That is
still the right posture and this module honours it: nothing imports it except
the registry, and the roles it fills in ``series.yaml`` are ones the engine
already degrades without.

It exists because the alternatives ran out. FRED's ``SP500`` is licence-capped
to a rolling ten-year window, and Stooq — the only other free source of daily
S&P history — fronts its CSV with a JavaScript proof-of-work challenge that
blocks a developer network and GitHub's runners alike (verified, twice). Yahoo
serves ``^GSPC`` daily from 1927, which is what makes a regime model fittable
on the S&P itself rather than on a more volatile proxy.

What it is not: a contract. The endpoint is undocumented, unversioned, and
rate-limits aggressively by source IP — a 429 from a cloud range is routine and
says nothing about whether a laptop can reach it. So the adapter validates the
response shape rather than trusting it, and every structural surprise raises
:class:`ParseError` naming what was missing instead of an ``IndexError`` three
frames deep.

Dates, the part that is easy to get wrong
-----------------------------------------

Yahoo timestamps each daily bar with the session *open*, as epoch seconds. Read
as UTC that happens to be right for US equities — 09:30 in New York is 13:30 or
14:30 UTC, still the same calendar day — and wrong the moment the same code
meets an exchange whose open falls on the other side of midnight UTC. The bar
is a session on the exchange's calendar, so it is converted on the exchange's
calendar, using the ``gmtoffset`` Yahoo returns alongside it.
"""

from __future__ import annotations

import logging
from datetime import UTC, date, datetime, timedelta
from typing import Any

from findynamics.data.providers.base import (
    Observation,
    ParseError,
    Provider,
    SeriesMetadata,
)
from findynamics.data.providers.resilience import Transport

log = logging.getLogger("findynamics.data.providers.yahoo")

BASE_URL = "https://query1.finance.yahoo.com/v8/finance/chart"

#: series_id -> (symbol, human title, unit)
_SYMBOLS: dict[str, tuple[str, str, str]] = {
    "YAHOO:^GSPC": ("^GSPC", "S&P 500 Index (daily close)", "index"),
    "YAHOO:^NDX": ("^NDX", "Nasdaq 100 Index (daily close)", "index"),
    "YAHOO:^DJI": ("^DJI", "Dow Jones Industrial Average (daily close)", "index"),
    "YAHOO:^VIX": ("^VIX", "CBOE Volatility Index (daily close)", "index"),
}


#: 1900-01-01. A floor rather than the real start: Yahoo clamps to whatever it
#: holds, so this asks for everything without hard-coding a per-symbol epoch.
HISTORY_FLOOR_EPOCH = -2208988800

#: Below this many bars the median gap says nothing — a fortnight of daily data
#: and a fortnight of weekly data are not distinguishable at that length.
MIN_BARS_FOR_DENSITY_CHECK = 30

#: Daily bars are 1 day apart, 3 across a weekend, so the median is 1. Allowing
#: 4 leaves room for a holiday-heavy stretch without admitting weekly bars.
MAX_DAILY_MEDIAN_GAP_DAYS = 4


def _require(value: Any, what: str) -> Any:
    if value is None:
        raise ParseError("yahoo", f"response has no {what}")
    return value


class YahooProvider(Provider):
    id = "yahoo"
    requires_api_key = False

    def __init__(self, transport: Transport, *, base_url: str = BASE_URL) -> None:
        self.transport = transport
        self.base_url = base_url.rstrip("/")

    def available_series(self) -> list[str]:
        return sorted(_SYMBOLS)

    def _symbol(self, series_id: str) -> str:
        self._require_known_series(series_id)
        return _SYMBOLS[series_id][0]

    def _result(self, symbol: str, params: dict[str, str]) -> dict[str, Any]:
        """The single ``chart.result`` entry, or a ParseError explaining its absence."""
        response = self.transport.get(f"{self.base_url}/{symbol}", params=params, cache_ttl=3600)
        try:
            payload = response.json()
        except ValueError as err:
            raise ParseError(self.id, f"non-JSON response for {symbol}: {err}") from err
        if not isinstance(payload, dict):
            raise ParseError(self.id, f"unexpected payload type for {symbol}")

        chart = _require(payload.get("chart"), "'chart' object")
        # Yahoo reports symbol-level problems in-band, with HTTP 200.
        if error := chart.get("error"):
            description = error.get("description") if isinstance(error, dict) else error
            raise ParseError(self.id, f"{symbol}: upstream reported {description!r}")

        results = chart.get("result")
        if not results:
            raise ParseError(self.id, f"{symbol}: response carries no result")
        return results[0]

    @staticmethod
    def _session_dates(timestamps: list[int], gmtoffset: int) -> list[date]:
        """Epoch seconds -> the exchange-local calendar date of each session."""
        return [
            (datetime.fromtimestamp(ts, UTC) + timedelta(seconds=gmtoffset)).date()
            for ts in timestamps
        ]

    def _observations(self, series_id: str, unit: str, result: dict[str, Any]) -> list[Observation]:
        meta = result.get("meta") or {}
        timestamps = result.get("timestamp") or []
        indicators = _require(result.get("indicators"), "'indicators' object")
        quotes = indicators.get("quote")
        if not quotes:
            raise ParseError(self.id, f"{series_id}: indicators carry no quote block")
        closes = quotes[0].get("close")
        if closes is None:
            raise ParseError(self.id, f"{series_id}: quote block has no close series")

        if len(closes) != len(timestamps):
            raise ParseError(
                self.id,
                f"{series_id}: {len(timestamps)} timestamps but {len(closes)} closes; "
                "the arrays are positional and must agree",
            )

        dates = self._session_dates(timestamps, int(meta.get("gmtoffset") or 0))

        observations: list[Observation] = []
        for observation_date, close in zip(dates, closes, strict=True):
            # Yahoo pads holidays and halted sessions with null rather than
            # omitting them; a null close is an absent observation, not a zero.
            if close is None:
                continue
            try:
                value = float(close)
            except (TypeError, ValueError):
                continue
            observations.append(
                Observation(
                    series_id=series_id,
                    provider=self.id,
                    frequency="daily",
                    unit=unit,
                    observation_date=observation_date,
                    # A daily close is public the same session it prints.
                    release_date=observation_date,
                    revision_date=observation_date,
                    value=value,
                )
            )
        observations.sort(key=lambda o: o.observation_date)
        return observations

    def _require_daily(self, series_id: str, observations: list[Observation]) -> None:
        """Refuse a series the API quietly coarsened.

        Daily bars sit one calendar day apart, three across a weekend. The
        median gap is therefore 1 and a handful of holidays cannot move it.
        Anything larger means the response is not what ``interval=1d`` asked
        for, and admitting it would put quarterly closes into a series every
        consumer downstream reads as daily — a corruption no later check looks
        for, because nothing downstream re-examines spacing.
        """
        if len(observations) < MIN_BARS_FOR_DENSITY_CHECK:
            return
        gaps = sorted(
            (b.observation_date - a.observation_date).days
            for a, b in zip(observations, observations[1:], strict=False)
        )
        median = gaps[len(gaps) // 2]
        if median > MAX_DAILY_MEDIAN_GAP_DAYS:
            raise ParseError(
                self.id,
                f"{series_id}: asked for interval=1d but the bars are a median "
                f"{median} days apart across {len(observations)} of them — the API "
                "answered with a coarser interval than requested",
            )

    def fetch_metadata(self, series_id: str) -> SeriesMetadata:
        symbol = self._symbol(series_id)
        _, title, unit = _SYMBOLS[series_id]
        # One bar is enough to read the meta block, and cheap on a throttled API.
        result = self._result(symbol, {"range": "5d", "interval": "1d"})
        meta = result.get("meta") or {}

        first: date | None = None
        if (raw := meta.get("firstTradeDate")) is not None:
            with_offset = self._session_dates([int(raw)], int(meta.get("gmtoffset") or 0))
            first = with_offset[0]

        return SeriesMetadata(
            series_id=series_id,
            provider=self.id,
            title=str(meta.get("longName") or title),
            frequency="daily",
            # The catalogue's unit, not Yahoo's ``currency``. Yahoo reports
            # currency USD for ^GSPC, but an index level is not a price in
            # dollars — and taking it would disagree with the unit every
            # observation carries, which the quality engine rejects as a
            # unit_mismatch rather than guessing which side is right.
            unit=unit,
            first_observation=first,
            notes=(
                "Yahoo Finance chart API. No API key; undocumented and unversioned, "
                "so treat a schema change as expected rather than exceptional."
            ),
        )

    def fetch_observations(
        self,
        series_id: str,
        *,
        start: date | None = None,
        end: date | None = None,
    ) -> list[Observation]:
        symbol = self._symbol(series_id)
        _, _, unit = _SYMBOLS[series_id]

        # Always an explicit window, never ``range=max``. Asked for max with a
        # daily interval, the live API answered with 168 bars starting in 1984 —
        # roughly quarterly, for a series it holds daily from 1927. It does not
        # report the downgrade; it just returns coarser data under the interval
        # you asked for. An explicit period1/period2 is honoured. Verified
        # 2026-07-31, and the density check below is what would catch it
        # recurring rather than letting quarterly bars enter a daily series.
        #
        # Bounds are widened a day at each end before clipping below: the window
        # is in UTC but the sessions are on the exchange's calendar, and a
        # boundary session must not fall out of the request that asked for it.
        params: dict[str, str] = {"interval": "1d"}
        params["period1"] = (
            str(
                int(
                    datetime.combine(
                        start - timedelta(days=1), datetime.min.time(), UTC
                    ).timestamp()
                )
            )
            if start
            else str(HISTORY_FLOOR_EPOCH)
        )
        params["period2"] = str(
            int(datetime.combine(end + timedelta(days=1), datetime.min.time(), UTC).timestamp())
            if end
            else int(datetime.now(UTC).timestamp()) + 86400
        )

        observations = self._observations(series_id, unit, self._result(symbol, params))
        self._require_daily(series_id, observations)

        # The API honours the window loosely; enforce it so a caller's cutoff
        # means the same thing here as it does for every other provider.
        if start is not None:
            observations = [o for o in observations if o.observation_date >= start]
        if end is not None:
            observations = [o for o in observations if o.observation_date <= end]
        return observations
