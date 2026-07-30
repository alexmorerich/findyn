"""Stooq daily price CSV.

Intended as the keyless bulk source for index and ETF history
(FINDYN_V1_SPEC.md §5.1 source 6, backfill role).

Operational caveat, verified 2026-07-30: stooq.com now fronts the CSV endpoint
with a JavaScript proof-of-work interstitial. A plain HTTP client — including
one sending a browser User-Agent — receives that HTML challenge with HTTP 200
instead of the CSV. The adapter therefore detects the challenge explicitly and
raises :class:`BotChallengeError`, which is non-retryable so the circuit opens
immediately and the caller falls back rather than spending its retry budget on
a page that will never contain data.

The adapter is kept complete and tested against fixtures because the challenge
is applied per-network and may not be present from every egress; where it is
present, Shiller supplies monthly index history without a key.
"""

from __future__ import annotations

import csv
import logging
from datetime import date, datetime
from io import StringIO

from findynamics.data.providers.base import (
    BotChallengeError,
    Observation,
    ParseError,
    Provider,
    SeriesMetadata,
)
from findynamics.data.providers.resilience import Transport

log = logging.getLogger("findynamics.data.providers.stooq")

BASE_URL = "https://stooq.com/q/d/l/"

EXPECTED_HEADER = ["Date", "Open", "High", "Low", "Close", "Volume"]

#: symbol -> (series_id, human title, unit)
_SYMBOLS: dict[str, tuple[str, str, str]] = {
    "^spx": ("STOOQ:^SPX", "S&P 500 Index (daily close)", "index"),
    "^ndx": ("STOOQ:^NDX", "Nasdaq 100 Index (daily close)", "index"),
    "^dji": ("STOOQ:^DJI", "Dow Jones Industrial Average (daily close)", "index"),
    "spy.us": ("STOOQ:SPY.US", "SPDR S&P 500 ETF Trust (daily close)", "usd"),
    "voo.us": ("STOOQ:VOO.US", "Vanguard S&P 500 ETF (daily close)", "usd"),
    "tlt.us": ("STOOQ:TLT.US", "iShares 20+ Year Treasury Bond ETF (daily close)", "usd"),
    "gld.us": ("STOOQ:GLD.US", "SPDR Gold Shares (daily close)", "usd"),
}

_BY_SERIES_ID = {series_id: symbol for symbol, (series_id, _, _) in _SYMBOLS.items()}


def looks_like_bot_challenge(payload: str) -> bool:
    """True when the body is an interstitial rather than CSV.

    Checked on content, not status code: the challenge is served with HTTP 200,
    so status-based handling would treat it as a successful fetch and the parser
    would report a confusing 'missing Date column' instead of the real cause.
    """
    head = payload.lstrip()[:2048].lower()
    if head.startswith("date,"):
        return False
    return (
        "<!doctype html" in head
        or "<html" in head
        or "requires javascript" in head
        or "crypto.subtle.digest" in head
    )


class StooqProvider(Provider):
    id = "stooq"
    requires_api_key = False

    def __init__(self, transport: Transport, *, base_url: str = BASE_URL) -> None:
        self.transport = transport
        self.base_url = base_url

    def available_series(self) -> list[str]:
        return sorted(_BY_SERIES_ID)

    def _symbol(self, series_id: str) -> str:
        self._require_known_series(series_id)
        return _BY_SERIES_ID[series_id]

    def _download(self, symbol: str) -> str:
        response = self.transport.get(self.base_url, params={"s": symbol, "i": "d"}, cache_ttl=3600)
        text = response.text
        if looks_like_bot_challenge(text):
            raise BotChallengeError(
                self.id,
                "endpoint returned a JavaScript proof-of-work challenge instead of CSV; "
                "this egress is being bot-filtered",
            )
        return text

    def _parse(self, series_id: str, unit: str, payload: str) -> list[Observation]:
        reader = csv.reader(StringIO(payload))
        try:
            header = next(reader)
        except StopIteration:
            raise ParseError(self.id, "empty CSV response") from None

        if header[: len(EXPECTED_HEADER)] != EXPECTED_HEADER:
            raise ParseError(self.id, f"unexpected CSV header: {header}")

        close_idx = header.index("Close")
        observations: list[Observation] = []
        for row in reader:
            if len(row) <= close_idx or not row[0]:
                continue
            try:
                obs_date = datetime.strptime(row[0], "%Y-%m-%d").date()
                value = float(row[close_idx])
            except ValueError:
                # Stooq pads missing sessions with 'N/D'; skip rather than abort.
                continue
            observations.append(
                Observation(
                    series_id=series_id,
                    provider=self.id,
                    frequency="daily",
                    unit=unit,
                    observation_date=obs_date,
                    # A daily close is public the same session it prints.
                    release_date=obs_date,
                    revision_date=obs_date,
                    value=value,
                )
            )
        observations.sort(key=lambda o: o.observation_date)
        return observations

    def fetch_metadata(self, series_id: str) -> SeriesMetadata:
        symbol = self._symbol(series_id)
        _, title, unit = _SYMBOLS[symbol]
        return SeriesMetadata(
            series_id=series_id,
            provider=self.id,
            title=title,
            frequency="daily",
            unit=unit,
            notes="stooq.com daily CSV. No API key required; subject to bot filtering.",
        )

    def fetch_observations(
        self,
        series_id: str,
        *,
        start: date | None = None,
        end: date | None = None,
    ) -> list[Observation]:
        symbol = self._symbol(series_id)
        _, _, unit = _SYMBOLS[symbol]
        observations = self._parse(series_id, unit, self._download(symbol))
        if start is not None:
            observations = [o for o in observations if o.observation_date >= start]
        if end is not None:
            observations = [o for o in observations if o.observation_date <= end]
        return observations
