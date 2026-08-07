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
from pathlib import Path

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

#: Columns every Stooq daily CSV carries. ``Volume`` follows on instrument
#: exports but is absent on index exports (^SPX has no volume to report), so it
#: is not required — the parser reads ``Date`` and ``Close`` and nothing else.
REQUIRED_HEADER = ["Date", "Open", "High", "Low", "Close"]

#: symbol -> (series_id, human title, unit)
_SYMBOLS: dict[str, tuple[str, str, str]] = {
    "^spx": ("STOOQ:^SPX", "S&P 500 Index (daily close)", "index"),
    "^ndx": ("STOOQ:^NDX", "Nasdaq 100 Index (daily close)", "index"),
    "^dji": ("STOOQ:^DJI", "Dow Jones Industrial Average (daily close)", "index"),
    "spy.us": ("STOOQ:SPY.US", "SPDR S&P 500 ETF Trust (daily close)", "usd"),
    "voo.us": ("STOOQ:VOO.US", "Vanguard S&P 500 ETF (daily close)", "usd"),
    "tlt.us": ("STOOQ:TLT.US", "iShares 20+ Year Treasury Bond ETF (daily close)", "usd"),
    "gld.us": ("STOOQ:GLD.US", "SPDR Gold Shares (daily close)", "usd"),
    # P5. A currency pair rather than an instrument, so the export carries no
    # Volume column — which costs nothing, since the parser reads Date and Close
    # and nothing else. Bitcoin has no exchange calendar: the series has an
    # observation every calendar day including weekends, and everything
    # downstream annualizes on 365 rather than 252 because of it.
    "btcusd": ("STOOQ:BTCUSD", "Bitcoin / US Dollar (daily close)", "usd"),
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

        if header[: len(REQUIRED_HEADER)] != REQUIRED_HEADER:
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


class StooqFileProvider(StooqProvider):
    """The same CSV, read from disk instead of fetched.

    The proof-of-work challenge above blocks every automated egress available to
    this project — a developer network and GitHub's runners both, verified. A
    browser passes it, because passing it is what a browser is for, so the one
    remaining route to Stooq's daily history is a person downloading the file
    and pointing this at it.

    That is a reasonable trade for this particular series: daily index history
    before 2016 is fixed, so it is fetched once and never again. It is a bad
    trade for anything that needs refreshing, which is why this is a deliberate
    ``--from-file`` argument rather than a fallback the daily job could reach.

    Everything downstream is unchanged — same parser, same release-date
    convention, same quality checks — so a file-sourced backfill and a network
    one produce identical rows.
    """

    def __init__(self, path: Path) -> None:
        # Deliberately no transport: there is nothing to rate-limit or retry,
        # and constructing one would imply this can reach the network.
        self.path = path

    def _download(self, symbol: str) -> str:
        try:
            text = self.path.read_text(encoding="utf-8", errors="replace")
        except OSError as err:
            raise ParseError(self.id, f"cannot read {self.path}: {err}") from err

        if looks_like_bot_challenge(text):
            raise BotChallengeError(
                self.id,
                f"{self.path} contains the bot-challenge page, not CSV — the browser "
                "saved the interstitial. Open the download URL in a real browser tab, "
                "let it finish, and save the file it offers.",
            )
        return text

    def fetch_metadata(self, series_id: str) -> SeriesMetadata:
        metadata = super().fetch_metadata(series_id)
        return SeriesMetadata(
            **{**metadata.__dict__, "notes": f"stooq.com daily CSV, ingested from {self.path.name}"}
        )
