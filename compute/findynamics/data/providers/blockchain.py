"""Bitcoin on-chain metrics from blockchain.info's charts API.

The keyless half of FinCrypto's on-chain panel (P5). Every metric here is a
*network* measurement — how many transactions confirmed, how much value moved,
how much hardware is pointed at the chain — rather than a price or a derivative
of one. That is the whole reason the crypto engine wants them: the price is
already in the price, and the only inputs that are not are the ones describing
the network the price is supposed to be about.

Why this source and not one of the analytics vendors
----------------------------------------------------

blockchain.info needs no API key, publishes the full history back to 2009, and
serves a stable two-field JSON. Glassnode and Coin Metrics have the richer
metrics — MVRV, SOPR, realized cap, coin-days-destroyed — and every one of them
is behind a paid key. Those ids are declared in ``series.yaml`` with their lags
so the config is a complete statement of what the model wants, and they are
**not** given a provider that pretends to fetch them: see the TODO in
``data/providers/registry.py``. An engine that reads them degrades and says so,
which is the same contract gold has with ``ENGINE:equity.rii``.

Response shape
--------------

``GET /charts/<chart>?format=json&timespan=all`` returns::

    {"status": "ok", "name": ..., "unit": ..., "period": "day",
     "values": [{"x": 1231459200, "y": 1.0}, ...]}

``x`` is epoch seconds at UTC midnight; ``y`` is the value. Verified live
2026-08-05.

Dates and release dates
-----------------------

A daily chart point covers a UTC day and is only complete once that day has
ended, so it is not public until the following day: ``release_date`` is
``observation_date + 1``. This is one of the few places a lag is asserted by the
adapter rather than left to the configured fallback, because the reason is
structural (a day's transaction count cannot be known mid-day) rather than a
publisher's schedule.

Bitcoin has no exchange calendar. These series carry an observation every
calendar day, weekends included, which is why everything downstream annualizes
on 365.
"""

from __future__ import annotations

import logging
from datetime import UTC, date, datetime, timedelta

from findynamics.data.providers.base import (
    Observation,
    ParseError,
    Provider,
    SeriesMetadata,
)
from findynamics.data.providers.resilience import Transport

log = logging.getLogger("findynamics.data.providers.blockchain")

BASE_URL = "https://api.blockchain.info/charts"

#: series_id -> (chart slug, human title, unit)
#:
#: Deliberately small. Each entry answers a question the engine actually asks;
#: blockchain.info publishes about forty charts and importing all of them would
#: be building a data catalogue rather than an engine.
_CHARTS: dict[str, tuple[str, str, str]] = {
    # The deep price history, and the only keyless route to it. Stooq is
    # bot-filtered and Yahoo's BTC-USD starts 2014-09-17, which cuts the 2011 and
    # 2013 cycles out of the sample — half the cycles bitcoin has had.
    #
    # This is a **volume-weighted daily average across exchanges**, not a close.
    # That difference is not cosmetic and is handled explicitly in
    # engines/crypto/prices.py rather than papered over here: the two statistics
    # agree on the level (mean gap -0.18% over 4,341 shared days, no step at the
    # seam, volatility ratio 1.016) and disagree hard on high-range days
    # (2020-03-12: a 4,971 close against a 7,937 daily average). The splice is
    # validated on the properties that matter and the per-date provenance is
    # published, so nothing downstream has to guess which statistic it is reading.
    "BLOCKCHAIN:MARKET_PRICE": (
        "market-price",
        "Bitcoin market price, volume-weighted daily average across exchanges (USD)",
        "usd",
    ),
    # The volume leg of the speculation index. USD rather than BTC on purpose:
    # a constant BTC volume at ten times the price is ten times the speculation,
    # and the BTC-denominated series cannot say so.
    "BLOCKCHAIN:TX_VOLUME_USD": (
        "estimated-transaction-volume-usd",
        "Estimated on-chain transaction volume (USD)",
        "usd",
    ),
    # Network usage, independent of price entirely.
    "BLOCKCHAIN:N_TRANSACTIONS": (
        "n-transactions",
        "Confirmed transactions per day",
        "count",
    ),
    "BLOCKCHAIN:N_UNIQUE_ADDRESSES": (
        "n-unique-addresses",
        "Unique addresses used per day",
        "count",
    ),
    # Security spend — the one on-chain series with a real cost floor behind it.
    "BLOCKCHAIN:HASH_RATE": (
        "hash-rate",
        "Estimated network hash rate (TH/s)",
        "hashes_th_per_s",
    ),
}

#: Series whose non-positive values mean "no observation", not "a value of zero".
#:
#: ``market-price`` is padded with 0.0 from the genesis block on 2009-01-03 to
#: 2010-08-17, which is the API saying there was no market yet rather than that
#: bitcoin was worth nothing. Ingesting those as observations would put 588 zeros
#: into a price series, and the first log return out of that window would be
#: infinite. The network-activity charts are not filtered: a day with genuinely
#: zero transactions is a fact about the chain.
_POSITIVE_ONLY: frozenset[str] = frozenset({"BLOCKCHAIN:MARKET_PRICE"})


class BlockchainProvider(Provider):
    id = "blockchain"
    requires_api_key = False

    def __init__(self, transport: Transport, *, base_url: str = BASE_URL) -> None:
        self.transport = transport
        self.base_url = base_url.rstrip("/")

    def available_series(self) -> list[str]:
        return sorted(_CHARTS)

    def _chart(self, series_id: str) -> str:
        self._require_known_series(series_id)
        return _CHARTS[series_id][0]

    def _values(self, chart: str) -> list[dict[str, float]]:
        response = self.transport.get(
            f"{self.base_url}/{chart}",
            params={"format": "json", "timespan": "all", "sampled": "false"},
            cache_ttl=3600,
        )
        try:
            payload = response.json()
        except ValueError as err:
            raise ParseError(self.id, f"non-JSON response for {chart}: {err}") from err
        if not isinstance(payload, dict):
            raise ParseError(self.id, f"unexpected payload type for {chart}")

        # The API reports chart-level problems in-band with HTTP 200.
        status = payload.get("status")
        if status is not None and status != "ok":
            raise ParseError(self.id, f"{chart}: upstream reported status {status!r}")

        # A daily chart that came back at another cadence is not the series the
        # caller asked for, and nothing downstream re-examines spacing.
        period = payload.get("period")
        if period is not None and period != "day":
            raise ParseError(
                self.id,
                f"{chart}: asked for a daily chart but the response says period={period!r}",
            )

        values = payload.get("values")
        if not isinstance(values, list):
            raise ParseError(self.id, f"{chart}: response carries no 'values' array")
        return values

    def fetch_metadata(self, series_id: str) -> SeriesMetadata:
        self._require_known_series(series_id)
        _, title, unit = _CHARTS[series_id]
        return SeriesMetadata(
            series_id=series_id,
            provider=self.id,
            title=title,
            frequency="daily",
            unit=unit,
            notes=(
                "blockchain.info charts API. No API key. Daily UTC network metrics; "
                "a day's value is only complete once that day has ended, so the "
                "release date is the following day."
            ),
        )

    def fetch_observations(
        self,
        series_id: str,
        *,
        start: date | None = None,
        end: date | None = None,
    ) -> list[Observation]:
        chart = self._chart(series_id)
        _, _, unit = _CHARTS[series_id]

        observations: list[Observation] = []
        for point in self._values(chart):
            if not isinstance(point, dict):
                continue
            raw_x, raw_y = point.get("x"), point.get("y")
            if raw_x is None or raw_y is None:
                continue
            try:
                observation_date = datetime.fromtimestamp(int(raw_x), UTC).date()
                value = float(raw_y)
            except (TypeError, ValueError, OSError, OverflowError):
                # One malformed point is not a reason to discard a 6,000-day
                # history; a systematically malformed payload fails the shape
                # checks above instead.
                continue
            if series_id in _POSITIVE_ONLY and value <= 0.0:
                continue
            if start is not None and observation_date < start:
                continue
            if end is not None and observation_date > end:
                continue
            observations.append(
                Observation(
                    series_id=series_id,
                    provider=self.id,
                    frequency="daily",
                    unit=unit,
                    observation_date=observation_date,
                    release_date=observation_date + timedelta(days=1),
                    revision_date=observation_date + timedelta(days=1),
                    value=value,
                )
            )

        observations.sort(key=lambda o: o.observation_date)
        log.debug("blockchain: %d observations for %s", len(observations), series_id)
        return observations


__all__ = ["BASE_URL", "BlockchainProvider"]
