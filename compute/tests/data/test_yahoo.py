"""Yahoo chart-API adapter.

The endpoint is undocumented and unversioned, so these tests pin the shape the
adapter was written against. If the live API drifts, the fixture below is the
statement of what it used to return, and the mismatch is the diff to read.

Routed through a scripted fetcher — no network.
"""

from __future__ import annotations

import json
from datetime import date

import pytest

from findynamics.data.providers.base import NotFoundError, ParseError
from findynamics.data.providers.resilience import (
    CircuitBreaker,
    MemoryCache,
    RateLimiter,
    RetryPolicy,
    Transport,
)
from findynamics.data.providers.yahoo import HISTORY_FLOOR_EPOCH, YahooProvider
from tests.conftest import ok

#: New York is UTC-4 in summer. Each timestamp is a session *open* (09:30 local
#: = 13:30 UTC), which is the detail that makes naive UTC conversion look
#: correct for US equities and quietly wrong elsewhere.
NY_OFFSET = -14400

#: 2025-07-28 09:30 and 2025-07-29 09:30, New York, as epoch seconds.
T1 = 1753709400
T2 = 1753795800


def chart(
    *,
    timestamps: list[int] | None = None,
    closes: list[float | None] | None = None,
    gmtoffset: int = NY_OFFSET,
    error: object = None,
    meta_extra: dict | None = None,
) -> str:
    result = {
        "meta": {
            "currency": "USD",
            "symbol": "^GSPC",
            "instrumentType": "INDEX",
            "firstTradeDate": -1325583000,
            "gmtoffset": gmtoffset,
            "exchangeTimezoneName": "America/New_York",
            "longName": "S&P 500",
            **(meta_extra or {}),
        },
        "timestamp": timestamps if timestamps is not None else [T1, T2],
        "indicators": {
            "quote": [{"close": closes if closes is not None else [6300.5, 6350.25]}],
        },
    }
    return json.dumps({"chart": {"result": [result], "error": error}})


def build(body: str | list[str]) -> YahooProvider:
    bodies = [body] if isinstance(body, str) else list(body)

    def fetcher(url, *, params=None, headers=None, timeout=30.0):
        return ok(bodies.pop(0) if len(bodies) > 1 else bodies[0])

    transport = Transport(
        "yahoo",
        fetcher,
        rate_limiter=RateLimiter(capacity=1000, refill_per_second=1000.0, min_interval=0.0),
        breaker=CircuitBreaker("yahoo", failure_threshold=100, cooldown=1.0),
        retry=RetryPolicy(max_attempts=1),
        cache=MemoryCache(),
    )
    return YahooProvider(transport)


class TestParsing:
    def test_daily_closes_are_read(self):
        observations = build(chart()).fetch_observations("YAHOO:^GSPC")
        assert [o.value for o in observations] == [6300.5, 6350.25]

    def test_a_close_is_knowable_the_session_it_prints(self):
        first = build(chart()).fetch_observations("YAHOO:^GSPC")[0]
        assert first.release_date == first.observation_date

    def test_null_closes_are_skipped_not_zeroed(self):
        """Yahoo pads holidays with null; a null is an absent bar, not a 0.0."""
        observations = build(chart(closes=[6300.5, None])).fetch_observations("YAHOO:^GSPC")
        assert [o.value for o in observations] == [6300.5]

    def test_observations_come_back_ascending(self):
        observations = build(chart()).fetch_observations("YAHOO:^GSPC")
        assert observations == sorted(observations, key=lambda o: o.observation_date)


class TestSessionDates:
    """The bar is a session on the exchange's calendar, not a UTC instant."""

    def test_a_us_session_lands_on_its_own_calendar_day(self):
        observations = build(chart()).fetch_observations("YAHOO:^GSPC")
        assert [o.observation_date for o in observations] == [
            date(2025, 7, 28),
            date(2025, 7, 29),
        ]

    def test_an_open_before_midnight_utc_does_not_roll_forward(self):
        """REGRESSION — reading the timestamp as UTC breaks east of Greenwich.

        Tokyo opens 09:00 JST, which is 00:00 UTC the *same* date; an exchange
        an hour further east opens the UTC day before. Converting on the
        exchange's offset is what keeps the session on its own trading day.
        """
        tokyo_open = 1753747200  # 2025-07-29 09:00 JST == 2025-07-29 00:00 UTC
        observations = build(
            chart(timestamps=[tokyo_open], closes=[40000.0], gmtoffset=32400)
        ).fetch_observations("YAHOO:^GSPC")
        assert observations[0].observation_date == date(2025, 7, 29)

    def test_a_missing_offset_does_not_crash(self):
        observations = build(chart(meta_extra={"gmtoffset": None})).fetch_observations(
            "YAHOO:^GSPC"
        )
        assert len(observations) == 2


class TestMalformedResponses:
    """Unversioned API: every structural surprise names what was missing."""

    def test_an_in_band_error_is_surfaced(self):
        """Yahoo reports symbol-level failures with HTTP 200 and error set."""
        body = chart(error={"code": "Not Found", "description": "No data found for symbol"})
        with pytest.raises(ParseError, match="No data found"):
            build(body).fetch_observations("YAHOO:^GSPC")

    def test_an_empty_result_list_is_reported(self):
        body = json.dumps({"chart": {"result": [], "error": None}})
        with pytest.raises(ParseError, match="no result"):
            build(body).fetch_observations("YAHOO:^GSPC")

    def test_a_missing_chart_object_is_reported(self):
        with pytest.raises(ParseError, match="'chart' object"):
            build(json.dumps({"finance": {}})).fetch_observations("YAHOO:^GSPC")

    def test_a_missing_close_series_is_reported(self):
        body = json.dumps(
            {
                "chart": {
                    "result": [
                        {"meta": {}, "timestamp": [T1], "indicators": {"quote": [{"open": [1.0]}]}}
                    ],
                    "error": None,
                }
            }
        )
        with pytest.raises(ParseError, match="no close series"):
            build(body).fetch_observations("YAHOO:^GSPC")

    def test_misaligned_arrays_are_refused_rather_than_zipped_short(self):
        """The arrays are positional. Silently truncating would pair a close
        with the wrong session — a date error that no later check would catch."""
        with pytest.raises(ParseError, match="must agree"):
            build(chart(timestamps=[T1, T2], closes=[6300.5])).fetch_observations("YAHOO:^GSPC")

    def test_non_json_is_reported_as_such(self):
        with pytest.raises(ParseError, match="non-JSON"):
            build("Too Many Requests").fetch_observations("YAHOO:^GSPC")


class TestMetadataAndCatalogue:
    def test_metadata_reads_the_first_trade_date(self):
        metadata = build(chart()).fetch_metadata("YAHOO:^GSPC")
        assert metadata.first_observation == date(1927, 12, 30)
        assert metadata.frequency == "daily"

    def test_metadata_and_observations_agree_on_the_unit(self):
        """REGRESSION — an index level is not a price in dollars.

        Yahoo reports currency USD for ^GSPC. Taking that for the metadata
        while every observation carries 'index' is a real disagreement, and
        the quality engine withholds the whole series over it rather than
        guessing which side is right. Caught on the first live fetch.
        """
        provider = build(chart())
        metadata = provider.fetch_metadata("YAHOO:^GSPC")
        observations = provider.fetch_observations("YAHOO:^GSPC")
        assert metadata.unit == "index"
        assert {o.unit for o in observations} == {metadata.unit}

    def test_metadata_says_the_source_is_unversioned(self):
        """A consumer reading this series should know what it is resting on."""
        assert "unversioned" in (build(chart()).fetch_metadata("YAHOO:^GSPC").notes or "")

    def test_an_unknown_series_is_refused(self):
        with pytest.raises(NotFoundError):
            build(chart()).fetch_observations("YAHOO:^NOPE")

    def test_the_catalogue_is_explicit(self):
        assert "YAHOO:^GSPC" in build(chart()).available_series()


class TestWindowing:
    def test_a_range_filter_is_enforced_locally(self):
        """The API honours a window loosely; a caller's cutoff must not."""
        observations = build(chart()).fetch_observations("YAHOO:^GSPC", start=date(2025, 7, 29))
        assert [o.observation_date for o in observations] == [date(2025, 7, 29)]

    def _params_used(self, body: str, **kwargs) -> dict:
        seen: dict = {}

        def fetcher(url, *, params=None, headers=None, timeout=30.0):
            seen.update(params or {})
            return ok(body)

        transport = Transport(
            "yahoo",
            fetcher,
            rate_limiter=RateLimiter(capacity=1000, refill_per_second=1000.0),
            breaker=CircuitBreaker("yahoo", failure_threshold=100, cooldown=1.0),
            retry=RetryPolicy(max_attempts=1),
            cache=MemoryCache(),
        )
        YahooProvider(transport).fetch_observations("YAHOO:^GSPC", **kwargs)
        return seen

    def test_an_unbounded_fetch_asks_for_an_explicit_window_not_max(self):
        """REGRESSION — ``range=max`` is answered with coarser bars.

        Asked for max at interval=1d, the live API returned 168 bars starting
        in 1984 for a series it holds daily from 1927: roughly quarterly, with
        no indication it had substituted anything. An explicit period1/period2
        is honoured, so that is what is always sent.
        """
        seen = self._params_used(chart())
        assert "range" not in seen
        assert int(seen["period1"]) == HISTORY_FLOOR_EPOCH
        assert seen["interval"] == "1d"

    def test_the_floor_reaches_back_past_the_first_trade(self):
        """^GSPC starts 1927; the floor must not clip it."""
        assert HISTORY_FLOOR_EPOCH < -1325583000


class TestIntervalDowngradeIsRefused:
    """The API can answer interval=1d with coarser bars and not say so."""

    def _spaced(self, n: int, step_days: int) -> str:
        day = 86400
        return chart(
            timestamps=[T1 + i * step_days * day for i in range(n)],
            closes=[100.0 + i for i in range(n)],
        )

    def test_daily_bars_pass(self):
        observations = build(self._spaced(60, 1)).fetch_observations("YAHOO:^GSPC")
        assert len(observations) == 60

    def test_quarterly_bars_are_refused(self):
        """The exact shape the live API returned for range=max."""
        with pytest.raises(ParseError, match="coarser interval"):
            build(self._spaced(168, 91)).fetch_observations("YAHOO:^GSPC")

    def test_weekly_bars_are_refused(self):
        with pytest.raises(ParseError, match="median 7 days"):
            build(self._spaced(60, 7)).fetch_observations("YAHOO:^GSPC")

    def test_a_short_series_is_not_judged_on_its_spacing(self):
        """Too few bars to tell daily from weekly; refusing would be a guess."""
        assert len(build(self._spaced(5, 7)).fetch_observations("YAHOO:^GSPC")) == 5

    def test_a_holiday_heavy_stretch_still_passes(self):
        """Weekends and holidays must not trip it: the median stays 1."""
        day = 86400
        gaps = [1, 1, 3, 1, 4, 1, 1, 3, 1, 1] * 6  # weekends + a long weekend
        stamps, t = [T1], T1
        for g in gaps:
            t += g * day
            stamps.append(t)
        body = chart(timestamps=stamps, closes=[100.0 + i for i in range(len(stamps))])
        assert len(build(body).fetch_observations("YAHOO:^GSPC")) == len(stamps)
