"""REGRESSION — FRED's vintage-window cap.

FRED rejects an observations request whose real-time window spans more than
2000 vintage dates. DGS10 has over 5000, so the single unsplit request the
provider used to make returned HTTP 400 and the rates engine could not be fed
at all. Every test here fails if the window stops being split.

Routed through a scripted fetcher — no network, no API key.
"""

from __future__ import annotations

import json
from datetime import date, timedelta

import pytest

from findynamics.data.providers.base import ProviderError
from findynamics.data.providers.fred import (
    MAX_VINTAGE_DATES,
    REALTIME_END,
    REALTIME_START,
    FredProvider,
    _dedupe_vintages,
)
from findynamics.data.providers.resilience import (
    CircuitBreaker,
    MemoryCache,
    RateLimiter,
    RetryPolicy,
    Transport,
)
from tests.conftest import ok

METADATA = {
    "seriess": [
        {
            "id": "DGS10",
            "title": "10-Year Treasury Constant Maturity Rate",
            "frequency_short": "D",
            "units_short": "%",
            "observation_start": "1962-01-02",
            "observation_end": "2026-07-28",
        }
    ]
}


def daily_vintages(n: int) -> list[date]:
    """``n`` distinct vintage dates, one per day from 2005.

    ``date`` objects, matching what ``vintage_dates()`` returns —
    ``_vintage_windows`` consumes that output directly.
    """
    base = date(2005, 1, 3)
    return [base + timedelta(days=i) for i in range(n)]


def as_iso(vintages: list[date]) -> list[str]:
    """The wire form the vintagedates endpoint returns."""
    return [v.isoformat() for v in vintages]


class ScriptedFred:
    """Answers FRED's three endpoints from in-memory data.

    Enforces the real cap: a window spanning more than MAX_VINTAGE_DATES
    vintages gets the same 400 the live API returns.
    """

    def __init__(
        self,
        vintages: list[date],
        observations: list[dict] | None = None,
        *,
        enforce_cap: bool = True,
    ) -> None:
        self.vintages = as_iso(vintages)
        self.observations = observations or []
        self.enforce_cap = enforce_cap
        self.windows: list[tuple[str, str]] = []

    def __call__(self, url, *, params=None, headers=None, timeout=30.0):
        params = params or {}
        if url.endswith("/series"):
            return ok(json.dumps(METADATA))

        if url.endswith("/series/vintagedates"):
            offset = int(params.get("offset", 0))
            limit = int(params.get("limit", 10_000))
            return ok(json.dumps({"vintage_dates": self.vintages[offset : offset + limit]}))

        start = str(params.get("realtime_start", REALTIME_START))
        end = str(params.get("realtime_end", REALTIME_END))
        self.windows.append((start, end))

        covered = [v for v in self.vintages if start <= v <= end]
        if self.enforce_cap and len(covered) > MAX_VINTAGE_DATES:
            return ok(
                json.dumps(
                    {
                        "error_code": 400,
                        "error_message": (
                            f"Bad Request.  There are {len(covered)} vintage dates in the "
                            f"specified real-time period: {start} to {end}.  This exceeds "
                            f"the maximum number of vintage dates allowed for this file "
                            f"type ({MAX_VINTAGE_DATES})."
                        ),
                    }
                ),
                status=400,
            )

        # FRED clamps each row's realtime_start to the requested window start.
        rows = [
            {**row, "realtime_start": max(row["realtime_start"], start)}
            for row in self.observations
            if row["realtime_start"] <= end
        ]
        return ok(json.dumps({"observations": rows}))


def build(fetcher) -> FredProvider:
    transport = Transport(
        "fred",
        fetcher,
        rate_limiter=RateLimiter(capacity=1000, refill_per_second=1000.0, min_interval=0.0),
        breaker=CircuitBreaker("fred", failure_threshold=100, cooldown=1.0),
        retry=RetryPolicy(max_attempts=1),
        cache=MemoryCache(),
    )
    return FredProvider(transport, api_key="test-key")


def observations_for(vintages: list[date]) -> list[dict]:
    """One period per vintage, each first issued on its own vintage date."""
    return [
        {
            "date": f"2005-01-{(i % 28) + 1:02d}",
            "realtime_start": v.isoformat(),
            "value": str(4.0 + i * 0.001),
        }
        for i, v in enumerate(vintages)
    ]


class TestVintageWindows:
    def test_a_series_inside_the_cap_is_one_request(self):
        assert FredProvider._vintage_windows(daily_vintages(500)) == [
            (REALTIME_START, REALTIME_END)
        ]

    def test_a_series_over_the_cap_is_split(self):
        """DGS10's real shape: over 5000 vintages against a cap of 2000."""
        assert len(FredProvider._vintage_windows(daily_vintages(5076))) == 3

    def test_no_window_exceeds_the_cap(self):
        vintages = daily_vintages(5076)
        iso = as_iso(vintages)
        for start, end in FredProvider._vintage_windows(vintages):
            assert len([v for v in iso if start <= v <= end]) <= MAX_VINTAGE_DATES

    def test_the_windows_stay_open_ended_at_both_extremes(self):
        """Closing them would drop the oldest and newest vintages."""
        windows = FredProvider._vintage_windows(daily_vintages(5076))
        assert windows[0][0] == REALTIME_START
        assert windows[-1][1] == REALTIME_END

    def test_windows_leave_no_vintage_uncovered(self):
        vintages = daily_vintages(5076)
        windows = FredProvider._vintage_windows(vintages)
        for vintage in as_iso(vintages):
            assert any(start <= vintage <= end for start, end in windows), vintage

    def test_no_vintages_at_all_still_yields_one_window(self):
        assert FredProvider._vintage_windows([]) == [(REALTIME_START, REALTIME_END)]


class TestFetchAgainstTheCap:
    def test_a_series_over_the_cap_is_fetched_successfully(self):
        vintages = daily_vintages(5076)
        scripted = ScriptedFred(vintages, observations_for(vintages))

        observations = build(scripted).fetch_observations("FRED:DGS10")

        assert observations
        # Three windows, so three observation requests — not one that 400s.
        assert len(scripted.windows) == 3

    def test_the_unsplit_request_really_would_have_failed(self):
        """Guards the test itself: without splitting, the scripted API 400s.

        If this ever stops raising, the fake has drifted from the real API and
        the tests above prove nothing.
        """
        vintages = daily_vintages(5076)
        provider = build(ScriptedFred(vintages, observations_for(vintages)))

        with pytest.raises(ProviderError, match="400"):
            provider._get(
                "series/observations",
                {
                    "series_id": "DGS10",
                    "realtime_start": REALTIME_START,
                    "realtime_end": REALTIME_END,
                },
            )

    def test_a_series_inside_the_cap_makes_a_single_request(self):
        vintages = daily_vintages(500)
        scripted = ScriptedFred(vintages, observations_for(vintages))

        build(scripted).fetch_observations("FRED:DGS10")

        assert len(scripted.windows) == 1

    def test_release_date_is_the_earliest_issue_across_all_windows(self):
        """Splitting must not move a release date; that would be lookahead."""
        vintages = [date(2005, 1, 3), date(2010, 6, 1), date(2020, 7, 21)]
        observations = [
            {"date": "2004-12-31", "realtime_start": "2005-01-03", "value": "4.0"},
            {"date": "2004-12-31", "realtime_start": "2010-06-01", "value": "4.1"},
        ]

        rows = build(ScriptedFred(vintages, observations, enforce_cap=False)).fetch_observations(
            "FRED:DGS10"
        )

        assert {r.release_date for r in rows} == {date(2005, 1, 3)}

    def test_vintage_dates_are_returned_in_order(self):
        vintages = daily_vintages(500)
        assert build(ScriptedFred(vintages)).vintage_dates("FRED:DGS10") == vintages

    def test_vintage_dates_page_beyond_the_first_request(self):
        """A series with more vintages than one page still enumerates fully."""
        vintages = daily_vintages(12_000)
        assert build(ScriptedFred(vintages)).vintage_dates("FRED:DGS10") == vintages

    def test_vintages_are_skipped_entirely_when_disabled(self):
        vintages = daily_vintages(5076)
        scripted = ScriptedFred(vintages, observations_for(vintages), enforce_cap=False)
        provider = build(scripted)
        provider.use_vintages = False

        provider.fetch_observations("FRED:DGS10")

        assert scripted.windows == [(REALTIME_START, REALTIME_END)]


class TestDedupeVintages:
    """Splitting clamps each chunk's realtime_start to its boundary, so an
    unchanged figure reappears looking like a fresh issue. Left in, the vintage
    count would depend on how the fetch happened to be split."""

    def test_a_reissue_of_the_same_value_is_dropped(self):
        rows = [
            (date(2005, 1, 3), date(2005, 1, 4), 4.0),
            (date(2005, 1, 3), date(2010, 6, 1), 4.0),  # clamped boundary artifact
        ]
        assert _dedupe_vintages(rows) == [(date(2005, 1, 3), date(2005, 1, 4), 4.0)]

    def test_a_genuine_revision_survives(self):
        rows = [
            (date(2005, 1, 3), date(2005, 1, 4), 4.0),
            (date(2005, 1, 3), date(2010, 6, 1), 4.2),
        ]
        assert len(_dedupe_vintages(rows)) == 2

    def test_a_value_that_returns_to_an_earlier_figure_is_kept(self):
        """4.0 -> 4.2 -> 4.0 is three real issues, not two."""
        rows = [
            (date(2005, 1, 3), date(2005, 1, 4), 4.0),
            (date(2005, 1, 3), date(2010, 6, 1), 4.2),
            (date(2005, 1, 3), date(2015, 6, 1), 4.0),
        ]
        assert len(_dedupe_vintages(rows)) == 3

    def test_periods_are_deduplicated_independently(self):
        rows = [
            (date(2005, 1, 3), date(2005, 1, 4), 4.0),
            (date(2005, 1, 4), date(2005, 1, 5), 4.0),
        ]
        assert len(_dedupe_vintages(rows)) == 2

    def test_the_first_issue_is_always_kept(self):
        rows = [(date(2005, 1, 3), date(2005, 1, 4), 4.0)]
        assert _dedupe_vintages(rows) == rows
