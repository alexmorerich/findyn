"""Provider adapters: canonical model, parsing, and per-source quirks."""

from __future__ import annotations

import json
from datetime import date

import numpy as np
import pandas as pd
import pytest

from findynamics.data.providers.base import (
    AuthError,
    BotChallengeError,
    NotFoundError,
    Observation,
    ProviderError,
    month_end,
    period_end,
    synthesize_release_date,
)
from findynamics.data.providers.bls import BlsProvider
from findynamics.data.providers.fred import FredProvider
from findynamics.data.providers.mock import MockProvider
from findynamics.data.providers.registry import (
    KEYLESS_PROVIDERS,
    available_providers,
    build_provider,
)
from findynamics.data.providers.resilience import MemoryCache, RateLimiter, RetryPolicy, Transport
from findynamics.data.providers.shiller import ShillerProvider, parse_shiller_date
from findynamics.data.providers.stooq import (
    StooqFileProvider,
    StooqProvider,
    looks_like_bot_challenge,
)
from tests.conftest import FakeFetcher, ok


def transport(fetcher, clock, sleeper, name="test") -> Transport:
    return Transport(
        name,
        fetcher,
        rate_limiter=RateLimiter(
            capacity=100, refill_per_second=100.0, clock=clock, sleeper=sleeper
        ),
        retry=RetryPolicy(max_attempts=2, base_delay=0.0, jitter=False),
        cache=MemoryCache(clock=clock),
        clock=clock,
        sleeper=sleeper,
    )


# --------------------------------------------------------------------------
# Canonical model
# --------------------------------------------------------------------------


def observation(**overrides) -> Observation:
    base = {
        "series_id": "X:Y",
        "provider": "test",
        "frequency": "monthly",
        "unit": "index",
        "observation_date": date(2025, 1, 1),
        "release_date": date(2025, 2, 15),
        "value": 1.0,
    }
    return Observation(**{**base, **overrides})


def test_observation_rejects_release_before_observation():
    with pytest.raises(ValueError, match="lookahead"):
        observation(release_date=date(2024, 12, 1))


def test_observation_rejects_revision_before_release():
    with pytest.raises(ValueError, match="revision_date"):
        observation(revision_date=date(2025, 1, 5))


def test_observation_wire_shape_matches_the_admin_endpoint():
    wire = observation(revision_date=date(2025, 3, 1)).to_wire()
    assert wire == {
        "series_id": "X:Y",
        "obs_date": "2025-01-01",
        "release_date": "2025-02-15",
        "revision_date": "2025-03-01",
        "value": 1.0,
        "source": "test",
    }


@pytest.mark.parametrize(
    ("day", "expected"),
    [
        (date(2025, 1, 15), date(2025, 1, 31)),
        (date(2024, 2, 1), date(2024, 2, 29)),  # leap year
        (date(2025, 2, 1), date(2025, 2, 28)),
    ],
)
def test_month_end(day, expected):
    assert month_end(day) == expected


@pytest.mark.parametrize(
    ("day", "frequency", "expected"),
    [
        (date(2025, 1, 1), "daily", date(2025, 1, 1)),
        (date(2025, 1, 1), "weekly", date(2025, 1, 7)),
        (date(2025, 1, 1), "monthly", date(2025, 1, 31)),
        (date(2025, 2, 1), "quarterly", date(2025, 3, 31)),
        (date(2025, 11, 1), "quarterly", date(2025, 12, 31)),
    ],
)
def test_period_end(day, frequency, expected):
    assert period_end(day, frequency) == expected


def test_release_lag_is_measured_from_the_end_of_the_period():
    # A monthly figure for January cannot be published mid-January, however
    # short the stated lag.
    assert synthesize_release_date(date(2025, 1, 1), "monthly", 14) == date(2025, 2, 14)


def test_negative_lag_is_rejected():
    with pytest.raises(ValueError, match="lookahead"):
        synthesize_release_date(date(2025, 1, 1), "monthly", -1)


# --------------------------------------------------------------------------
# Shiller
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (1871.01, date(1871, 1, 1)),
        (1871.09, date(1871, 9, 1)),
        # The trap: .1 is October, not "a tenth of a year".
        (1871.1, date(1871, 10, 1)),
        (1871.11, date(1871, 11, 1)),
        (1871.12, date(1871, 12, 1)),
        (2024.09, date(2024, 9, 1)),
    ],
)
def test_shiller_float_dates_decode_the_month(raw, expected):
    assert parse_shiller_date(raw) == expected


@pytest.mark.parametrize("raw", [None, float("nan"), 1871.13, 1871.99])
def test_shiller_rejects_impossible_months(raw):
    assert parse_shiller_date(raw) is None


def shiller_workbook(rows: list[list[float | None]]) -> pd.DataFrame:
    """Synthetic `Data` sheet: 8 header rows, observations, then a footnote row."""
    width = 22
    # object dtype: the real sheet mixes numbers, header labels and a prose
    # footnote in the same columns, and a float64 frame rejects the strings.
    frame = pd.DataFrame(np.nan, index=range(8 + len(rows) + 1), columns=range(width), dtype=object)
    frame.iloc[7, 0] = "Date"
    for offset, row in enumerate(rows):
        for col, value in enumerate(row):
            frame.iloc[8 + offset, col] = value
    # Shiller's real file ends with prose in the data columns.
    frame.iloc[8 + len(rows), 1] = "Sept price is Sept 4th close"
    return frame


def shiller_row(
    date_float: float, price: float, dividend: float, cape: float
) -> list[float | None]:
    row: list[float | None] = [None] * 22
    row[0] = date_float
    row[1] = price
    row[2] = dividend
    row[3] = 5.0
    row[4] = 100.0
    row[6] = 4.0
    row[7] = price
    row[8] = dividend
    row[10] = 5.0
    row[12] = cape
    row[16] = 2.0
    return row


@pytest.fixture
def shiller(monkeypatch, clock, sleeper) -> ShillerProvider:
    workbook = shiller_workbook(
        [
            shiller_row(2024.01, 4800.0, 70.0, 33.0),
            shiller_row(2024.02, 4900.0, 70.5, 33.5),
            shiller_row(2024.1, 5100.0, 71.0, 34.0),  # October
        ]
    )
    monkeypatch.setattr(pd, "read_excel", lambda *a, **k: workbook)
    return ShillerProvider(transport(FakeFetcher(ok(b"fake-xls")), clock, sleeper, "shiller"))


def test_shiller_parses_observations_and_drops_the_footnote(shiller):
    observations = shiller.fetch_observations("SHILLER:CAPE")
    assert [o.observation_date for o in observations] == [
        date(2024, 1, 1),
        date(2024, 2, 1),
        date(2024, 10, 1),
    ]
    assert [o.value for o in observations] == [33.0, 33.5, 34.0]


def test_shiller_synthesizes_release_dates_from_period_end(shiller):
    first = shiller.fetch_observations("SHILLER:CAPE")[0]
    # January's figure is dated the 1st but describes the whole month, so the
    # lag runs from 2024-01-31; +30 days lands in March across a 29-day February.
    assert first.observation_date == date(2024, 1, 1)
    assert first.release_date == date(2024, 3, 1)


def test_shiller_derives_dividend_yield(shiller):
    observations = shiller.fetch_observations("SHILLER:DIVIDEND_YIELD")
    assert observations[0].value == pytest.approx(70.0 / 4800.0 * 100)
    assert observations[0].unit == "percent"


def test_shiller_needs_no_api_key(shiller):
    assert shiller.requires_api_key is False
    assert "shiller" in KEYLESS_PROVIDERS


def test_shiller_range_filters_apply(shiller):
    observations = shiller.fetch_observations("SHILLER:CAPE", start=date(2024, 2, 1))
    assert [o.observation_date for o in observations] == [date(2024, 2, 1), date(2024, 10, 1)]


def test_shiller_metadata_spans_the_available_data(shiller):
    metadata = shiller.fetch_metadata("SHILLER:CAPE")
    assert metadata.frequency == "monthly"
    assert metadata.unit == "ratio"
    assert metadata.first_observation == date(2024, 1, 1)
    assert metadata.last_observation == date(2024, 10, 1)


def test_shiller_rejects_an_unknown_series(shiller):
    with pytest.raises(NotFoundError):
        shiller.fetch_observations("SHILLER:NOPE")


def test_shiller_downloads_the_workbook_once(monkeypatch, clock, sleeper):
    workbook = shiller_workbook([shiller_row(2024.01, 4800.0, 70.0, 33.0)])
    monkeypatch.setattr(pd, "read_excel", lambda *a, **k: workbook)
    fetcher = FakeFetcher(ok(b"fake-xls"))
    provider = ShillerProvider(transport(fetcher, clock, sleeper, "shiller"))
    provider.fetch_observations("SHILLER:CAPE")
    provider.fetch_observations("SHILLER:REAL_PRICE")
    assert fetcher.call_count == 1


# --------------------------------------------------------------------------
# Stooq
# --------------------------------------------------------------------------

STOOQ_CSV = (
    "Date,Open,High,Low,Close,Volume\n"
    "2025-01-02,5900.0,5950.0,5880.0,5920.5,1000\n"
    "2025-01-03,5920.0,5980.0,5910.0,5975.25,1100\n"
    "2025-01-06,N/D,N/D,N/D,N/D,0\n"
)

CHALLENGE_HTML = (
    '<!DOCTYPE html><html><head><meta charset="utf-8"></head><body>'
    "<noscript>This site requires JavaScript to verify your browser.</noscript>"
    '<script>crypto.subtle.digest("SHA-256", x)</script></body></html>'
)


def test_stooq_parses_daily_closes(clock, sleeper):
    provider = StooqProvider(transport(FakeFetcher(ok(STOOQ_CSV)), clock, sleeper, "stooq"))
    observations = provider.fetch_observations("STOOQ:^SPX")
    assert [o.observation_date for o in observations] == [date(2025, 1, 2), date(2025, 1, 3)]
    assert observations[0].value == 5920.5


def test_stooq_prices_are_knowable_the_day_they_print(clock, sleeper):
    provider = StooqProvider(transport(FakeFetcher(ok(STOOQ_CSV)), clock, sleeper, "stooq"))
    first = provider.fetch_observations("STOOQ:^SPX")[0]
    assert first.release_date == first.observation_date


def test_stooq_skips_padded_sessions(clock, sleeper):
    provider = StooqProvider(transport(FakeFetcher(ok(STOOQ_CSV)), clock, sleeper, "stooq"))
    assert len(provider.fetch_observations("STOOQ:^SPX")) == 2


@pytest.mark.parametrize(
    "payload",
    [CHALLENGE_HTML, "<html><body>nope</body></html>", "  <!doctype html>"],
)
def test_bot_challenge_is_recognised(payload):
    assert looks_like_bot_challenge(payload) is True


def test_real_csv_is_not_mistaken_for_a_challenge():
    assert looks_like_bot_challenge(STOOQ_CSV) is False


#: An index export: no Volume column, because an index has no volume. This is
#: the shape the manual ^SPX download actually has, and the strict six-column
#: header check rejected it.
STOOQ_INDEX_CSV = (
    "Date,Open,High,Low,Close\n"
    "1928-01-03,17.76,17.76,17.76,17.76\n"
    "1928-01-04,17.72,17.72,17.72,17.72\n"
)


def test_stooq_accepts_an_index_export_with_no_volume_column(clock, sleeper):
    provider = StooqProvider(transport(FakeFetcher(ok(STOOQ_INDEX_CSV)), clock, sleeper, "stooq"))
    observations = provider.fetch_observations("STOOQ:^SPX")
    assert [o.value for o in observations] == [17.76, 17.72]


class TestStooqFromFile:
    """Reading the CSV off disk after a human downloaded it past the challenge."""

    def _write(self, tmp_path, body, name="^spx_d.csv"):
        path = tmp_path / name
        path.write_text(body)
        return path

    def test_a_downloaded_file_parses_identically_to_a_fetch(self, tmp_path, clock, sleeper):
        """Same parser, so file-sourced and network-sourced rows must match."""
        fetched = StooqProvider(
            transport(FakeFetcher(ok(STOOQ_CSV)), clock, sleeper, "stooq")
        ).fetch_observations("STOOQ:^SPX")

        from_file = StooqFileProvider(self._write(tmp_path, STOOQ_CSV)).fetch_observations(
            "STOOQ:^SPX"
        )

        assert from_file == fetched

    def test_it_reads_the_deep_index_history(self, tmp_path):
        provider = StooqFileProvider(self._write(tmp_path, STOOQ_INDEX_CSV))
        observations = provider.fetch_observations("STOOQ:^SPX")
        assert observations[0].observation_date == date(1928, 1, 3)
        assert observations[0].release_date == date(1928, 1, 3)

    def test_range_filters_still_apply(self, tmp_path):
        provider = StooqFileProvider(self._write(tmp_path, STOOQ_INDEX_CSV))
        assert len(provider.fetch_observations("STOOQ:^SPX", start=date(1928, 1, 4))) == 1

    def test_a_saved_challenge_page_is_reported_as_such(self, tmp_path):
        """The likeliest way this goes wrong: the browser saved the interstitial.

        Reported as a bot challenge with what to do about it, rather than as
        'unexpected CSV header <html>', which names the symptom not the cause.
        """
        provider = StooqFileProvider(self._write(tmp_path, CHALLENGE_HTML))

        with pytest.raises(BotChallengeError, match="saved the interstitial"):
            provider.fetch_observations("STOOQ:^SPX")

    def test_a_missing_file_is_a_clean_error(self, tmp_path):
        provider = StooqFileProvider(tmp_path / "absent.csv")
        with pytest.raises(ProviderError, match="cannot read"):
            provider.fetch_observations("STOOQ:^SPX")

    def test_metadata_records_where_the_rows_came_from(self, tmp_path):
        """A file-sourced series must be traceable to the file that supplied it."""
        provider = StooqFileProvider(self._write(tmp_path, STOOQ_CSV))
        assert "^spx_d.csv" in (provider.fetch_metadata("STOOQ:^SPX").notes or "")

    def test_it_cannot_reach_the_network(self, tmp_path):
        """No transport, deliberately — this must never become a silent fallback."""
        provider = StooqFileProvider(self._write(tmp_path, STOOQ_CSV))
        assert not hasattr(provider, "transport")


def test_stooq_challenge_fails_fast_rather_than_retrying(clock, sleeper):
    # The challenge arrives with HTTP 200, so only content inspection catches it.
    fetcher = FakeFetcher(ok(CHALLENGE_HTML, 200))
    provider = StooqProvider(transport(fetcher, clock, sleeper, "stooq"))
    with pytest.raises(ProviderError) as excinfo:
        provider.fetch_observations("STOOQ:^SPX")
    assert "challenge" in str(excinfo.value)
    assert excinfo.value.retryable is False
    assert fetcher.call_count == 1


# --------------------------------------------------------------------------
# FRED
# --------------------------------------------------------------------------

FRED_META = {
    "seriess": [
        {
            "id": "CPIAUCSL",
            "title": "Consumer Price Index for All Urban Consumers",
            "frequency_short": "M",
            "units_short": "Index 1982-1984=100",
            "seasonal_adjustment_short": "SA",
            "observation_start": "1947-01-01",
            "observation_end": "2025-06-01",
        }
    ]
}

FRED_VINTAGES = {
    "observations": [
        # March 2025 CPI: first print, then a revision a month later.
        {"date": "2025-03-01", "realtime_start": "2025-04-10", "value": "319.6"},
        {"date": "2025-03-01", "realtime_start": "2025-05-13", "value": "319.8"},
        {"date": "2025-04-01", "realtime_start": "2025-05-13", "value": "320.8"},
        {"date": "2025-05-01", "realtime_start": "2025-06-11", "value": "."},
    ]
}


@pytest.fixture
def fred(clock, sleeper) -> FredProvider:
    def fetcher(url, *, params=None, headers=None, timeout=30.0):
        body = FRED_VINTAGES if url.endswith("series/observations") else FRED_META
        return ok(json.dumps(body))

    return FredProvider(transport(fetcher, clock, sleeper, "fred"), api_key="test-key")


def test_fred_uses_real_vintages_for_release_and_revision_dates(fred):
    observations = fred.fetch_observations("FRED:CPIAUCSL")
    march = [o for o in observations if o.observation_date == date(2025, 3, 1)]
    assert len(march) == 2
    # Both vintages share the date the period first became knowable...
    assert {o.release_date for o in march} == {date(2025, 4, 10)}
    # ...but each keeps its own issue date.
    assert [o.revision_date for o in march] == [date(2025, 4, 10), date(2025, 5, 13)]
    assert [o.value for o in march] == [319.6, 319.8]


def test_fred_skips_periods_with_no_data(fred):
    dates = {o.observation_date for o in fred.fetch_observations("FRED:CPIAUCSL")}
    assert date(2025, 5, 1) not in dates  # value was "."


def test_fred_maps_frequency_and_units(fred):
    metadata = fred.fetch_metadata("FRED:CPIAUCSL")
    assert metadata.frequency == "monthly"
    assert metadata.unit == "Index 1982-1984=100"
    assert metadata.seasonal_adjustment == "SA"


def test_fred_without_a_key_fails_as_auth_not_as_a_network_error(clock, sleeper):
    provider = FredProvider(transport(FakeFetcher(ok("{}")), clock, sleeper, "fred"), api_key=None)
    with pytest.raises(AuthError, match="FRED_API_KEY"):
        provider.fetch_metadata("FRED:CPIAUCSL")


# --------------------------------------------------------------------------
# BLS
# --------------------------------------------------------------------------


def test_bls_reports_a_quota_message_as_retryable(clock, sleeper):
    body = json.dumps(
        {"status": "REQUEST_NOT_PROCESSED", "message": ["daily threshold for requests exceeded"]}
    )
    provider = BlsProvider(transport(FakeFetcher(ok(body)), clock, sleeper, "bls"))
    with pytest.raises(ProviderError) as excinfo:
        provider.fetch_observations("BLS:CES0500000003")
    # BLS signals quota exhaustion inside an HTTP 200 body.
    assert excinfo.value.retryable is True


def test_bls_parses_monthly_periods_and_skips_annual_averages(clock, sleeper):
    body = json.dumps(
        {
            "status": "REQUEST_SUCCEEDED",
            "Results": {
                "series": [
                    {
                        "seriesID": "CES0500000003",
                        "data": [
                            {"year": "2025", "period": "M03", "value": "35.10"},
                            {"year": "2025", "period": "M02", "value": "35.00"},
                            {"year": "2025", "period": "M13", "value": "34.90"},
                        ],
                    }
                ]
            },
        }
    )
    provider = BlsProvider(transport(FakeFetcher(ok(body)), clock, sleeper, "bls"))
    observations = provider.fetch_observations("BLS:CES0500000003")
    assert [o.observation_date for o in observations] == [date(2025, 2, 1), date(2025, 3, 1)]
    assert observations[0].release_date == date(2025, 3, 7)  # Feb 28 + 7 days


# --------------------------------------------------------------------------
# Mock
# --------------------------------------------------------------------------


def test_mock_is_deterministic():
    a = MockProvider().fetch_observations("MOCK:CPI")
    b = MockProvider().fetch_observations("MOCK:CPI")
    assert [o.value for o in a] == [o.value for o in b]


def test_mock_labels_itself_as_synthetic():
    assert "SYNTHETIC" in (MockProvider().fetch_metadata("MOCK:CPI").notes or "")


def test_mock_series_are_namespaced():
    assert all(s.startswith("MOCK:") for s in MockProvider().available_series())


# --------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------


def test_registry_refuses_the_mock_provider_by_default():
    # Synthetic values are indistinguishable from measurements once stored.
    with pytest.raises(ProviderError, match="explicitly"):
        build_provider("mock")


def test_registry_builds_the_mock_provider_when_asked():
    assert isinstance(build_provider("mock", allow_mock=True), MockProvider)


def test_registry_rejects_an_unknown_provider():
    with pytest.raises(ProviderError, match="unknown provider"):
        build_provider("bloomberg")


def test_registry_builds_keyless_providers_without_credentials():
    for provider_id in sorted(KEYLESS_PROVIDERS):
        assert build_provider(provider_id, env={}).requires_api_key is False


def test_availability_reflects_configured_keys():
    status = available_providers(env={"FRED_API_KEY": "k"})
    assert status["shiller"] is True
    assert status["stooq"] is True
    assert status["fred"] is True
    assert status["bea"] is False


def test_every_network_provider_has_a_documented_quota():
    from findynamics.data.providers.registry import NETWORK_PROVIDERS, QUOTAS

    assert set(QUOTAS) >= NETWORK_PROVIDERS
    for provider_id in NETWORK_PROVIDERS:
        assert QUOTAS[provider_id].note


def test_every_provider_the_config_accepts_is_buildable():
    """VALID_PROVIDERS may only name providers that actually exist.

    The config once accepted `alphavantage` and `yahoo` with no adapter behind
    them, so a series pointed at one passed validation and failed at fetch time.

    This assertion used to read ``VALID_PROVIDERS - {"derived"}``, carrying the
    note "computed downstream, never built". Nothing computed it. Two factor
    inputs went unfetched for months while this test stayed green, because the
    exemption made the gap look deliberate and gave every later reader a reason
    not to check. A test that encodes a bug as expected behaviour is worse than
    no test: it actively defends the bug.

    The set equality is now exact, and it is what proves `derived` is real.
    """
    from findynamics.core.config import VALID_PROVIDERS
    from findynamics.data.providers.registry import NETWORK_PROVIDERS

    assert VALID_PROVIDERS == NETWORK_PROVIDERS
