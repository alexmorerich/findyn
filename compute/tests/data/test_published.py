"""The published-output provider: reading one engine's work as another's input.

This adapter is the seam that makes the engine-independence rule survivable, so
what matters is that it behaves like any other provider — same frame shape, same
release-date honesty — and that it fails soft. An unreachable serving plane must
cost the consumer its long-horizon curve, never the run.
"""

from __future__ import annotations

import json
from datetime import date

import pytest

from findynamics.core.config import SeriesSpec
from findynamics.core.contracts.vocab import engine_series_id, parse_engine_series_id
from findynamics.data.providers import build_provider
from findynamics.data.providers.base import NotFoundError, ParseError
from findynamics.data.providers.published import (
    ADMIN_URL_ENV,
    API_URL_ENV,
    PublishedOutputProvider,
    resolve_api_base,
)
from findynamics.data.providers.registry import available_providers, build_transport
from findynamics.data.vintages import repair_pre_archive_releases
from tests.conftest import FakeFetcher, ok

BASE = "https://findyn.example.workers.dev"
SERIES = "ENGINE:rates.ns_level"


def envelope(points: list[dict], *, asset: str = "rates", metric: str = "ns_level") -> str:
    return json.dumps(
        {
            "as_of": points[-1]["as_of"] if points else None,
            "model_version": "rates-1.0.0",
            "stale": False,
            "data": {"asset": asset, "metric": metric, "count": len(points), "points": points},
            "disclaimer": "…",
        }
    )


def provider(body: str, *, base_url: str | None = BASE) -> PublishedOutputProvider:
    transport = build_transport("engine_output", fetcher=FakeFetcher(ok(body)))
    return PublishedOutputProvider(transport, base_url=base_url)


class TestSeriesIdVocabulary:
    def test_round_trip(self):
        assert engine_series_id("rates", "ns_level") == SERIES
        assert parse_engine_series_id(SERIES) == ("rates", "ns_level")

    def test_an_ordinary_series_is_classified_not_rejected(self):
        assert parse_engine_series_id("FRED:DGS10") is None

    def test_a_malformed_engine_id_is_an_error(self):
        with pytest.raises(ValueError, match="malformed engine series id"):
            parse_engine_series_id("ENGINE:rates")

    def test_an_unknown_asset_is_rejected_at_both_ends(self):
        with pytest.raises(ValueError, match="not one of"):
            engine_series_id("bonds", "ns_level")
        with pytest.raises(ValueError, match="unknown asset"):
            parse_engine_series_id("ENGINE:bonds.ns_level")


class TestResolveApiBase:
    def test_the_explicit_variable_wins(self):
        assert resolve_api_base({API_URL_ENV: f"{BASE}/"}) == BASE

    def test_it_is_derived_from_the_admin_endpoint_otherwise(self):
        """One configured location, so reads and writes cannot diverge."""
        assert resolve_api_base({ADMIN_URL_ENV: f"{BASE}/admin/v1/results"}) == BASE

    def test_nothing_configured_is_none_rather_than_a_guess(self):
        assert resolve_api_base({}) is None
        assert resolve_api_base({API_URL_ENV: "  "}) is None

    def test_availability_reflects_whether_there_is_somewhere_to_read(self):
        assert available_providers({})["engine_output"] is False
        assert available_providers({API_URL_ENV: BASE})["engine_output"] is True


class TestFetching:
    def test_points_become_observations(self):
        body = envelope(
            [
                {"as_of": "2026-07-27", "value": 5.11, "meta": None, "written_at": "2026-07-28"},
                {"as_of": "2026-07-28", "value": 5.13, "meta": None, "written_at": "2026-07-29"},
            ]
        )
        observations = provider(body).fetch_observations(SERIES)

        assert [o.observation_date for o in observations] == [
            date(2026, 7, 27),
            date(2026, 7, 28),
        ]
        assert [o.value for o in observations] == [5.11, 5.13]
        assert all(o.series_id == SERIES for o in observations)
        assert all(o.provider == "engine_output" for o in observations)

    def test_written_at_is_the_release_date(self):
        """The genuine vintage: the run that published the row."""
        body = envelope(
            [{"as_of": "2026-07-01", "value": 5.0, "written_at": "2026-07-20T03:00:00Z"}]
        )
        observation = provider(body).fetch_observations(SERIES)[0]
        assert observation.observation_date == date(2026, 7, 1)
        assert observation.release_date == date(2026, 7, 20)

    def test_a_missing_written_at_falls_back_to_a_conservative_next_day(self):
        body = envelope([{"as_of": "2026-07-27", "value": 5.11}])
        observation = provider(body).fetch_observations(SERIES)[0]
        assert observation.release_date == date(2026, 7, 28)

    def test_a_written_at_before_the_period_is_clamped_not_rejected(self):
        """Clock skew must not make the frame unloadable."""
        body = envelope([{"as_of": "2026-07-27", "value": 5.11, "written_at": "2026-07-01"}])
        observation = provider(body).fetch_observations(SERIES)[0]
        assert observation.release_date == date(2026, 7, 27)

    def test_it_requests_the_right_endpoint_and_metric(self):
        fetcher = FakeFetcher(ok(envelope([{"as_of": "2026-07-27", "value": 1.0}])))
        transport = build_transport("engine_output", fetcher=fetcher)
        PublishedOutputProvider(transport, base_url=BASE).fetch_observations(
            "ENGINE:rates.ns_curvature", start=date(2020, 1, 1)
        )

        url, params = fetcher.calls[0]
        assert url == f"{BASE}/api/v1/assets/rates/history"
        assert params["metric"] == "ns_curvature"
        assert params["from"] == "2020-01-01"

    def test_unusable_points_are_skipped_rather_than_failing_the_series(self):
        body = envelope(
            [
                {"as_of": "2026-07-27", "value": 5.11},
                {"as_of": None, "value": 5.12},
                {"as_of": "2026-07-29", "value": None},
                {"as_of": "not-a-date", "value": 5.14},
                {"as_of": "2026-07-30", "value": "5.15"},
            ]
        )
        observations = provider(body).fetch_observations(SERIES)
        assert [o.observation_date for o in observations] == [date(2026, 7, 27)]

    def test_metadata_describes_the_producing_engine(self):
        metadata = provider(envelope([])).fetch_metadata(SERIES)
        assert metadata.provider == "engine_output"
        assert "rates" in metadata.title
        assert metadata.frequency == "daily"

    def test_an_empty_history_is_an_empty_list_not_an_error(self):
        """An engine that has never run is normal, not broken."""
        assert provider(envelope([])).fetch_observations(SERIES) == []


class TestFailingSoft:
    def test_no_configured_base_url_is_reported_as_not_found(self):
        """The first-ever run of the system takes this path."""
        with pytest.raises(NotFoundError, match=API_URL_ENV):
            provider(envelope([]), base_url=None).fetch_observations(SERIES)

    def test_a_non_engine_series_id_is_rejected(self):
        with pytest.raises(NotFoundError, match="not a published-output id"):
            provider(envelope([])).fetch_observations("FRED:DGS10")

    def test_a_non_envelope_response_is_a_parse_error(self):
        with pytest.raises(ParseError, match="not a FinDyn envelope"):
            provider(json.dumps({"points": []})).fetch_observations(SERIES)

    def test_a_missing_points_array_is_a_parse_error(self):
        body = json.dumps({"data": {"asset": "rates", "metric": "ns_level"}})
        with pytest.raises(ParseError, match="no points array"):
            provider(body).fetch_observations(SERIES)

    def test_non_json_is_a_parse_error(self):
        with pytest.raises(ParseError, match="non-JSON"):
            provider("<html>nope</html>").fetch_observations(SERIES)

    def test_build_provider_constructs_it_with_the_resolved_base(self):
        built = build_provider(
            "engine_output", env={API_URL_ENV: BASE}, fetcher=FakeFetcher(ok("{}"))
        )
        assert isinstance(built, PublishedOutputProvider)
        assert built.base_url == BASE

    def test_build_provider_tolerates_no_configuration(self):
        """Constructing must not raise; only fetching reports the problem."""
        built = build_provider("engine_output", env={}, fetcher=FakeFetcher(ok("{}")))
        assert isinstance(built, PublishedOutputProvider)
        assert built.base_url is None


class TestVintageRepairIsSkipped:
    """A daily run republishing a five-year window is not an archive artifact."""

    def test_shared_publication_dates_survive_the_repair(self):
        """Exactly the 'bulk seeded' pattern — and here it is the truth.

        Repairing it would move these release dates earlier and license a
        consumer to read a curve factor at a cutoff before the run existed.
        """
        body = envelope(
            [
                {"as_of": "2024-01-02", "value": 4.0, "written_at": "2026-07-29"},
                {"as_of": "2025-01-02", "value": 4.5, "written_at": "2026-07-29"},
                {"as_of": "2026-07-28", "value": 5.0, "written_at": "2026-07-29"},
            ]
        )
        observations = provider(body).fetch_observations(SERIES)
        spec = SeriesSpec(
            id=SERIES, provider="engine_output", frequency="daily", publication_lag_days=0
        )

        repaired = repair_pre_archive_releases(observations, spec)
        assert [o.release_date for o in repaired] == [date(2026, 7, 29)] * 3
        assert repaired == observations

    def test_an_ordinary_provider_still_gets_repaired(self):
        """Guard the guard: the skip must be provider-specific, not global."""
        from findynamics.data.providers.base import Observation

        rows = [
            Observation(
                series_id="FRED:DGS20",
                provider="fred",
                frequency="daily",
                unit="pct",
                observation_date=date(1988, 1, 4),
                release_date=date(2020, 6, 1),
                value=8.0,
            ),
            Observation(
                series_id="FRED:DGS20",
                provider="fred",
                frequency="daily",
                unit="pct",
                observation_date=date(1988, 1, 5),
                release_date=date(2020, 6, 1),
                value=8.1,
            ),
        ]
        spec = SeriesSpec(
            id="FRED:DGS20", provider="fred", frequency="daily", publication_lag_days=1
        )
        repaired = repair_pre_archive_releases(rows, spec)
        assert [o.release_date for o in repaired] == [date(1988, 1, 5), date(1988, 1, 6)]
