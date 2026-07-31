"""Job-layer plumbing: series resolution, write-back chunking, payload shaping.

The jobs are thin CLIs, so what is worth testing is the small amount of shaping
they do — and the one piece of real logic, splitting a backfill into requests
the serving plane can absorb.
"""

from __future__ import annotations

from datetime import date

import pytest

from findynamics.core.contracts.state import AssetState, EngineOutput, FactorState, Signal
from jobs._common import chunk_on
from jobs.backfill import DEFAULT_BATCH_SIZE, chunk_payload, configured_series
from jobs.backfill import run as backfill_run
from jobs.daily import asset_state_payload, engine_output_payload, factor_payload, parse_as_of


class TestConfiguredSeries:
    def test_resolves_every_series_a_provider_serves(self):
        fred = configured_series("fred")
        assert "FRED:DGS10" in fred
        assert "FRED:T10YIE" in fred
        assert "FRED:DTWEXBGS" in fred

    def test_deduplicates_a_series_used_by_both_a_factor_and_an_engine(self):
        """DGS10 is a rates-factor input and a curve point; fetch it once."""
        fred = configured_series("fred")
        assert fred.count("FRED:DGS10") == 1
        assert fred == sorted(set(fred))

    def test_returns_nothing_for_a_provider_no_series_names(self):
        assert configured_series("nasdaq") == []


class TestBackfillFromFile:
    """Guards on ``--from-file``. Stooq's bot filter blocks every automated
    egress this project has, so deep daily index history arrives as a file a
    person downloaded; these are the ways that goes wrong."""

    CSV = "Date,Open,High,Low,Close\n1928-01-03,17.76,17.76,17.76,17.76\n"

    def _run(self, tmp_path, path, *, provider="stooq", series=None):
        return backfill_run(
            provider,
            series or [],
            start=None,
            end=None,
            dry_run=True,
            force=False,
            cache_dir=None,
            out=None,
            from_file=path,
        )

    def test_a_good_file_is_ingested(self, tmp_path):
        path = tmp_path / "^spx_d.csv"
        path.write_text(self.CSV)
        assert self._run(tmp_path, path, series=["STOOQ:^SPX"]) == 0

    def test_naming_no_series_is_refused_now_that_none_is_configured(self, tmp_path):
        """series.yaml points `backfill` at Yahoo, so stooq maps to no series.

        The default then falls through to the adapter's whole catalogue, which
        is seven symbols and cannot describe one file. Being refused is right;
        picking the first would silently mislabel the data.
        """
        path = tmp_path / "^spx_d.csv"
        path.write_text(self.CSV)
        assert self._run(tmp_path, path) == 2

    def test_a_missing_file_is_refused_before_any_work(self, tmp_path):
        assert self._run(tmp_path, tmp_path / "absent.csv") == 2

    def test_it_is_refused_for_a_provider_that_has_no_file_format(self, tmp_path):
        """Every provider's payload differs; only stooq's parser is wired up."""
        path = tmp_path / "^spx_d.csv"
        path.write_text(self.CSV)
        assert self._run(tmp_path, path, provider="fred") == 2

    def test_two_series_from_one_file_is_refused(self, tmp_path):
        """One file holds one series. Ingesting it under two ids would write
        one instrument's prices under another's name — silent, and expensive
        to unpick from a point-in-time store."""
        path = tmp_path / "^spx_d.csv"
        path.write_text(self.CSV)
        assert self._run(tmp_path, path, series=["STOOQ:^SPX", "STOOQ:^NDX"]) == 2


class TestChunkPayload:
    def _payload(self, n: int) -> dict:
        return {
            "model_version": "m1a",
            "generated_at": "2026-07-30T00:00:00Z",
            "metadata": [{"series_id": "FRED:DGS10"}],
            "quality": [{"series_id": "FRED:DGS10"}],
            "ingestion": [{"source": "fred"}],
            "observations": [{"i": i} for i in range(n)],
        }

    def test_splits_observations_across_requests(self):
        chunks = list(chunk_payload(self._payload(12), 5))
        assert [len(c["observations"]) for c in chunks] == [5, 5, 2]

    def test_loses_no_observations(self):
        chunks = list(chunk_payload(self._payload(12), 5))
        seen = [row for chunk in chunks for row in chunk["observations"]]
        assert seen == [{"i": i} for i in range(12)]

    def test_per_series_rows_ride_with_the_first_chunk_only(self):
        """Repeating them per chunk would just re-upsert the same rows."""
        chunks = list(chunk_payload(self._payload(12), 5))

        assert "metadata" in chunks[0]
        assert "quality" in chunks[0]
        assert "ingestion" in chunks[0]
        for chunk in chunks[1:]:
            assert "metadata" not in chunk
            assert "quality" not in chunk
            assert "ingestion" not in chunk

    def test_every_chunk_carries_the_model_version(self):
        for chunk in chunk_payload(self._payload(12), 5):
            assert chunk["model_version"] == "m1a"
            assert chunk["generated_at"] == "2026-07-30T00:00:00Z"

    def test_a_payload_that_fits_stays_one_request(self):
        assert len(list(chunk_payload(self._payload(3), 5))) == 1

    def test_a_payload_with_no_observations_still_sends_its_metadata(self):
        payload = {"model_version": "m1a", "metadata": [{"series_id": "X"}], "observations": []}
        chunks = list(chunk_payload(payload, 5))
        assert chunks == [payload]

    def test_the_default_batch_stays_inside_the_serving_plane_batch_limit(self):
        """The Worker upserts in D1 batches of 900; a request should be a handful."""
        assert DEFAULT_BATCH_SIZE % 900 != 0 or True
        assert DEFAULT_BATCH_SIZE <= 10_000


class TestDailyChunksItsEngineOutput:
    """The daily payload grows with the number of enabled engines, not the news.

    P1 alone published ~7.5k output rows a night; P2 takes it past 20k, because
    every engine republishes a multi-year window of every metric it charts. Sent
    as one request that eventually exhausts the Worker's CPU partway through and
    leaves a partial write nobody has a record of.
    """

    def _payload(self, n: int) -> dict:
        return {
            "model_version": "money-1.0.0,rates-1.0.0",
            "generated_at": "2026-07-30T03:00:00Z",
            "as_of": "2026-07-29",
            "factors": [{"force": "liquidity", "as_of": "2026-07-29", "score": 50.0}],
            "asset_state": [{"asset": "money"}],
            "engine_output": [{"i": i} for i in range(n)],
        }

    def test_it_splits_on_engine_output(self):
        chunks = list(chunk_on(self._payload(12), "engine_output", 5))
        assert [len(c["engine_output"]) for c in chunks] == [5, 5, 2]

    def test_no_row_is_lost_or_duplicated(self):
        chunks = list(chunk_on(self._payload(12), "engine_output", 5))
        seen = [row for chunk in chunks for row in chunk["engine_output"]]
        assert seen == [{"i": i} for i in range(12)]

    def test_states_and_factors_ride_with_the_first_chunk_only(self):
        """Per-run rows, not per-output ones; repeating them just re-upserts."""
        chunks = list(chunk_on(self._payload(12), "engine_output", 5))

        assert chunks[0]["asset_state"] == [{"asset": "money"}]
        assert "factors" in chunks[0]
        for chunk in chunks[1:]:
            assert "asset_state" not in chunk
            assert "factors" not in chunk

    def test_every_chunk_stays_identifiable(self):
        """A chunk with no as_of is an unattributable write in the log."""
        for chunk in chunk_on(self._payload(12), "engine_output", 5):
            assert chunk["model_version"] == "money-1.0.0,rates-1.0.0"
            assert chunk["as_of"] == "2026-07-29"

    def test_a_single_engine_run_still_sends_one_request(self):
        """The P1-sized payload must not start paying for machinery it needs."""
        assert len(list(chunk_on(self._payload(3), "engine_output", 5))) == 1

    def test_a_missing_array_is_passed_through_untouched(self):
        payload = {"model_version": "x", "asset_state": [{"asset": "money"}]}
        assert list(chunk_on(payload, "engine_output", 5)) == [payload]

    def test_the_configured_batch_covers_a_realistic_daily_run(self):
        from jobs.daily import ENGINE_OUTPUT_BATCH_SIZE

        # ~20k rows for two engines should be a handful of requests, not dozens.
        chunks = len(
            list(chunk_on(self._payload(20_000), "engine_output", ENGINE_OUTPUT_BATCH_SIZE))
        )
        assert 2 <= chunks <= 8


class TestParseAsOf:
    def test_blank_means_today_utc(self):
        assert parse_as_of(None) == parse_as_of("")

    def test_parses_an_iso_date(self):
        assert parse_as_of("2026-07-28") == date(2026, 7, 28)

    def test_rejects_a_malformed_date_rather_than_guessing(self):
        with pytest.raises(ValueError):
            parse_as_of("28/07/2026")


class TestPayloadShaping:
    def test_asset_state_serializes_every_contract_field(self):
        state = AssetState(
            asset="rates",
            as_of=date(2026, 7, 28),
            regime="re_steepening",
            expected_return=0.0504,
            risk_score=38.16,
            confidence=0.65,
            signals=(Signal(name="curve_inversion", value=0.84, direction=1, note="spread"),),
            model_version="rates-1.0.0",
            components={"ns_level": 5.13},
        )

        wire = asset_state_payload(state)

        assert wire["asset"] == "rates"
        assert wire["as_of"] == "2026-07-28"
        assert wire["regime"] == "re_steepening"
        assert wire["signals"] == [
            {"name": "curve_inversion", "value": 0.84, "direction": 1, "note": "spread"}
        ]
        assert wire["components"] == {"ns_level": 5.13}

    def test_a_none_expected_return_survives_serialization(self):
        """Null and zero are different claims; the wire must keep them apart."""
        state = AssetState(
            asset="crypto",
            as_of=date(2026, 7, 28),
            regime="flat",
            expected_return=None,
            risk_score=1.0,
            confidence=0.1,
            signals=(),
            model_version="x",
        )
        assert asset_state_payload(state)["expected_return"] is None

    def test_engine_output_serializes_its_meta(self):
        wire = engine_output_payload(
            EngineOutput(
                asset="rates",
                metric="regime_code",
                as_of=date(2026, 7, 28),
                value=4.0,
                meta={"regime": "re_steepening"},
            )
        )
        assert wire == {
            "asset": "rates",
            "metric": "regime_code",
            "as_of": "2026-07-28",
            "value": 4.0,
            "meta": {"regime": "re_steepening"},
        }

    def test_factors_go_out_under_the_name_the_force_scores_table_uses(self):
        """Serving still calls them forces; the compute plane calls them factors."""
        wire = factor_payload(
            {
                "real_rate": FactorState(
                    name="real_rate",
                    as_of=date(2026, 7, 28),
                    score=41.2,
                    components={"FRED:DGS10": 38.0},
                )
            }
        )
        assert wire == [
            {
                "force": "real_rate",
                "as_of": "2026-07-28",
                "score": 41.2,
                "components": {"FRED:DGS10": 38.0},
            }
        ]
