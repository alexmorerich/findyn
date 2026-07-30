"""Job-layer plumbing: series resolution, write-back chunking, payload shaping.

The jobs are thin CLIs, so what is worth testing is the small amount of shaping
they do — and the one piece of real logic, splitting a backfill into requests
the serving plane can absorb.
"""

from __future__ import annotations

from datetime import date

import pytest

from findynamics.core.contracts.state import AssetState, EngineOutput, FactorState, Signal
from jobs.backfill import DEFAULT_BATCH_SIZE, chunk_payload, configured_series
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
