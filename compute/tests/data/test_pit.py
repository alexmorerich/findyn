"""Point-in-time join tests — the no-lookahead contract (FINDYN_V1_SPEC.md §14.1)."""

from __future__ import annotations

import pandas as pd
import pytest

from findynamics.data.pit import (
    LookaheadError,
    assert_no_lookahead,
    pit_join,
    synthesize_release_date,
)


def frame(rows: list[tuple[str, str, str, float]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["series_id", "obs_date", "release_date", "value"])


def test_drops_observations_not_yet_released():
    """CPI for March exists in the table but was not published until mid-April."""
    obs = frame(
        [
            ("FRED:CPIAUCSL", "2026-02-01", "2026-03-11", 310.0),
            ("FRED:CPIAUCSL", "2026-03-01", "2026-04-10", 311.5),
        ]
    )

    result = pit_join(obs, "2026-04-01")

    assert len(result) == 1
    assert result.loc["FRED:CPIAUCSL", "value"] == 310.0
    assert result.loc["FRED:CPIAUCSL", "obs_date"] == pd.Timestamp("2026-02-01")


def test_includes_observation_released_exactly_on_cutoff():
    obs = frame([("FRED:UNRATE", "2026-03-01", "2026-04-03", 4.1)])
    result = pit_join(obs, "2026-04-03")
    assert result.loc["FRED:UNRATE", "value"] == 4.1


def test_prefers_latest_vintage_of_the_same_period():
    """A revision known at the cutoff supersedes the original print."""
    obs = frame(
        [
            ("FRED:PAYEMS", "2026-03-01", "2026-04-03", 150.0),
            ("FRED:PAYEMS", "2026-03-01", "2026-05-08", 142.0),  # revised down
        ]
    )

    assert pit_join(obs, "2026-04-20").loc["FRED:PAYEMS", "value"] == 150.0
    assert pit_join(obs, "2026-05-20").loc["FRED:PAYEMS", "value"] == 142.0


def test_returns_one_row_per_series():
    obs = frame(
        [
            ("FRED:DGS10", "2026-04-01", "2026-04-02", 4.2),
            ("FRED:DGS10", "2026-04-02", "2026-04-03", 4.25),
            ("FRED:VIXCLS", "2026-04-02", "2026-04-03", 18.0),
        ]
    )

    result = pit_join(obs, "2026-04-10")

    assert sorted(result.index) == ["FRED:DGS10", "FRED:VIXCLS"]
    assert result.loc["FRED:DGS10", "value"] == 4.25


def test_reports_staleness_in_days():
    obs = frame([("BEA:CORPORATE_PROFITS", "2026-01-01", "2026-03-01", 2.5)])
    result = pit_join(obs, "2026-04-01")
    assert result.loc["BEA:CORPORATE_PROFITS", "staleness_days"] == 90


def test_missing_series_degrades_instead_of_raising():
    """A provider outage should shrink the feature set, not abort the run (§14.2)."""
    obs = frame([("FRED:DGS10", "2026-04-01", "2026-04-02", 4.2)])
    result = pit_join(obs, "2026-04-10", series_ids=["FRED:DGS10", "FRED:MISSING"])
    assert list(result.index) == ["FRED:DGS10"]


def test_empty_result_has_the_expected_shape():
    obs = frame([("FRED:DGS10", "2026-04-01", "2026-04-02", 4.2)])
    result = pit_join(obs, "2026-01-01")
    assert result.empty
    assert list(result.columns) == ["obs_date", "release_date", "value", "staleness_days"]


def test_release_before_observation_is_a_data_error():
    obs = frame([("FRED:CPIAUCSL", "2026-03-01", "2026-02-01", 311.0)])
    with pytest.raises(LookaheadError, match="precedes obs_date"):
        pit_join(obs, "2026-06-01")


def test_missing_columns_are_rejected():
    bad = pd.DataFrame({"series_id": ["x"], "obs_date": ["2026-01-01"], "value": [1.0]})
    with pytest.raises(ValueError, match="release_date"):
        pit_join(bad, "2026-06-01")


def test_assert_no_lookahead_catches_contaminated_frames():
    contaminated = frame([("FRED:DGS10", "2026-04-01", "2026-05-01", 4.2)])
    with pytest.raises(LookaheadError, match="release_date after as_of"):
        assert_no_lookahead(contaminated, "2026-04-15")


def test_synthesized_release_date_applies_the_lag():
    assert synthesize_release_date("2026-03-01", 14) == pd.Timestamp("2026-03-15")


def test_negative_lag_is_rejected():
    with pytest.raises(ValueError, match="lookahead"):
        synthesize_release_date("2026-03-01", -1)
