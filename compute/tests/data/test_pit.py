"""Point-in-time join tests — the no-lookahead contract (FINDYN_V1_SPEC.md §14.1)."""

from __future__ import annotations

import pandas as pd
import pytest

from findynamics.data.pit import (
    LookaheadError,
    assert_no_lookahead,
    pit_history,
    pit_join,
    synthesize_release_date,
)


def frame(rows: list[tuple[str, str, str, float]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["series_id", "obs_date", "release_date", "value"])


def vintaged(rows: list[tuple[str, str, str, str, float]]) -> pd.DataFrame:
    """(series_id, obs_date, release_date, revision_date, value)."""
    return pd.DataFrame(
        rows, columns=["series_id", "obs_date", "release_date", "revision_date", "value"]
    )


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


class TestRevisionDateIsWhatMakesAFigureKnowable:
    """REGRESSION — the defect found in P1.

    ``release_date`` says when the *period* became observable and is therefore
    constant across every revision of that period. Filtering on it alone admits
    a later revision the moment the original print is published: a value nobody
    could have seen. FRED reissues under the same release_date and DGS10 carries
    over 5000 vintages, so this contaminated real data, not a hypothetical.

    These tests are written to fail against the old rule. That needs care: the
    old code broke ties on release_date, which is identical across revisions, so
    a stable sort returned whichever row happened to come first in the frame. A
    fixture in ascending order passed by luck. Hence the row-order parametrization
    below — the honest statement of the contract is that the answer depends on
    the information set and on nothing else, least of all on row order.
    """

    REVISED = [
        ("FRED:GDP", "2026-01-01", "2026-01-30", "2026-01-30", 100.0),
        ("FRED:GDP", "2026-01-01", "2026-01-30", "2026-02-27", 101.5),
        ("FRED:GDP", "2026-01-01", "2026-01-30", "2026-03-26", 102.1),
    ]

    @pytest.mark.parametrize("order", ["ascending", "descending", "shuffled"])
    def test_a_revision_is_invisible_until_it_is_issued(self, order):
        rows = {
            "ascending": self.REVISED,
            "descending": list(reversed(self.REVISED)),
            "shuffled": [self.REVISED[2], self.REVISED[0], self.REVISED[1]],
        }[order]

        result = pit_join(vintaged(rows), "2026-02-10")

        # Only the original print had been issued by 2026-02-10.
        assert result.loc["FRED:GDP", "value"] == 100.0

    @pytest.mark.parametrize("order", ["ascending", "descending"])
    def test_the_answer_does_not_depend_on_row_order(self, order):
        """The old rule's tie-break made this frame-order dependent."""
        rows = self.REVISED if order == "ascending" else list(reversed(self.REVISED))
        assert pit_join(vintaged(rows), "2026-02-10").loc["FRED:GDP", "value"] == pytest.approx(
            pit_join(vintaged(self.REVISED), "2026-02-10").loc["FRED:GDP", "value"]
        )

    def test_each_revision_appears_on_its_own_issue_date(self):
        data = vintaged(self.REVISED)
        assert pit_join(data, "2026-02-26").loc["FRED:GDP", "value"] == 100.0
        assert pit_join(data, "2026-02-27").loc["FRED:GDP", "value"] == 101.5
        assert pit_join(data, "2026-03-25").loc["FRED:GDP", "value"] == 101.5
        assert pit_join(data, "2026-03-26").loc["FRED:GDP", "value"] == 102.1

    def test_history_hides_unissued_revisions_too(self):
        """pit_history shares the filter; fixing one and not the other is worse."""
        history = pit_history(vintaged(list(reversed(self.REVISED))), "2026-02-10")
        assert list(history["value"]) == [100.0]

    def test_a_frame_without_revision_dates_still_works(self):
        """Sources with no vintage information are unaffected."""
        result = pit_join(
            frame(
                [
                    ("FRED:PAYEMS", "2026-03-01", "2026-04-03", 150.0),
                    ("FRED:PAYEMS", "2026-03-01", "2026-05-08", 142.0),
                ]
            ),
            "2026-04-20",
        )
        assert result.loc["FRED:PAYEMS", "value"] == 150.0

    def test_a_null_revision_date_falls_back_to_the_release_date(self):
        data = vintaged([("FRED:GDP", "2026-01-01", "2026-01-30", None, 100.0)])
        assert pit_join(data, "2026-02-10").loc["FRED:GDP", "value"] == 100.0


class TestPitHistory:
    OBS = [
        ("FRED:DGS10", "2026-01-05", "2026-01-06", "2026-01-06", 4.0),
        ("FRED:DGS10", "2026-01-06", "2026-01-07", "2026-01-07", 4.1),
        ("FRED:DGS10", "2026-01-07", "2026-01-08", "2026-01-08", 4.2),
        ("FRED:DGS2", "2026-01-05", "2026-01-06", "2026-01-06", 3.0),
    ]

    def test_returns_every_period_not_just_the_last(self):
        history = pit_history(vintaged(self.OBS), "2026-01-10")

        assert len(history) == 4
        assert list(history.columns) == ["series_id", "obs_date", "release_date", "value"]

    def test_is_sorted_by_series_then_period_ascending(self):
        history = pit_history(vintaged(self.OBS), "2026-01-10")
        dgs10 = history[history["series_id"] == "FRED:DGS10"]
        assert list(dgs10["obs_date"]) == sorted(dgs10["obs_date"])

    def test_applies_the_same_cutoff_as_pit_join(self):
        history = pit_history(vintaged(self.OBS), "2026-01-07")
        assert history["obs_date"].max() == pd.Timestamp("2026-01-06")

    def test_keeps_one_row_per_period_at_its_newest_knowable_vintage(self):
        history = pit_history(
            vintaged(TestRevisionDateIsWhatMakesAFigureKnowable.REVISED), "2026-03-01"
        )
        assert len(history) == 1
        assert history.iloc[0]["value"] == 101.5

    def test_can_be_restricted_by_series(self):
        history = pit_history(vintaged(self.OBS), "2026-01-10", series_ids=["FRED:DGS2"])
        assert set(history["series_id"]) == {"FRED:DGS2"}

    def test_can_be_restricted_by_start_date(self):
        history = pit_history(vintaged(self.OBS), "2026-01-10", start="2026-01-06")
        assert history["obs_date"].min() == pd.Timestamp("2026-01-06")

    def test_empty_result_has_the_expected_shape(self):
        history = pit_history(vintaged(self.OBS), "2020-01-01")
        assert history.empty
        assert list(history.columns) == ["series_id", "obs_date", "release_date", "value"]

    def test_its_last_row_per_series_agrees_with_pit_join(self):
        """Two functions, one definition of 'knowable' — or the two will drift."""
        data = vintaged(self.OBS)
        history = pit_history(data, "2026-01-10")
        latest = pit_join(data, "2026-01-10")

        for series_id, group in history.groupby("series_id"):
            assert group.iloc[-1]["value"] == latest.loc[series_id, "value"]
