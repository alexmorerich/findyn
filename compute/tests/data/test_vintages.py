"""Repairing release dates that predate the vintage archive.

ALFRED starts recording a series on some date and stamps everything older with
that date. Left alone, that claims the 1962-2005 Treasury curve became knowable
in December 2005 — which would make every point-in-time query before 2006 return
nothing, and the sanity backtest impossible to write.
"""

from __future__ import annotations

from datetime import date

import pytest

from findynamics.core.config import SeriesSpec
from findynamics.data.providers.base import Observation
from findynamics.data.vintages import archive_epoch, repair_pre_archive_releases

DAILY = SeriesSpec(id="FRED:DGS10", provider="fred", frequency="daily", publication_lag_days=1)
MONTHLY = SeriesSpec(
    id="FRED:CPIAUCSL", provider="fred", frequency="monthly", publication_lag_days=14
)


def observation(
    obs: str,
    release: str,
    *,
    revision: str | None = None,
    value: float = 1.0,
    frequency: str = "daily",
):
    return Observation(
        series_id="FRED:DGS10",
        provider="fred",
        frequency=frequency,
        unit="%",
        observation_date=date.fromisoformat(obs),
        release_date=date.fromisoformat(release),
        revision_date=date.fromisoformat(revision) if revision else None,
        value=value,
    )


def seeded(*pre_archive: str, epoch: str = "2005-12-20"):
    """Rows sharing one seed vintage, plus a genuine post-epoch print.

    More than one period must share the epoch or it is an ordinary first
    release, not a bulk seed.
    """
    return [observation(obs, epoch) for obs in pre_archive] + [
        observation("2010-01-04", "2010-01-05")
    ]


class TestArchiveEpoch:
    def test_is_the_earliest_release_present(self):
        rows = [observation("1990-01-02", "2005-12-20"), observation("2010-01-04", "2010-01-05")]
        assert archive_epoch(rows) == date(2005, 12, 20)

    def test_is_none_for_no_observations(self):
        assert archive_epoch([]) is None


class TestBulkSeeding:
    """The archive stamped a span of history with the day it started tracking."""

    def test_replaces_the_seed_stamp_with_the_configured_lag(self):
        rows = seeded("1962-01-02", "1962-01-03")

        repaired = repair_pre_archive_releases(rows, DAILY)

        assert repaired[0].release_date == date(1962, 1, 3)
        assert repaired[1].release_date == date(1962, 1, 4)

    def test_measures_the_lag_from_the_end_of_the_period(self):
        """A January monthly figure cannot be published before January ends."""
        rows = [
            observation("1950-01-01", "1960-03-15", frequency="monthly"),
            observation("1950-02-01", "1960-03-15", frequency="monthly"),
            observation("1990-01-01", "1990-02-14", frequency="monthly"),
        ]

        repaired = repair_pre_archive_releases(rows, MONTHLY)

        # End of January + 14 days, not start of January + 14.
        assert repaired[0].release_date == date(1950, 2, 14)

    def test_a_lone_first_print_is_not_a_seed(self):
        """Every series' first row has a period older than its release date."""
        rows = [observation("2010-01-04", "2010-01-05"), observation("2010-01-05", "2010-01-06")]
        assert repair_pre_archive_releases(rows, DAILY) == rows

    def test_leaves_everything_after_the_epoch_untouched(self):
        """The archive is the better authority wherever it has an opinion."""
        rows = seeded("1990-01-02", "1990-01-03") + [observation("2020-06-01", "2020-06-02")]

        repaired = repair_pre_archive_releases(rows, DAILY)

        assert repaired[2].release_date == date(2010, 1, 5)
        assert repaired[3].release_date == date(2020, 6, 2)

    def test_leaves_a_genuine_long_lag_alone(self):
        """Overriding a real publication lag with a guess manufactures lookahead."""
        rows = [
            # Released two weeks after the period, in order, with no artifact:
            # a real lag the archive genuinely recorded.
            observation("2010-01-04", "2010-01-18"),
            observation("2010-01-05", "2010-01-19"),
        ]
        assert repair_pre_archive_releases(rows, DAILY) == rows

    def test_never_moves_a_release_later_than_the_archive_proves(self):
        """Whatever the lag says, it was knowable by the date the archive shows."""
        spec = SeriesSpec(
            id="FRED:DGS10", provider="fred", frequency="daily", publication_lag_days=10_000
        )
        rows = seeded("2005-12-01", "2005-12-02")

        repaired = repair_pre_archive_releases(rows, spec)

        assert repaired[0].release_date == date(2005, 12, 20)


class TestRetroactiveLoading:
    """A discontinued series revived later: its old history was loaded in bulk
    afterwards, so those rows claim a release date decades after the fact.

    This is DGS20 (gap 1987-1993) and DGS30 (gap 2002-2006), whose pre-gap
    history FRED loaded into ALFRED in 2020.
    """

    def test_detects_a_later_period_released_earlier(self):
        rows = [
            observation("1988-01-29", "2020-07-21"),  # loaded retroactively
            observation("1994-01-03", "1994-01-04"),  # published normally
        ]

        repaired = repair_pre_archive_releases(rows, DAILY)

        assert repaired[0].release_date == date(1988, 1, 30)
        assert repaired[1].release_date == date(1994, 1, 4)

    def test_caps_the_repair_at_the_earliest_later_release(self):
        spec = SeriesSpec(
            id="FRED:DGS20", provider="fred", frequency="daily", publication_lag_days=10_000
        )
        rows = [
            observation("1988-01-29", "2020-07-21"),
            observation("1994-01-03", "1994-01-04"),
        ]

        repaired = repair_pre_archive_releases(rows, spec)

        assert repaired[0].release_date == date(1994, 1, 4)

    def test_monotone_publication_is_left_alone(self):
        rows = [
            observation("2010-01-04", "2010-01-05"),
            observation("2010-01-05", "2010-01-06"),
            observation("2010-01-06", "2010-01-07"),
        ]
        assert repair_pre_archive_releases(rows, DAILY) == rows


class TestRevisions:
    def test_the_revision_date_follows_the_release_it_revises(self):
        rows = [
            observation("1990-01-02", "2005-12-20", revision="2005-12-20"),
            observation("1990-01-03", "2005-12-20", revision="2005-12-20"),
            observation("2010-01-04", "2010-01-05", revision="2010-01-05"),
        ]

        repaired = repair_pre_archive_releases(rows, DAILY)

        assert repaired[0].release_date == date(1990, 1, 3)
        assert repaired[0].revision_date == date(1990, 1, 3)
        # The contract forbids a revision before its release.
        assert repaired[0].revision_date >= repaired[0].release_date

    def test_a_genuine_later_revision_survives_the_repair(self):
        rows = [
            observation("1990-01-02", "2005-12-20", revision="2018-04-01"),
            observation("1990-01-03", "2005-12-20"),
            observation("2010-01-04", "2010-01-05"),
        ]

        repaired = repair_pre_archive_releases(rows, DAILY)

        assert repaired[0].release_date == date(1990, 1, 3)
        assert repaired[0].revision_date == date(2018, 4, 1)


class TestEdgeCases:
    def test_no_observations_is_not_an_error(self):
        assert repair_pre_archive_releases([], DAILY) == []

    def test_the_result_still_satisfies_the_observation_contract(self):
        """Observation.__post_init__ rejects a release before its period."""
        rows = seeded("1962-01-02", "1962-01-03")
        for repaired in repair_pre_archive_releases(rows, DAILY):
            assert repaired.release_date >= repaired.observation_date
            if repaired.revision_date is not None:
                assert repaired.revision_date >= repaired.release_date

    def test_the_repair_is_idempotent(self):
        rows = seeded("1962-01-02", "1962-01-03")
        once = repair_pre_archive_releases(rows, DAILY)
        assert repair_pre_archive_releases(once, DAILY) == once


class TestAgainstTheRealSnapshot:
    def test_the_committed_fixture_is_knowable_across_its_whole_span(self, treasury_observations):
        """The repair is what makes a 1988-2026 point-in-time replay possible."""
        early = treasury_observations[treasury_observations["obs_date"] < "1995-01-01"]

        assert len(early) > 0
        # Each early figure is knowable within days of its period, not decades
        # later. The bound is loose enough to leave genuine publication delays
        # alone — the point is that nothing claims a thirty-year lag.
        gap = (early["release_date"] - early["obs_date"]).dt.days
        assert gap.max() <= 21

    def test_a_1990_cutoff_sees_a_curve(self, treasury_observations):
        from findynamics.data.accessor import PandasPITAccessor

        accessor = PandasPITAccessor(treasury_observations, date(1990, 6, 30))
        latest = accessor.latest()

        assert len(latest) >= 5

    @pytest.mark.parametrize("cutoff", [date(1990, 6, 30), date(2000, 6, 30), date(2010, 6, 30)])
    def test_nothing_visible_at_a_cutoff_was_released_after_it(self, treasury_observations, cutoff):
        from findynamics.data.accessor import PandasPITAccessor

        history = PandasPITAccessor(treasury_observations, cutoff).history()
        assert history["release_date"].max().date() <= cutoff
