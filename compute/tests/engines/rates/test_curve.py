"""Curve assembly from point-in-time series.

The interesting cases are all about raggedness: the Treasury curve is missing
tenors on most dates in its history, and the rule "skip the date" would throw
away decades.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from findynamics.data.accessor import PandasPITAccessor
from findynamics.engines.rates.curve import (
    curve_frame,
    curve_on,
    iter_curves,
    latest_curve,
    tenor_map,
)

TENORS = {"FRED:DGS3MO": 0.25, "FRED:DGS2": 2.0, "FRED:DGS5": 5.0, "FRED:DGS10": 10.0}


def observations(rows: list[tuple[str, str, float]]) -> pd.DataFrame:
    """(series_id, obs_date, value); released the next day."""
    return pd.DataFrame(
        [
            {
                "series_id": series_id,
                "obs_date": pd.Timestamp(obs_date),
                "release_date": pd.Timestamp(obs_date) + pd.Timedelta(days=1),
                "revision_date": pd.Timestamp(obs_date) + pd.Timedelta(days=1),
                "value": value,
            }
            for series_id, obs_date, value in rows
        ]
    )


def accessor(frame: pd.DataFrame, as_of: str) -> PandasPITAccessor:
    return PandasPITAccessor(frame, as_of)


class TestTenorMap:
    def test_reads_maturities_from_config(self):
        assert tenor_map({"tenors": {"FRED:DGS10": 10.0}}) == {"FRED:DGS10": 10.0}

    def test_rejects_a_missing_or_empty_map(self):
        with pytest.raises(ValueError, match="non-empty mapping"):
            tenor_map({})

    def test_rejects_a_non_positive_maturity(self):
        with pytest.raises(ValueError, match="must be positive"):
            tenor_map({"tenors": {"FRED:DGS10": 0.0}})


class TestCurveFrame:
    def test_columns_are_maturities_in_ascending_order(self):
        frame = curve_frame(
            accessor(
                observations(
                    [
                        ("FRED:DGS10", "2026-01-05", 4.0),
                        ("FRED:DGS3MO", "2026-01-05", 3.0),
                        ("FRED:DGS2", "2026-01-05", 3.5),
                    ]
                ),
                "2026-01-10",
            ),
            TENORS,
        )
        assert list(frame.columns) == [0.25, 2.0, 10.0]

    def test_keeps_partial_dates_and_leaves_the_gaps_as_nan(self):
        """Forward-filling would invent a quote nobody made."""
        frame = curve_frame(
            accessor(
                observations(
                    [
                        ("FRED:DGS3MO", "2026-01-05", 3.0),
                        ("FRED:DGS10", "2026-01-05", 4.0),
                        ("FRED:DGS3MO", "2026-01-06", 3.1),
                    ]
                ),
                "2026-01-10",
            ),
            TENORS,
        )
        assert len(frame) == 2
        assert pd.isna(frame.loc[pd.Timestamp("2026-01-06"), 10.0])

    def test_drops_dates_where_nothing_quoted(self):
        frame = curve_frame(
            accessor(observations([("FRED:DGS10", "2026-01-05", 4.0)]), "2026-01-10"),
            TENORS,
        )
        assert list(frame.index) == [pd.Timestamp("2026-01-05")]

    def test_respects_the_information_set(self):
        """A quote released after the cutoff is not in the frame at all."""
        frame = curve_frame(
            accessor(
                observations(
                    [("FRED:DGS10", "2026-01-05", 4.0), ("FRED:DGS10", "2026-01-20", 4.5)]
                ),
                "2026-01-10",
            ),
            TENORS,
        )
        assert list(frame.index) == [pd.Timestamp("2026-01-05")]

    def test_empty_input_yields_an_empty_frame_not_an_error(self):
        empty = pd.DataFrame(columns=["series_id", "obs_date", "release_date", "value"])
        assert curve_frame(accessor(empty, "2026-01-10"), TENORS).empty

    def test_series_outside_the_tenor_map_are_ignored(self):
        """The breakeven series is configured but is not a point on the curve."""
        frame = curve_frame(
            accessor(
                observations(
                    [
                        ("FRED:DGS10", "2026-01-05", 4.0),
                        ("FRED:T10YIE", "2026-01-05", 2.3),
                    ]
                ),
                "2026-01-10",
            ),
            TENORS,
        )
        assert list(frame.columns) == [10.0]


class TestCurveOn:
    def test_returns_the_curve_for_an_exact_observation_date(self):
        frame = curve_frame(
            accessor(
                observations(
                    [
                        ("FRED:DGS3MO", "2026-01-05", 3.0),
                        ("FRED:DGS2", "2026-01-05", 3.5),
                        ("FRED:DGS5", "2026-01-05", 3.8),
                        ("FRED:DGS10", "2026-01-05", 4.0),
                    ]
                ),
                "2026-01-10",
            ),
            TENORS,
        )
        curve = curve_on(frame, date(2026, 1, 5), min_tenors=4)

        assert curve is not None
        assert curve.maturities == (0.25, 2.0, 5.0, 10.0)
        assert curve.yields == (3.0, 3.5, 3.8, 4.0)
        assert curve.yield_at(10.0) == 4.0
        assert curve.yield_at(30.0) is None
        assert len(curve) == 4

    def test_a_non_trading_day_has_no_curve(self):
        """Nothing is interpolated onto a Sunday; that would be a fabrication."""
        frame = curve_frame(
            accessor(observations([("FRED:DGS10", "2026-01-05", 4.0)]), "2026-01-10"), TENORS
        )
        assert curve_on(frame, date(2026, 1, 4)) is None

    def test_a_thin_date_is_reported_as_no_curve(self):
        frame = curve_frame(
            accessor(
                observations(
                    [("FRED:DGS3MO", "2026-01-05", 3.0), ("FRED:DGS10", "2026-01-05", 4.0)]
                ),
                "2026-01-10",
            ),
            TENORS,
        )
        assert curve_on(frame, date(2026, 1, 5), min_tenors=4) is None
        assert curve_on(frame, date(2026, 1, 5), min_tenors=2) is not None

    def test_a_zero_yield_is_a_quote_not_a_gap(self):
        """Bills traded at 0.00% for years; dropping them would rewrite 2011."""
        frame = curve_frame(
            accessor(
                observations(
                    [
                        ("FRED:DGS3MO", "2026-01-05", 0.0),
                        ("FRED:DGS2", "2026-01-05", 0.2),
                        ("FRED:DGS5", "2026-01-05", 1.0),
                    ]
                ),
                "2026-01-10",
            ),
            TENORS,
        )
        curve = curve_on(frame, date(2026, 1, 5), min_tenors=3)
        assert curve is not None
        assert curve.yields[0] == 0.0


class TestLatestCurve:
    def test_walks_back_past_a_half_published_day(self):
        """The right answer for a two-tenor Monday is Friday's full curve."""
        frame = curve_frame(
            accessor(
                observations(
                    [
                        ("FRED:DGS3MO", "2026-01-05", 3.0),
                        ("FRED:DGS2", "2026-01-05", 3.5),
                        ("FRED:DGS5", "2026-01-05", 3.8),
                        ("FRED:DGS3MO", "2026-01-06", 3.1),
                    ]
                ),
                "2026-01-10",
            ),
            TENORS,
        )
        curve = latest_curve(frame, min_tenors=3)

        assert curve is not None
        assert curve.as_of == date(2026, 1, 5)

    def test_returns_none_when_no_date_is_usable(self):
        frame = curve_frame(
            accessor(observations([("FRED:DGS10", "2026-01-05", 4.0)]), "2026-01-10"), TENORS
        )
        assert latest_curve(frame, min_tenors=3) is None


class TestIterCurves:
    def test_yields_usable_curves_oldest_first(self):
        frame = curve_frame(
            accessor(
                observations(
                    [
                        ("FRED:DGS3MO", "2026-01-05", 3.0),
                        ("FRED:DGS2", "2026-01-05", 3.5),
                        ("FRED:DGS3MO", "2026-01-06", 3.1),
                        ("FRED:DGS2", "2026-01-06", 3.6),
                        ("FRED:DGS10", "2026-01-07", 4.0),
                    ]
                ),
                "2026-01-10",
            ),
            TENORS,
        )
        curves = list(iter_curves(frame, min_tenors=2))

        assert [c.as_of for c in curves] == [date(2026, 1, 5), date(2026, 1, 6)]


class TestAgainstRealHistory:
    def test_handles_the_dgs1mo_start_in_2001(self, treasury_observations):
        """Before 2001 the curve is ten tenors, after it is eleven."""
        from tests.engines.rates.conftest import TENOR_IDS

        frame = curve_frame(PandasPITAccessor(treasury_observations, "2026-07-30"), TENOR_IDS)

        before = curve_on(frame, date(1995, 5, 31))
        after = latest_curve(frame)

        assert before is not None
        assert 0.08333 not in before.maturities
        assert after is not None
        assert 0.08333 in after.maturities

    def test_every_month_since_1988_produces_a_curve(self, treasury_observations):
        from tests.engines.rates.conftest import TENOR_IDS

        frame = curve_frame(PandasPITAccessor(treasury_observations, "2026-07-30"), TENOR_IDS)
        curves = list(iter_curves(frame))

        assert len(curves) == len(frame)
        assert min(len(c) for c in curves) >= 5
