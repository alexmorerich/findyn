"""Factor scoring — the Layer 0 pipeline.

The property that matters is causality: the score of a 2008 reading must be its
rank among 1960-2008 and nothing later. A scoring function that quietly uses the
full sample looks fine on every spot check and inflates every backtest.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from findynamics.core.config import FactorSpec, SeriesSpec
from findynamics.core.contracts.state import FactorState
from findynamics.data.accessor import PandasPITAccessor
from findynamics.factors.compute import (
    MIN_OBSERVATIONS,
    WINSOR_Z,
    _expanding_percentile,
    _expanding_z,
    compute_factors,
    score_factor,
    score_series,
)


def series(values: list[float]) -> pd.Series:
    return pd.Series(values, index=pd.date_range("2000-01-01", periods=len(values), freq="D"))


def spec(*series_specs: SeriesSpec, name: str = "liquidity") -> FactorSpec:
    return FactorSpec(name=name, weight=1.0, series=tuple(series_specs))


def series_spec(series_id: str, direction: int = 1) -> SeriesSpec:
    return SeriesSpec(
        id=series_id,
        provider="fred",
        frequency="daily",
        publication_lag_days=1,
        direction=direction,
    )


class TestExpandingZ:
    def test_uses_only_the_history_up_to_each_point(self):
        """Row i sees rows 0..i. If it saw more, the early z-scores would move."""
        values = series([1.0, 2.0, 3.0, 100.0])
        z = _expanding_z(values)

        # The outlier at the end must not have changed the third point's score.
        assert z.iloc[2] == pytest.approx(_expanding_z(series([1.0, 2.0, 3.0])).iloc[2])

    def test_winsorizes_to_the_documented_bound(self):
        z = _expanding_z(series([1.0] * 30 + [10_000.0]))
        assert z.max() <= WINSOR_Z
        assert z.min() >= -WINSOR_Z

    def test_a_constant_history_has_no_scale_and_scores_zero(self):
        assert (_expanding_z(series([5.0] * 10)) == 0.0).all()


class TestExpandingPercentile:
    def test_the_first_point_has_no_history_and_sits_at_the_midpoint(self):
        assert _expanding_percentile(series([4.0])).iloc[0] == 0.5

    def test_a_rising_series_scores_at_the_top_each_time(self):
        pct = _expanding_percentile(series([1.0, 2.0, 3.0, 4.0]))
        assert list(pct[1:]) == [1.0, 1.0, 1.0]

    def test_a_falling_series_scores_at_the_bottom(self):
        pct = _expanding_percentile(series([4.0, 3.0, 2.0, 1.0]))
        assert list(pct[1:]) == [0.0, 0.0, 0.0]

    def test_ties_take_the_midpoint_of_their_range(self):
        pct = _expanding_percentile(series([1.0, 1.0, 1.0]))
        assert pct.iloc[1] == pytest.approx(0.5)
        assert pct.iloc[2] == pytest.approx(0.5)

    def test_matches_a_naive_expanding_rank(self):
        """The Fenwick tree is an optimization; it must agree with the obvious way."""
        rng = np.random.default_rng(7)
        values = rng.normal(size=200).round(3)
        fast = _expanding_percentile(series(list(values)))

        for i in range(1, len(values)):
            history = values[:i]
            below = np.sum(history < values[i])
            equal = np.sum(history == values[i])
            expected = (below + (below + equal)) / 2.0 / i
            assert fast.iloc[i] == pytest.approx(expected)

    def test_appending_the_future_never_changes_a_past_score(self):
        rng = np.random.default_rng(11)
        values = list(rng.normal(size=120).round(3))

        short = _expanding_percentile(series(values[:60]))
        long = _expanding_percentile(series(values))

        assert list(short) == pytest.approx(list(long[:60]))


class TestScoreSeries:
    def test_produces_0_to_100(self):
        rng = np.random.default_rng(3)
        scored = score_series(series(list(rng.normal(size=200))), 1)
        assert scored.min() >= 0.0
        assert scored.max() <= 100.0

    def test_direction_flips_the_axis(self):
        values = series(list(np.linspace(0, 10, 100)))
        up = score_series(values, 1)
        down = score_series(values, -1)
        assert up.iloc[-1] == pytest.approx(100.0 - down.iloc[-1])

    def test_too_little_history_scores_nothing(self):
        """A percentile from a handful of points is noise wearing a number."""
        assert score_series(series([1.0] * (MIN_OBSERVATIONS - 1)), 1).empty
        assert not score_series(series(list(range(MIN_OBSERVATIONS))), 1).empty

    def test_nans_are_dropped_not_scored(self):
        values = series([1.0, float("nan")] * 40)
        scored = score_series(values, 1)
        assert len(scored) == 40
        assert scored.notna().all()


def wide(**columns: list[float]) -> pd.DataFrame:
    length = len(next(iter(columns.values())))
    return pd.DataFrame(
        columns, index=pd.date_range("2000-01-01", periods=length, freq="D", name="obs_date")
    )


class TestScoreFactor:
    def test_averages_its_series_and_records_the_trace(self):
        frame = wide(
            **{"FRED:A": list(np.linspace(0, 10, 60)), "FRED:B": list(np.linspace(10, 0, 60))}
        )
        state = score_factor(
            spec(series_spec("FRED:A"), series_spec("FRED:B")), frame, date(2000, 3, 1)
        )

        assert isinstance(state, FactorState)
        assert 0.0 <= state.score <= 100.0
        assert "FRED:A" in state.components
        assert "FRED:B" in state.components

    def test_carries_the_raw_level_beside_the_score(self):
        """A dashboard that can only show "liquidity is 34" cannot show why."""
        frame = wide(**{"FRED:A": list(np.linspace(0, 10, 60))})
        state = score_factor(spec(series_spec("FRED:A")), frame, date(2000, 3, 1))

        assert state is not None
        assert state.components["FRED:A:level"] == pytest.approx(10.0)

    def test_a_missing_series_degrades_the_factor_rather_than_failing(self):
        frame = wide(**{"FRED:A": list(np.linspace(0, 10, 60))})
        state = score_factor(
            spec(series_spec("FRED:A"), series_spec("FRED:MISSING")), frame, date(2000, 3, 1)
        )

        assert state is not None
        assert "FRED:MISSING" not in state.components

    def test_a_factor_with_nothing_usable_is_omitted(self):
        state = score_factor(
            spec(series_spec("FRED:MISSING")), wide(**{"FRED:A": [1.0] * 60}), date(2000, 3, 1)
        )
        assert state is None

    def test_direction_puts_a_headwind_at_the_bottom_of_the_axis(self):
        """Every factor score reads on one axis: 100 is maximally supportive."""
        rising = wide(**{"FRED:A": list(np.linspace(0, 10, 60))})

        supportive = score_factor(spec(series_spec("FRED:A", 1)), rising, date(2000, 3, 1))
        headwind = score_factor(spec(series_spec("FRED:A", -1)), rising, date(2000, 3, 1))

        assert supportive is not None and headwind is not None
        assert supportive.score > 90
        assert headwind.score < 10


class TestComputeFactors:
    def _observations(self, series_ids: list[str], n: int = 80) -> pd.DataFrame:
        rows = []
        for i in range(n):
            obs_date = pd.Timestamp("2000-01-01") + pd.Timedelta(days=i)
            for series_id in series_ids:
                rows.append(
                    {
                        "series_id": series_id,
                        "obs_date": obs_date,
                        "release_date": obs_date + pd.Timedelta(days=1),
                        "revision_date": obs_date + pd.Timedelta(days=1),
                        "value": float(i),
                    }
                )
        return pd.DataFrame(rows)

    def test_scores_the_configured_factors_from_a_pit_accessor(self, config):
        ids = sorted({s.id for spec_ in config.factors.values() for s in spec_.series})
        accessor = PandasPITAccessor(self._observations(ids), date(2000, 4, 1))

        states = compute_factors(accessor, config)

        assert set(states) == set(config.factors)
        for name, state in states.items():
            assert state.name == name
            assert state.as_of == date(2000, 4, 1)
            assert 0.0 <= state.score <= 100.0

    def test_takes_its_information_set_from_the_accessor_alone(self, config):
        """The function never sees a date, so it cannot be pointed at the wrong one."""
        ids = sorted({s.id for spec_ in config.factors.values() for s in spec_.series})
        observations = self._observations(ids)

        early = compute_factors(PandasPITAccessor(observations, date(2000, 2, 1)), config)
        late = compute_factors(PandasPITAccessor(observations, date(2000, 3, 20)), config)

        assert early["liquidity"].as_of == date(2000, 2, 1)
        assert late["liquidity"].as_of == date(2000, 3, 20)

    def test_factors_with_no_data_are_left_out_rather_than_zeroed(self, config):
        """A missing factor and a factor scoring zero are different statements."""
        accessor = PandasPITAccessor(self._observations(["FRED:M2SL"]), date(2000, 4, 1))

        states = compute_factors(accessor, config)

        assert "liquidity" in states
        assert "valuation" not in states

    def test_the_new_p1_factors_are_in_the_vocabulary(self, config):
        assert "real_rate" in config.factors
        assert "usd_strength" in config.factors

    def test_real_rate_reads_nominal_against_breakeven(self, config):
        ids = {s.id for s in config.factors["real_rate"].series}
        assert ids == {"FRED:DGS10", "FRED:T10YIE"}

        directions = {s.id: s.direction for s in config.factors["real_rate"].series}
        # On the supportive axis, a higher nominal yield is a headwind and a
        # higher breakeven (lower real rate) is support.
        assert directions["FRED:DGS10"] == -1
        assert directions["FRED:T10YIE"] == 1
