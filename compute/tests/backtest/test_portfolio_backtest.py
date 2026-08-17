"""The portfolio walk-forward backtest (task §4).

The metric helpers and PIT state builders are tested directly and cheaply; the
full 2000-2024 walk is a marked-slow smoke test over a short window, because
refitting gold across a century of month-ends is not something to do on every
`pytest` invocation.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from findynamics.backtest import portfolio as bt

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


class TestMetrics:
    def test_max_drawdown_of_a_pure_decline(self):
        # -10% then -10% compounds to a 19% drawdown.
        returns = pd.Series([-0.10, -0.10])
        assert bt._max_drawdown(returns) == pytest.approx(-0.19, abs=1e-9)

    def test_a_monotonic_gain_has_no_drawdown(self):
        assert bt._max_drawdown(pd.Series([0.01, 0.02, 0.03])) == pytest.approx(0.0)

    def test_metrics_annualise(self):
        # Twelve months of +1% each: total ~12.7%, and that is the CAGR too.
        returns = pd.Series([0.01] * 12)
        cash = pd.Series([0.0] * 12)
        m = bt._metrics(returns, cash)
        assert m["cagr"] == pytest.approx(0.01 ** 0 * ((1.01**12) - 1.0), rel=1e-6)
        assert m["months"] == 12
        assert m["max_drawdown"] == pytest.approx(0.0)


class TestPITStateBuilders:
    def test_reduced_form_equity_is_point_in_time(self):
        frame = pd.read_csv(FIXTURES / "equity_prices.csv")
        for col in ("obs_date", "release_date", "revision_date"):
            frame[col] = pd.to_datetime(frame[col])

        state = bt.reduced_form_equity_state(frame, date(2008, 12, 31))
        assert state is not None
        assert state.asset == "equity"
        assert state.as_of == date(2008, 12, 31)
        # Late 2008: trailing 12m return is deeply negative and vol is high, so the
        # expected return is depressed and the risk score elevated.
        assert state.expected_return < 0.05
        assert state.risk_score > 40

    def test_reduced_form_equity_declines_a_thin_history(self):
        frame = pd.read_csv(FIXTURES / "equity_prices.csv")
        for col in ("obs_date", "release_date", "revision_date"):
            frame[col] = pd.to_datetime(frame[col])
        # Before the S&P series begins there is nothing to speak about.
        assert bt.reduced_form_equity_state(frame, date(1860, 1, 31)) is None

    def test_short_rate_state_reads_the_pit_short_rate(self):
        frame = pd.read_csv(FIXTURES / "treasury_monthly.csv")
        for col in ("obs_date", "release_date", "revision_date"):
            frame[col] = pd.to_datetime(frame[col])
        state = bt.short_rate_state(frame, date(2019, 6, 28))
        assert state is not None
        assert state.asset == "money"
        assert 0.0 < state.expected_return < 0.10  # a plausible 2019 short rate
        assert state.risk_score == 0.0


@pytest.mark.slow
class TestWalkForward:
    def test_a_short_walk_beats_the_benchmark_drawdown_through_2008(self):
        """The headline robustness claim, on a short window to stay affordable."""
        result = bt.run_backtest(
            FIXTURES, start=date(2006, 1, 31), end=date(2009, 12, 31)
        )
        assert len(result.dates) >= 24
        # Every month is a valid allocation summing to 1.
        for weights in result.weights:
            assert sum(weights.values()) == pytest.approx(1.0, abs=1e-6)

        gfc = bt.window_metrics(result, date(2007, 10, 1), date(2009, 3, 31))
        # The balanced tilt into gold cushions the 2008 drawdown vs the static mix.
        assert gfc["balanced"]["max_drawdown"] > gfc["benchmark_60_30_10"]["max_drawdown"]

    def test_the_report_and_frame_render(self):
        result = bt.run_backtest(FIXTURES, start=date(2006, 1, 31), end=date(2009, 12, 31))
        report = bt.render_report(result)
        assert "balanced profile vs a static 60/30/10 benchmark" in report
        assert "Not investment advice" in report
        frame = bt.returns_frame(result)
        assert {"balanced_return", "benchmark_return", "w_equity"}.issubset(frame.columns)
