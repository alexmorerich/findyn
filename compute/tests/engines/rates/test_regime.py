"""Rate-regime classification rules.

Constructed inputs throughout: the point is to pin the branch each combination
of spread, trend and history lands on, so a later refactor cannot quietly
change what "flat" means.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from findynamics.engines.rates.domain import RATE_REGIMES, regime_code
from findynamics.engines.rates.regime import (
    RegimeInputs,
    RegimeRules,
    build_inputs,
    classify,
    classify_history,
)

RULES = RegimeRules()


def inputs(
    spread: float,
    *,
    spread_delta: float | None = None,
    level_delta: float | None = None,
    recently_inverted: bool = False,
) -> RegimeInputs:
    return RegimeInputs(
        spread=spread,
        spread_delta=spread_delta,
        level_delta=level_delta,
        recently_inverted=recently_inverted,
    )


class TestClassify:
    def test_negative_spread_is_inverted(self):
        assert classify(inputs(-0.5), RULES) == "inverted"

    def test_inversion_outranks_flat(self):
        """A curve at -0.05 is both flat by magnitude and inverted by sign."""
        assert classify(inputs(-0.05), RULES) == "inverted"

    def test_exactly_zero_is_not_inverted(self):
        assert classify(inputs(0.0), RULES) == "flat"

    def test_small_positive_spread_is_flat(self):
        assert classify(inputs(0.3), RULES) == "flat"

    def test_steep_with_rising_level_is_tightening(self):
        assert classify(inputs(1.5, level_delta=0.8), RULES) == "steep_tightening"

    def test_steep_with_falling_level_is_easing(self):
        assert classify(inputs(1.5, level_delta=-0.8), RULES) == "steep_easing"

    def test_steep_with_unknown_level_trend_reads_as_easing(self):
        """Ties and unknowns go to the less alarming call, by documented choice."""
        assert classify(inputs(1.5, level_delta=None), RULES) == "steep_easing"
        assert classify(inputs(1.5, level_delta=0.0), RULES) == "steep_easing"

    def test_re_steepening_needs_both_a_recent_inversion_and_a_rising_spread(self):
        rising_after_inversion = inputs(0.4, spread_delta=0.9, recently_inverted=True)
        assert classify(rising_after_inversion, RULES) == "re_steepening"

        # Rising, but the curve has not been inverted in living memory.
        assert classify(inputs(0.4, spread_delta=0.9, recently_inverted=False), RULES) == "flat"

        # Recently inverted, but the spread is not actually rising.
        assert classify(inputs(0.4, spread_delta=0.05, recently_inverted=True), RULES) == "flat"

    def test_re_steepening_outranks_flat_because_the_exit_is_the_signal(self):
        assert (
            classify(inputs(0.1, spread_delta=0.8, recently_inverted=True), RULES)
            == "re_steepening"
        )

    def test_a_currently_inverted_curve_is_inverted_not_re_steepening(self):
        assert classify(inputs(-0.2, spread_delta=0.9, recently_inverted=True), RULES) == "inverted"

    def test_every_branch_is_reachable_and_in_the_vocabulary(self):
        produced = {
            classify(inputs(-0.5), RULES),
            classify(inputs(0.2), RULES),
            classify(inputs(0.4, spread_delta=0.9, recently_inverted=True), RULES),
            classify(inputs(1.5, level_delta=0.8), RULES),
            classify(inputs(1.5, level_delta=-0.8), RULES),
        }
        assert produced == set(RATE_REGIMES)


class TestRules:
    def test_thresholds_come_from_config(self):
        rules = RegimeRules.from_params({"regime": {"flat_below": 1.2, "trend_days": 60}})
        assert rules.flat_below == 1.2
        assert rules.trend_days == 60

    def test_missing_keys_fall_back_to_defaults(self):
        assert RegimeRules.from_params({}) == RegimeRules()

    def test_a_widened_flat_band_reclassifies(self):
        wide = RegimeRules.from_params({"regime": {"flat_below": 2.0}})
        assert classify(inputs(1.5, level_delta=0.8), RULES) == "steep_tightening"
        assert classify(inputs(1.5, level_delta=0.8), wide) == "flat"


def factor_frame(spreads: list[float], levels: list[float] | None = None) -> pd.DataFrame:
    index = pd.date_range("2020-01-01", periods=len(spreads), freq="D", name="obs_date")
    return pd.DataFrame(
        {"spread": spreads, "ns_level": levels if levels is not None else [4.0] * len(spreads)},
        index=index,
    )


class TestBuildInputs:
    def test_reads_only_up_to_the_cutoff(self):
        frame = factor_frame([0.5, 0.6, -0.9])
        built = build_inputs(frame, pd.Timestamp("2020-01-02"), RULES)

        assert built is not None
        assert built.spread == 0.6  # not -0.9, which is in the future

    def test_reports_no_trend_when_history_is_shorter_than_the_window(self):
        built = build_inputs(factor_frame([0.5, 0.6]), pd.Timestamp("2020-01-02"), RULES)
        assert built is not None
        assert built.spread_delta is None
        assert built.level_delta is None

    def test_computes_the_trailing_delta_over_the_window(self):
        rules = RegimeRules(trend_days=2)
        frame = factor_frame([0.1, 0.2, 0.5], levels=[4.0, 4.2, 4.9])
        built = build_inputs(frame, pd.Timestamp("2020-01-03"), rules)

        assert built is not None
        assert built.spread_delta == pytest.approx(0.4)
        assert built.level_delta == pytest.approx(0.9)

    def test_recent_inversion_excludes_today(self):
        """ "Recently inverted" is about where the curve has come from."""
        frame = factor_frame([0.5, 0.5, -0.5])
        today_inverted = build_inputs(frame, pd.Timestamp("2020-01-03"), RULES)
        assert today_inverted is not None
        assert today_inverted.recently_inverted is False

        after = factor_frame([0.5, -0.5, 0.5])
        built = build_inputs(after, pd.Timestamp("2020-01-03"), RULES)
        assert built is not None
        assert built.recently_inverted is True

    def test_an_inversion_outside_the_lookback_does_not_count(self):
        rules = RegimeRules(re_steepening_lookback_days=2)
        index = pd.to_datetime(["2020-01-01", "2020-06-01", "2020-06-02"])
        frame = pd.DataFrame({"spread": [-0.5, 0.4, 0.5], "ns_level": [4.0] * 3}, index=index)

        built = build_inputs(frame, pd.Timestamp("2020-06-02"), rules)
        assert built is not None
        assert built.recently_inverted is False

    def test_empty_history_yields_nothing_rather_than_raising(self):
        assert build_inputs(pd.DataFrame(), pd.Timestamp("2020-01-01"), RULES) is None


class TestClassifyHistory:
    def test_labels_every_date_causally(self):
        labels = classify_history(factor_frame([-0.5, -0.2, 0.4, 1.6]), RegimeRules(trend_days=1))

        assert list(labels[:2]) == ["inverted", "inverted"]
        assert set(labels.dropna()) <= set(RATE_REGIMES)

    def test_a_label_never_depends_on_a_later_date(self):
        """Truncating the future must not change any past label."""
        spreads = [0.6, 0.2, -0.4, -0.6, 0.1, 0.9, 1.7, 1.9]
        rules = RegimeRules(trend_days=2)

        full = classify_history(factor_frame(spreads), rules)
        truncated = classify_history(factor_frame(spreads[:5]), rules)

        assert list(full.iloc[:5]) == list(truncated)

    def test_matches_the_single_date_classifier(self):
        rules = RegimeRules(trend_days=2)
        frame = factor_frame([0.6, 0.2, -0.4, -0.6, 0.1, 0.9])
        labels = classify_history(frame, rules)

        for timestamp in frame.index:
            built = build_inputs(frame, timestamp, rules)
            expected = classify(built, rules) if built else None
            assert labels.loc[timestamp] == expected

    def test_non_finite_spreads_are_left_unlabelled(self):
        frame = factor_frame([0.5, float("nan"), 0.6])
        labels = classify_history(frame, RULES)
        assert labels.iloc[1] is None

    def test_empty_input_is_empty_output(self):
        assert classify_history(pd.DataFrame(), RULES).empty


class TestRegimeCode:
    def test_codes_round_trip_through_the_vocabulary(self):
        for regime in RATE_REGIMES:
            assert RATE_REGIMES[regime_code(regime)] == regime

    def test_an_unknown_regime_is_an_error(self):
        with pytest.raises(ValueError, match="unknown rate regime"):
            regime_code("contango")

    def test_codes_are_finite_floats_for_the_engine_output_table(self):
        assert all(np.isfinite(float(regime_code(r))) for r in RATE_REGIMES)
