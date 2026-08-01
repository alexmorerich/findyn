"""The driver panel: the two splices, the standardization, and causality."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from findynamics.engines.gold import drivers as drivers_mod
from findynamics.engines.gold.drivers import DriverRules
from tests.engines.gold.conftest import (
    BREAKEVEN_10Y,
    CPI,
    NOMINAL_10Y,
    PRICE,
    SERIES_IDS,
    USD_INDEX,
    USD_INDEX_LEGACY,
    wide_frame,
)

RULES = DriverRules()


@pytest.fixture(scope="module")
def panel(request):
    observations = request.getfixturevalue("gold_observations")
    return drivers_mod.build_panel(wide_frame(observations), SERIES_IDS, RULES)


# --------------------------------------------------------------------------
# Shape
# --------------------------------------------------------------------------


def test_panel_spans_the_whole_lbma_record(panel):
    assert not panel.empty
    assert panel.daily.index[0].date() <= date(1968, 4, 30)
    assert panel.daily.index[-1].date() >= date(2026, 7, 1)


def test_every_driver_and_its_z_score_is_present(panel):
    for name in drivers_mod.DRIVER_COLUMNS:
        assert name in panel.daily.columns
        assert f"z_{name}" in panel.daily.columns


def test_the_price_is_the_spine(panel, gold_observations):
    """The panel speaks about dates gold actually fixed on, not calendar days."""
    frame = wide_frame(gold_observations)
    fixings = frame[PRICE].dropna()
    assert panel.daily.index.equals(fixings.index)


def test_monthly_returns_are_month_end_log_changes(panel):
    monthly = panel.monthly
    assert "ret" in monthly
    assert monthly["ret"].notna().all()
    # Sanity: a 58-year history of monthly gold returns.
    assert 500 < len(monthly) < 800
    assert abs(monthly["ret"].mean()) < 2.0


def test_an_absent_optional_driver_leaves_a_column_of_nan(panel):
    """`equity_rii` is not in the snapshot — the panel must still build."""
    assert panel.available["equity_rii"] is False
    assert panel.daily["equity_rii"].isna().all()
    assert panel.available["real_rate"] is True


def test_no_price_means_no_panel():
    empty = drivers_mod.build_panel(pd.DataFrame(), SERIES_IDS, RULES)
    assert empty.empty
    assert empty.explain() == {}


# --------------------------------------------------------------------------
# The real-rate splice
# --------------------------------------------------------------------------


def test_real_rate_uses_the_breakeven_once_tips_exist(gold_observations):
    """After 2003 the rate must be exactly nominal minus breakeven."""
    frame = wide_frame(gold_observations)
    rate, is_ex_post = drivers_mod.real_rate(frame, SERIES_IDS, RULES)

    day = pd.Timestamp("2020-06-15")
    expected = frame[NOMINAL_10Y].ffill()[day] - frame[BREAKEVEN_10Y].ffill()[day]
    assert rate[day] == pytest.approx(expected)
    assert not is_ex_post[day]


def test_real_rate_falls_back_to_realized_inflation_before_tips(gold_observations):
    frame = wide_frame(gold_observations)
    rate, is_ex_post = drivers_mod.real_rate(frame, SERIES_IDS, RULES)

    day = pd.Timestamp("1990-06-15")
    assert np.isfinite(rate[day])
    assert is_ex_post[day]
    # 1990: 10y nominal ~8.5%, CPI ~4.7% -> a real rate near 4%.
    assert 1.0 < rate[day] < 7.0


def test_the_splice_is_flagged_rather_than_blended(panel):
    """A reader must be able to tell which definition produced a date."""
    assert 0.4 < panel.ex_post_share < 0.8


def test_real_rate_survives_the_breakeven_being_absent(gold_observations):
    frame = wide_frame(gold_observations)
    ids = {k: v for k, v in SERIES_IDS.items() if k != "breakeven_10y"}
    rate, is_ex_post = drivers_mod.real_rate(frame, ids, RULES)
    assert rate.notna().any()
    assert is_ex_post[rate.notna()].all()


def test_no_nominal_yield_means_no_real_rate():
    rate, _ = drivers_mod.real_rate(pd.DataFrame(), SERIES_IDS, RULES)
    assert rate.empty


# --------------------------------------------------------------------------
# The dollar splice
# --------------------------------------------------------------------------


def test_usd_trend_splices_the_change_not_the_level(gold_observations):
    """The two indices sit at different levels; splicing those would print a step.

    The assertion is that the spliced trend has no discontinuity at the 2006
    handover — a level splice between DTWEXM (~85) and DTWEXBGS (~105) would put
    a ~20% one-day move there, which is the failure this is guarding.
    """
    frame = wide_frame(gold_observations)
    trend = drivers_mod.usd_trend(frame, SERIES_IDS, RULES).dropna()

    handover = trend.loc["2005-06":"2007-06"]
    assert handover.diff().abs().max() < 0.05, "the splice put a step change in the trend"

    # And the levels really are different, so the guard is not vacuous.
    levels = frame[[USD_INDEX, USD_INDEX_LEGACY]].loc["2006-01-03":"2006-01-31"].dropna()
    assert abs(levels[USD_INDEX].mean() - levels[USD_INDEX_LEGACY].mean()) > 5.0


def test_usd_trend_prefers_the_broad_index_where_both_exist(gold_observations):
    frame = wide_frame(gold_observations)
    spliced = drivers_mod.usd_trend(frame, SERIES_IDS, RULES)
    broad_only = drivers_mod.usd_trend(
        frame, {k: v for k, v in SERIES_IDS.items() if k != "usd_index_legacy"}, RULES
    )
    overlap = broad_only.dropna().index
    pd.testing.assert_series_equal(spliced.loc[overlap], broad_only.loc[overlap])


def test_usd_trend_survives_either_index_being_absent(gold_observations):
    frame = wide_frame(gold_observations)
    for dropped in ("usd_index", "usd_index_legacy"):
        ids = {k: v for k, v in SERIES_IDS.items() if k != dropped}
        assert drivers_mod.usd_trend(frame, ids, RULES).notna().any()


# --------------------------------------------------------------------------
# Causality — the no-lookahead law, asserted on the transforms
# --------------------------------------------------------------------------


def test_the_panel_is_a_function_of_its_own_past(gold_observations):
    """Truncating the input must not change any earlier date's driver values.

    Every transform here is expanding or trailing, so this holds exactly.
    Without it the PIT replay test could not pass at all: if an expanding mean
    were a full-sample mean, every z-score in history would move whenever a new
    observation arrived.

    The truncation is applied to the **observation frame**, not to the accessor
    cutoff. Moving the cutoff would also change which *vintage* of each figure is
    visible — DGS10 and CPI are both revised — and the resulting differences
    would be point-in-time working correctly rather than a transform reaching
    forward. Holding the vintages fixed and varying only the dates is what
    isolates the property being asserted.
    """
    frame = wide_frame(gold_observations, date(2026, 7, 31))
    cutoff = pd.Timestamp("2015-06-30")

    full = drivers_mod.build_panel(frame, SERIES_IDS, RULES)
    early = drivers_mod.build_panel(frame.loc[:cutoff], SERIES_IDS, RULES)

    shared = early.daily.index
    assert len(shared) > 10000
    for column in ("real_rate", "usd_trend", "z_stress", "z_real_rate_change_12m"):
        pd.testing.assert_series_equal(
            full.daily.loc[shared, column],
            early.daily[column],
            check_names=False,
        )


def test_z_scores_are_clipped_but_not_winsorized_away(panel):
    z = panel.daily["z_real_rate_change_12m"].dropna()
    assert z.min() >= -RULES.z_clip
    assert z.max() <= RULES.z_clip
    # 1980 must still register as extreme rather than being averaged flat.
    assert panel.daily["z_real_rate_change_12m"].loc["1980":"1982"].max() > 1.5


def test_z_scores_wait_for_a_real_sample(panel):
    """A z-score over twenty observations is a number, not a standardization."""
    early = panel.daily["z_stress"].iloc[: RULES.z_min_observations - 1]
    assert early.isna().all()


def test_explain_reports_the_newest_row(panel):
    explained = panel.explain()
    assert "real_rate" in explained
    assert "z_stress" in explained
    assert all(np.isfinite(v) for v in explained.values())


def test_config_rejects_a_non_mapping_block():
    with pytest.raises(ValueError, match="must be a mapping"):
        DriverRules.from_params({"drivers": "nope"})


def test_config_overrides_reach_the_rules():
    rules = DriverRules.from_params({"drivers": {"trend_days": 126, "z_clip": 3.0}})
    assert rules.trend_days == 126
    assert rules.z_clip == 3.0
    assert CPI in SERIES_IDS.values()
