"""The observable block: what each transform does, and what it refuses to do.

Synthetic where a closed form is the assertion — a constant series has no
volatility, a monotone ramp has no drawdown — and real where the question is
whether the availability check does its job on the data the track will actually
run on.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from findynamics.research.omega import build_feature_frame
from findynamics.research.omega.features import (
    FeatureParams,
    OmegaFeatureError,
    available_columns,
)
from tests.research.omega.conftest import (
    SNAPSHOT_AS_OF,
    accessor_at,
    observation_rows,
    price_frame,
    price_only_config,
    with_params,
)


def test_a_constant_price_has_no_realized_volatility(omega_config):
    """Zero, not a small number: a flat tape has no dispersion to measure."""
    config = price_only_config(omega_config)
    frame = build_feature_frame(accessor_at(price_frame([100.0] * 600), date(2003, 1, 1)), config)

    assert np.nanmax(frame["realized_vol"].to_numpy()) == 0.0
    assert np.nanmax(frame["vol_of_vol"].to_numpy()) == 0.0


def test_a_monotone_ramp_never_draws_down(omega_config):
    """``cummax`` tracks a rising series exactly, so the depth below it is zero.

    The test that would fail if ``drawdown`` were ever computed against a global
    maximum: a ramp's global max is its last point, and every earlier date would
    show a drawdown that no one could have measured at the time.
    """
    config = price_only_config(omega_config)
    ramp = [100.0 * (1.0005**i) for i in range(600)]
    frame = build_feature_frame(accessor_at(price_frame(ramp), date(2003, 1, 1)), config)

    assert np.nanmax(frame["drawdown"].to_numpy()) == 0.0


def test_one_level_shift_moves_abs_return_on_exactly_one_date(omega_config):
    """The EWMA jumps once and then decays; it never rises again on its own."""
    config = price_only_config(omega_config)
    values = [100.0] * 300 + [120.0] * 300
    frame = build_feature_frame(accessor_at(price_frame(values), date(2003, 1, 1)), config)

    changes = frame["abs_return"].dropna().diff().dropna()
    rises = changes[changes > 1e-12]
    assert len(rises) == 1, f"abs_return rose on {len(rises)} dates, expected exactly one"
    # And every later change is a decay, which is what distinguishes a smoothed
    # absolute return from one that keeps re-firing on its own history.
    after = changes[changes.index > rises.index[0]]
    assert (after <= 1e-15).all()


def test_drawdown_is_a_non_negative_depth(omega_config, omega_observations):
    """Sign convention, asserted rather than left to a reader of the source.

    ``drawdown`` is the *depth* below the running max, so a bigger number is a
    worse drawdown. Getting this backwards would invert the column's loading and
    the resulting Ω would still look entirely plausible.
    """
    frame = build_feature_frame(accessor_at(omega_observations), omega_config)
    column = frame["drawdown"].dropna()

    assert (column >= 0.0).all()
    assert (
        column.loc["2009-03-01":"2009-03-31"].mean() > column.loc["2006-01-01":"2006-12-31"].mean()
    )


def test_the_thin_credit_column_is_dropped_and_named(omega_config, omega_observations):
    """The finding the whole availability check exists for.

    ``FRED:BAMLH0A0HYM2`` starts 2023-08-01 in the committed fixture. Over the
    2000-2026 primary window it covers 730 of 6,683 rows — 11% — so it is
    dropped. The number that matters is what happens without the check: a plain
    ``dropna()`` over all nine columns leaves **730 rows starting 2023-08-31**,
    an Ω with no 2008 and no 2020 in it, and nothing anywhere saying so.
    """
    frame = build_feature_frame(accessor_at(omega_observations), omega_config)
    params = FeatureParams.from_config(omega_config)

    surviving, dropped = available_columns(
        frame, params.min_column_observations, params.min_column_coverage
    )

    assert dropped == ("credit_velocity",)
    assert "credit_velocity" not in surviving
    assert len(surviving) == 8

    naive = frame.dropna()
    kept = frame[list(surviving)].dropna()
    assert len(naive) < 1000
    assert naive.index[0].year == 2023
    assert len(kept) > 6000
    assert kept.index[0].year == 2000


def test_a_column_short_in_absolute_terms_is_dropped_too(omega_config, omega_observations):
    """Both tests apply, and the absolute floor is not decorative.

    Raising the floor above every column's count drops all of them, which is the
    degenerate case the estimator has to refuse rather than fit.
    """
    frame = build_feature_frame(accessor_at(omega_observations), omega_config)

    surviving, dropped = available_columns(frame, min_observations=10_000, min_coverage=0.0)

    assert surviving == ()
    assert len(dropped) == len(frame.columns)


def test_an_unconfigured_role_becomes_an_all_nan_column_not_a_missing_one(omega_config):
    """The frame keeps its width so the drop is *reported* rather than implied.

    A frame that quietly narrowed would put the same information in
    ``len(columns)``, where nothing reads it, instead of in
    ``OmegaSpec.dropped``, where the report prints it.
    """
    config = price_only_config(omega_config)
    frame = build_feature_frame(accessor_at(price_frame([100.0] * 600), date(2003, 1, 1)), config)

    assert list(frame.columns) == [
        "realized_vol",
        "vol_of_vol",
        "abs_return",
        "drawdown",
        "dispersion",
        "rate_level_chg",
        "curve_slope",
        "liquidity_stress",
        "credit_velocity",
    ]
    for column in ("dispersion", "rate_level_chg", "curve_slope", "liquidity_stress"):
        assert frame[column].isna().all()


def test_a_weekly_series_is_placed_on_the_calendar_by_its_release_date(omega_config):
    """The reason this module does not use ``PITAccessor.wide()``.

    ``FRED:NFCI``'s median publication lag in the committed fixture is 13 days.
    An observation-date alignment would put a reading into the feature frame
    nine trading days before anyone could have read it — invisible, and enough
    to make every historical Ω slightly too well informed.
    """
    prices = price_frame([100.0 + i for i in range(40)], start=date(2020, 1, 1))
    # One weekly observation, published a fortnight after the week it describes.
    liquidity = pd.DataFrame(observation_rows("SYNTH:NFCI", {date(2020, 1, 8): 1.5}, lag_days=14))
    observations = pd.concat([prices, liquidity], ignore_index=True)

    config = with_params(
        omega_config,
        series={"price": "SYNTH:PRICE", "liquidity": "SYNTH:NFCI"},
        features={"start": None, "max_staleness_days": 21},
    )
    frame = build_feature_frame(accessor_at(observations, date(2020, 3, 1)), config)
    column = frame["liquidity_stress"]

    release = pd.Timestamp(2020, 1, 22)
    assert column.loc[column.index < release].isna().all(), (
        "the NFCI reading was visible before its release date"
    )
    assert column.loc[release] == 1.5


def test_a_value_stops_standing_in_once_it_goes_stale(omega_config):
    """Forward fill is for the days between releases, not for a dead series."""
    prices = price_frame([100.0 + i for i in range(120)], start=date(2020, 1, 1))
    liquidity = pd.DataFrame(observation_rows("SYNTH:NFCI", {date(2020, 1, 6): 1.5}, lag_days=0))
    observations = pd.concat([prices, liquidity], ignore_index=True)

    config = with_params(
        omega_config,
        series={"price": "SYNTH:PRICE", "liquidity": "SYNTH:NFCI"},
        features={"start": None, "max_staleness_days": 21},
    )
    column = build_feature_frame(accessor_at(observations, date(2020, 7, 1)), config)[
        "liquidity_stress"
    ]

    assert column.loc[pd.Timestamp(2020, 1, 20)] == 1.5
    assert pd.isna(column.loc[pd.Timestamp(2020, 3, 2)])


def test_a_gap_in_one_column_does_not_blank_the_others(omega_config, omega_observations):
    """The documented fill rule, and its blast radius.

    A late or missing input costs the rows it actually covers and nothing else.
    The alternative — one NaN propagating across the frame — would make Ω
    disappear on every date any one of six series happened to be late.
    """
    frame = build_feature_frame(accessor_at(omega_observations), omega_config)

    # credit_velocity is absent for 5,953 of 6,683 rows and the rest of the
    # block is unaffected on exactly those dates.
    blank = frame["credit_velocity"].isna()
    assert blank.sum() > 5_000
    assert frame.loc[blank, "realized_vol"].notna().mean() > 0.99


def test_a_price_the_accessor_cannot_supply_is_a_named_failure(omega_config):
    """Ω has no spine without a price, and says so rather than returning empty."""
    config = with_params(omega_config, series={"price": "SYNTH:ABSENT"})
    empty = pd.DataFrame(
        columns=["series_id", "obs_date", "release_date", "revision_date", "value"]
    )

    with pytest.raises(OmegaFeatureError, match="no knowable observations"):
        build_feature_frame(accessor_at(empty, SNAPSHOT_AS_OF), config)


def test_an_unknown_column_name_fails_at_config_load(omega_config):
    """A typo in ``features.columns`` must not silently build eight columns."""
    config = with_params(omega_config, features={"columns": ["realized_vol", "complexity"]})

    with pytest.raises(OmegaFeatureError, match="unknown column"):
        FeatureParams.from_config(config)


def test_a_negative_staleness_budget_is_refused(omega_config):
    """It would be lookahead, and it is the kind of typo a minus sign makes."""
    config = with_params(omega_config, features={"max_staleness_days": -5})

    with pytest.raises(OmegaFeatureError, match="lookahead"):
        FeatureParams.from_config(config)


def test_a_non_positive_window_is_refused(omega_config):
    """It would not raise downstream — it would quietly become two observations.

    ``FeatureParams.periods`` floors at two, so a zero month count produces a
    two-day realized-volatility window under a config that says one month. That
    is the class of failure config validation exists to catch: not a crash, a
    transform that silently is not the one described.
    """
    config = with_params(omega_config, features={"realized_vol_months": 0.0})

    with pytest.raises(OmegaFeatureError, match="must be > 0"):
        FeatureParams.from_config(config)
