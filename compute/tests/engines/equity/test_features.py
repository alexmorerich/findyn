"""The causal feature path: Kalman, FFD, kinematics.

Split by what each kind of input can prove. The FFD weights and the filter's
causality have closed forms or exact properties, so they are asserted against
synthetic series where the right answer is known independently. Anything whose
answer depends on what markets actually did — which ``d`` a series needs, whether
one ``d`` could have served two series — is asserted against the committed
snapshot.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from findynamics.engines.equity.features.ffd import (
    FfdError,
    adf_p_value,
    frac_diff,
    frac_diff_weights,
    search_d,
)
from findynamics.engines.equity.features.kalman import (
    InsufficientHistoryError,
    KalmanParams,
    filter_state,
    fit_params,
)
from findynamics.engines.equity.features.kinematics import (
    JERK_ELEVATED_Z,
    JERK_EXTREME_Z,
    baseline_window,
    expanding_z,
    jerk_lamp,
    kinematics,
)
from findynamics.engines.equity.features.pipeline import (
    CORE_COLUMNS,
    FeatureParams,
    FrozenFeatureParams,
    compute_features,
)
from findynamics.engines.equity.prices import PriceSeries

DAILY = PriceSeries(
    role="publication",
    source_role="primary",
    series_id="TEST:DAILY",
    frequency="daily",
    observations=1500,
)
MONTHLY = PriceSeries(
    role="deep_history",
    source_role="deep_history",
    series_id="TEST:MONTHLY",
    frequency="monthly",
    observations=1500,
)


def walk(n: int = 1500, *, drift: float = 0.0003, vol: float = 0.01, seed: int = 3) -> pd.Series:
    rng = np.random.default_rng(seed)
    index = pd.bdate_range("2010-01-04", periods=n)
    return pd.Series(1000.0 * np.exp(np.cumsum(rng.normal(drift, vol, n))), index=index)


# --- FFD weights: closed form ----------------------------------------------


def test_d_zero_is_the_identity():
    """(1-B)^0 = 1: one weight, and the series comes back unchanged."""
    weights = frac_diff_weights(0.0)
    assert weights.tolist() == [1.0]

    series = walk(200)
    np.testing.assert_allclose(frac_diff(series, 0.0).to_numpy(), series.to_numpy())


def test_d_one_is_the_first_difference():
    """(1-B)^1 = 1 - B, so the weights are exactly [-1, 1] oldest-first."""
    np.testing.assert_allclose(frac_diff_weights(1.0), [-1.0, 1.0])

    series = walk(200)
    differenced = frac_diff(series, 1.0)
    np.testing.assert_allclose(differenced.to_numpy()[1:], series.diff().to_numpy()[1:], atol=1e-12)
    assert math.isnan(differenced.iloc[0]), "no full window on the first position"


def test_weights_follow_the_binomial_recursion():
    d = 0.4
    weights = frac_diff_weights(d, threshold=1e-4)[::-1]  # newest-first
    assert weights[0] == 1.0
    for k in range(1, len(weights)):
        assert weights[k] == pytest.approx(-weights[k - 1] * (d - k + 1.0) / k)


def test_weights_alternate_and_shrink_for_fractional_d():
    weights = frac_diff_weights(0.4, threshold=1e-4)[::-1]
    assert all(w < 0 for w in weights[1:]), "fractional d gives one positive lead weight"
    assert all(abs(weights[k]) < abs(weights[k - 1]) for k in range(2, len(weights)))


def test_the_memory_budget_caps_the_window():
    """Without a cap, d=0.3 needs thousands of lags to clear 1e-5."""
    uncapped = frac_diff_weights(0.3, threshold=1e-5)
    capped = frac_diff_weights(0.3, threshold=1e-5, max_width=252)
    assert len(uncapped) > 1000
    assert len(capped) == 252
    # The cap only truncates; the weights that survive are unchanged.
    np.testing.assert_allclose(capped, uncapped[-252:])


def test_the_window_is_fixed_width_not_expanding():
    """Every value uses the same number of lags, so dates are comparable."""
    series = walk(600)
    differenced = frac_diff(series, 0.4, threshold=1e-4)
    width = len(frac_diff_weights(0.4, threshold=1e-4))
    assert differenced.iloc[: width - 1].isna().all()
    assert differenced.iloc[width - 1 :].notna().all()


def test_a_value_depends_only_on_its_own_trailing_window():
    """The causality property, checked directly: changing a *later* observation
    must not move an earlier FFD value."""
    series = walk(400)
    before = frac_diff(series, 0.4, threshold=1e-3)

    tampered = series.copy()
    tampered.iloc[-1] *= 1.5
    after = frac_diff(tampered, 0.4, threshold=1e-3)

    pd.testing.assert_series_equal(before.iloc[:-1], after.iloc[:-1])


def test_negative_d_is_rejected():
    with pytest.raises(FfdError, match="non-negative"):
        frac_diff_weights(-0.1)


def test_search_returns_the_smallest_passing_d():
    """Smallest, not lowest p-value: memory is what the transform is for."""
    series = np.log(walk(1200))
    fit = search_d(series, series_id="TEST:DAILY", max_width=252)

    assert fit.passed
    assert fit.p_value < 0.05
    # Everything below the chosen d failed, or it was not the smallest.
    for d in [round(x, 2) for x in np.arange(0.0, fit.d, 0.05)]:
        assert adf_p_value(frac_diff(series, d, max_width=252)) >= 0.05


def test_a_failed_search_says_so_rather_than_returning_a_passing_looking_d():
    flat = pd.Series(np.linspace(0.0, 1.0, 300), index=pd.bdate_range("2020-01-01", periods=300))
    fit = search_d(flat, series_id="TEST:FLAT", grid=(0.0, 0.05), max_width=64)
    assert not fit.passed


# --- Kalman: filtered only --------------------------------------------------


def test_the_filter_is_causal():
    """The property the whole feature path rests on: a future observation cannot
    change a past filtered value. This is what the RTS smoother would break."""
    series = np.log(walk(700))
    params, _ = fit_params(series)

    before = filter_state(series, params)
    tampered = series.copy()
    tampered.iloc[-1] += 0.25
    after = filter_state(tampered, params)

    np.testing.assert_allclose(before.level.to_numpy()[:-1], after.level.to_numpy()[:-1])
    np.testing.assert_allclose(before.slope.to_numpy()[:-1], after.slope.to_numpy()[:-1])


def test_filtering_never_computes_a_smoothed_state():
    """Structural, not conventional.

    ``filter()`` runs the forward recursion only, so the results object carries
    no smoothed state — ``smoothed_state`` is ``None``, not an array. Had the
    module used ``smooth()``, this would be populated, and every downstream
    feature would silently improve by having seen the future.
    """
    series = np.log(walk(700))
    params, _ = fit_params(series)
    state = filter_state(series, params)
    assert len(state) == len(series)

    from findynamics.engines.equity.features.kalman import _model

    results = _model(series).filter(params.to_array())
    assert results.filtered_state is not None
    assert results.smoothed_state is None, "the feature path must never smooth"


def test_the_filter_tracks_a_deterministic_trend():
    """A noiseless exponential has slope log(1+g) per period, and the filter
    should converge to it once the diffuse start-up has washed out."""
    growth = 0.0004
    index = pd.bdate_range("2015-01-01", periods=800)
    series = pd.Series(np.log(100.0) + growth * np.arange(800), index=index)

    state = filter_state(series, KalmanParams(irregular=1e-8, level=1e-10, trend=1e-14))
    assert state.slope.iloc[-1] == pytest.approx(growth, abs=5e-5)
    assert state.level.iloc[-1] == pytest.approx(series.iloc[-1], abs=1e-3)


def test_too_short_a_series_cannot_identify_three_variances():
    with pytest.raises(InsufficientHistoryError):
        fit_params(np.log(walk(100)))


def test_frozen_parameters_reproduce_the_path_exactly():
    series = np.log(walk(700))
    params, _ = fit_params(series)
    np.testing.assert_array_equal(
        filter_state(series, params).level.to_numpy(),
        filter_state(series, params).level.to_numpy(),
    )


# --- kinematics -------------------------------------------------------------


def test_velocity_is_the_annualized_slope():
    slope = pd.Series([0.001] * 400, index=pd.bdate_range("2020-01-01", periods=400))
    kin = kinematics(slope, periods_per_year=252.0)
    assert kin.velocity.iloc[-1] == pytest.approx(0.252)
    assert kin.acceleration.iloc[-1] == pytest.approx(0.0)


def test_daily_and_monthly_slopes_annualize_differently():
    """The same per-period slope means very different things at two frequencies."""
    slope = pd.Series([0.001] * 400, index=pd.bdate_range("2020-01-01", periods=400))
    assert kinematics(slope, periods_per_year=252.0).velocity.iloc[-1] == pytest.approx(0.252)
    assert kinematics(slope, periods_per_year=12.0).velocity.iloc[-1] == pytest.approx(0.012)


def test_the_z_baseline_is_expanding_never_centred():
    """A centred window would read the future (§14.1 rule 3)."""
    series = pd.Series(np.arange(100.0))
    z = expanding_z(series, min_periods=10)
    assert z.iloc[:9].isna().all()
    # Truncating the series must not change the values that survive.
    truncated = expanding_z(series.iloc[:50], min_periods=10)
    pd.testing.assert_series_equal(z.iloc[:50], truncated)


def test_the_baseline_degrades_when_ten_years_is_not_available():
    """FRED:SP500 is capped at roughly ten years, so demanding a full decade
    would produce a column of NaN and a lamp that never lights."""
    full, short = baseline_window(4000, 252.0), baseline_window(2512, 252.0)
    assert full == (2520, False)
    assert short[0] < 2520 and short[1] is True


def test_the_lamp_thresholds_are_the_spec_ones():
    assert jerk_lamp(0.0) == "calm"
    assert jerk_lamp(JERK_ELEVATED_Z) == "elevated"
    assert jerk_lamp(-JERK_ELEVATED_Z) == "elevated", "the reading is on |z|"
    assert jerk_lamp(JERK_EXTREME_Z) == "extreme"
    assert jerk_lamp(float("nan")) == "unknown"


def test_jerk_is_only_ever_published_as_a_z_score():
    """§3.1 demotes the third derivative to an indicator. The pipeline must not
    expose a raw one under any column name."""
    features = compute_features(walk(1200), DAILY)
    assert "jerk" not in features.frame.columns
    assert "jerk_z" in features.frame.columns


# --- the pipeline as a whole ------------------------------------------------


def test_the_pipeline_produces_the_documented_columns():
    features = compute_features(walk(1200), DAILY)
    assert set(CORE_COLUMNS) <= set(features.frame.columns)
    assert {"momentum_1m", "momentum_3m", "momentum_12m"} <= set(features.frame.columns)


def test_the_same_transform_runs_over_a_monthly_series():
    """Sub-milestone B's transfer argument requires one pipeline, not two."""
    monthly = pd.Series(
        np.exp(np.cumsum(np.random.default_rng(1).normal(0.004, 0.04, 900))) * 100,
        index=pd.date_range("1950-01-01", periods=900, freq="MS"),
    )
    features = compute_features(monthly, MONTHLY)
    assert set(CORE_COLUMNS) <= set(features.frame.columns)
    # One year of memory is twelve lags here, not 252.
    assert features.ffd.width <= 12


def test_a_frozen_d_from_another_series_is_refused():
    """One shared d across series with different memory is a silent error, so it
    is made a loud one."""
    fitted = compute_features(walk(1200), DAILY).frozen()
    with pytest.raises(ValueError, match="frozen per series"):
        compute_features(walk(1200, seed=9), MONTHLY, frozen=fitted)


def test_frozen_parameters_reproduce_the_frame():
    path = walk(1200)
    first = compute_features(path, DAILY)
    second = compute_features(path, DAILY, frozen=first.frozen())
    pd.testing.assert_frame_equal(first.frame, second.frame)
    assert second.used_frozen_params


def test_an_empty_path_is_refused_rather_than_producing_empty_features():
    with pytest.raises(ValueError, match="no observations"):
        compute_features(pd.Series(dtype="float64"), DAILY)


def test_feature_params_come_from_the_engine_yaml():
    params = FeatureParams.from_params(
        {"features": {"ffd_significance": 0.01, "momentum_months": [2, 6]}}
    )
    assert params.ffd_significance == 0.01
    assert params.momentum_months == (2, 6)


def test_frozen_params_round_trip_through_json_shapes():
    frozen = compute_features(walk(1200), DAILY).frozen()
    restored = FrozenFeatureParams.from_dict(frozen.as_dict())
    assert restored.ffd == frozen.ffd
    assert restored.kalman == frozen.kalman


# --- against the real snapshot ----------------------------------------------


def test_each_series_needs_its_own_d(equity_engine, equity_observations):
    """The reason ``d`` is frozen per series rather than once for the engine.

    Ten years of S&P, forty of NASDAQ and a century and a half of monthly Shiller
    do not share a minimum stationary ``d`` — asserting that they differ is
    asserting that a single shared ``d`` would have been wrong for two of them.
    """
    from findynamics.engines.equity.engine import ALL_ROLES
    from tests.engines.equity.conftest import world_from

    analysis = equity_engine.analyze(world_from(equity_observations), roles=ALL_ROLES)
    chosen = {role: fs.ffd.d for role, fs in analysis.features.items()}

    assert len(analysis.features) == 3
    assert len(set(chosen.values())) > 1, chosen
    for role, fs in analysis.features.items():
        assert fs.ffd.passed, f"{role} never reached stationarity"
        assert fs.ffd.series_id == fs.series.series_id
