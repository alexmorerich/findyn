"""The three coupling estimators, their safeguards, and the leak a regression adds.

The truncation test reappears here and it is the important one. An expanding
regression introduces a new leakage surface that the KK1 tests cannot see: the
coefficient at date *t* is a function of a *sample*, and a sample is exactly the
kind of thing that quietly grows.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from findynamics.research.omega.coupling import (
    CouplingParams,
    OmegaCouplingError,
    all_couplings,
    interaction_coupling,
    newey_west_lag,
    primary_coupling,
    ratio_coupling,
    regression_coupling,
)
from tests.research.omega.conftest import with_params


@pytest.fixture
def params(omega_config) -> CouplingParams:
    return CouplingParams.from_config(omega_config)


def _index(n: int, start: date = date(2000, 1, 3)) -> pd.DatetimeIndex:
    return pd.bdate_range(start=start, periods=n)


# ---------------------------------------------------------------------------
# The lag rule
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "n,expected",
    [(100, 4), (504, 5), (6661, 11), (1, 1), (0, 0)],
)
def test_the_newey_west_lag_is_a_pure_function_of_the_sample_size(n, expected):
    """``floor(4·(n/100)^0.25)``.

    Pure because a lag that depended on anything but the count would make two
    runs on the same data report different significance — and significance is
    the only thing the standard error is there to supply.
    """
    assert newey_west_lag(n) == expected


# ---------------------------------------------------------------------------
# 1a. The ratio estimator
# ---------------------------------------------------------------------------


def test_a_denominator_crossing_zero_produces_nan_and_never_an_infinity(params):
    """The failure the floor exists to prevent, on a denominator built to hit zero.

    Unguarded, one ``ΔΩ`` near zero produces a ±10⁹ spike that dominates every
    downstream z-score. The assertion is on the *absence of infinities*, not on
    the magnitude: a huge finite number is the same failure wearing a float.
    """
    n = 2_000
    index = _index(n)
    # A sawtooth through zero: every 50th observation has ΔΩ ≈ 0.
    omega = pd.Series(np.sin(np.arange(n) * np.pi / 50.0), index=index)
    velocity = pd.Series(np.linspace(0.0, 1.0, n), index=index)
    tuned = CouplingParams.from_config(
        with_params(
            pytest.importorskip("findynamics.research.omega").load_omega_config(),
            coupling={"min_observations": 100},
        )
    )

    result = ratio_coupling(velocity, omega, params=tuned)

    assert not result.declined
    values = result.coupling.to_numpy(dtype=float)
    assert np.isfinite(values[~np.isnan(values)]).all()
    assert not np.isinf(values).any()
    # NaN below the floor, never zero: zero is a claim about the coupling.
    below = result.coupling.isna() & result.detail["coupling_floor"].notna()
    assert below.sum() > 0
    assert (result.coupling.dropna() != 0.0).any()


def test_the_undefined_share_is_reported(params):
    """The share is the whole point of publishing NaN instead of 0."""
    n = 2_000
    index = _index(n)
    rng = np.random.default_rng(3)
    omega = pd.Series(np.cumsum(rng.normal(size=n)), index=index)
    velocity = pd.Series(np.cumsum(rng.normal(size=n)), index=index)
    tuned = CouplingParams.from_config(
        with_params(
            pytest.importorskip("findynamics.research.omega").load_omega_config(),
            coupling={"min_observations": 100},
        )
    )

    result = ratio_coupling(velocity, omega, params=tuned)

    share = result.diagnostics["coupling_undefined_share"]
    # A 25th-percentile floor makes roughly a quarter undefined BY DESIGN.
    assert 0.15 < share < 0.40
    assert result.diagnostics["coupling_eligible"] > 1_000


def test_the_ratio_estimator_declines_rather_than_publishing_a_mostly_nan_column(
    omega_config,
):
    """A ceiling below what the floor produces by construction forces the decline.

    Declining is a *result* and comes back on the object; it is not an error.
    A caller that got a mostly-NaN series back instead would have to discover
    the problem by counting.
    """
    n = 2_000
    index = _index(n)
    rng = np.random.default_rng(5)
    omega = pd.Series(np.cumsum(rng.normal(size=n)), index=index)
    velocity = pd.Series(np.cumsum(rng.normal(size=n)), index=index)
    strict = CouplingParams.from_config(
        with_params(
            omega_config,
            coupling={"min_observations": 100, "max_undefined_share": 0.05},
        )
    )

    result = ratio_coupling(velocity, omega, params=strict)

    assert result.declined
    assert result.coupling.empty
    assert "ceiling" in result.decline_reason
    assert result.diagnostics["coupling_undefined_share"] > 0.05


def test_winsorization_clips_and_reports_what_it_clipped(omega_config):
    """It can destroy the signal it protects, so the share is published."""
    n = 2_000
    index = _index(n)
    rng = np.random.default_rng(7)
    omega = pd.Series(np.cumsum(rng.normal(size=n)), index=index)
    velocity = pd.Series(np.cumsum(rng.normal(size=n)), index=index)
    tuned = CouplingParams.from_config(
        with_params(omega_config, coupling={"min_observations": 100})
    )

    wide = ratio_coupling(velocity, omega, params=tuned)
    narrow = ratio_coupling(
        velocity,
        omega,
        params=CouplingParams.from_config(
            with_params(
                omega_config,
                coupling={"min_observations": 100, "winsor_lower": 10.0, "winsor_upper": 90.0},
            )
        ),
    )

    assert (
        narrow.diagnostics["coupling_winsorized_share"]
        > wide.diagnostics["coupling_winsorized_share"]
    )
    # And the clip actually binds wherever its bounds exist. Asserted this way
    # rather than on the series maximum: the largest value falls inside the
    # warm-up, where the expanding quantile has not produced a bound yet, so the
    # maximum is identical under both settings and a test on it would pass for
    # the wrong reason — which is how it was written first.
    # Recomputed here with plain pandas rather than imported from the module, so
    # the bound the test checks against is not the one the module produced.
    upper = narrow.detail["coupling_raw"].expanding(min_periods=100).quantile(0.90)
    bound = upper.notna() & narrow.coupling.notna()
    assert bound.sum() > 500
    assert (narrow.coupling[bound] <= upper[bound] + 1e-12).all()


# ---------------------------------------------------------------------------
# 1b. The regression estimator
# ---------------------------------------------------------------------------


def _synthetic_regression(
    n: int = 3_000,
    beta1: float = 1.75,
    seed: int = 11,
    noise: float = 0.0,
):
    """``v = 0.3 + β1·Ω̇ + 0.4·Ω + ε`` with a known ``β1``.

    ``noise=0`` by default, and that is the point rather than a shortcut. The
    exactness test below asks whether the **solver** is the normal equations;
    with residuals in play, the recoverable precision is the sampling error
    (≈ σ/√n, about 1e-3 here), and a 1e-6 assertion would be measuring the
    random seed. Noise is switched on where the question is statistical.
    """
    rng = np.random.default_rng(seed)
    index = _index(n)
    omega = pd.Series(rng.normal(size=n), index=index)
    omega_velocity = pd.Series(rng.normal(size=n), index=index)
    omega_acceleration = pd.Series(rng.normal(size=n), index=index)
    error = rng.normal(scale=noise, size=n) if noise else np.zeros(n)
    velocity = pd.Series(0.3 + beta1 * omega_velocity + 0.4 * omega + error, index=index)
    acceleration = pd.Series(
        -0.1 + 0.9 * omega_velocity + 0.5 * omega_acceleration + 0.2 * omega + error,
        index=index,
    )
    return velocity, acceleration, omega, omega_velocity, omega_acceleration


def test_the_expanding_regression_recovers_a_known_beta(params):
    """Closed form, so the answer is the normal equations and not an optimizer's.

    Noiseless, so 1e-6 is a statement about the solve. The neighbouring test
    puts residuals back and asks the statistical question instead.
    """
    velocity, acceleration, omega, omega_velocity, omega_acceleration = _synthetic_regression()

    result = regression_coupling(
        velocity, acceleration, omega, omega_velocity, omega_acceleration, params=params
    )

    assert result.latest() == pytest.approx(1.75, abs=1e-6)
    assert not result.declined
    # Exact at every date it publishes, not only at the end: a solve that drifted
    # with the sample size would still land on the truth by 3,000 observations.
    published = result.coupling.dropna()
    assert np.abs(published - 1.75).max() < 1e-6


def test_with_residuals_the_estimate_lands_within_its_own_standard_error(params):
    """The statistical question, kept apart from the numerical one.

    σ/√n with σ = 0.05 over 3,000 observations is about 1e-3, so this is the
    precision the data can support — and a test that demanded 1e-6 here would
    pass or fail on the seed.
    """
    velocity, acceleration, omega, omega_velocity, omega_acceleration = _synthetic_regression(
        noise=0.05
    )

    result = regression_coupling(
        velocity, acceleration, omega, omega_velocity, omega_acceleration, params=params
    )

    assert result.latest() == pytest.approx(1.75, abs=5e-3)
    assert result.diagnostics["regression_velocity_r2_latest"] > 0.99


def test_the_sample_size_grows_by_one_each_date(params):
    """``n`` is monotone in the date, which is what makes the window expanding."""
    velocity, acceleration, omega, omega_velocity, omega_acceleration = _synthetic_regression()

    result = regression_coupling(
        velocity, acceleration, omega, omega_velocity, omega_acceleration, params=params
    )
    counts = result.detail["velocity__n"].dropna()

    assert counts.is_monotonic_increasing
    assert int(counts.diff().dropna().max()) == 1
    assert int(counts.iloc[0]) == params.min_observations


def test_a_coefficient_does_not_move_when_later_data_is_appended(params):
    """The truncation test, for the leakage surface a regression introduces.

    Every coefficient dated ``<= T`` must be bit-identical whether or not the
    frame ran past ``T``. If it is not, the expanding window is not expanding —
    it is the whole sample wearing a date index.
    """
    velocity, acceleration, omega, omega_velocity, omega_acceleration = _synthetic_regression()
    cut = 2_000

    short = regression_coupling(
        velocity.iloc[:cut],
        acceleration.iloc[:cut],
        omega.iloc[:cut],
        omega_velocity.iloc[:cut],
        omega_acceleration.iloc[:cut],
        params=params,
    )
    long = regression_coupling(
        velocity, acceleration, omega, omega_velocity, omega_acceleration, params=params
    )

    pd.testing.assert_series_equal(
        short.coupling,
        long.coupling.iloc[:cut],
        check_exact=True,
    )
    # And the supporting columns, which is where a sloppy R² would show up.
    for column in ("velocity__r_squared", "velocity__n", "velocity__intercept"):
        pd.testing.assert_series_equal(
            short.detail[column], long.detail[column].iloc[:cut], check_exact=True
        )


def test_both_specifications_are_fitted_and_reported(params):
    """Fitting only the configured one would make the report a claim about it."""
    velocity, acceleration, omega, omega_velocity, omega_acceleration = _synthetic_regression()

    result = regression_coupling(
        velocity, acceleration, omega, omega_velocity, omega_acceleration, params=params
    )

    assert result.diagnostics["regression_velocity_beta_latest"] == pytest.approx(1.75, abs=1e-6)
    assert result.diagnostics["regression_acceleration_beta_latest"] == pytest.approx(0.9, abs=1e-6)
    assert "velocity__omega_velocity" in result.detail.columns
    assert "acceleration__omega_velocity" in result.detail.columns


def test_a_short_sample_declines_rather_than_publishing_a_two_point_slope(params):
    velocity, acceleration, omega, omega_velocity, omega_acceleration = _synthetic_regression(n=100)

    result = regression_coupling(
        velocity, acceleration, omega, omega_velocity, omega_acceleration, params=params
    )

    assert result.declined
    assert "504" in result.decline_reason


def test_the_tstat_is_published_on_a_cadence_and_never_forward_filled(params):
    """A forward-filled significance reading implies a test that was not run."""
    velocity, acceleration, omega, omega_velocity, omega_acceleration = _synthetic_regression()

    result = regression_coupling(
        velocity, acceleration, omega, omega_velocity, omega_acceleration, params=params
    )
    tstat = result.detail["velocity__tstat"]

    evaluated = tstat.notna().sum()
    published = result.detail["velocity__omega_velocity"].notna().sum()
    assert 0 < evaluated < published
    assert evaluated == pytest.approx(published / params.tstat_cadence, rel=0.1)
    # The last date is always evaluated, because that is the one a report quotes.
    assert np.isfinite(tstat.dropna().iloc[-1])


# ---------------------------------------------------------------------------
# 1c. The interaction control
# ---------------------------------------------------------------------------


def test_the_interaction_control_is_defined_wherever_both_z_scores_are(omega_config):
    """Always available, which is what makes it a control."""
    n = 3_000
    index = _index(n)
    rng = np.random.default_rng(13)
    velocity = pd.Series(rng.normal(size=n), index=index)
    omega_velocity = pd.Series(rng.normal(size=n), index=index)
    tuned = CouplingParams.from_config(
        with_params(omega_config, coupling={"zscore_min_years": 2.0})
    )

    result = interaction_coupling(velocity, omega_velocity, params=tuned, periods_per_year=252.0)

    assert not result.declined
    assert result.defined_share > 0.75
    assert set(result.detail.columns) == {"z_velocity", "z_omega_velocity"}


# ---------------------------------------------------------------------------
# Config and orchestration
# ---------------------------------------------------------------------------


def test_all_three_estimators_come_back_together(params):
    velocity, acceleration, omega, omega_velocity, omega_acceleration = _synthetic_regression()

    results = all_couplings(
        velocity,
        acceleration,
        omega,
        omega_velocity,
        omega_acceleration,
        params=params,
        periods_per_year=252.0,
    )

    assert set(results) == {"ratio", "regression", "interaction"}
    pd.testing.assert_series_equal(
        primary_coupling(results, params), results["regression"].coupling
    )


@pytest.mark.parametrize(
    "block,message",
    [
        ({"primary": "autoencoder"}, "coupling.primary"),
        ({"specification": "jerk"}, "coupling.specification"),
        ({"denominator_percentile": 0.0}, "denominator_percentile"),
        ({"winsor_lower": 99.0, "winsor_upper": 1.0}, "winsorization"),
        ({"max_undefined_share": 1.5}, "max_undefined_share"),
        ({"min_observations": 2}, "min_observations"),
        ({"tstat_cadence": 0}, "tstat_cadence"),
        ({"newey_west_lag": -1}, "newey_west_lag"),
    ],
)
def test_a_malformed_coupling_config_fails_at_load(omega_config, block, message):
    """Strict at load, like every other config in the repo."""
    with pytest.raises(OmegaCouplingError, match=message):
        CouplingParams.from_config(with_params(omega_config, coupling=block))
