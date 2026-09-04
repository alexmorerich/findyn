"""Ω̇, Ω̈, and the banding — the derivative conventions, not the model.

The assertions here are about phase, burn-in and thresholds: things with known
answers on a synthetic path, which is exactly where synthetic data earns its
place. Whether Ω̇ says anything about markets is KK3's question.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from findynamics.engines.equity.features.kinematics import burn_in_window
from findynamics.research.omega import OMEGA_REGIMES, omega_dynamics, omega_regime
from findynamics.research.omega.contracts import OmegaContractError
from findynamics.research.omega.dynamics import DynamicsParams, OmegaDynamicsError
from tests.research.omega.conftest import accessor_at, dummy_spec, with_params

#: Six full cycles over twelve years of business days — long enough for the
#: filter to settle and for the burn-in to discard less than one period.
CYCLE_PERIODS = 504
PATH_LENGTH = 3_024


@pytest.fixture(scope="module")
def sine_path():
    index = pd.bdate_range(date(2000, 1, 3), periods=PATH_LENGTH)
    phase = 2.0 * np.pi * np.arange(PATH_LENGTH) / CYCLE_PERIODS
    return pd.Series(np.sin(phase), index=index), pd.Series(np.cos(phase), index=index)


@pytest.fixture(scope="module")
def sine_dynamics(sine_path):
    from findynamics.research.omega import load_omega_config

    omega, _ = sine_path
    return omega_dynamics(
        omega,
        dummy_spec(n_observations=PATH_LENGTH),
        periods_per_year=252.0,
        config=load_omega_config(),
    )


def test_omega_velocity_leads_omega_by_a_quarter_cycle(sine_path, sine_dynamics):
    """On Ω = sin(t), Ω̇ is cos(t): correlated with the cosine, orthogonal to Ω.

    Asserted on correlation signs rather than on values, because the filter
    applies a small phase lag that depends on the fitted variances and the
    quantity under test is the *relationship*, not the constant.
    """
    omega, cosine = sine_path
    velocity = sine_dynamics.omega_velocity
    usable = velocity.notna()

    assert float(velocity[usable].corr(cosine[usable])) > 0.95
    assert abs(float(omega[usable].corr(velocity[usable]))) < 0.10


def test_omega_acceleration_is_antiphase_with_omega(sine_path, sine_dynamics):
    """And Ω̈ is −sin(t): the second derivative of a sine inverts it."""
    omega, _ = sine_path
    acceleration = sine_dynamics.omega_acceleration
    usable = acceleration.notna()

    assert float(omega[usable].corr(acceleration[usable])) < -0.95


def test_the_filtered_slope_is_far_smoother_than_a_naive_difference():
    """Why Ω̇ is not ``omega.diff()``, measured rather than asserted in prose.

    ``features/kinematics.py`` opens with the argument: differencing a noisy
    series amplifies its noise by roughly the differencing order. Ω is a PC1 of
    nine noisy columns, so it is exactly the kind of series that argument is
    about. On a sine buried in noise, the filtered slope recovers the underlying
    cycle and the raw difference does not.
    """
    from findynamics.research.omega import load_omega_config

    rng = np.random.default_rng(11)
    index = pd.bdate_range(date(2000, 1, 3), periods=PATH_LENGTH)
    phase = 2.0 * np.pi * np.arange(PATH_LENGTH) / CYCLE_PERIODS
    truth = pd.Series(np.cos(phase), index=index)
    noisy = pd.Series(np.sin(phase) + rng.normal(0.0, 0.5, PATH_LENGTH), index=index)

    path = omega_dynamics(
        noisy,
        dummy_spec(n_observations=PATH_LENGTH),
        periods_per_year=252.0,
        config=load_omega_config(),
    )
    usable = path.omega_velocity.notna()

    filtered = abs(float(path.omega_velocity[usable].corr(truth[usable])))
    naive = abs(float(noisy.diff()[usable].corr(truth[usable])))

    assert filtered > 0.5, f"the filtered slope lost the cycle (rho={filtered:.3f})"
    assert filtered > 4 * naive, (
        f"the filtered slope ({filtered:.3f}) is not meaningfully better than a raw "
        f"difference ({naive:.3f}); the extra machinery is not earning its place"
    )


def test_the_filter_start_up_is_discarded_from_the_slope_but_not_from_omega(sine_dynamics):
    """One year of Ω̇, and Ω keeps its full span.

    The level is pinned by the first observation; only the slope has to be
    estimated out of a diffuse prior, which is the split ``kinematics.py`` makes
    for exactly the same reason.
    """
    expected, is_short = burn_in_window(PATH_LENGTH, 252.0, years=1.0)

    assert not is_short
    assert sine_dynamics.diagnostics["burn_in_periods"] == float(expected)
    assert sine_dynamics.omega_velocity.iloc[:expected].isna().all()
    assert sine_dynamics.omega_velocity.iloc[expected:].notna().any()
    assert sine_dynamics.omega.notna().all()


def test_the_regime_bands_split_the_percentile_where_config_says(omega_config):
    """Boundaries closed on the left: a value on a threshold reads as the worse state."""
    params = DynamicsParams.from_config(omega_config)
    percentile = pd.Series(
        [0.0, 59.9, 60.0, 84.9, 85.0, 100.0, np.nan],
        index=pd.bdate_range(date(2020, 1, 1), periods=7),
    )

    labels = omega_regime(
        percentile,
        transition_percentile=params.transition_percentile,
        stress_percentile=params.stress_percentile,
    )

    assert list(labels[:6]) == [
        "latent_calm",
        "latent_calm",
        "latent_transition",
        "latent_transition",
        "latent_stress",
        "latent_stress",
    ]
    assert pd.isna(labels.iloc[6]), "an unscored date has no reading, not a calm one"


def test_overlapping_regime_thresholds_are_refused(omega_config):
    """Bands that overlap would make the state depend on evaluation order."""
    config = with_params(
        omega_config, regime={"transition_percentile": 90.0, "stress_percentile": 85.0}
    )

    with pytest.raises(OmegaDynamicsError, match="transition < stress"):
        DynamicsParams.from_config(config)


def test_every_series_on_a_path_shares_one_calendar(sine_dynamics):
    """Checked by the contract, and checked here so the contract stays checked."""
    for name in (*sine_dynamics.SERIES_FIELDS, "omega_regime"):
        pd.testing.assert_index_equal(getattr(sine_dynamics, name).index, sine_dynamics.omega.index)


def test_a_path_with_a_stray_label_is_refused(sine_dynamics):
    """``omega_regime`` carries the module's vocabulary and nothing else."""
    from dataclasses import replace

    stray = sine_dynamics.omega_regime.copy()
    stray.iloc[-1] = "crisis"

    with pytest.raises(OmegaContractError, match="outside the vocabulary"):
        replace(sine_dynamics, omega_regime=stray)


def test_a_path_too_short_to_differentiate_is_a_named_failure(omega_config):
    index = pd.bdate_range(date(2020, 1, 1), periods=3)
    omega = pd.Series([np.nan, np.nan, 1.0], index=index)

    with pytest.raises(OmegaDynamicsError, match="at least two"):
        omega_dynamics(omega, dummy_spec(), periods_per_year=252.0, config=omega_config)


def test_the_real_path_puts_the_crises_in_latent_stress(omega_engine, omega_observations):
    """The sanity check that the banding describes something.

    In-sample and therefore not a result — it is fitted on the whole record. It
    is here because a coordinate that could not tell 2008 from 2017 even with
    hindsight would not be worth taking out of sample.
    """
    path = omega_engine.fit_transform(accessor_at(omega_observations))
    stress = path.omega_regime == OMEGA_REGIMES[2]

    assert stress.loc["2008-09-15":"2008-12-31"].mean() > 0.8
    assert stress.loc["2020-03-01":"2020-04-30"].mean() > 0.8
    assert stress.loc["2017-01-01":"2017-12-31"].mean() < 0.05


def test_a_non_positive_dynamics_window_is_refused(omega_config):
    """Same argument as the feature windows: floored downstream, so silent."""
    with pytest.raises(OmegaDynamicsError, match="volatility_months must be > 0"):
        DynamicsParams.from_config(with_params(omega_config, dynamics={"volatility_months": 0.0}))

    with pytest.raises(OmegaDynamicsError, match="zscore_min_years must be > 0"):
        DynamicsParams.from_config(with_params(omega_config, dynamics={"zscore_min_years": -1.0}))
