"""The projection, and the one property that would silently invent a signal.

The sign tests are the important ones here. An unpinned principal component
does not *fail* — it produces a Ω̇ spike train aligned to the refit calendar
that is large, regular, robust to winsorization, and entirely an artifact of
whichever LAPACK driver the wheel was built against.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from findynamics.research.omega import OmegaEngine, PriceOnlyOmegaEstimator, build_estimator
from findynamics.research.omega.estimator import (
    SIGN_TOLERANCE,
    EstimatorParams,
    OmegaFitError,
)
from findynamics.research.omega.features import FeatureParams
from tests.research.omega.conftest import accessor_at, price_only_config, with_params


@pytest.fixture(scope="module")
def block(omega_observations_module):
    """The full feature block on the committed snapshot, built once."""
    from findynamics.research.omega import build_feature_frame, load_omega_config

    return build_feature_frame(accessor_at(omega_observations_module), load_omega_config())


@pytest.fixture(scope="module")
def omega_observations_module() -> pd.DataFrame:
    from tests.conftest import FIXTURE_DIR

    frame = pd.read_csv(FIXTURE_DIR / "equity_prices.csv")
    for column in ("obs_date", "release_date", "revision_date"):
        frame[column] = pd.to_datetime(frame[column])
    return frame


def test_transform_reproduces_the_projection_the_fit_describes(omega_config, block):
    """The spec is the whole transform: mean, scale, loadings and nothing else.

    Recomputed here from the stored numbers rather than compared against the
    estimator's own intermediate state, because the property that matters is
    that a reader of ``OmegaSpec`` — the KK3 report, a year from now — can
    reproduce Ω without the object that produced it.
    """
    estimator = build_estimator(omega_config)
    spec = estimator.fit(block)
    omega = estimator.transform(block, spec)

    matrix = block[list(spec.columns)]
    standardized = (matrix - np.asarray(spec.mean)) / np.asarray(spec.scale)
    expected = standardized.to_numpy() @ np.asarray(spec.loadings)

    finite = np.isfinite(expected)
    np.testing.assert_allclose(omega.to_numpy()[finite], expected[finite], rtol=0.0, atol=1e-12)


def test_transform_is_pure_in_row_and_spec(omega_config, block):
    """Ω on a slice equals Ω on the whole path restricted to that slice.

    If this ever fails, ``transform`` has picked up a dependence on the rows
    around it — which is the shape every leak in this module would take.
    """
    estimator = build_estimator(omega_config)
    spec = estimator.fit(block)

    whole = estimator.transform(block, spec)
    piece = estimator.transform(block.iloc[1_000:1_500], spec)

    pd.testing.assert_series_equal(piece, whole.iloc[1_000:1_500], check_exact=True)


def test_pc1_is_oriented_so_that_higher_omega_is_more_stress(omega_config, block):
    """The convention, asserted where it is established rather than assumed."""
    estimator = build_estimator(omega_config)
    spec = estimator.fit(block)

    assert spec.sign_reference == "realized_vol"
    assert spec.loading("realized_vol") > 0.0
    # And the orientation reaches the values, not only the loadings.
    omega = estimator.transform(block, spec)
    assert omega.loc["2008-09-01":"2008-12-31"].mean() > omega.loc["2006-01-01":"2006-12-31"].mean()


def test_a_negative_reference_loading_negates_the_whole_vector(omega_config, block):
    """The negation branch, exercised directly.

    Reaching it through the public path would mean finding data whose PC1
    happens to come back inverted, which is a property of the LAPACK build
    rather than of the model — precisely the thing this rule exists to stop
    mattering. So the rule is tested as the pure function it is.
    """
    estimator = build_estimator(omega_config)
    columns = ("abs_return", "drawdown", "realized_vol")
    loadings = np.array([0.6, 0.5, -0.62])

    reference, pinned = estimator._pin_sign(columns, loadings)

    assert reference == "realized_vol"
    np.testing.assert_allclose(pinned, -loadings)
    assert pinned[columns.index("realized_vol")] > 0.0


def test_a_zero_loading_pins_nothing_and_the_search_moves_on(omega_config):
    """Negating a vector leaves a zero where it was, so a zero is not a pin."""
    estimator = build_estimator(omega_config)
    columns = ("abs_return", "drawdown", "realized_vol")
    loadings = np.array([-0.7, 0.5, SIGN_TOLERANCE / 10.0])

    reference, pinned = estimator._pin_sign(columns, loadings)

    # Precedence is realized_vol, vol_of_vol, abs_return, ... — the first is
    # orthogonal, the second was never in this block, so the third pins it.
    assert reference == "abs_return"
    assert pinned[columns.index("abs_return")] > 0.0
    np.testing.assert_allclose(pinned, -loadings)


def test_the_sign_survives_extending_the_fit_window(omega_config, block):
    """The stability test the whole track depends on.

    Two windows differing by a year must not disagree about which way Ω points.
    A failure here means every Ω̇, Ω̈, C and K result downstream is an artifact
    of the refit calendar.
    """
    estimator = build_estimator(omega_config)
    short = block.iloc[:3_000]
    long = block.iloc[:3_250]

    first = estimator.fit(short)
    second = estimator.fit(long)

    assert first.sign_reference == second.sign_reference
    assert np.sign(first.loading(first.sign_reference)) == np.sign(
        second.loading(second.sign_reference)
    )

    shared = short.index
    a = estimator.transform(block.loc[shared], first).dropna()
    b = estimator.transform(block.loc[shared], second).dropna()
    common = a.index.intersection(b.index)
    correlation = float(np.corrcoef(a.loc[common], b.loc[common])[0, 1])
    assert correlation > 0.9, f"two neighbouring fits disagree about Ω (rho={correlation:.3f})"


def test_a_window_below_the_minimum_raises_rather_than_returning_a_short_omega(omega_config, block):
    """A two-point projection is not a coordinate, and must not look like one."""
    estimator = build_estimator(omega_config)
    minimum = EstimatorParams.from_config(omega_config).min_fit_observations

    with pytest.raises(OmegaFitError, match="below the"):
        estimator.fit(block.iloc[: minimum - 10])


def test_a_dropped_column_is_named_and_does_not_move_the_surviving_omega(omega_config, block):
    """Dropping a column must be a *report*, not a change to the model.

    Ω from the shipped config — where ``credit_velocity`` fails the availability
    check — must equal Ω from a config that never built the column at all. If
    the two differ, the drop is leaving a trace somewhere it should not.
    """
    estimator = build_estimator(omega_config)
    with_thin_column = estimator.fit(block)
    assert "credit_velocity" in with_thin_column.dropped

    without = estimator.fit(block.drop(columns=["credit_velocity"]))

    assert without.columns == with_thin_column.columns
    assert without.loadings == with_thin_column.loadings
    assert without.mean == with_thin_column.mean
    assert without.scale == with_thin_column.scale
    pd.testing.assert_series_equal(
        estimator.transform(block, with_thin_column),
        estimator.transform(block, without),
        check_exact=True,
    )


def test_a_constant_column_is_dropped_rather_than_divided_by_zero(omega_config, block):
    """A column that never moved has no scale, and the contract refuses one."""
    estimator = build_estimator(omega_config)
    frozen = block.copy()
    frozen["curve_slope"] = 1.0

    spec = estimator.fit(frozen)

    assert "curve_slope" in spec.dropped
    assert "curve_slope" not in spec.columns
    assert all(value > 0.0 for value in spec.scale)


def test_the_price_only_specification_reaches_1927(omega_config, omega_observations_module):
    """A different specification, and the reason it exists.

    Back past 2000 only the four price-derived columns exist. Running the
    full-column estimator there would drop five columns and quietly become this
    one; naming it separately is what keeps the two from being averaged in a
    table.
    """
    config = with_params(
        omega_config, estimator={"name": "pca_price_only"}, features={"start": None}
    )
    engine = OmegaEngine(config=config)
    assert isinstance(engine.estimator, PriceOnlyOmegaEstimator)

    z = engine.features(accessor_at(omega_observations_module))
    spec = engine.fit(z)

    assert spec.estimator == "pca_price_only"
    assert spec.columns == ("abs_return", "drawdown", "realized_vol", "vol_of_vol")
    assert spec.fit_start.year == 1928
    assert spec.n_observations > 24_000

    omega = engine.transform(z, spec)
    # 1929, 1987 and 2008 all read as stress against a calm mid-sixties.
    calm = omega.loc["1964-01-01":"1964-12-31"].mean()
    for window in (
        ("1929-09-01", "1929-12-31"),
        ("1987-10-01", "1987-12-31"),
        ("2008-09-01", "2008-12-31"),
    ):
        assert omega.loc[window[0] : window[1]].mean() > calm + 3.0


def test_an_unknown_estimator_name_fails_with_the_known_ones(omega_config):
    config = with_params(omega_config, estimator={"name": "autoencoder"})

    with pytest.raises(OmegaFitError, match="unknown estimator"):
        build_estimator(config)


def test_a_sign_precedence_naming_an_unknown_column_is_refused(omega_config):
    """A typo here would silently fall through to the documented fallback."""
    config = with_params(
        omega_config, estimator={"sign_reference_precedence": ["realized_vol", "entropy"]}
    )

    with pytest.raises(OmegaFitError, match="unknown column"):
        EstimatorParams.from_config(config)


def test_transform_refuses_a_block_missing_a_column_the_spec_was_fitted_on(omega_config, block):
    """A projection onto eight of nine axes is a different coordinate."""
    estimator = build_estimator(omega_config)
    spec = estimator.fit(block)

    with pytest.raises(OmegaFitError, match="missing column"):
        estimator.transform(block.drop(columns=["realized_vol"]), spec)


def test_an_all_nan_block_cannot_be_fitted(omega_config):
    """The degenerate case: every candidate fails the availability check."""
    config = price_only_config(omega_config)
    estimator = build_estimator(config)
    features = FeatureParams.from_config(config)
    frame = pd.DataFrame(
        {name: [np.nan] * 600 for name in features.columns},
        index=pd.bdate_range(date(2000, 1, 3), periods=600),
    )

    with pytest.raises(OmegaFitError, match="availability check"):
        estimator.fit(frame)
