"""The instability composite: its weights, its coverage floor, and its orientation.

The two properties that matter are that the weights cannot be fitted on the
whole sample, and that a date missing four of five terms does not get published
as if it had them.
"""

from __future__ import annotations

import math
from datetime import date

import numpy as np
import pandas as pd
import pytest

from findynamics.research.omega.curvature import (
    CURVATURE_TERMS,
    CurvatureParams,
    OmegaCurvatureError,
    financial_curvature,
)
from tests.research.omega.conftest import with_params

PERIODS = 252.0


def _terms(n: int = 4_000, seed: int = 17, names: tuple[str, ...] = CURVATURE_TERMS):
    """Five correlated instability-shaped series on one calendar."""
    rng = np.random.default_rng(seed)
    index = pd.bdate_range(date(2000, 1, 3), periods=n)
    common = rng.normal(size=n).cumsum() / 30.0
    return {name: pd.Series(common + rng.normal(scale=0.8, size=n), index=index) for name in names}


@pytest.fixture
def params(omega_config) -> CurvatureParams:
    return CurvatureParams.from_config(
        with_params(omega_config, curvature={"zscore_min_years": 2.0})
    )


def test_equal_weights_sum_to_one_over_the_terms_present(params):
    """``math.fsum``, not ``sum``.

    CPython 3.12 gave ``sum`` Neumaier compensation and 3.11 sums naively, so
    the same five weights land on exactly 1.0 on one interpreter and
    0.9999999999999999 on the other — which is a CI failure, not a rounding
    nicety (commit bdf5e2d).
    """
    result = financial_curvature(_terms(), params=params, periods_per_year=PERIODS)

    assert len(result.weights) == 5
    assert math.fsum(result.weights.values()) == 1.0
    assert set(result.weights.values()) == {0.2}


def test_a_missing_term_renormalizes_the_rest_and_is_named(params):
    """The ``compute_rii`` pattern: four components, with the fifth named."""
    terms = _terms(names=tuple(t for t in CURVATURE_TERMS if t != "coupling"))

    result = financial_curvature(terms, params=params, periods_per_year=PERIODS)

    assert result.missing == ("coupling",)
    assert set(result.weights) == set(CURVATURE_TERMS) - {"coupling"}
    assert math.fsum(result.weights.values()) == 1.0
    assert all(weight == pytest.approx(0.25) for weight in result.weights.values())


def test_an_all_nan_term_is_dropped_rather_than_zero_filled(params):
    """Zero on an instability axis means "maximally stable", which is a claim."""
    terms = _terms()
    terms["coupling"] = pd.Series(np.nan, index=terms["coupling"].index)

    result = financial_curvature(terms, params=params, periods_per_year=PERIODS)

    assert "coupling" in result.missing
    assert "coupling" not in result.components


def test_a_term_too_short_to_score_is_dropped_and_named(params):
    terms = _terms()
    terms["coupling"] = terms["coupling"].iloc[:50]

    result = financial_curvature(terms, params=params, periods_per_year=PERIODS)

    assert "coupling" in result.missing


def test_every_term_enters_as_a_magnitude(params):
    """Sign-flipping one input must not change the composite.

    The rule ``rii.py`` states for jerk, applied to all five — and the test that
    would fail if a term were quietly left signed.
    """
    terms = _terms()
    flipped = {**terms, "omega_velocity": -terms["omega_velocity"]}

    original = financial_curvature(terms, params=params, periods_per_year=PERIODS)
    negated = financial_curvature(flipped, params=params, periods_per_year=PERIODS)

    pd.testing.assert_series_equal(original.curvature, negated.curvature, check_exact=True)


def test_the_coverage_floor_blanks_a_partial_composite(omega_config):
    """The departure from the RII, and the reason for it.

    With no floor, a date holding two of five terms publishes a renormalized
    two-term reading called curvature. On the shipped configuration the price
    terms start in 1927 and the Ω terms in 2000, so that is not a corner case —
    it was the largest reading in the whole series.
    """
    terms = _terms()
    # Half the calendar has only the two price terms.
    cut = 2_000
    for name in ("omega_velocity", "omega_acceleration", "coupling"):
        terms[name] = terms[name].iloc[cut:]

    strict = CurvatureParams.from_config(
        with_params(omega_config, curvature={"zscore_min_years": 2.0, "min_coverage": 0.999})
    )
    loose = CurvatureParams.from_config(
        with_params(omega_config, curvature={"zscore_min_years": 2.0, "min_coverage": 0.2})
    )

    blanked = financial_curvature(terms, params=strict, periods_per_year=PERIODS)
    partial = financial_curvature(terms, params=loose, periods_per_year=PERIODS)

    assert blanked.curvature.iloc[:cut].isna().all()
    assert partial.curvature.iloc[:cut].notna().any()
    assert blanked.diagnostics["curvature_blanked_by_coverage"] > 1_000
    # The coverage series is what makes the difference visible either way.
    assert blanked.coverage.iloc[:cut].max() < 0.999
    assert blanked.coverage.iloc[-1] == pytest.approx(1.0)


def test_the_fitted_mode_records_its_window_and_freezes_the_weights(omega_config):
    """Fitted on the FIRST N rows, so appending later data cannot move it.

    A fraction of the sample would move with the sample, which is the quiet way
    a frozen weighting becomes a full-sample one.
    """
    terms = _terms(n=4_000)
    fitted = CurvatureParams.from_config(
        with_params(
            omega_config,
            curvature={"weights": "fitted", "zscore_min_years": 2.0, "fit_observations": 800},
        )
    )

    short = financial_curvature(
        {name: series.iloc[:3_000] for name, series in terms.items()},
        params=fitted,
        periods_per_year=PERIODS,
    )
    long = financial_curvature(terms, params=fitted, periods_per_year=PERIODS)

    assert short.fit_start is not None and short.fit_end is not None
    assert (short.fit_start, short.fit_end) == (long.fit_start, long.fit_end)
    assert short.weights == long.weights
    assert math.fsum(long.weights.values()) == pytest.approx(1.0, abs=1e-15)
    # The raw loadings travel too, so a term moving opposite to the others is
    # visible rather than absorbed by the absolute value.
    assert all(f"curvature_loading_{term}" in long.diagnostics for term in long.weights)


def test_the_fitted_mode_refuses_a_window_it_cannot_fill(omega_config):
    terms = _terms(n=1_000)
    fitted = CurvatureParams.from_config(
        with_params(
            omega_config,
            curvature={"weights": "fitted", "zscore_min_years": 2.0, "fit_observations": 5_000},
        )
    )

    with pytest.raises(OmegaCurvatureError, match="complete row"):
        financial_curvature(terms, params=fitted, periods_per_year=PERIODS)


def test_there_is_no_third_weighting_mode(omega_config):
    """Anything else is lookahead, and it fails at config load rather than later."""
    with pytest.raises(OmegaCurvatureError, match="lookahead"):
        CurvatureParams.from_config(with_params(omega_config, curvature={"weights": "optimal"}))


def test_an_unknown_term_is_refused(params):
    terms = _terms()
    terms["entropy"] = terms["jerk"]

    with pytest.raises(OmegaCurvatureError, match="unknown curvature term"):
        financial_curvature(terms, params=params, periods_per_year=PERIODS)


def test_a_composite_with_no_usable_term_raises(params):
    with pytest.raises(OmegaCurvatureError, match="at least one usable term"):
        financial_curvature({}, params=params, periods_per_year=PERIODS)


def test_the_percentile_is_expanding_and_therefore_causal(params):
    """The percentile at date *t* ranks against ``0..t`` and nothing after.

    Scored by ``factors/compute.py::score_series``, the same function every
    factor and every RII component uses, so there is one definition of what 50
    means rather than two.
    """
    terms = _terms()
    result = financial_curvature(terms, params=params, periods_per_year=PERIODS)

    from findynamics.factors.compute import score_series

    defined = result.curvature.dropna()
    pd.testing.assert_series_equal(
        score_series(defined.iloc[:2_000], 1),
        score_series(defined, 1).iloc[:2_000],
        check_exact=True,
    )
    scored = result.percentile.dropna()
    assert scored.min() >= 0.0
    assert scored.max() <= 100.0


def test_contributions_decompose_the_composite(params):
    """A composite whose biggest reading comes from one term has one component."""
    terms = _terms()
    result = financial_curvature(terms, params=params, periods_per_year=PERIODS)

    peak = result.curvature.dropna().idxmax()
    contributions = result.contributions(peak)

    assert set(contributions) == set(result.weights)
    assert math.fsum(contributions.values()) == pytest.approx(
        float(result.curvature.loc[peak]), rel=1e-9
    )


@pytest.mark.parametrize(
    "block,message",
    [
        ({"weights": "fitted_on_everything"}, "lookahead"),
        ({"fit_observations": 1}, "fit_observations"),
        ({"zscore_min_years": 0.0}, "zscore_min_years"),
        ({"min_coverage": 0.0}, "min_coverage"),
        ({"min_coverage": 1.5}, "min_coverage"),
    ],
)
def test_a_malformed_curvature_config_fails_at_load(omega_config, block, message):
    with pytest.raises(OmegaCurvatureError, match=message):
        CurvatureParams.from_config(with_params(omega_config, curvature=block))
