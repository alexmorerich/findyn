"""The scale-free transfer — the property sub-milestone B rests on.

"Fit on ``calibration``, infer on ``publication``" is only sound if a feature
value means the same thing on both series.

**This is now a sanity check rather than the phase\'s central risk, and the
tests are deliberately not weakened to match.** A daily S&P source landed
(``YAHOO:^GSPC``, 1927+), so ``backfill`` takes the calibration role and the
model is fitted on the same index it publishes against. But ``regime_proxy``
remains the configured fallback for the day that source breaks — Yahoo is
unversioned and the spec calls it a fallback — and on that path the transfer
risk is exactly as sharp as it ever was: ``FRED:NASDAQ100`` is a third more
volatile than ``FRED:SP500``, and a model fitted in units carrying that
difference would sit its ``crisis`` state where the S&P never goes and under-call
crisis forever, for a mechanical reason indistinguishable from a market
judgement.

So the measurements below run against the **NASDAQ proxy specifically**, not
against whichever series happens to win the precedence today. Testing the
resolved calibration would silently stop testing anything the moment the
configuration improved — which is precisely when a fallback path starts rotting
unobserved.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from findynamics.engines.equity.domain import REGIMES
from findynamics.engines.equity.engine import ALL_ROLES
from findynamics.engines.equity.regime.design import (
    build_design,
    dispersion_ratio,
    distribution_summary,
)
from findynamics.engines.equity.regime.hmm import RegimeModel, fit_hmm
from tests.engines.equity.conftest import world_from

#: A feature's dispersion may differ this much between the two series before the
#: transfer is unsound. Measured 0.83-1.18 on the shipped snapshot; the raw
#: features run to 0.57 and 1.94, so the band separates the two cases with room
#: to spare rather than being drawn around the answer.
DISPERSION_BAND = (0.70, 1.40)

#: Episodes the transferred model must still find, and the share of each year's
#: sessions it has to call bear-or-crisis to count as finding them.
TRANSFER_STRESS_YEARS = {2018: 0.25, 2020: 0.25, 2022: 0.50}

#: Years with no material S&P drawdown. A transferred model that flags these is
#: not carrying across; it is just pessimistic.
TRANSFER_CALM_YEARS = {2021: 0.15, 2024: 0.15}

QUANTILES = (0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99)


@pytest.fixture(scope="module")
def designs(equity_observations, config_module):
    """Design matrices for the publication path and for the NASDAQ proxy.

    The proxy is built explicitly rather than taken from whichever series the
    precedence resolves to, so this file keeps measuring the fallback path after
    a better calibration source arrives.
    """
    from findynamics.core.artifacts import ArtifactStore
    from findynamics.engines.equity import prices as prices_mod
    from findynamics.engines.equity.engine import EquityEngine
    from findynamics.engines.equity.features.pipeline import compute_features

    world = world_from(equity_observations)
    engine = EquityEngine(config_module, ArtifactStore(directory=None))
    analysis = engine.analyze(world, roles=ALL_ROLES)

    built = {role: build_design(fs) for role, fs in analysis.features.items()}

    # The proxy, whether or not it won the role today.
    proxy_spec = prices_mod.configured_roles(config_module)["regime_proxy"]
    proxy_series = prices_mod.PriceSeries(
        role="calibration",
        source_role="regime_proxy",
        series_id=proxy_spec.id,
        frequency=proxy_spec.frequency,
        observations=0,
    )
    built["proxy"] = build_design(
        compute_features(prices_mod.price_path(world.series, proxy_series), proxy_series)
    )
    return built


@pytest.fixture(scope="module")
def config_module():
    from findynamics.core.config import load_series_config

    return load_series_config()


@pytest.fixture(scope="module")
def separate_fits(designs):
    """The same pipeline fitted independently on the S&P and on the proxy."""
    return {
        "publication": fit_hmm(designs["publication"], is_proxy=False),
        "proxy": fit_hmm(designs["proxy"], is_proxy=True),
    }


# --- the measurement ---------------------------------------------------------


def test_the_two_series_really_do_have_different_volatility(designs):
    """The premise. If they did not, this whole file would be vacuous."""
    pub = designs["publication"].returns.std() * np.sqrt(252)
    cal = designs["proxy"].returns.std() * np.sqrt(252)
    assert cal > pub * 1.2, (
        f"the calibration series ({cal:.1%}) is not materially more volatile than "
        f"the publication series ({pub:.1%}); the transfer risk this file tests "
        "for does not exist on this data and the assertions below prove nothing"
    )


def test_every_feature_has_comparable_dispersion_across_the_two_series(designs):
    """The blocker.

    Measured on common dates only: comparing a 1987-2026 window against a
    2018-2026 one would measure which episodes each series lived through rather
    than whether a value means the same thing on both.
    """
    ratios = dispersion_ratio(designs["publication"], designs["proxy"])
    low, high = DISPERSION_BAND

    offenders = {
        feature: round(float(ratio), 3)
        for feature, ratio in ratios.items()
        if not low <= ratio <= high
    }
    assert not offenders, (
        f"these features do not carry across the two indices: {offenders}. "
        "A Gaussian HMM fitted on one and applied to the other would place its "
        "states in the wrong region of feature space."
    )


def test_the_distribution_shapes_agree_in_the_tails(designs):
    """Dispersion alone would miss the failure that matters.

    A crisis state lives in the tails, so two features can share a standard
    deviation and still disagree about where 'extreme' is. The quantiles are
    compared over the common window for the same reason as above.
    """
    pub, cal = designs["publication"], designs["proxy"]
    common = pub.frame.index.intersection(cal.frame.index)

    def on_common(design):
        return replace(design, frame=design.frame.loc[common])

    gap = (
        (
            distribution_summary(on_common(pub), QUANTILES)
            - distribution_summary(on_common(cal), QUANTILES)
        )
        .abs()
        .max()
    )
    offenders = {f: round(float(v), 3) for f, v in gap.items() if v > 1.5}
    assert not offenders, f"quantile shapes diverge on {offenders}"


def test_a_proxy_fitted_model_still_finds_the_published_index_episodes(designs, separate_fits):
    """The transfer, measured by what it is *for*.

    An earlier version of this compared fitted state means and required them to
    sit within a fixed distance. That was a proxy for the real question and a bad
    one: it passed for a feature set whose trend column was nearly constant —
    degenerate features are trivially "comparable" — and failed once the columns
    actually discriminated, because two different indices genuinely lived through
    different episodes and their states land in different places. It was
    penalising the model for working.

    The operational question is the one worth asserting: fit on the proxy, apply
    to the S&P, and check the result still identifies the S&P's own stress and
    stays quiet the rest of the time. That is exactly what the fallback path
    would do in production.
    """
    transferred = RegimeModel(separate_fits["proxy"]).states(designs["publication"])
    adverse = transferred.isin(["bear", "crisis"])

    for year, floor in TRANSFER_STRESS_YEARS.items():
        share = float(adverse[adverse.index.year == year].mean())
        assert share >= floor, (
            f"a model fitted on {separate_fits['proxy'].fitted_on} called only "
            f"{share:.0%} of {year} adverse on the S&P (need {floor:.0%}); the "
            "fitted parameters are not carrying across the two indices"
        )

    for year, ceiling in TRANSFER_CALM_YEARS.items():
        share = float(adverse[adverse.index.year == year].mean())
        assert share <= ceiling, (
            f"the transferred model called {share:.0%} of {year} adverse "
            f"(limit {ceiling:.0%}); {year} had no material S&P drawdown, so this "
            "is pessimism rather than transfer"
        )


def test_the_scale_free_property_is_what_makes_the_transfer_possible(designs):
    """Dispersion is the mechanical property, and it is still the blocker.

    The episode test above can only pass if a feature value means the same thing
    on both series. This asserts the underlying reason directly, so a failure
    says *why* rather than only *that*.
    """
    ratios = dispersion_ratio(designs["publication"], designs["proxy"])
    low, high = DISPERSION_BAND
    offenders = {
        feature: round(float(ratio), 3)
        for feature, ratio in ratios.items()
        if not low <= ratio <= high
    }
    assert not offenders, f"these features do not carry across the two indices: {offenders}"


def test_the_assertions_reject_raw_dimensional_features(designs):
    """The control. A test that cannot fail is not evidence.

    Velocity and realized volatility in their own units are exactly what the
    design matrix avoids, and exactly what a plausible-looking implementation
    would have used. Both assertions above must reject them.
    """
    pub, cal = designs["publication"], designs["proxy"]
    common = pub.frame.index.intersection(cal.frame.index)

    def raw(design) -> pd.DataFrame:
        # Undo the normalization: multiply the dimensionless columns back by the
        # volatility they were divided by.
        vol = design.realized_vol.loc[common]
        return pd.DataFrame(
            {
                "trend": design.frame.loc[common, "trend_fast"] * vol,
                "drawdown": design.frame.loc[common, "drawdown_sigma"] * vol,
                "realized_vol": vol,
            }
        )

    ratios = raw(cal).std() / raw(pub).std()
    low, high = DISPERSION_BAND
    assert any(not low <= r <= high for r in ratios), (
        f"the raw features passed the dispersion band ({ratios.round(3).to_dict()}); "
        "the band is too wide to detect the problem it exists for"
    )


# --- what the transfer produces ----------------------------------------------


def test_the_publication_window_is_too_thin_to_fit_five_states_on(separate_fits, designs):
    """Why the fit runs on a longer series at all.

    The first version of this asserted that the S&P-only fit's ``bear`` state is
    *not* meaningfully negative. That passed locally and failed on CI at −10.0%,
    which is the correct lesson rather than a flaky test: whether ten years of
    one index happen to yield a negative fifth state is a property of one EM
    local optimum, not of the design.

    The durable claim is about sample size. A five-state model needs enough
    observations of each state for its covariance to mean anything, and the
    publication window has an order of magnitude fewer — which is the actual
    reason the fit runs elsewhere, and does not depend on where EM landed.
    """
    publication = designs["publication"]
    proxy = designs["proxy"]

    assert len(proxy) > 4 * len(publication), (
        f"the fitting series has {len(proxy)} rows against the publication "
        f"series' {len(publication)}; if that gap has closed, the split deserves "
        "revisiting"
    )

    cal_bear = separate_fits["proxy"].stats_for("bear")
    assert cal_bear is not None
    assert cal_bear.mean_return < 0.0, (
        f"the fitting series' bear state returns {cal_bear.mean_return:.1%}; a "
        "bear market that makes money is a labelling failure, not a finding"
    )


@pytest.mark.parametrize("which", ["publication", "proxy"])
def test_the_worst_states_actually_lose_money(separate_fits, which):
    """The vocabulary has to mean what it says on both series.

    Monotonicity in return is true by construction of the labelling rule; that
    ``bear`` and ``crisis`` come out *negative* is not, and it is the claim the
    names make.
    """
    fit = separate_fits[which]
    assert fit.stats_for("bull_expansion").mean_return > 0.10  # type: ignore[union-attr]
    assert fit.stats_for("crisis").mean_return < 0.0  # type: ignore[union-attr]


@pytest.mark.parametrize("which", ["publication", "proxy"])
def test_crisis_is_a_high_volatility_state(separate_fits, which):
    """A consequence of the labelling rule, checked rather than assumed.

    ``crisis`` is chosen by worst mean return, with volatility deliberately kept
    out of the key — the two rules that used it each mislabelled a real series
    (see the module docstring in regime/hmm.py). So the fact that the worst
    state also turns out to be a violent one is evidence the states mean what
    their names say, and it is asserted as median-or-above rather than as the
    maximum: on the 1927+ S&P the ``late_cycle`` state is a hair more volatile.
    """
    fit = separate_fits[which]
    vols = sorted(fit.stats_for(label).volatility for label in REGIMES)  # type: ignore[union-attr]
    crisis = fit.stats_for("crisis").volatility  # type: ignore[union-attr]
    assert crisis >= vols[len(vols) // 2], dict(
        zip(REGIMES, (fit.stats_for(x).volatility for x in REGIMES), strict=True)  # type: ignore[union-attr]
    )
