"""The scale-free transfer — the assumption sub-milestone B rests on.

"Fit on ``calibration``, infer on ``publication``" is only sound if a feature
value means the same thing on both series. With ``FRED:NASDAQ100`` a third more
volatile than ``FRED:SP500``, a model fitted in units that carry that difference
would sit its ``crisis`` state where the S&P never goes, and would then under-call
crisis forever — for a mechanical reason indistinguishable, from outside, from a
considered market judgement.

So this file does not assume it. It measures it, on the real series, and it
includes a control that fails: the same assertions applied to the raw
(dimensional) features must reject, or the tolerances are too loose to mean
anything.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from findynamics.engines.equity.domain import REGIMES
from findynamics.engines.equity.engine import ALL_ROLES
from findynamics.engines.equity.regime.design import (
    HMM_FEATURES,
    build_design,
    dispersion_ratio,
    distribution_summary,
)
from findynamics.engines.equity.regime.hmm import fit_hmm
from tests.engines.equity.conftest import world_from

#: A feature's dispersion may differ this much between the two series before the
#: transfer is unsound. Measured 0.83-1.18 on the shipped snapshot; the raw
#: features run to 0.57 and 1.94, so the band separates the two cases with room
#: to spare rather than being drawn around the answer.
DISPERSION_BAND = (0.70, 1.40)

#: Maximum acceptable distance between like-labelled state means when the
#: pipeline is fitted on each series separately. Measured 0.66; the
#: rolling-standardized variant that loses direction reaches 2.46.
MAX_STATE_MEAN_GAP = 1.20

QUANTILES = (0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99)


@pytest.fixture(scope="module")
def designs(equity_observations, config_module):
    """Design matrices for all three roles, built once — the fits are slow."""
    from findynamics.core.artifacts import ArtifactStore
    from findynamics.engines.equity.engine import EquityEngine

    engine = EquityEngine(config_module, ArtifactStore(directory=None))
    analysis = engine.analyze(world_from(equity_observations), roles=ALL_ROLES)
    return {role: build_design(fs) for role, fs in analysis.features.items()}


@pytest.fixture(scope="module")
def config_module():
    from findynamics.core.config import load_series_config

    return load_series_config()


@pytest.fixture(scope="module")
def separate_fits(designs):
    """The same pipeline fitted independently on each daily series."""
    return {
        role: fit_hmm(designs[role], is_proxy=(role == "calibration"))
        for role in ("publication", "calibration")
    }


# --- the measurement ---------------------------------------------------------


def test_the_two_series_really_do_have_different_volatility(designs):
    """The premise. If they did not, this whole file would be vacuous."""
    pub = designs["publication"].returns.std() * np.sqrt(252)
    cal = designs["calibration"].returns.std() * np.sqrt(252)
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
    ratios = dispersion_ratio(designs["publication"], designs["calibration"])
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
    pub, cal = designs["publication"], designs["calibration"]
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


def test_state_means_agree_when_each_series_is_fitted_separately(separate_fits):
    """The prompt's assertion, stated as a number.

    Fit the identical pipeline on each series alone. If the features are
    comparable the two models place their like-labelled states in nearly the same
    place; if they are not, the more volatile series pushes its states outward
    and the gap blows up.
    """
    pub, cal = separate_fits["publication"], separate_fits["calibration"]

    worst_label, worst_feature, worst_gap = "", "", 0.0
    for label in REGIMES:
        pub_mean = pub.means[pub.labels.index(label)]
        cal_mean = cal.means[cal.labels.index(label)]
        for feature, left, right in zip(HMM_FEATURES, pub_mean, cal_mean, strict=True):
            gap = abs(left - right)
            if gap > worst_gap:
                worst_label, worst_feature, worst_gap = label, feature, gap

    assert worst_gap <= MAX_STATE_MEAN_GAP, (
        f"{worst_label}.{worst_feature} differs by {worst_gap:.3f} between the two "
        f"fits (limit {MAX_STATE_MEAN_GAP}); the features are not carrying across "
        "the two indices and the fitted model must not be transferred"
    )


def test_the_assertions_reject_raw_dimensional_features(designs):
    """The control. A test that cannot fail is not evidence.

    Velocity and realized volatility in their own units are exactly what the
    design matrix avoids, and exactly what a plausible-looking implementation
    would have used. Both assertions above must reject them.
    """
    pub, cal = designs["publication"], designs["calibration"]
    common = pub.frame.index.intersection(cal.frame.index)

    def raw(design) -> pd.DataFrame:
        # Undo the normalization: multiply the dimensionless columns back by the
        # volatility they were divided by.
        vol = design.realized_vol.loc[common]
        return pd.DataFrame(
            {
                "velocity": design.frame.loc[common, "trend_to_noise"] * vol,
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


def test_the_publication_series_alone_cannot_define_a_bear_state(separate_fits):
    """Why the fit runs on the calibration series at all.

    Ten years containing one sustained drawdown do not describe five regimes.
    Asserting the S&P-only fit's ``bear`` state is *not* meaningfully negative is
    asserting the premise of the whole design — and it is a claim about this
    window, so it is checked rather than assumed.
    """
    pub_bear = separate_fits["publication"].stats_for("bear")
    cal_bear = separate_fits["calibration"].stats_for("bear")
    assert pub_bear is not None and cal_bear is not None

    assert pub_bear.mean_return > -0.05, (
        f"the S&P-only fit produced a genuinely negative bear state "
        f"({pub_bear.mean_return:.1%}); if the publication window has grown enough "
        "history to define one, the calibration split deserves revisiting"
    )
    assert cal_bear.mean_return < 0.0, (
        f"the calibration fit's bear state returns {cal_bear.mean_return:.1%}; a "
        "bear market that makes money is a labelling failure, not a finding"
    )


def test_the_calibration_fit_is_monotone_in_return(separate_fits):
    """The vocabulary is ordered most to least risk-on, so the fit must be too."""
    fit = separate_fits["calibration"]
    returns = [fit.stats_for(label).mean_return for label in REGIMES]  # type: ignore[union-attr]
    assert returns == sorted(returns, reverse=True), dict(zip(REGIMES, returns, strict=True))


def test_crisis_is_the_most_volatile_state_on_the_calibration_fit(separate_fits):
    """Not the labelling rule — a consequence of it, and worth checking.

    ``crisis`` is chosen by worst reward-per-risk, not by highest volatility. On
    a long series carrying real crises the two coincide, and their coinciding is
    evidence the states mean what their names say.
    """
    fit = separate_fits["calibration"]
    vols = {label: fit.stats_for(label).volatility for label in REGIMES}  # type: ignore[union-attr]
    assert max(vols, key=vols.get) == "crisis", vols
