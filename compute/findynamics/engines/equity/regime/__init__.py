"""Regime detection — FINDYN_V1_SPEC.md §9 layers L2 and L3.

L2 is a five-state Gaussian HMM over the kinematic features; L3 is a calibrated
classifier for the probability of *transitioning* into the bad states within a
horizon. They answer different questions and the difference matters: the HMM
says where the market is now, the classifier says where it is likely to go.

**Fit on ``calibration``, infer on ``publication``.** The publication series
holds ten years and one drawdown; a five-state model fitted on it does not have
a crisis state, it has an outlier. So the parameters come from the longest daily
series available and are applied to the published index's features.

That transfer is only sound if a feature value means the same thing on both
series, which is what :mod:`.design` exists to guarantee and
``tests/engines/equity/test_transfer.py`` exists to prove. It is not an
implementation detail — with the calibration series a more volatile index than
the publication series, a Gaussian HMM fitted in raw feature space would encode
that volatility in its state means and covariances, and applied to S&P features
it would under-call ``crisis`` for a mechanical reason that reads as a market
judgement.
"""

from findynamics.engines.equity.regime.design import (
    HMM_FEATURES,
    DesignError,
    RegimeDesign,
    build_design,
    dispersion_ratio,
    distribution_summary,
    realized_vol,
)
from findynamics.engines.equity.regime.hmm import (
    HmmFit,
    RegimeFitError,
    RegimeModel,
    StateStats,
    fit_hmm,
    label_states,
    reward_to_risk,
)

__all__ = [
    "HMM_FEATURES",
    "DesignError",
    "HmmFit",
    "RegimeDesign",
    "RegimeFitError",
    "RegimeModel",
    "StateStats",
    "build_design",
    "dispersion_ratio",
    "distribution_summary",
    "fit_hmm",
    "label_states",
    "realized_vol",
    "reward_to_risk",
]
