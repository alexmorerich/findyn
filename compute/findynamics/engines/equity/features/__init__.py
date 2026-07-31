"""FinEquity's causal feature path (FINDYN_V1_SPEC.md §8).

    raw close -> log price -> Kalman (filtered) -> kinematics
                           -> fractional differentiation -> stationary memory

Every transform in this package sees only information at or before the date it
produces a value for. That is not a convention here, it is the package's reason
to exist: §8.1 permits Savitzky-Golay in the dashboard display layer and nowhere
else, because a centred window reads the future. There is no SG function under
``engines/equity`` at all, and CI greps for one.

The RTS smoother is banned for the same reason and is easier to import by
accident, so :mod:`.kalman` never calls ``smooth()`` — only ``filter()``, whose
results object does not carry a smoothed state to reach for.

Nothing in here knows which series it is looking at. The pipeline takes a series
and its frequency as parameters and runs identically over the publication path,
the calibration path and the monthly deep history — which is what makes
sub-milestone B's "fit on calibration, infer on publication" a matter of calling
the same function twice rather than of two code paths that have to be kept
honest by hand.
"""

from __future__ import annotations

from findynamics.engines.equity.features.ffd import (
    FfdFit,
    frac_diff,
    frac_diff_weights,
    search_d,
)
from findynamics.engines.equity.features.kalman import (
    KalmanParams,
    KalmanState,
    filter_state,
    fit_params,
)
from findynamics.engines.equity.features.kinematics import (
    JERK_ELEVATED_Z,
    JERK_EXTREME_Z,
    Kinematics,
    jerk_lamp,
    kinematics,
)
from findynamics.engines.equity.features.pipeline import (
    FeatureSet,
    compute_features,
)

__all__ = [
    "JERK_ELEVATED_Z",
    "JERK_EXTREME_Z",
    "FeatureSet",
    "FfdFit",
    "KalmanParams",
    "KalmanState",
    "Kinematics",
    "compute_features",
    "filter_state",
    "fit_params",
    "frac_diff",
    "frac_diff_weights",
    "jerk_lamp",
    "kinematics",
    "search_d",
]
