"""Ω̇, Ω̈, and the banding of Ω.

The temptation here is ``omega.diff()`` twice. ``features/kinematics.py`` opens
by explaining why that is wrong for price and the argument transfers exactly:
differencing a noisy series amplifies its noise by roughly the differencing
order, so a second difference of a PC1 is mostly microstructure wearing an
acceleration's name.

So Ω goes through the **same local-linear-trend Kalman filter price goes
through** — ``engines/equity/features/kalman.py::filter_state``, which calls
``filter()`` and never ``smooth()`` because the RTS smoother conditions on the
whole sample. Ω̇ is the filtered slope annualized by ``periods_per_year``; Ω̈ is
its first difference, annualized again, so the two are on consistent time units
the way ``kinematics.py`` puts velocity and acceleration on them.

Reusing that module rather than copying it is the point of placing this package
above ``engines`` in the layer stack. Two implementations of one filter drift,
and the drift looks like a market signal.

Ω is not a price, and the filter does not care
----------------------------------------------

``filter_state`` is named for its usual argument and its model is agnostic: a
local linear trend is a level plus a slope plus three variances, and nothing in
it assumes positivity or a log scale. Ω is centred near zero over its fit window
and takes both signs. What *would* be wrong is taking a log of it first, which
is why nothing here does.

Burn-in, for the same reason velocity has one
---------------------------------------------

The filter is initialized diffusely — the slope's standard error starts at 1e3
and shrinks as observations arrive. ``kinematics.py`` measured the consequence
on the century price path: the first published velocity was **+142% annualized**,
and because the z-baseline is expanding, that transient widens the denominator
for every later date and never washes out. The leading ``kalman_burn_in_years``
of Ω̇ is therefore discarded, using that module's own
:func:`~findynamics.engines.equity.features.kinematics.burn_in_window` so the
cap and the short-path behaviour cannot drift apart from it.

Ω itself keeps its full span. Only the *slope* has to be estimated out of a
diffuse prior; the level is pinned by the first observation.

Everything else is expanding, never rolling
-------------------------------------------

``omega_zscore`` is scored by ``kinematics.expanding_z`` against a baseline from
``kinematics.baseline_window`` — §8.3's expanding minimum-ten-years rule, the
same one ``jerk_z`` is held to, degrading and reporting itself on a short path
rather than publishing a column of NaN. ``omega_volatility`` is a trailing
standard deviation of Ω̇ and is the one genuinely windowed quantity here,
because "how fast is Ω moving lately" is a question about a span.

``omega_regime``
----------------

A three-state banding of Ω's **expanding percentile**, scored by the shared
causal scorer ``factors/compute.py::score_series``. Reused rather than
reimplemented for the reason ``rii.py`` gives for reusing it: two
implementations of "0-100 score" would eventually disagree about what 50 means.

This is a display and diagnostic banding of one continuous variable. It carries
no dynamics of its own — no transition matrix, no posterior, no persistence
prior — and it is **not** a second regime model. Whether it says anything the
existing five-state HMM does not is measured in KK3 §5 against the episode list
already in ``engines/equity/backtest.py::EPISODES``, and the answer is allowed
to be "it fires later and more often", which would be a complete result.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from findynamics.engines.equity.features.kalman import KalmanParams, filter_state
from findynamics.engines.equity.features.kinematics import (
    baseline_window,
    burn_in_window,
    expanding_z,
)
from findynamics.factors.compute import score_series
from findynamics.research.omega.config import OmegaConfig
from findynamics.research.omega.contracts import OmegaPath, OmegaSpec
from findynamics.research.omega.domain import OMEGA_REGIMES

log = logging.getLogger("findynamics.research.omega.dynamics")


class OmegaDynamicsError(ValueError):
    """Raised when Ω's derivatives cannot be taken from the path given."""


@dataclass(frozen=True)
class DynamicsParams:
    """The dynamics half of ``omega.yaml``."""

    kalman_burn_in_years: float = 1.0
    kalman_maxiter: int = 200
    volatility_months: float = 3.0
    zscore_min_years: float = 10.0
    transition_percentile: float = 60.0
    stress_percentile: float = 85.0

    @classmethod
    def from_config(cls, config: OmegaConfig) -> DynamicsParams:
        block = config.block("dynamics")
        regime = config.block("regime")

        transition = float(regime.get("transition_percentile", 60.0))
        stress = float(regime.get("stress_percentile", 85.0))
        if not 0.0 < transition < stress < 100.0:
            raise OmegaDynamicsError(
                "regime thresholds must satisfy 0 < transition < stress < 100, got "
                f"transition={transition} stress={stress}; overlapping bands would make the "
                "state a function of evaluation order rather than of Ω"
            )

        burn_in = float(block.get("kalman_burn_in_years", 1.0))
        if burn_in < 0.0:
            raise OmegaDynamicsError(f"dynamics.kalman_burn_in_years must be >= 0, got {burn_in}")

        # Positive, for the reason `FeatureParams.from_config` gives: both spans
        # are floored downstream, so a non-positive value here does not raise —
        # it produces a two-observation window under a config that says three
        # months, and nothing anywhere reports the discrepancy.
        volatility_months = float(block.get("volatility_months", 3.0))
        zscore_min_years = float(block.get("zscore_min_years", 10.0))
        for name, value in (
            ("volatility_months", volatility_months),
            ("zscore_min_years", zscore_min_years),
        ):
            if value <= 0.0:
                raise OmegaDynamicsError(f"dynamics.{name} must be > 0, got {value}")

        return cls(
            kalman_burn_in_years=burn_in,
            kalman_maxiter=int(block.get("kalman_maxiter", 200)),
            volatility_months=volatility_months,
            zscore_min_years=zscore_min_years,
            transition_percentile=transition,
            stress_percentile=stress,
        )


def omega_regime(
    percentile: pd.Series,
    *,
    transition_percentile: float,
    stress_percentile: float,
) -> pd.Series:
    """Band a 0-100 expanding percentile of Ω into the three latent states.

    Boundaries are closed on the left (``>=``), so a value exactly on a
    threshold takes the more stressed reading. Arbitrary but fixed: an
    open/closed choice made independently at two call sites is how a boundary
    date ends up labelled two different ways in two panels.
    """
    values = percentile.astype(float)
    labels = pd.Series(np.nan, index=values.index, dtype=object)
    known = values.notna()
    labels[known] = OMEGA_REGIMES[0]
    labels[known & (values >= transition_percentile)] = OMEGA_REGIMES[1]
    labels[known & (values >= stress_percentile)] = OMEGA_REGIMES[2]
    return labels.rename("omega_regime")


def omega_dynamics(
    omega: pd.Series,
    spec: OmegaSpec,
    *,
    periods_per_year: float,
    config: OmegaConfig,
    kalman_params: KalmanParams | None = None,
) -> OmegaPath:
    """The full :class:`OmegaPath` for one stitched causal Ω path.

    ``omega`` is the *stitched* out-of-sample path from the walk-forward, not a
    single window's transform: derivatives taken per window would restart at
    every refit boundary and put a discontinuity where the model has none.

    ``kalman_params`` frozen from an earlier window makes Ω̇ a **pure forward
    recursion**, so its value at *t* depends on ``Ω[0..t]`` and nothing after.
    Left as ``None`` the variances are re-estimated by maximum likelihood on the
    whole path handed in, which is correct expanding-window behaviour for a
    whole-history diagnostic and is *not* good enough for the walk-forward: an
    MLE that saw 2026 would be in every 2008 slope. KK3 always passes them;
    :meth:`OmegaEngine.fit_transform` deliberately does not, and says so.
    """
    params = DynamicsParams.from_config(config)

    clean = omega.dropna()
    if len(clean) < 2:
        raise OmegaDynamicsError(
            f"Ω has {len(clean)} finite value(s); a filtered slope needs at least two"
        )

    state = filter_state(clean, kalman_params, maxiter=params.kalman_maxiter, label="omega")

    burn_in, burn_in_is_short = burn_in_window(
        len(state.slope), periods_per_year, years=params.kalman_burn_in_years
    )
    settled = state.slope.copy()
    if burn_in:
        settled.iloc[:burn_in] = np.nan
    if burn_in_is_short:
        log.info(
            "omega dynamics: %d observation(s) cannot give up %.0f year(s) to the filter's "
            "start-up; discarding %d instead, so the earliest Ω̇ carries more of the diffuse "
            "prior than usual",
            len(state.slope),
            params.kalman_burn_in_years,
            burn_in,
        )

    velocity = (settled * periods_per_year).rename("omega_velocity")
    # diff() of an annualized rate is per period; scaling again gives per year
    # squared, so Ω̈ and Ω̇ are on consistent time units — kinematics.py's
    # convention, so the two blocks can be plotted on comparable axes.
    acceleration = (velocity.diff() * periods_per_year).rename("omega_acceleration")

    volatility_window = max(int(round(params.volatility_months * periods_per_year / 12.0)), 2)
    volatility = (
        velocity.rolling(window=volatility_window, min_periods=volatility_window)
        .std()
        .rename("omega_volatility")
    )

    baseline_periods, baseline_is_short = baseline_window(
        len(clean), periods_per_year, min_years=params.zscore_min_years
    )
    if baseline_is_short:
        log.info(
            "omega dynamics: %d observation(s) is under %.0f years; the Ω z-score baseline "
            "uses %d periods instead",
            len(clean),
            params.zscore_min_years,
            baseline_periods,
        )
    zscore = expanding_z(clean, baseline_periods).rename("omega_zscore")

    # The same causal 0-100 scorer every factor and every RII component uses.
    # direction=+1: a higher Ω is a more stressed reading by the sign-pinning
    # convention, so 100 is the stressed end of the axis.
    percentile = score_series(clean, 1)
    regime = omega_regime(
        percentile.reindex(clean.index),
        transition_percentile=params.transition_percentile,
        stress_percentile=params.stress_percentile,
    )

    index = omega.index
    path = OmegaPath(
        omega=omega.rename("omega"),
        omega_velocity=velocity.reindex(index),
        omega_acceleration=acceleration.reindex(index),
        omega_volatility=volatility.reindex(index),
        omega_zscore=zscore.reindex(index),
        omega_regime=regime.reindex(index),
        spec=spec,
        diagnostics={
            "observations": float(len(clean)),
            "burn_in_periods": float(burn_in),
            "burn_in_is_short": float(burn_in_is_short),
            "zscore_baseline_periods": float(baseline_periods),
            "zscore_baseline_is_short": float(baseline_is_short),
            "volatility_window": float(volatility_window),
            "kalman_sigma2_irregular": state.params.irregular,
            "kalman_sigma2_level": state.params.level,
            "kalman_sigma2_trend": state.params.trend,
            "kalman_converged": float(state.converged),
            "explained_variance_ratio": spec.explained_variance_ratio,
            "columns_used": float(len(spec.columns)),
            "columns_dropped": float(len(spec.dropped)),
        },
    )
    log.info(
        "omega dynamics: %d row(s) %s → %s; %d discarded to filter start-up; "
        "Ω̇ over %d finite row(s); latent_stress on %.1f%% of scored dates",
        len(path),
        index[0].date(),
        index[-1].date(),
        burn_in,
        int(path.omega_velocity.notna().sum()),
        100.0 * float((regime == OMEGA_REGIMES[2]).sum()) / max(int(regime.notna().sum()), 1),
    )
    return path


__all__ = ["DynamicsParams", "OmegaDynamicsError", "omega_dynamics", "omega_regime"]
