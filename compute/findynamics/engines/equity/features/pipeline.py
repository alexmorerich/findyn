"""The feature path end to end, over whichever series it is handed.

    close -> log -> Kalman (filtered) -> velocity/acceleration/jerk_z
                 -> FFD(d frozen per series) -> momentum

One function, three callers. The publication path is what the live state
describes; the calibration path is what sub-milestone B fits the HMM on; the
monthly deep-history path is what sub-milestone C fits the tail on. They are the
same transform because "fit on calibration, infer on publication" is only
meaningful if the two feature spaces were built the same way — two pipelines
kept in step by hand would drift, and the drift would look like a market signal.

The series is a parameter and so is everything frequency-dependent: annualization
and momentum windows come from :attr:`PriceSeries.periods_per_year`, and ``d``
comes from that series' own :class:`FfdFit`. There is no module-level default
that quietly means "daily S&P".
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

# Names, not modules. ``features/__init__`` re-exports a function called
# ``kinematics`` alongside the module of the same name, so ``from ... import
# kinematics`` would resolve to whichever the package bound last — an ambiguity
# that is not worth carrying for the sake of a module alias.
from findynamics.engines.equity.features.ffd import (
    DEFAULT_D_GRID,
    DEFAULT_MAX_MEMORY_YEARS,
    DEFAULT_SIGNIFICANCE,
    DEFAULT_THRESHOLD,
    FfdFit,
    frac_diff,
    search_d,
)
from findynamics.engines.equity.features.kalman import (
    DEFAULT_MAXITER,
    KalmanParams,
    KalmanState,
    filter_state,
)
from findynamics.engines.equity.features.kinematics import (
    DEFAULT_MOMENTUM_MONTHS,
    DEFAULT_Z_MIN_YEARS,
    Kinematics,
)
from findynamics.engines.equity.features.kinematics import kinematics as compute_kinematics
from findynamics.engines.equity.prices import PriceSeries

log = logging.getLogger("findynamics.engines.equity.features.pipeline")

#: Written to ``derived_features``; the vocabulary is fixed by
#: engines/equity/domain.py::KINEMATIC_FEATURES.
CORE_COLUMNS: tuple[str, ...] = (
    "price_filtered",
    "velocity",
    "acceleration",
    "jerk_z",
    "ffd_price",
)


@dataclass(frozen=True)
class FeatureParams:
    """Everything configurable about the transform, from ``equity.yaml``."""

    ffd_threshold: float = DEFAULT_THRESHOLD
    ffd_significance: float = DEFAULT_SIGNIFICANCE
    ffd_grid: tuple[float, ...] = DEFAULT_D_GRID
    #: Memory budget in years, converted to lags against the series' own
    #: frequency — so "one year" is 252 lags on a daily path and 12 on a monthly
    #: one, rather than one number meaning two different spans.
    ffd_max_memory_years: float = DEFAULT_MAX_MEMORY_YEARS
    z_min_years: float = DEFAULT_Z_MIN_YEARS
    momentum_months: tuple[int, ...] = DEFAULT_MOMENTUM_MONTHS
    kalman_maxiter: int = DEFAULT_MAXITER

    @classmethod
    def from_params(cls, params: dict[str, Any] | None) -> FeatureParams:
        block = (params or {}).get("features") or {}
        if not isinstance(block, dict):
            return cls()
        grid = block.get("ffd_grid")
        months = block.get("momentum_months")
        return cls(
            ffd_threshold=float(block.get("ffd_threshold", DEFAULT_THRESHOLD)),
            ffd_significance=float(block.get("ffd_significance", DEFAULT_SIGNIFICANCE)),
            ffd_grid=tuple(float(d) for d in grid) if grid else DEFAULT_D_GRID,
            ffd_max_memory_years=float(block.get("ffd_max_memory_years", DEFAULT_MAX_MEMORY_YEARS)),
            z_min_years=float(block.get("z_min_years", DEFAULT_Z_MIN_YEARS)),
            momentum_months=(tuple(int(m) for m in months) if months else DEFAULT_MOMENTUM_MONTHS),
            kalman_maxiter=int(block.get("kalman_maxiter", DEFAULT_MAXITER)),
        )


@dataclass(frozen=True)
class FrozenFeatureParams:
    """What a refit froze for one series, read back by the daily run.

    Both halves are per series and neither is optional in spirit: ``d`` searched
    on one series and applied to another imposes the wrong memory, and Kalman
    variances estimated on one index and applied to another impose the wrong
    signal-to-noise. Absent, the daily run re-estimates on its own information
    set — causal, reproducible, and slower — and says so.
    """

    kalman: KalmanParams | None = None
    ffd: FfdFit | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "kalman": None if self.kalman is None else self.kalman.as_dict(),
            "ffd": None if self.ffd is None else self.ffd.as_dict(),
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> FrozenFeatureParams:
        body = raw or {}
        kalman = body.get("kalman")
        ffd = body.get("ffd")
        return cls(
            kalman=KalmanParams.from_dict(kalman) if kalman else None,
            ffd=FfdFit.from_dict(ffd) if ffd else None,
        )


@dataclass(frozen=True)
class FeatureSet:
    """The computed feature path for one series at one information set."""

    series: PriceSeries
    #: Index ``obs_date``; :data:`CORE_COLUMNS` plus one column per momentum window.
    frame: pd.DataFrame
    #: Raw log price, kept beside the filtered level for the chart and for the
    #: FFD input — the transform runs on the observed series, not on the filter's
    #: output, so a filter misspecification cannot leak into the memory features.
    log_price: pd.Series
    kalman: KalmanState
    ffd: FfdFit
    kinematics: Kinematics
    #: True when ``d`` and the Kalman variances came from a refit rather than
    #: being re-estimated on this run's window.
    used_frozen_params: bool
    diagnostics: dict[str, float] = field(default_factory=dict)

    @property
    def as_of(self) -> date | None:
        """Newest date the features can speak about."""
        usable = self.frame.dropna(subset=["price_filtered"])
        if usable.empty:
            return None
        return usable.index[-1].date()

    def latest(self) -> dict[str, float]:
        """The newest finite value of every feature column."""
        out: dict[str, float] = {}
        for column in self.frame.columns:
            series = self.frame[column].dropna()
            if not series.empty and np.isfinite(series.iloc[-1]):
                out[column] = float(series.iloc[-1])
        return out

    def frozen(self) -> FrozenFeatureParams:
        """What a refit should persist so the next daily run reproduces this."""
        return FrozenFeatureParams(kalman=self.kalman.params, ffd=self.ffd)


def compute_features(
    path: pd.Series,
    series: PriceSeries,
    *,
    params: FeatureParams | None = None,
    frozen: FrozenFeatureParams | None = None,
) -> FeatureSet:
    """Run the whole feature path over one price series.

    ``path`` is a close history, oldest first, already clipped to an information
    set by the caller — this function has no cutoff of its own, which is exactly
    why it is safe to reuse across the publication, calibration and deep-history
    roles.
    """
    resolved = params or FeatureParams()
    frozen_params = frozen or FrozenFeatureParams()

    clean = path.dropna().sort_index()
    if clean.empty:
        raise ValueError(f"{series.series_id}: no observations to build features from")

    log_price = np.log(clean).rename("log_price")

    state = filter_state(
        log_price,
        frozen_params.kalman,
        maxiter=resolved.kalman_maxiter,
        label=series.series_id,
    )

    # The budget in lags, against this series' own frequency.
    max_width = max(int(round(resolved.ffd_max_memory_years * series.periods_per_year)), 1)

    ffd_fit = frozen_params.ffd
    if ffd_fit is None:
        ffd_fit = search_d(
            log_price,
            series_id=series.series_id,
            grid=resolved.ffd_grid,
            threshold=resolved.ffd_threshold,
            significance=resolved.ffd_significance,
            max_width=max_width,
        )
    elif ffd_fit.series_id != series.series_id:
        # A frozen d belongs to the series it was searched on. Applying one
        # series' memory to another is the silent error this guard exists for.
        raise ValueError(
            f"frozen FFD d was searched on {ffd_fit.series_id!r} but is being "
            f"applied to {series.series_id!r}; d is frozen per series"
        )

    # The frozen fit's own width is the ceiling, not the current config's: a
    # refit froze a transform, and reproducing it means reproducing its window
    # even if the budget has since been re-tuned.
    ffd_price = frac_diff(
        log_price, ffd_fit.d, threshold=ffd_fit.threshold, max_width=ffd_fit.width
    ).rename("ffd_price")

    kin = compute_kinematics(
        state.slope,
        periods_per_year=series.periods_per_year,
        ffd=ffd_price,
        momentum_months=resolved.momentum_months,
        z_min_years=resolved.z_min_years,
    )

    columns: dict[str, pd.Series] = {
        # The spec's "filtered log index level" (§2.1). Logs, not index points:
        # every derivative below is taken in log space, and a chart that wants
        # index points exponentiates one column rather than the engine
        # publishing the same quantity twice in two units.
        "price_filtered": state.level,
        "velocity": kin.velocity,
        "acceleration": kin.acceleration,
        "jerk_z": kin.jerk_z,
        "ffd_price": ffd_price,
    }
    columns.update(kin.momentum)

    frame = pd.DataFrame(columns)
    frame.index.name = "obs_date"

    diagnostics = {
        "ffd_d": ffd_fit.d,
        "ffd_p_value": ffd_fit.p_value,
        "ffd_width": float(ffd_fit.width),
        "ffd_passed": float(ffd_fit.passed),
        "kalman_sigma2_irregular": state.params.irregular,
        "kalman_sigma2_level": state.params.level,
        "kalman_sigma2_trend": state.params.trend,
        "kalman_converged": float(state.converged),
        "jerk_baseline_periods": float(kin.baseline_periods),
        "jerk_baseline_is_short": float(kin.baseline_is_short),
        "observations": float(len(clean)),
    }

    log.info(
        "%s features: %d rows, d=%.2f (p=%.3g), %d momentum window(s), frozen=%s",
        series.series_id,
        len(frame),
        ffd_fit.d,
        ffd_fit.p_value,
        len(kin.momentum),
        frozen_params.kalman is not None or frozen_params.ffd is not None,
    )

    return FeatureSet(
        series=series,
        frame=frame,
        log_price=log_price,
        kalman=state,
        ffd=ffd_fit,
        kinematics=kin,
        used_frozen_params=frozen_params.kalman is not None and frozen_params.ffd is not None,
        diagnostics=diagnostics,
    )


__all__ = [
    "CORE_COLUMNS",
    "FeatureParams",
    "FeatureSet",
    "FrozenFeatureParams",
    "compute_features",
]
