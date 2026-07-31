"""Fixed-width fractional differentiation (FINDYN_V1_SPEC.md §8.2).

López de Prado's method, and the reason it is here rather than a first
difference: a log price is non-stationary and a log *return* is stationary but
has thrown away every trace of where the level came from. Fractional
differencing at the smallest ``d`` that buys stationarity keeps as much of that
memory as the test will allow.

The weights come from the binomial expansion of ``(1 - B)^d``::

    w_0 = 1,    w_k = -w_{k-1} * (d - k + 1) / k

and are truncated where ``|w_k|`` falls below ``threshold``. Fixed-width, not
expanding: every value is a dot product over the same number of lags, so the
series has constant memory and is comparable across dates. The expanding variant
weights early observations differently from late ones, which shows up in a
regime fit as a drift that is really an artefact of the window growing.

**``d`` is searched, then frozen — per series.** The search runs on the training
window and the result is stored per series in the engine artifact. One shared
``d`` across series with different memory is a silent error: it would impose the
publication series' memory on the calibration series, and the difference between
"minimum d that passes ADF" for a 10-year window and for a 40-year window is
exactly the kind of quantity that has no business being assumed equal.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

log = logging.getLogger("findynamics.engines.equity.features.ffd")

#: Weights below this are dropped, fixing the window width.
DEFAULT_THRESHOLD = 1e-5

#: ADF significance the chosen ``d`` must clear (§8.2: 95% confidence).
DEFAULT_SIGNIFICANCE = 0.05

#: Search grid: ``d`` from 0 (untouched) to 1 (first difference) inclusive.
DEFAULT_D_GRID: tuple[float, ...] = tuple(round(0.05 * i, 2) for i in range(21))

#: Hard ceiling on the weight expansion, independent of any memory budget.
MAX_WIDTH = 5000

#: Default memory budget in years. The threshold alone does not bound the window
#: usefully: at ``d=0.3`` the weights decay like ``k**-1.3``, so ``1e-5`` is not
#: reached until roughly 2,300 lags — wider than the whole ten-year publication
#: series, which would leave a couple of hundred usable rows and no 12-month
#: momentum at all.
#:
#: So the width is capped explicitly. Truncating at N lags is the same operation
#: as raising the threshold to whatever ``|w_N|`` happens to be; stating it as a
#: span in years just makes the trade-off legible — this is how far back a value
#: is allowed to remember, and it costs that much history off the front of the
#: series. One year keeps the long-memory property that motivates FFD while
#: leaving a decade of usable features.
DEFAULT_MAX_MEMORY_YEARS = 1.0


class FfdError(ValueError):
    """Raised when the transform cannot be applied to the series given."""


@dataclass(frozen=True)
class FfdFit:
    """The frozen result of a ``d`` search on one series."""

    series_id: str
    d: float
    #: ADF p-value the winning ``d`` achieved on the search window.
    p_value: float
    threshold: float
    #: Number of lags the truncated weight vector spans.
    width: int
    #: Observations the search ran on — part of the audit trail, since the
    #: answer is only meaningful relative to the window it was searched on.
    observations: int
    #: False when no ``d`` on the grid reached significance and the fallback
    #: (full first differencing) was taken.
    passed: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "series_id": self.series_id,
            "d": self.d,
            "p_value": self.p_value,
            "threshold": self.threshold,
            "width": self.width,
            "observations": self.observations,
            "passed": self.passed,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> FfdFit:
        return cls(
            series_id=str(raw["series_id"]),
            d=float(raw["d"]),
            p_value=float(raw["p_value"]),
            threshold=float(raw.get("threshold", DEFAULT_THRESHOLD)),
            width=int(raw["width"]),
            observations=int(raw.get("observations", 0)),
            passed=bool(raw.get("passed", True)),
        )


def frac_diff_weights(
    d: float,
    *,
    threshold: float = DEFAULT_THRESHOLD,
    max_width: int | None = None,
) -> np.ndarray:
    """Truncated weights for ``(1 - B)^d``, newest lag last.

    Ordered so that the array lines up with a trailing window read oldest-first,
    which is what :func:`frac_diff` convolves against. ``max_width`` is the
    memory budget in lags; the expansion stops at whichever of the threshold and
    the budget bites first.
    """
    if d < 0.0:
        raise FfdError(f"d must be non-negative, got {d}")
    if not 0.0 < threshold < 1.0:
        raise FfdError(f"threshold must be in (0, 1), got {threshold}")

    ceiling = MAX_WIDTH if max_width is None else max(int(max_width), 1)
    ceiling = min(ceiling, MAX_WIDTH)

    weights = [1.0]
    for k in range(1, ceiling):
        weight = -weights[-1] * (d - k + 1.0) / k
        if abs(weight) < threshold:
            break
        weights.append(weight)
    # Reversed: index 0 multiplies the oldest lag in the window.
    return np.asarray(weights[::-1], dtype=float)


def frac_diff(
    series: pd.Series,
    d: float,
    *,
    threshold: float = DEFAULT_THRESHOLD,
    max_width: int | None = None,
) -> pd.Series:
    """Fixed-width fractionally differenced ``series``.

    The first ``width - 1`` positions have no full window and come back as NaN
    rather than being computed on a short one — a value built from fewer lags is
    a different transform wearing the same column name.
    """
    weights = frac_diff_weights(d, threshold=threshold, max_width=max_width)
    width = len(weights)
    values = np.asarray(series, dtype=float)

    out = np.full(values.shape, np.nan, dtype=float)
    if len(values) >= width:
        # Sliding dot product. correlate('valid') slides `weights` over `values`
        # without reversing it, which is what lines the oldest lag up with
        # weights[0]; convolve would flip it and invert the memory.
        out[width - 1 :] = np.correlate(values, weights, mode="valid")

    return pd.Series(out, index=series.index, name=f"ffd_{series.name}" if series.name else "ffd")


def adf_p_value(series: pd.Series) -> float:
    """Augmented Dickey-Fuller p-value, lag order by AIC.

    Returns 1.0 (maximally non-stationary) when the series is too short or
    degenerate for the test rather than raising: the caller is scanning a grid,
    and a ``d`` the test cannot evaluate is simply a ``d`` that does not win.
    """
    clean = series.dropna()
    if len(clean) < 20 or float(np.nanstd(clean)) == 0.0:
        return 1.0
    try:
        return float(adfuller(clean.to_numpy(), regression="c", autolag="AIC")[1])
    except (ValueError, np.linalg.LinAlgError) as err:
        log.warning("ADF failed on %s: %s", series.name, err)
        return 1.0


def search_d(
    series: pd.Series,
    *,
    series_id: str | None = None,
    grid: tuple[float, ...] = DEFAULT_D_GRID,
    threshold: float = DEFAULT_THRESHOLD,
    significance: float = DEFAULT_SIGNIFICANCE,
    max_width: int | None = None,
) -> FfdFit:
    """Smallest ``d`` on ``grid`` whose FFD series passes ADF at ``significance``.

    Smallest, not best: a lower p-value at higher ``d`` is more stationarity
    bought with more memory destroyed, and the memory is the reason for using
    this transform at all. The scan stops at the first grid point that clears the
    bar.

    When nothing on the grid clears it — a short or badly behaved window — the
    fit falls back to the top of the grid with ``passed=False``. The caller
    decides what to do with a failed search; silently returning a ``d`` that did
    not pass would put a non-stationary series into an HMM that assumes one.
    """
    name = series_id or (str(series.name) if series.name is not None else "series")
    clean = series.dropna()
    if clean.empty:
        raise FfdError(f"{name}: cannot search d on an empty series")

    for d in grid:
        differenced = frac_diff(clean, d, threshold=threshold, max_width=max_width)
        p_value = adf_p_value(differenced)
        if p_value < significance:
            width = len(frac_diff_weights(d, threshold=threshold, max_width=max_width))
            log.info(
                "ffd: %s settled on d=%.2f (ADF p=%.4g, %d lags, %d observations)",
                name,
                d,
                p_value,
                width,
                len(clean),
            )
            return FfdFit(
                series_id=name,
                d=float(d),
                p_value=p_value,
                threshold=threshold,
                width=width,
                observations=len(clean),
                passed=True,
            )

    fallback = float(grid[-1])
    p_value = adf_p_value(frac_diff(clean, fallback, threshold=threshold, max_width=max_width))
    log.warning(
        "ffd: no d in [%.2f, %.2f] made %s stationary at p<%.3g; falling back to d=%.2f (p=%.4g)",
        grid[0],
        grid[-1],
        name,
        significance,
        fallback,
        p_value,
    )
    return FfdFit(
        series_id=name,
        d=fallback,
        p_value=p_value,
        threshold=threshold,
        width=len(frac_diff_weights(fallback, threshold=threshold, max_width=max_width)),
        observations=len(clean),
        passed=False,
    )


__all__ = [
    "DEFAULT_D_GRID",
    "DEFAULT_SIGNIFICANCE",
    "DEFAULT_THRESHOLD",
    "DEFAULT_MAX_MEMORY_YEARS",
    "MAX_WIDTH",
    "FfdError",
    "FfdFit",
    "adf_p_value",
    "frac_diff",
    "frac_diff_weights",
    "search_d",
]
