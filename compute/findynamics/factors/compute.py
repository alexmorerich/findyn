"""Factor scoring — winsorized z, expanding percentile, 0-100 (§8 step 4).

Layer 0. Every factor is computed once per run from point-in-time frames and
handed to every engine through :class:`WorldState`, so two engines can never
disagree about what liquidity was doing on a given day.

Three properties are load-bearing:

* **Expanding, never centred.** The percentile of a value uses only the history
  up to and including that observation (§14.1 rule 3). A rolling *centred*
  window, or a percentile taken against the full sample, would let 2024 decide
  what counted as extreme in 2008.
* **Winsorized before standardizing.** One bad print — a provider glitch, a
  1970s inflation spike — otherwise sets the scale for every other observation.
* **One axis.** 100 is maximally risk-supportive, 0 maximally hostile;
  ``SeriesSpec.direction`` is what orients each series onto it. The raw levels
  survive in ``FactorState.components`` for consumers that want them.

Missing series degrade a factor rather than failing the run (§14.2); a factor
with no usable series is omitted from the result entirely, and
``WorldState.factor_score`` returns ``None`` for it.
"""

from __future__ import annotations

import logging
import math
from datetime import date

import numpy as np
import pandas as pd

from findynamics.core.config import FactorSpec, SeriesConfig, get_series_config
from findynamics.core.contracts.pit import PITAccessor
from findynamics.core.contracts.state import FactorState
from findynamics.factors.definitions import factor_specs

log = logging.getLogger("findynamics.factors")

#: Z-scores are clipped here before scoring. Wide enough to keep genuine stress
#: distinguishable, narrow enough that one outlier cannot flatten the rest.
WINSOR_Z = 3.0

#: An expanding percentile computed from a handful of points is noise wearing a
#: number. Below this many observations a series contributes nothing.
MIN_OBSERVATIONS = 24


def _expanding_z(series: pd.Series) -> pd.Series:
    """Z-score of each point against the history up to and including it.

    ``expanding()`` is what keeps this causal: mean and standard deviation at
    row *i* see rows ``0..i`` and nothing after.
    """
    mean = series.expanding(min_periods=2).mean()
    std = series.expanding(min_periods=2).std(ddof=0)
    # A constant history has no scale; calling that "zero deviations from the
    # mean" is the honest reading, not an error.
    z = (series - mean) / std.replace(0.0, np.nan)
    return z.fillna(0.0).clip(-WINSOR_Z, WINSOR_Z)


class _Fenwick:
    """Prefix-count tree — how many values seen so far fall at or below a rank."""

    def __init__(self, size: int) -> None:
        self._tree = [0] * (size + 1)

    def add(self, index: int) -> None:
        i = index + 1
        while i < len(self._tree):
            self._tree[i] += 1
            i += i & -i

    def count_upto(self, index: int) -> int:
        """Number of added values with rank <= ``index``."""
        i = index + 1
        total = 0
        while i > 0:
            total += self._tree[i]
            i -= i & -i
        return total


def _expanding_percentile(series: pd.Series) -> pd.Series:
    """Fraction of the history up to each point that the point exceeds, in 0-1.

    Expanding, so the percentile of a 2008 reading is its rank among 1960-2008
    and not among everything through today — that difference is the whole
    no-lookahead point. Ranks are accumulated in a Fenwick tree because the
    obvious ``expanding().rank()`` is quadratic, and these series run to tens of
    thousands of daily observations.

    Ties take the midpoint of their range, so equal values score identically
    regardless of arrival order.
    """
    values = series.to_numpy(dtype=float)
    if len(values) == 0:
        return pd.Series(dtype=float, index=series.index)

    # Rank-compress once: the tree indexes positions in the sorted distinct set,
    # which is a labelling of the values, not a use of their future order.
    distinct = np.unique(values)
    ranks = np.searchsorted(distinct, values)

    tree = _Fenwick(len(distinct))
    out = np.empty(len(values), dtype=float)
    for i, rank in enumerate(ranks):
        if i == 0:
            out[i] = 0.5
        else:
            below = tree.count_upto(int(rank) - 1) if rank > 0 else 0
            at_or_below = tree.count_upto(int(rank))
            out[i] = (below + at_or_below) / 2.0 / i
        tree.add(int(rank))
    return pd.Series(out, index=series.index)


def score_series(values: pd.Series, direction: int) -> pd.Series:
    """One series' contribution to a factor, as a 0-100 causal percentile.

    Winsorizing happens in z-space and the percentile is taken on the winsorized
    z rather than the raw level, so a series whose scale drifts over decades
    (M2, PAYEMS) is still comparable with one that does not.
    """
    clean = values.dropna()
    if len(clean) < MIN_OBSERVATIONS:
        return pd.Series(dtype=float)
    z = _expanding_z(clean)
    pct = _expanding_percentile(z)
    oriented = pct if direction >= 0 else 1.0 - pct
    return (oriented * 100.0).clip(0.0, 100.0)


def score_factor(
    spec: FactorSpec,
    frame: pd.DataFrame,
    as_of: date,
) -> FactorState | None:
    """Score one factor from a wide PIT frame, or ``None`` if nothing usable.

    ``frame`` is indexed by observation date with one column per series id, as
    produced by :meth:`PITAccessor.wide`. Series observe on different calendars,
    so each is scored on its own observations and only then aligned to the
    factor's own last-known values — resampling first would invent observations
    a daily series never had.
    """
    contributions: dict[str, float] = {}
    for series_spec in spec.series:
        column = frame.get(series_spec.id)
        if column is None:
            continue
        scored = score_series(column, series_spec.direction)
        if scored.empty:
            continue
        contributions[series_spec.id] = float(scored.iloc[-1])

    if not contributions:
        log.debug("factor %s has no usable series as of %s", spec.name, as_of)
        return None

    score = float(np.mean(list(contributions.values())))
    if not math.isfinite(score):
        return None

    # Raw levels alongside the scores: a dashboard that can only show "liquidity
    # is 34" cannot show why, and the score alone is not auditable.
    components = dict(contributions)
    for series_spec in spec.series:
        column = frame.get(series_spec.id)
        if column is None:
            continue
        level = column.dropna()
        if not level.empty and math.isfinite(float(level.iloc[-1])):
            components[f"{series_spec.id}:level"] = float(level.iloc[-1])

    return FactorState(
        name=spec.name,
        as_of=as_of,
        score=round(score, 4),
        components={k: round(v, 6) for k, v in components.items()},
    )


def compute_factors(
    series: PITAccessor,
    config: SeriesConfig | None = None,
) -> dict[str, FactorState]:
    """Every factor that can be scored from the information set behind ``series``.

    The accessor's cutoff is the information set: this function never sees a
    date, so it cannot be pointed at the wrong one.
    """
    resolved = config or get_series_config()
    specs = factor_specs(resolved)
    wanted = sorted({s.id for spec in specs.values() for s in spec.series})

    frame = series.wide(wanted)
    as_of = series.as_of

    states: dict[str, FactorState] = {}
    for name, spec in specs.items():
        state = score_factor(spec, frame, as_of)
        if state is not None:
            states[name] = state

    missing = sorted(set(specs) - set(states))
    if missing:
        log.warning("factors with no usable input as of %s: %s", as_of, ", ".join(missing))
    return states
