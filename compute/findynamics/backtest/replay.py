"""Replay: recompute a past state from a past information set.

FINDYN_V1_SPEC.md §14.1 rule 5 asks for a test that recomputes historical output
using only data with ``release_date <= cutoff`` and checks it matches what was
stored. That test is worth more than every other no-lookahead guard combined,
because it is the only one that can catch lookahead which entered through model
code rather than through the data layer: if a feature secretly depends on the
future, the value computed at the cutoff will differ from the value computed
today for that same date.

The helper is engine-agnostic — it takes an engine and an observation frame and
walks the cutoffs — so P2-P6 reuse it rather than each writing their own.

What replay can and cannot prove:

* It **can** prove that a state is a function of its information set alone.
* It **cannot** prove the model is right. A model that is consistently wrong
  replays perfectly.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import date

import pandas as pd

from findynamics.core.config import SeriesConfig, get_series_config
from findynamics.core.contracts.state import AssetState, WorldState
from findynamics.core.engine import AssetEngine
from findynamics.data.accessor import PandasPITAccessor
from findynamics.factors.compute import compute_factors

log = logging.getLogger("findynamics.backtest.replay")


@dataclass(frozen=True)
class ReplayPoint:
    """One cutoff's result."""

    cutoff: date
    state: AssetState | None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.state is not None


@dataclass(frozen=True)
class Mismatch:
    """A field that differs between the replayed and the stored state."""

    cutoff: date
    field: str
    replayed: object
    stored: object

    def __str__(self) -> str:
        return f"{self.cutoff}: {self.field} replayed={self.replayed!r} stored={self.stored!r}"


def world_at(
    observations: pd.DataFrame,
    cutoff: date,
    *,
    config: SeriesConfig | None = None,
    with_factors: bool = True,
) -> WorldState:
    """A :class:`WorldState` clamped to ``cutoff``.

    The accessor binds the cutoff, so nothing downstream can widen it. Factor
    scoring is optional because it is the expensive part and most engine replays
    do not read factors — but when an engine does, scoring them from a *later*
    information set would be exactly the contamination this is testing for.
    """
    accessor = PandasPITAccessor(observations, cutoff)
    resolved = config or get_series_config()
    factors = compute_factors(accessor, resolved) if with_factors else {}
    return WorldState(as_of=cutoff, factors=factors, series=accessor)


def replay(
    engine: AssetEngine,
    observations: pd.DataFrame,
    cutoffs: Iterable[date],
    *,
    config: SeriesConfig | None = None,
    with_factors: bool = True,
) -> list[ReplayPoint]:
    """Run ``engine.predict`` once per cutoff, each on its own information set.

    A cutoff at which the engine cannot speak (too little history, no curve yet)
    is recorded rather than raised: the early years of any series legitimately
    have nothing to say, and a replay that aborts on the first of them tests
    nothing.
    """
    points: list[ReplayPoint] = []
    for cutoff in cutoffs:
        world = world_at(observations, cutoff, config=config, with_factors=with_factors)
        try:
            points.append(ReplayPoint(cutoff=cutoff, state=engine.predict(world)))
        except Exception as err:
            log.info("replay at %s produced no state: %s", cutoff, err)
            points.append(ReplayPoint(cutoff=cutoff, state=None, error=str(err)))
    return points


#: Fields compared by default. ``model_version`` is included on purpose: a state
#: that matches numerically but came from a different model version is not the
#: same state.
COMPARED_FIELDS: tuple[str, ...] = (
    "asset",
    "as_of",
    "regime",
    "expected_return",
    "risk_score",
    "confidence",
    "model_version",
)


def compare(
    replayed: AssetState,
    stored: AssetState,
    *,
    cutoff: date,
    tolerance: float = 1e-9,
    fields: Sequence[str] = COMPARED_FIELDS,
    components: bool = True,
) -> list[Mismatch]:
    """Field-by-field difference between two states.

    Floats compare within ``tolerance`` rather than exactly: the same
    computation over the same inputs can differ in the last bit depending on
    how BLAS decided to order a sum, and calling that lookahead would make the
    test cry wolf. Anything larger is a real divergence.
    """
    mismatches: list[Mismatch] = []

    for field in fields:
        left, right = getattr(replayed, field), getattr(stored, field)
        if _differs(left, right, tolerance):
            mismatches.append(Mismatch(cutoff, field, left, right))

    if components:
        left_components = replayed.components or {}
        right_components = stored.components or {}
        for key in sorted(set(left_components) | set(right_components)):
            left, right = left_components.get(key), right_components.get(key)
            if _differs(left, right, tolerance):
                mismatches.append(Mismatch(cutoff, f"components[{key}]", left, right))

    return mismatches


def _differs(left: object, right: object, tolerance: float) -> bool:
    if isinstance(left, int | float) and isinstance(right, int | float):
        if isinstance(left, bool) or isinstance(right, bool):
            return left is not right
        return abs(float(left) - float(right)) > tolerance
    return left != right


def assert_replays(
    engine: AssetEngine,
    observations: pd.DataFrame,
    stored: dict[date, AssetState],
    *,
    config: SeriesConfig | None = None,
    tolerance: float = 1e-9,
    with_factors: bool = True,
) -> None:
    """Assert every stored state is reproducible from its own information set.

    Raises ``AssertionError`` listing every divergence, not just the first —
    when lookahead is present it usually shows up in several fields at once, and
    seeing all of them is what points at the cause.
    """
    points = replay(engine, observations, sorted(stored), config=config, with_factors=with_factors)

    failures: list[str] = []
    for point in points:
        expected = stored[point.cutoff]
        if point.state is None:
            failures.append(f"{point.cutoff}: no state was produced ({point.error})")
            continue
        failures.extend(
            str(m) for m in compare(point.state, expected, cutoff=point.cutoff, tolerance=tolerance)
        )

    if failures:
        raise AssertionError(
            "replayed state differs from the stored run at "
            f"{len(failures)} field(s) — this is what lookahead looks like:\n  "
            + "\n  ".join(failures)
        )
