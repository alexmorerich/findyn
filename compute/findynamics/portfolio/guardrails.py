"""Hard constraints every allocation must satisfy (Phase 6).

These are not the model — they are the fence around it. The allocator in
``allocate.py`` proposes weights; this module makes them legal:

* weights are non-negative and sum to 1;
* no asset exceeds its configured cap;
* the experimental engine (crypto) never carries weight unless explicitly
  configured in (``01-target-architecture.md`` §3 rule 5);
* a missing or stale engine's asset falls back to the profile's neutral weight,
  and the run is flagged ``degraded``.

The functions are pure and operate on plain ``{asset: weight}`` maps, so they can
be unit-tested against contrived inputs the allocator would never produce — a
guardrail is only worth having if it holds for weights nobody meant to make.
"""

from __future__ import annotations

import math
from collections.abc import Collection, Mapping

#: Engines whose class declares ``experimental = True``. Mirrors
#: ``core.registry.experimental_engines`` and ``serving/src/domain.ts``; kept as
#: a constant here so a guardrail check needs no registry lookup, and the parity
#: test asserts it against the registry.
EXPERIMENTAL_ASSETS: tuple[str, ...] = ("crypto",)

#: Sum-to-one and cap checks tolerate this much floating-point slack. A tenth of
#: a basis point — larger than accumulated rounding, far smaller than any tilt.
DEFAULT_TOLERANCE = 1e-9


class GuardrailError(ValueError):
    """Raised when an allocation violates a hard constraint."""


def enforce_caps(
    weights: Mapping[str, float],
    caps: Mapping[str, float],
    *,
    max_iter: int = 64,
    eps: float = 1e-12,
) -> dict[str, float]:
    """Clip weights to their caps and redistribute the freed budget, summing to 1.

    Water-filling: any asset over its cap is set to the cap, and the excess is
    spread across the assets still below theirs, in proportion to their current
    weight. Redistributing can push another asset over its cap, so the step
    repeats until nothing is over — which happens within a handful of passes for
    any realistic cap set.

    The config loader guarantees feasibility (every cap is at least the asset's
    neutral weight, and the neutral mix sums to 1, so the caps sum to at least
    1). If a caller passes infeasible caps anyway, the loop leaves the residual
    unallocated rather than looping forever, and :func:`assert_sums_to_one`
    catches it downstream.
    """
    w = {k: max(float(v), 0.0) for k, v in weights.items()}
    total = sum(w.values())
    if total <= eps:
        raise GuardrailError("cannot cap an all-zero weight vector")
    w = {k: v / total for k, v in w.items()}

    for _ in range(max_iter):
        excess = 0.0
        for asset, value in w.items():
            cap = caps.get(asset, math.inf)
            if value > cap + eps:
                excess += value - cap
                w[asset] = cap
        if excess <= eps:
            break
        free_sum = sum(v for a, v in w.items() if v < caps.get(a, math.inf) - eps)
        if free_sum <= eps:
            break  # infeasible caps; reported by assert_sums_to_one
        scale = 1.0 + excess / free_sum
        for asset, value in list(w.items()):
            if value < caps.get(asset, math.inf) - eps:
                w[asset] = value * scale
    return w


def assert_sums_to_one(
    weights: Mapping[str, float], *, tolerance: float = DEFAULT_TOLERANCE
) -> None:
    total = sum(weights.values())
    if abs(total - 1.0) > tolerance:
        raise GuardrailError(f"weights must sum to 1, got {total:.9f}: {dict(weights)}")


def assert_within_caps(
    weights: Mapping[str, float],
    caps: Mapping[str, float],
    *,
    tolerance: float = DEFAULT_TOLERANCE,
) -> None:
    for asset, weight in weights.items():
        if weight < -tolerance:
            raise GuardrailError(f"{asset} weight is negative: {weight}")
        cap = caps.get(asset, math.inf)
        if weight > cap + tolerance:
            raise GuardrailError(f"{asset} weight {weight:.6f} exceeds its cap {cap}")


def assert_experimental_excluded(
    weights: Mapping[str, float],
    *,
    excluded: Collection[str] = EXPERIMENTAL_ASSETS,
    tolerance: float = DEFAULT_TOLERANCE,
) -> None:
    """No experimental engine carries weight (§3 rule 5).

    The last line of defence, behind the registry filter that keeps crypto out of
    the universe and the cap of 0 that would zero it anyway. Cheap, and it turns a
    silent leak into a loud failure if either of the other two ever regresses.
    """
    for asset in excluded:
        weight = weights.get(asset, 0.0)
        if weight > tolerance:
            raise GuardrailError(
                f"{asset} is experimental and excluded from allocations, but carries "
                f"weight {weight:.6f}"
            )


def validate(
    weights: Mapping[str, float],
    caps: Mapping[str, float],
    *,
    excluded: Collection[str] = EXPERIMENTAL_ASSETS,
    tolerance: float = DEFAULT_TOLERANCE,
) -> None:
    """Run every hard constraint. Raises :class:`GuardrailError` on the first miss."""
    assert_sums_to_one(weights, tolerance=tolerance)
    assert_within_caps(weights, caps, tolerance=tolerance)
    assert_experimental_excluded(weights, excluded=excluded, tolerance=tolerance)


def merge_with_neutral(
    tilted: Mapping[str, float],
    neutral: Mapping[str, float],
    degraded: Collection[str],
) -> dict[str, float]:
    """Combine the tilted usable weights with neutral fallbacks for degraded assets.

    A degraded asset (missing or stale engine) is held at exactly its neutral
    weight. The usable assets share the rest of the budget — which is
    ``1 - sum(neutral of degraded)`` — among themselves as ``tilted`` prescribes.
    ``tilted`` is expected to already sum to that remaining budget; this function
    just unions the two and is where the sum-to-1 guarantee is assembled:

        sum(neutral of degraded) + (1 - sum(neutral of degraded)) = 1.

    With every asset degraded, ``tilted`` is empty and the result is the whole
    neutral mix — the honest all-engines-down answer, and still a valid
    allocation because the neutral mix sums to 1 by construction.
    """
    out = {asset: float(tilted[asset]) for asset in tilted}
    for asset in degraded:
        out[asset] = float(neutral.get(asset, 0.0))
    return out
