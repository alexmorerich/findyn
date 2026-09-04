"""``K`` — a documented instability composite. **Scaffold (KK0).**

    K_t = w1·Z(a_t) + w2·Z(j_t) + w3·Z(Ω̇_t) + w4·Z(Ω̈_t) + w5·Z(C_t)

**This is not Riemann curvature.** There is no connection here and no second
derivative of a metric; ``curvature`` is a label on a composite that stands for
regime instability. The precedent is exact: ``FINDYN_V1_SPEC.md`` §3.1 replaced
*snap* — the fourth derivative of price — with the RII, because a fourth
derivative of a daily index is essentially all microstructure, and the honest
response was to build a composite of individually measurable things rather than
a derivative nobody can estimate. ``K`` inherits that obligation, and it also
inherits the obligation to be checked against the RII: if the two correlate
above 0.9, ``K`` is a re-derivation of an index this repo already publishes, and
KK2's redundancy panel is required to say so in one sentence at the top.

Three rules, none of them negotiable.

**Every ``Z`` is an expanding z-score**, with the same minimum-baseline
convention ``jerk_z`` uses (``features/kinematics.py::expanding_z``). A
full-sample z here would put the whole future into every historical curvature
value, and it would be invisible: the series would look entirely plausible and
be slightly too good everywhere.

**Terms enter as magnitudes where direction is not meaningful**, following the
RII's treatment of jerk ("jerk enters as a magnitude. Direction is the market's
business; instability is about the size of the change in trend, either way").
Each term documents whether it is signed or absolute and why.

**Weights are equal by default and are never fitted on the full sample.**
Config offers exactly two modes and there is no third:

* ``weights: equal`` — ``w_i = 1/5``, the documented rule. Equal is a real
  choice, not a placeholder, for the reason ``rii.py::DEFAULT_WEIGHTS`` gives:
  nothing in the record justifies asserting that one term leads another by a
  factor of 1.4, and a weight vector fitted on eight episodes is fitted on eight
  observations.
* ``weights: fitted`` — estimated on the **first walk-forward training window
  only** and frozen for the entire out-of-sample path, with the fitting window
  recorded in the artifact.

Anything else is lookahead. Wanting to optimize the weights over the whole
history is the specific failure this rule exists to prevent.

Missing terms degrade with **renormalized** weights and the surviving list is
reported — ``compute_rii``'s pattern, "six components, with the seventh named".
The renormalization sums with ``math.fsum``: CPython 3.12 gave ``sum`` Neumaier
compensation and 3.11 sums naively, so the same five weights land on exactly 1.0
on one interpreter and 0.9999999999999999 on the other, which is a CI failure
rather than a rounding nicety (commit ``bdf5e2d``).

Published: ``financial_curvature`` and ``financial_curvature_percentile``
(expanding).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - scaffold
    import pandas as pd

    from findynamics.research.omega.config import OmegaConfig


def financial_curvature(
    components: dict[str, pd.Series],
    *,
    config: OmegaConfig,
) -> object:
    """``K`` and its parts, with weights renormalized over surviving terms."""
    raise NotImplementedError("KK2: implement the curvature composite")


def curvature_weights(
    components: dict[str, pd.Series],
    *,
    config: OmegaConfig,
) -> dict[str, float]:
    """The weights actually used, renormalized with ``math.fsum`` to sum to 1."""
    raise NotImplementedError("KK2: implement equal and frozen-fit weighting")


__all__ = ["curvature_weights", "financial_curvature"]
