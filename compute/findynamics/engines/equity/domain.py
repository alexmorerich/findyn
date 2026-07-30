"""FinEquity's private vocabulary.

These names are fixed by FINDYN_V1_SPEC.md and mirrored in
serving/src/domain.ts. Changing one is a schema change, not a refactor;
tests/test_domain.py guards the two copies against drift.

Only equity-specific names live here. The vocabulary shared across engines —
horizons, quantiles, factors, frequencies — is in
:mod:`findynamics.core.contracts.vocab`, and no other engine may import this
module (``01-target-architecture.md`` §3 rule 2).
"""

from __future__ import annotations

from typing import Final

#: §9 L2 — the five HMM regimes, ordered from most to least risk-on.
REGIMES: Final[tuple[str, ...]] = (
    "bull_expansion",
    "normal_expansion",
    "late_cycle",
    "bear",
    "crisis",
)

#: §5.2 — feature names written to derived_features.
KINEMATIC_FEATURES: Final[tuple[str, ...]] = (
    "price_filtered",
    "velocity",
    "acceleration",
    "jerk_z",
    "ffd_price",
)

#: §11 — shock taxonomy for the Monte Carlo overlay. Deliberately generic:
#: the engine models classes of shock, not replays of 2008 or 2020.
SHOCK_CLASSES: Final[tuple[str, ...]] = (
    "financial_crisis",
    "liquidity_crisis",
    "technology_disruption",
    "geopolitical_shock",
    "policy_regime_change",
)
