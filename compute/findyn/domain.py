"""Shared domain vocabulary.

These names are fixed by FINDYN_V1_SPEC.md and mirrored in
serving/src/domain.ts. Changing one is a schema change, not a refactor;
tests/test_domain.py guards the two copies against drift.
"""

from __future__ import annotations

from typing import Final

#: §2.2 — Market Force State F(t).
FORCES: Final[tuple[str, ...]] = (
    "valuation",
    "earnings",
    "liquidity",
    "rates",
    "credit",
    "inflation",
    "labor",
    "risk_appetite",
    "sentiment",
)

#: §9 L2 — the five HMM regimes, ordered from most to least risk-on.
REGIMES: Final[tuple[str, ...]] = (
    "bull_expansion",
    "normal_expansion",
    "late_cycle",
    "bear",
    "crisis",
)

#: §10 — forecast horizons.
HORIZONS: Final[tuple[str, ...]] = (
    "tactical",
    "strategic",
    "generational",
    "educational_30y",
    "educational_50y",
)

#: §10 — excluded from every accuracy evaluation; scenario simulation only.
EDUCATIONAL_HORIZONS: Final[frozenset[str]] = frozenset({"educational_30y", "educational_50y"})

#: §10 — quantiles stored for every forecast distribution.
QUANTILES: Final[tuple[float, ...]] = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)

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
