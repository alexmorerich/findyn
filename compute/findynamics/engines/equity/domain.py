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

#: §5.2 — feature names written to derived_features, in **model units**:
#: ``price_filtered`` is the filtered *log* level, ``velocity`` and
#: ``acceleration`` are annualized log rates, ``jerk_z`` is a z-score and
#: ``ffd_price`` is the fractionally differenced log price.
#:
#: Not the whole table: the momentum windows are configurable
#: (``features.momentum_months``) and so are named at runtime. These five are the
#: fixed part, and the ones §2.1 defines.
KINEMATIC_FEATURES: Final[tuple[str, ...]] = (
    "price_filtered",
    "velocity",
    "acceleration",
    "jerk_z",
    "ffd_price",
)

#: Metrics published to ``engine_output`` — the same information as
#: :data:`KINEMATIC_FEATURES` but in the units a reader expects: ``price_close``
#: and ``price_filtered`` are index points, not logs.
#:
#: Two tables for one pipeline, on purpose. ``derived_features`` is what the
#: model was fitted on and is keyed by ``model_version``; this is what the
#: dashboard draws and is keyed by date alone. Publishing model units to a chart
#: would need the page to know the transform, and publishing chart units to the
#: feature store would mean a refit could not reproduce its own inputs.
CHART_METRICS: Final[tuple[str, ...]] = (
    "price_close",
    "price_filtered",
    "velocity",
    "acceleration",
    "jerk_z",
    "jerk_lamp",
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
