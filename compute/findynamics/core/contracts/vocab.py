"""Shared vocabulary — names owned by no single engine.

These constants are fixed by FINDYN_V1_SPEC.md and mirrored in
serving/src/domain.ts. Changing one is a schema change, not a refactor;
tests/test_domain.py guards the two copies against drift.

Engine-private vocabulary does not belong here: the five HMM regimes and the
kinematic feature names live in :mod:`findynamics.engines.equity.domain`.
"""

from __future__ import annotations

from typing import Final

#: The five asset engines (``01-target-architecture.md`` §1). Registry keys,
#: `asset_state.asset` values, and the `ASSETS` array in serving/src/domain.ts.
ASSETS: Final[tuple[str, ...]] = ("money", "rates", "equity", "gold", "crypto")

#: §2.2 — the shared risk factors (v1 called them "forces"). Layer 0: computed
#: once per run and handed to every engine through ``WorldState``.
#: Re-exported by :mod:`findynamics.factors.definitions` as the initial factor set.
FACTORS: Final[tuple[str, ...]] = (
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

#: §5.2 — observation cadences a configured series may declare.
FREQUENCIES: Final[frozenset[str]] = frozenset({"daily", "weekly", "monthly", "quarterly"})
