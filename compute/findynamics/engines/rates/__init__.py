"""FinRates — interest-rate dynamics. Phase 1.

The observable Treasury curve reduced to Nelson-Siegel level, slope and
curvature, plus a rule-based regime over them. Phase 2 replaces the per-date
independent fits with a Kalman state space; Phase 3 adds the short-rate models
(``01-target-architecture.md`` §5). Nothing here predicts rates.

Importing this package registers :class:`RatesEngine` under the name ``rates``.
"""

from findynamics.engines.rates.engine import RatesEngine

__all__ = ["RatesEngine"]
