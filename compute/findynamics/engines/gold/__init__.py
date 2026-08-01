"""FinGold — trust and crisis protection. Phase 4.

Gold has no cash flow, so this engine models **drivers, not value**: the real
interest rate, the dollar, financial stress and the instability of the assets
gold is held against. What it publishes is a regime posterior, a hedge score, a
jump-driven crisis premium and the driver panel behind them — no price target,
and no valuation, because a discounted value of a non-yielding asset is not
merely hard to compute, it is undefined.

Reads FinEquity's instability index, but as *published data* through
``WorldState.series`` (``ENGINE:equity.rii``), never by importing it: the
engines are independent by contract (``01-target-architecture.md`` §3 rule 2) and
``lint-imports`` proves this package does not reach for ``engines.equity``.

Importing this package registers :class:`GoldEngine` under the name ``gold``.
"""

from findynamics.engines.gold.engine import GoldEngine

__all__ = ["GoldEngine"]
