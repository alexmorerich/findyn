"""FinMoney — time value of money. Phase 2.

The money-market account ``M(t) = M(0)·exp(∫r dt)``: cash carry, discount
factors, the risk-free benchmark and the liquidity state. Deliberately without a
model — nothing here is fitted, because the numeraire is arithmetic on observed
rates and a model of it would be a model of something already measured.

Reads FinRates' fitted curve to discount past a year, but as *published data*
through ``WorldState.series``, never by importing it: the engines are
independent by contract (``01-target-architecture.md`` §3 rule 2) and
``lint-imports`` proves this package does not reach for ``engines.rates``.

Importing this package registers :class:`MoneyEngine` under the name ``money``.
"""

from findynamics.engines.money.engine import MoneyEngine

__all__ = ["MoneyEngine"]
