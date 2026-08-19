"""Assemble the decision panel the allocator reads (Phase 6).

The portfolio layer consumes the *latest* ``AssetState`` per enabled,
non-experimental engine — nothing else. It reaches those states through the
registry (which engines to include) and D1's ``asset_state`` (the states
themselves), never through an engine's internals: this module imports no
``findynamics.engines`` module, and ``lint-imports`` proves it.

What the panel adds on top of the raw states is the two numbers an allocation
needs that a single state does not carry: the risk-free benchmark and the
discount curve, both of which come from the money engine's published output.
Excess return — an asset's expected return *over cash* — is what a weight should
respond to, and cash is FinMoney's job to price.

Money has a dual role and it is worth stating plainly: it is the risk-free leg
(its ``expected_return`` is the short rate every other asset is measured
against) *and*, in the conservative profile, an allocatable holding in its own
right (cash, with zero excess return by definition). Both roles read off the one
state.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date

from findynamics.core.contracts.state import AssetState, WorldState
from findynamics.core.contracts.vocab import DISCOUNT_HORIZONS
from findynamics.portfolio.config import PortfolioConfig

#: The engine whose state prices cash. Its ``expected_return`` is the risk-free
#: benchmark and its ``components`` carry the discount curve.
MONEY = "money"


@dataclass(frozen=True)
class AssetInput:
    """One asset's decision inputs, distilled from its ``AssetState``."""

    asset: str
    #: Annualized expected return, or ``None`` where the engine publishes none.
    expected_return: float | None
    #: Expected return over the risk-free benchmark, or ``None`` when either leg
    #: is missing. This is what the allocator tilts on.
    excess_return: float | None
    risk_score: float  # 0-100
    confidence: float  # 0-1
    regime: str
    #: The state's own information-set date.
    as_of: date
    #: Days the state's ``as_of`` is behind the panel's ``as_of``.
    age_days: int
    #: True when the engine did not report, or reported too long ago: the
    #: allocator holds this asset at its neutral weight and flags the run.
    stale: bool
    #: Why it is stale, for the degraded reason trace. Empty when it is not.
    reason: str = ""

    @property
    def usable(self) -> bool:
        """Can this asset carry a real tilt, or must it fall back to neutral?

        Usable means present, fresh, and carrying an excess return to respond
        to. An engine with a regime but no expected return (crypto, by design —
        though it never reaches the panel) cannot be tilted on and is held at
        neutral like a stale one.
        """
        return not self.stale and self.excess_return is not None


@dataclass(frozen=True)
class DecisionPanel:
    """Everything the allocator sees for one run — the states plus cash's price."""

    as_of: date
    #: Risk-free benchmark (annualized decimal), from the money engine, or
    #: ``None`` when FinMoney did not report.
    risk_free: float | None
    #: Discount factors D(t, h) from the money engine's published output, keyed
    #: by horizon. Carried for the "why" trace and downstream present-value work;
    #: the weight method does not read them.
    discount_factors: dict[str, float] = field(default_factory=dict)
    #: Per-asset decision inputs, keyed by engine name.
    assets: dict[str, AssetInput] = field(default_factory=dict)
    #: Shared factor scores worth surfacing beside the allocation (risk appetite,
    #: liquidity, ...). Explanation only; the weights do not read them.
    factors: dict[str, float] = field(default_factory=dict)

    def get(self, asset: str) -> AssetInput | None:
        return self.assets.get(asset)

    @property
    def degraded(self) -> bool:
        return any(a.stale for a in self.assets.values())

    @property
    def degraded_assets(self) -> tuple[str, ...]:
        return tuple(a for a, inp in self.assets.items() if inp.stale)


#: Factor scores carried onto the panel for context. A short, fixed list rather
#: than the whole factor set: these are the ones a reader looks at beside an
#: allocation, and copying twelve percentile scores onto every portfolio row
#: would bloat the D1 blob for no allocation that reads them.
CONTEXT_FACTORS = ("risk_appetite", "liquidity", "credit", "real_rate")


def assemble_panel(
    states: Mapping[str, AssetState],
    world: WorldState,
    universe: tuple[str, ...],
    config: PortfolioConfig,
) -> DecisionPanel:
    """Build the decision panel for one run.

    ``states`` is the latest ``AssetState`` per engine, keyed by asset name —
    whatever the caller could read from ``asset_state`` (or, in the daily job,
    hold in memory from this run's ``predict`` calls). ``universe`` is the set of
    engine names the portfolio is allowed to consume, from
    :func:`findynamics.core.registry.portfolio_asset_names`; an asset outside it
    is ignored even if a state was passed, and an asset inside it with no state
    becomes a stale input that falls back to neutral.

    An asset whose state is older than ``max_state_age_days`` behind the panel is
    treated as not reporting — that is the staleness the chaos test simulates by
    freezing one engine's data.
    """
    money_state = states.get(MONEY)
    risk_free = _risk_free(money_state, world.as_of, config.max_state_age_days)
    discount_factors = _discount_factors(money_state)

    assets: dict[str, AssetInput] = {}
    for asset in universe:
        assets[asset] = _asset_input(asset, states.get(asset), world.as_of, risk_free, config)

    factors = {
        name: score for name in CONTEXT_FACTORS if (score := world.factor_score(name)) is not None
    }

    return DecisionPanel(
        as_of=world.as_of,
        risk_free=risk_free,
        discount_factors=discount_factors,
        assets=assets,
        factors=factors,
    )


def _age_days(state_as_of: date, panel_as_of: date) -> int:
    """Calendar days the state is behind the panel. Never negative.

    A state stamped *after* the panel cutoff would be a lookahead bug upstream;
    clamped to 0 here so it reads as fresh rather than as a nonsensical negative
    age, and the replay test is what actually guards the cutoff.
    """
    return max((panel_as_of - state_as_of).days, 0)


def _risk_free(
    money_state: AssetState | None,
    panel_as_of: date,
    max_age_days: int,
) -> float | None:
    """The short rate from FinMoney, or ``None`` when it did not report freshly.

    A stale money state is no benchmark: discounting today's excess returns
    against a week-old rate is worse than admitting cash is unpriced and holding
    every asset at neutral. So a stale FinMoney degrades the whole panel by way
    of a missing risk-free, exactly as a stale equity engine degrades equity.
    """
    if money_state is None:
        return None
    if _age_days(money_state.as_of, panel_as_of) > max_age_days:
        return None
    return money_state.expected_return


def _discount_factors(money_state: AssetState | None) -> dict[str, float]:
    """Discount curve from the money state's components, on the standard grid."""
    if money_state is None or not money_state.components:
        return {}
    out: dict[str, float] = {}
    for horizon in DISCOUNT_HORIZONS:
        value = money_state.components.get(f"discount_{horizon}")
        if value is not None:
            out[horizon] = float(value)
    return out


def _asset_input(
    asset: str,
    state: AssetState | None,
    panel_as_of: date,
    risk_free: float | None,
    config: PortfolioConfig,
) -> AssetInput:
    """Distil one engine's state into decision inputs, or a stale placeholder."""
    if state is None:
        return AssetInput(
            asset=asset,
            expected_return=None,
            excess_return=None,
            risk_score=0.0,
            confidence=0.0,
            regime="unavailable",
            as_of=panel_as_of,
            age_days=0,
            stale=True,
            reason="no state published",
        )

    age = _age_days(state.as_of, panel_as_of)
    if age > config.max_state_age_days:
        return AssetInput(
            asset=asset,
            expected_return=state.expected_return,
            excess_return=None,
            risk_score=state.risk_score,
            confidence=state.confidence,
            regime=state.regime,
            as_of=state.as_of,
            age_days=age,
            stale=True,
            reason=f"state is {age}d old (> {config.max_state_age_days}d)",
        )

    # Cash's excess over the risk-free rate is zero by definition — it *is* the
    # risk-free leg — regardless of what FinMoney stamps as its expected return.
    if asset == MONEY:
        excess: float | None = 0.0
    elif state.expected_return is None or risk_free is None:
        excess = None
    else:
        excess = state.expected_return - risk_free

    return AssetInput(
        asset=asset,
        expected_return=state.expected_return,
        excess_return=excess,
        risk_score=state.risk_score,
        confidence=state.confidence,
        regime=state.regime,
        as_of=state.as_of,
        age_days=age,
        stale=False,
        # An asset that is fresh but carries no excess return (no expected return
        # from the engine, or no risk-free to net against) cannot be tilted; the
        # allocator will hold it at neutral, and the reason says which it was.
        reason=(
            ""
            if excess is not None
            else ("no risk-free benchmark" if risk_free is None else "engine publishes no return")
        ),
    )
