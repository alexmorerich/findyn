"""Portfolio layer — Phase 6.

Consumes the latest ``AssetState`` per enabled, non-experimental engine (which
ones, via ``core.registry``; the states themselves, from D1's ``asset_state`` or
this run's in-memory ``predict`` results) plus ``WorldState`` factors, and
produces target-weight *distributions* with a templated conditional implication.

It reaches engines only through the registry and the ``AssetState`` contract,
never through engine internals (``01-target-architecture.md`` §3 rule 3): no
module here imports ``findynamics.engines`` anything, and ``lint-imports`` proves
it. Experimental engines are excluded three times over — filtered out of the
universe by :func:`~findynamics.core.registry.portfolio_asset_names`, capped at
zero in every profile, and asserted absent by :mod:`.guardrails` — and the
import-linter quarantine forbids this package from importing
``findynamics.engines.crypto`` at all.

The non-goals are binding (FINDYN_V1_SPEC.md §0/§12): no trade commands, no
deterministic targets. Outputs are weight distributions and conditional
implications.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from findynamics.core.contracts.state import AssetState, WorldState
from findynamics.core.registry import portfolio_asset_names
from findynamics.portfolio.allocate import Allocation, allocate, weights_blob
from findynamics.portfolio.config import (
    PortfolioConfig,
    ProfileConfig,
    get_portfolio_config,
    load_portfolio_config,
)
from findynamics.portfolio.inputs import DecisionPanel, assemble_panel


def compute_allocations(
    states: Mapping[str, AssetState],
    world: WorldState,
    *,
    config: PortfolioConfig | None = None,
    profiles: Iterable[str] | None = None,
) -> dict[str, Allocation]:
    """Allocate every (selected) profile from one panel.

    ``states`` should already be filtered to the engines the portfolio may
    consume — read them for ``portfolio_asset_names(series_config)`` and pass
    only those. A state for an engine no profile allocates over is ignored, so a
    stray one does no harm, but relying on that instead of the registry filter
    would be relying on the profiles never naming crypto.

    Builds one :class:`DecisionPanel` over the union of the selected profiles'
    universes and hands it to :func:`~findynamics.portfolio.allocate.allocate`
    per profile — the panel is the expensive part (it prices cash once) and every
    profile reads the same states.
    """
    cfg = config or get_portfolio_config()
    names = tuple(profiles) if profiles is not None else tuple(cfg.profiles)
    unknown = [name for name in names if name not in cfg.profiles]
    if unknown:
        raise KeyError(f"unknown profile(s) {unknown}; configured: {sorted(cfg.profiles)}")

    universe = tuple(
        asset
        for profile in cfg.profiles.values()
        if profile.name in names
        for asset in profile.universe
    )
    # Deduplicate, preserving a stable order.
    universe = tuple(dict.fromkeys(universe))

    panel = assemble_panel(states, world, universe, cfg)
    return {name: allocate(panel, cfg.profile(name), cfg) for name in names}


__all__ = [
    "Allocation",
    "DecisionPanel",
    "PortfolioConfig",
    "ProfileConfig",
    "allocate",
    "assemble_panel",
    "compute_allocations",
    "get_portfolio_config",
    "load_portfolio_config",
    "portfolio_asset_names",
    "weights_blob",
]
