"""The asset engine interface (``03-contracts.md`` §2).

Every engine — money, rates, equity, gold, crypto — is a different stochastic
process behind the same three methods. The narrowness is deliberate: ``fit`` and
``predict`` receive nothing but a :class:`WorldState`, so an engine physically
cannot reach around the point-in-time gateway, and the portfolio layer can treat
five unrelated models as one type.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

from findynamics.core.contracts.state import AssetState, EngineOutput, WorldState


class AssetEngine(ABC):
    """Base class for the five asset physics engines."""

    #: Registry key, e.g. ``'rates'``. Must be a member of the ASSETS vocabulary.
    name: ClassVar[str]
    #: Stamped onto every ``AssetState`` this engine emits; bumped by monthly_refit.
    version: ClassVar[str]
    #: ``True`` quarantines the engine from the portfolio layer unless explicitly
    #: configured in (``01-target-architecture.md`` §3 rule 5).
    experimental: ClassVar[bool] = False

    @abstractmethod
    def required_series(self) -> tuple[str, ...]:
        """Series ids this engine needs, resolved against series.yaml."""

    @abstractmethod
    def fit(self, world: WorldState) -> None:
        """Expanding-window (re)fit (§14.1 rule 4); monthly_refit cadence.

        Persists parameters through the engine's own artifact store handle.
        """

    @abstractmethod
    def predict(self, world: WorldState) -> AssetState:
        """Pure function of (fitted params, world). Daily cadence."""

    def outputs(self, world: WorldState) -> tuple[EngineOutput, ...]:
        """Optional wide metrics for the ``engine_output`` table."""
        return ()
