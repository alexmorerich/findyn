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

from findynamics.core.contracts.state import (
    AssetState,
    DerivedFeature,
    EngineOutput,
    ForecastQuantile,
    RegimeProbability,
    WorldState,
)


# Deliberately not named *Error (hence the noqa): this is not a failure. It is
# an engine's correct answer when it has nothing to publish, and the job layer
# branches on that distinction.
class StateUnavailable(RuntimeError):  # noqa: N818
    """The engine has nothing to publish, and that is the expected answer.

    Distinct from a crash on purpose. An engine whose model is not fitted yet, or
    whose information set is too thin to speak about, has given a *correct*
    answer by declining — and the job layer treats it as one: the run is not
    failed, the engine's other outputs are still published, and the state
    endpoint keeps saying "no state" rather than serving a fabricated one.

    Anything else escaping ``predict`` is a bug and is logged as a failure.
    """


class AssetEngine(ABC):
    """Base class for the five asset physics engines."""

    #: Registry key, e.g. ``'rates'``. Must be a member of the ASSETS vocabulary.
    name: ClassVar[str]
    #: Stamped onto every ``AssetState`` this engine emits; bumped by monthly_refit.
    version: ClassVar[str]
    #: ``True`` quarantines the engine from the portfolio layer unless explicitly
    #: configured in (``01-target-architecture.md`` §3 rule 5).
    experimental: ClassVar[bool] = False

    #: When ``True``, the publishing methods reach back to the start of the data
    #: rather than the nightly window.
    #:
    #: An instance flag rather than config, because it is a property of *this
    #: run*: the nightly cron republishes a few years because all but the newest
    #: rows are unchanged upserts, and a deliberate backfill republishes a
    #: century once. One config number cannot mean both, and the failure mode of
    #: trying — a cron quietly writing 200k rows a night — is expensive and
    #: silent.
    full_history: bool = False

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
        """Pure function of (fitted params, world). Daily cadence.

        Raises :class:`StateUnavailable` when the engine is working correctly and
        still has nothing to say.
        """

    def outputs(self, world: WorldState) -> tuple[EngineOutput, ...]:
        """Optional wide metrics for the ``engine_output`` table."""
        return ()

    def derived_features(self, world: WorldState) -> tuple[DerivedFeature, ...]:
        """Optional per-date model inputs for the ``derived_features`` table.

        Separate from :meth:`outputs` because these rows are versioned by model:
        see :class:`~findynamics.core.contracts.state.DerivedFeature`.
        """
        return ()

    def forecasts(self, world: WorldState) -> tuple[ForecastQuantile, ...]:
        """Optional quantile forecasts for the ``forecast_distribution`` table.

        Its own hook rather than a metric on :meth:`outputs` because the key is
        (as_of, horizon, quantile) and because the row shape has no place to put
        a point forecast — which is the §0 non-goal made structural.
        """
        return ()

    def regime_states(self, world: WorldState) -> tuple[RegimeProbability, ...]:
        """Optional per-date regime posterior for the ``regime_state`` table.

        The third of the three v1 output tables that became engine-private
        (``01-target-architecture.md`` §6), and the third optional hook — one per
        table, because each is keyed differently and none of the three can be
        expressed as another without losing what its key protects.
        """
        return ()
