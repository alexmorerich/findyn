"""The state contracts every layer speaks (``03-contracts.md`` §1).

Four frozen records carry everything that crosses a layer boundary:
``FactorState`` (Layer 0 output), ``WorldState`` (what an engine is given),
``Signal`` and ``AssetState`` (what an engine returns). Validation lives in
``__post_init__`` rather than at the write-back boundary, because a nonsensical
score is cheapest to catch at the moment it is constructed — an out-of-range
``risk_score`` that reaches D1 has already been published to the dashboard.

No pandas object may be a field here. The only handle on tabular data is
``WorldState.series``, and that is a :class:`PITAccessor`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Literal

from findynamics.core.contracts.pit import PITAccessor
from findynamics.core.contracts.vocab import ASSETS


class ContractError(ValueError):
    """Raised when a state object violates its documented invariants."""


def _check_date(value: object, where: str) -> None:
    # bool is not a date, but datetime *is* a date subclass and carries a time
    # component that would quietly break the t-1 information-set convention.
    if type(value) is not date:
        raise ContractError(f"{where}: expected datetime.date, got {type(value).__name__}")


def _check_range(value: object, low: float, high: float, where: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ContractError(f"{where}: expected a number, got {value!r}")
    if not math.isfinite(value):
        raise ContractError(f"{where}: must be finite, got {value!r}")
    if not low <= value <= high:
        raise ContractError(f"{where}: must be within [{low}, {high}], got {value!r}")


def _check_components(components: object, where: str) -> None:
    if components is None:
        return
    if not isinstance(components, dict):
        raise ContractError(f"{where}: expected a mapping, got {type(components).__name__}")
    for key, value in components.items():
        if not isinstance(key, str):
            raise ContractError(f"{where}: keys must be str, got {key!r}")
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ContractError(f"{where}[{key!r}]: expected a number, got {value!r}")
        if not math.isfinite(value):
            raise ContractError(f"{where}[{key!r}]: must be finite, got {value!r}")


@dataclass(frozen=True)
class FactorState:
    """One shared risk factor, scored and explained (Layer 0)."""

    name: str
    as_of: date
    #: 0-100 expanding-window percentile. Expanding, never centred (§14.1 rule 3).
    score: float
    #: Explanation trace: series_id → contribution.
    components: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise ContractError("FactorState.name must be non-empty")
        _check_date(self.as_of, "FactorState.as_of")
        _check_range(self.score, 0.0, 100.0, "FactorState.score")
        _check_components(self.components, "FactorState.components")


@dataclass(frozen=True)
class Signal:
    """A named, directional observation an engine wants to surface."""

    name: str
    value: float
    direction: Literal[-1, 0, 1]
    note: str | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ContractError("Signal.name must be non-empty")
        if isinstance(self.value, bool) or not isinstance(self.value, int | float):
            raise ContractError(f"Signal.value: expected a number, got {self.value!r}")
        if not math.isfinite(self.value):
            raise ContractError(f"Signal.value: must be finite, got {self.value!r}")
        if self.direction not in (-1, 0, 1) or isinstance(self.direction, bool):
            raise ContractError(f"Signal.direction must be -1, 0 or 1, got {self.direction!r}")


@dataclass(frozen=True)
class WorldState:
    """Everything an engine is allowed to see for one run.

    ``series`` is the sole data handle: engines never touch a provider or
    ``macro_series`` directly (§14.1 rule 2). Its cutoff must agree with
    ``as_of``, otherwise the information set is not what the caller thinks.
    """

    #: Information-set cutoff — the ``INFO_SET = "t-1"`` convention (§14.1 rule 1).
    as_of: date
    factors: dict[str, FactorState]
    series: PITAccessor

    def __post_init__(self) -> None:
        _check_date(self.as_of, "WorldState.as_of")
        if not isinstance(self.factors, dict):
            raise ContractError(
                f"WorldState.factors: expected a mapping, got {type(self.factors).__name__}"
            )
        for name, factor in self.factors.items():
            if not isinstance(factor, FactorState):
                raise ContractError(f"WorldState.factors[{name!r}] is not a FactorState")
            if factor.name != name:
                raise ContractError(
                    f"WorldState.factors[{name!r}] is keyed as {name!r} but named {factor.name!r}"
                )
        accessor_as_of = getattr(self.series, "as_of", None)
        if accessor_as_of is not None and accessor_as_of != self.as_of:
            raise ContractError(
                f"WorldState.as_of={self.as_of} disagrees with its accessor cutoff "
                f"{accessor_as_of}; that gap is exactly how lookahead gets in"
            )

    def factor_score(self, name: str) -> float | None:
        """Score for ``name``, or ``None`` when the factor could not be computed."""
        factor = self.factors.get(name)
        return None if factor is None else factor.score


@dataclass(frozen=True)
class AssetState:
    """The universal engine output (``01-target-architecture.md`` §6)."""

    #: Engine name — must be a member of the ASSETS vocabulary.
    asset: str
    as_of: date
    #: Engine-defined vocabulary, documented per engine.
    regime: str
    #: Annualized. ``None`` where the notion is meaningless for the asset.
    expected_return: float | None
    risk_score: float  # 0-100
    confidence: float  # 0-1
    signals: tuple[Signal, ...]
    model_version: str
    components: dict[str, float] | None = None

    def __post_init__(self) -> None:
        if self.asset not in ASSETS:
            raise ContractError(f"AssetState.asset {self.asset!r} is not one of {list(ASSETS)}")
        _check_date(self.as_of, "AssetState.as_of")
        if not self.regime:
            raise ContractError("AssetState.regime must be non-empty")
        if not self.model_version:
            raise ContractError("AssetState.model_version must be non-empty")
        if self.expected_return is not None:
            if isinstance(self.expected_return, bool) or not isinstance(
                self.expected_return, int | float
            ):
                raise ContractError(
                    f"AssetState.expected_return: expected a number or None, "
                    f"got {self.expected_return!r}"
                )
            if not math.isfinite(self.expected_return):
                raise ContractError(
                    f"AssetState.expected_return must be finite, got {self.expected_return!r}"
                )
        _check_range(self.risk_score, 0.0, 100.0, "AssetState.risk_score")
        _check_range(self.confidence, 0.0, 1.0, "AssetState.confidence")
        if not isinstance(self.signals, tuple):
            raise ContractError(
                f"AssetState.signals must be a tuple, got {type(self.signals).__name__}"
            )
        for i, signal in enumerate(self.signals):
            if not isinstance(signal, Signal):
                raise ContractError(f"AssetState.signals[{i}] is not a Signal")
        _check_components(self.components, "AssetState.components")


@dataclass(frozen=True)
class DerivedFeature:
    """One per-date model input for the ``derived_features`` table.

    Distinct from :class:`EngineOutput` by exactly one thing, and it is the thing
    that matters: ``model_version`` is part of the key. ``engine_output`` is what
    an engine publishes for people to look at, so the newest run owns each date.
    A feature is what a *model was fitted on*, so a refit that changes the
    transform must land beside the old features rather than on top of them —
    otherwise every backtest of the previous model silently becomes a backtest of
    a feature set that model never saw.
    """

    asset: str
    feature: str
    as_of: date
    value: float
    #: Carried on the row, not taken from the run envelope: it is part of this
    #: table's key, and a run publishing two engines has two versions.
    model_version: str

    def __post_init__(self) -> None:
        if self.asset not in ASSETS:
            raise ContractError(f"DerivedFeature.asset {self.asset!r} is not one of {list(ASSETS)}")
        if not self.feature:
            raise ContractError("DerivedFeature.feature must be non-empty")
        if not self.model_version:
            raise ContractError("DerivedFeature.model_version must be non-empty")
        _check_date(self.as_of, "DerivedFeature.as_of")
        if isinstance(self.value, bool) or not isinstance(self.value, int | float):
            raise ContractError(f"DerivedFeature.value: expected a number, got {self.value!r}")
        if not math.isfinite(self.value):
            raise ContractError(f"DerivedFeature.value must be finite, got {self.value!r}")


@dataclass(frozen=True)
class RegimeProbability:
    """One regime's probability on one date, for the ``regime_state`` table.

    Deliberately a *distribution* rather than a winning label: §12's output
    contract shows the whole posterior, and the difference between a 0.95 bull
    call and a 0.35 one is most of the information. The argmax lives on
    :class:`AssetState`; this is what it was the argmax of.
    """

    asset: str
    as_of: date
    #: Engine-defined regime vocabulary — equity's is in engines/equity/domain.py.
    regime: str
    probability: float
    model_version: str

    def __post_init__(self) -> None:
        if self.asset not in ASSETS:
            raise ContractError(
                f"RegimeProbability.asset {self.asset!r} is not one of {list(ASSETS)}"
            )
        if not self.regime:
            raise ContractError("RegimeProbability.regime must be non-empty")
        if not self.model_version:
            raise ContractError("RegimeProbability.model_version must be non-empty")
        _check_date(self.as_of, "RegimeProbability.as_of")
        _check_range(self.probability, 0.0, 1.0, "RegimeProbability.probability")


@dataclass(frozen=True)
class ForecastQuantile:
    """One quantile of one horizon's forecast — the ``forecast_distribution`` row.

    A separate shape from :class:`EngineOutput` because its key is
    (as_of, horizon, quantile) and because of what it must *not* be able to
    express: there is no column here for a point forecast. §0's first non-goal
    forbids deterministic price targets, and a schema that cannot represent one
    enforces that better than a convention everybody has to remember.
    """

    asset: str
    as_of: date
    #: Forecast horizon name from the HORIZONS vocabulary.
    horizon: str
    #: 0-1. The published grid is core.contracts.vocab.QUANTILES.
    quantile: float
    #: Projected log index level at that quantile.
    value: float
    #: Excluded from every accuracy evaluation (§10). Carried on the row so a
    #: consumer cannot plot a 50-year scenario beside a 6-month forecast without
    #: having been told which is which.
    educational_only: bool
    model_version: str

    def __post_init__(self) -> None:
        if self.asset not in ASSETS:
            raise ContractError(
                f"ForecastQuantile.asset {self.asset!r} is not one of {list(ASSETS)}"
            )
        if not self.horizon:
            raise ContractError("ForecastQuantile.horizon must be non-empty")
        if not self.model_version:
            raise ContractError("ForecastQuantile.model_version must be non-empty")
        _check_date(self.as_of, "ForecastQuantile.as_of")
        _check_range(self.quantile, 0.0, 1.0, "ForecastQuantile.quantile")
        if isinstance(self.value, bool) or not isinstance(self.value, int | float):
            raise ContractError(f"ForecastQuantile.value: expected a number, got {self.value!r}")
        if not math.isfinite(self.value):
            raise ContractError(f"ForecastQuantile.value must be finite, got {self.value!r}")


@dataclass(frozen=True)
class EngineOutput:
    """One wide metric for the ``engine_output`` table (NS level/slope, carry, …).

    Kept separate from ``AssetState`` because these are time series an engine
    publishes for charting, not part of the state contract the portfolio layer
    consumes.
    """

    asset: str
    metric: str
    as_of: date
    value: float
    meta: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.asset not in ASSETS:
            raise ContractError(f"EngineOutput.asset {self.asset!r} is not one of {list(ASSETS)}")
        if not self.metric:
            raise ContractError("EngineOutput.metric must be non-empty")
        _check_date(self.as_of, "EngineOutput.as_of")
        if isinstance(self.value, bool) or not isinstance(self.value, int | float):
            raise ContractError(f"EngineOutput.value: expected a number, got {self.value!r}")
        if not math.isfinite(self.value):
            raise ContractError(f"EngineOutput.value must be finite, got {self.value!r}")
