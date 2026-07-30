"""State contract invariants (03-contracts.md §1).

These objects are the wire format for the dashboard: a risk_score of 140 or a
confidence of -0.2 renders as a plausible-looking tile rather than an error, so
the invariants are checked at construction and asserted here.
"""

from __future__ import annotations

import dataclasses
from datetime import date, datetime

import pandas as pd
import pytest

from findynamics.core.contracts import (
    AssetState,
    EngineOutput,
    FactorState,
    PITAccessor,
    Signal,
    WorldState,
)
from findynamics.core.contracts.state import ContractError
from findynamics.data.accessor import PandasPITAccessor

AS_OF = date(2026, 7, 29)


@pytest.fixture
def accessor() -> PandasPITAccessor:
    frame = pd.DataFrame(
        {
            "series_id": ["FRED:DGS10", "FRED:DGS10", "FRED:VIXCLS", "FRED:M2SL"],
            "obs_date": ["2026-07-01", "2026-07-28", "2026-07-28", "2026-07-01"],
            "release_date": ["2026-07-02", "2026-07-29", "2026-07-29", "2026-08-15"],
            "value": [4.10, 4.25, 17.3, 21_000.0],
        }
    )
    return PandasPITAccessor(frame, AS_OF)


def valid_asset_state(**overrides) -> AssetState:
    kwargs = {
        "asset": "rates",
        "as_of": AS_OF,
        "regime": "steepening",
        "expected_return": 0.041,
        "risk_score": 42.0,
        "confidence": 0.7,
        "signals": (Signal(name="curve_inversion", value=-0.35, direction=-1),),
        "model_version": "rates-0.1.0",
    }
    kwargs.update(overrides)
    return AssetState(**kwargs)


# --------------------------------------------------------------------------
# AssetState
# --------------------------------------------------------------------------


def test_a_well_formed_asset_state_is_accepted():
    state = valid_asset_state()
    assert state.asset == "rates"
    assert state.signals[0].direction == -1


def test_asset_state_is_frozen():
    state = valid_asset_state()
    with pytest.raises(dataclasses.FrozenInstanceError):
        state.risk_score = 99.0  # type: ignore[misc]


def test_asset_must_be_a_known_engine():
    with pytest.raises(ContractError, match="not one of"):
        valid_asset_state(asset="commodities")


@pytest.mark.parametrize("score", [-0.1, 100.1, float("nan"), float("inf")])
def test_risk_score_stays_within_0_100(score):
    with pytest.raises(ContractError, match="risk_score"):
        valid_asset_state(risk_score=score)


@pytest.mark.parametrize("confidence", [-0.01, 1.01, float("nan")])
def test_confidence_stays_within_0_1(confidence):
    with pytest.raises(ContractError, match="confidence"):
        valid_asset_state(confidence=confidence)


def test_boundary_values_are_inclusive():
    assert valid_asset_state(risk_score=0.0, confidence=0.0).risk_score == 0.0
    assert valid_asset_state(risk_score=100.0, confidence=1.0).confidence == 1.0


def test_expected_return_may_be_none_but_not_nan():
    assert valid_asset_state(expected_return=None).expected_return is None
    with pytest.raises(ContractError, match="expected_return"):
        valid_asset_state(expected_return=float("nan"))


def test_regime_and_model_version_must_be_non_empty():
    with pytest.raises(ContractError, match="regime"):
        valid_asset_state(regime="")
    with pytest.raises(ContractError, match="model_version"):
        valid_asset_state(model_version="")


def test_signals_must_be_a_tuple_of_signals():
    """A list would be mutable state escaping a frozen record."""
    with pytest.raises(ContractError, match="must be a tuple"):
        valid_asset_state(signals=[Signal(name="x", value=1.0, direction=1)])
    with pytest.raises(ContractError, match="not a Signal"):
        valid_asset_state(signals=({"name": "x"},))


def test_as_of_must_be_a_plain_date():
    """A datetime carries a time component that breaks the t-1 convention."""
    with pytest.raises(ContractError, match="datetime.date"):
        valid_asset_state(as_of=datetime(2026, 7, 29, 13, 0))
    with pytest.raises(ContractError, match="datetime.date"):
        valid_asset_state(as_of="2026-07-29")


def test_components_must_be_finite_numbers_keyed_by_string():
    assert valid_asset_state(components={"dgs10": 0.4}).components == {"dgs10": 0.4}
    with pytest.raises(ContractError, match="components"):
        valid_asset_state(components={"dgs10": float("inf")})
    with pytest.raises(ContractError, match="components"):
        valid_asset_state(components={"dgs10": "high"})


# --------------------------------------------------------------------------
# Signal / FactorState / EngineOutput
# --------------------------------------------------------------------------


@pytest.mark.parametrize("direction", [-2, 2, 0.5, True])
def test_signal_direction_is_trinary(direction):
    with pytest.raises(ContractError, match="direction"):
        Signal(name="carry", value=1.0, direction=direction)


def test_signal_value_must_be_finite():
    with pytest.raises(ContractError, match="finite"):
        Signal(name="carry", value=float("nan"), direction=1)


def test_factor_state_score_stays_within_0_100():
    assert FactorState(name="rates", as_of=AS_OF, score=61.0).components == {}
    with pytest.raises(ContractError, match="score"):
        FactorState(name="rates", as_of=AS_OF, score=101.0)


def test_factor_state_requires_a_name():
    with pytest.raises(ContractError, match="name"):
        FactorState(name="", as_of=AS_OF, score=50.0)


def test_engine_output_validates_asset_metric_and_value():
    output = EngineOutput(asset="rates", metric="ns_level", as_of=AS_OF, value=4.31)
    assert output.value == 4.31
    with pytest.raises(ContractError, match="not one of"):
        EngineOutput(asset="fx", metric="ns_level", as_of=AS_OF, value=1.0)
    with pytest.raises(ContractError, match="metric"):
        EngineOutput(asset="rates", metric="", as_of=AS_OF, value=1.0)
    with pytest.raises(ContractError, match="finite"):
        EngineOutput(asset="rates", metric="ns_level", as_of=AS_OF, value=float("inf"))


# --------------------------------------------------------------------------
# WorldState + PITAccessor
# --------------------------------------------------------------------------


def test_world_state_carries_factors_and_an_accessor(accessor):
    world = WorldState(
        as_of=AS_OF,
        factors={"rates": FactorState(name="rates", as_of=AS_OF, score=61.0)},
        series=accessor,
    )
    assert world.factor_score("rates") == 61.0
    assert world.factor_score("gold") is None
    assert world.series.value("FRED:DGS10") == 4.25


def test_world_state_rejects_a_mis_keyed_factor():
    with pytest.raises(ContractError, match="keyed as"):
        WorldState(
            as_of=AS_OF,
            factors={"liquidity": FactorState(name="rates", as_of=AS_OF, score=61.0)},
            series=None,  # type: ignore[arg-type]
        )


def test_world_state_rejects_an_accessor_with_a_different_cutoff(accessor):
    """The gap between the two dates is exactly where lookahead hides."""
    with pytest.raises(ContractError, match="disagrees"):
        WorldState(as_of=date(2026, 7, 30), factors={}, series=accessor)


def test_the_pandas_accessor_satisfies_the_core_protocol(accessor):
    assert isinstance(accessor, PITAccessor)


def test_the_accessor_hides_data_released_after_the_cutoff(accessor):
    """M2SL's release_date is 2026-08-15 — it does not exist yet at as_of."""
    frame = accessor.latest()
    assert set(frame.index) == {"FRED:DGS10", "FRED:VIXCLS"}
    assert accessor.value("FRED:M2SL") is None


def test_the_accessor_returns_the_newest_knowable_vintage(accessor):
    assert accessor.value("FRED:DGS10") == 4.25


def test_the_accessor_reports_a_missing_series_as_none(accessor):
    """A provider outage degrades the feature set, not the run (§14.2)."""
    assert accessor.value("FRED:NOPE") is None


def test_the_accessor_can_restrict_to_named_series(accessor):
    frame = accessor.latest(["FRED:VIXCLS"])
    assert list(frame.index) == ["FRED:VIXCLS"]


def test_the_accessor_cutoff_cannot_be_moved(accessor):
    """No setter, and no method takes an as_of — that is the whole guarantee."""
    with pytest.raises(AttributeError):
        accessor.as_of = date(2026, 8, 1)  # type: ignore[misc]
