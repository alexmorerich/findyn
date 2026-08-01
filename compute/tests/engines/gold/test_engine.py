"""GoldEngine: the contract, the outputs, and how it degrades."""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from findynamics.core.contracts.state import AssetState, ContractError
from findynamics.core.engine import StateUnavailable
from findynamics.core.registry import ENGINES, get_engine
from findynamics.engines.gold.domain import (
    GOLD_METRICS,
    GOLD_REGIMES,
    posterior_metric,
    regime_code,
)
from findynamics.engines.gold.engine import MODEL_VERSION, GoldEngine
from tests.engines.gold.conftest import (
    EQUITY_RII,
    PRICE,
    SNAPSHOT_AS_OF,
    observation_rows,
    world_from,
)

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")


@pytest.fixture
def fitted(gold_engine, gold_observations):
    """The engine with a chain fitted on the whole snapshot."""
    gold_engine.fit(world_from(gold_observations))
    return gold_engine


# --------------------------------------------------------------------------
# Registration and configuration
# --------------------------------------------------------------------------


def test_registers_itself_under_gold():
    assert ENGINES["gold"] is GoldEngine
    assert isinstance(get_engine("gold"), GoldEngine)


def test_is_not_experimental():
    """Only crypto is quarantined from the portfolio layer."""
    assert GoldEngine.experimental is False


def test_required_series_come_from_config(gold_engine):
    required = gold_engine.required_series()
    assert PRICE in required
    assert EQUITY_RII in required
    assert "FRED:DGS10" in required
    # Sorted and deduplicated: several roles resolve to the same id.
    assert list(required) == sorted(set(required))


def test_a_missing_required_role_is_a_config_error(config, artifacts):
    stripped = config.engines["gold"].series.copy()
    del stripped["price"]
    broken = config.engines["gold"].__class__(
        name="gold", enabled=True, series=stripped, params=config.engines["gold"].params
    )
    engine = GoldEngine(
        config.__class__(**{**config.__dict__, "engines": {"gold": broken}}), artifacts
    )
    with pytest.raises(ValueError, match="missing required role"):
        _ = engine.series_ids


# --------------------------------------------------------------------------
# predict
# --------------------------------------------------------------------------


@pytest.mark.slow
def test_publishes_a_valid_state(fitted, gold_observations):
    state = fitted.predict(world_from(gold_observations))

    assert isinstance(state, AssetState)
    assert state.asset == "gold"
    assert state.model_version == MODEL_VERSION
    assert state.regime in GOLD_REGIMES
    assert 0.0 <= state.risk_score <= 100.0
    assert 0.0 <= state.confidence <= 1.0
    # The state describes the newest date gold actually fixed on, which after the
    # one-day publication lag is before the cutoff.
    assert state.as_of < SNAPSHOT_AS_OF


@pytest.mark.slow
def test_signals_cover_the_three_the_spec_names(fitted, gold_observations):
    state = fitted.predict(world_from(gold_observations))
    names = {s.name for s in state.signals}
    assert {"hedge_score", "real_rate_headwind", "crisis_premium"} <= names


@pytest.mark.slow
def test_rising_real_rates_read_as_adverse(fitted, gold_observations):
    """The sign that would be invisible if it were wrong.

    A rising real rate is a *headwind* for a non-yielding asset, so the signal's
    direction must be -1 even though the number itself went up.
    """
    state = fitted.predict(world_from(gold_observations))
    signal = next(s for s in state.signals if s.name == "real_rate_headwind")
    if abs(signal.value) > 0.25:
        assert signal.direction == (-1 if signal.value > 0 else 1)


@pytest.mark.slow
def test_expected_return_is_labelled_as_a_historical_mean(fitted, gold_observations):
    """A number called expected_return is read as a forecast unless it says otherwise."""
    state = fitted.predict(world_from(gold_observations))
    components = state.components or {}

    assert components["expected_return_is_historical_mean"] == 1.0
    assert components["regime_conditional_mean_return"] == pytest.approx(state.expected_return)
    # And the confidence is capped well below certainty because of it.
    assert state.confidence <= 0.7


@pytest.mark.slow
def test_components_carry_the_whole_posterior_not_just_the_winner(fitted, gold_observations):
    state = fitted.predict(world_from(gold_observations))
    components = state.components or {}

    posteriors = {
        posterior_metric(name): components[posterior_metric(name)] for name in GOLD_REGIMES
    }
    assert sum(posteriors.values()) == pytest.approx(1.0, abs=1e-6)
    assert components["regime_code"] == regime_code(state.regime)
    # Both blocks are visible, so a reader can see which one moved the answer.
    assert "markov_violent_probability" in components
    assert "stress_gate" in components
    assert "carry_gate" in components


def test_without_a_fitted_chain_it_declines_rather_than_inventing_one(
    gold_engine, gold_observations
):
    """StateUnavailable, not an exception: the daily run must survive this."""
    with pytest.raises(StateUnavailable, match="no fitted regime model"):
        gold_engine.predict(world_from(gold_observations))


def test_without_any_price_it_declines(gold_engine):
    rows = observation_rows("FRED:DGS10", {date(2020, 1, 2): 1.8})
    world = world_from(pd.DataFrame(rows), date(2020, 6, 1))
    with pytest.raises(StateUnavailable, match="no price history"):
        gold_engine.predict(world)


# --------------------------------------------------------------------------
# outputs
# --------------------------------------------------------------------------


@pytest.mark.slow
def test_publishes_every_metric_it_declares(fitted, gold_observations):
    rows = fitted.outputs(world_from(gold_observations))
    published = {row.metric for row in rows}

    assert published <= set(GOLD_METRICS), f"undeclared metrics: {published - set(GOLD_METRICS)}"
    for metric in GOLD_METRICS:
        assert metric in published, f"{metric} is declared but never published"


@pytest.mark.slow
def test_the_three_posteriors_are_published_separately(fitted, gold_observations):
    """§4's rule, applied to gold: never a composite without its parts."""
    rows = fitted.outputs(world_from(gold_observations))
    for name in GOLD_REGIMES:
        assert any(row.metric == posterior_metric(name) for row in rows)


@pytest.mark.slow
def test_regime_code_travels_with_its_label(fitted, gold_observations):
    rows = [r for r in fitted.outputs(world_from(gold_observations)) if r.metric == "regime_code"]
    assert rows
    for row in rows:
        label = (row.meta or {}).get("regime")
        assert label in GOLD_REGIMES
        assert row.value == regime_code(label)


@pytest.mark.slow
def test_the_nightly_window_is_bounded(fitted, gold_observations):
    """A nightly run must not republish sixty years to add one row."""
    rows = fitted.outputs(world_from(gold_observations))
    oldest = min(row.as_of for row in rows)
    assert (SNAPSHOT_AS_OF - oldest).days <= fitted.history_days + 5


@pytest.mark.slow
def test_full_history_reaches_the_start_of_the_record(fitted, gold_observations):
    fitted.full_history = True
    rows = fitted.outputs(world_from(gold_observations))
    assert min(row.as_of for row in rows).year <= 1969


def test_outputs_without_a_price_are_empty_not_an_error(gold_engine):
    world = world_from(
        pd.DataFrame(observation_rows("FRED:DGS10", {date(2020, 1, 2): 1.8})), date(2020, 6, 1)
    )
    assert gold_engine.outputs(world) == ()
    assert gold_engine.regime_states(world) == ()


# --------------------------------------------------------------------------
# regime_states
# --------------------------------------------------------------------------


@pytest.mark.slow
def test_regime_states_publish_the_whole_distribution(fitted, gold_observations):
    rows = fitted.regime_states(world_from(gold_observations))
    assert rows

    by_date: dict[date, dict[str, float]] = {}
    for row in rows:
        assert row.asset == "gold"
        assert row.regime in GOLD_REGIMES
        assert row.model_version == MODEL_VERSION
        by_date.setdefault(row.as_of, {})[row.regime] = row.probability

    for day, probabilities in by_date.items():
        assert set(probabilities) == set(GOLD_REGIMES), f"{day} is missing a regime"
        assert sum(probabilities.values()) == pytest.approx(1.0, abs=1e-6)


# --------------------------------------------------------------------------
# fit
# --------------------------------------------------------------------------


@pytest.mark.slow
def test_fit_writes_an_artifact_the_engine_can_read_back(gold_engine, gold_observations, artifacts):
    gold_engine.fit(world_from(gold_observations))

    stored = artifacts.load("gold")
    assert stored["model_version"] == MODEL_VERSION
    assert "fitted_as_of" in stored

    model_fit = gold_engine.stored_fit()
    assert model_fit is not None
    assert model_fit.k_regimes == 3
    assert model_fit.n_observations > 240


def test_fit_on_a_thin_history_writes_nothing(gold_engine, artifacts):
    rows = observation_rows(
        PRICE,
        {d.date(): 1800.0 + i for i, d in enumerate(pd.bdate_range("2024-01-01", periods=60))},
    )
    gold_engine.fit(world_from(pd.DataFrame(rows), date(2024, 6, 1)))
    assert artifacts.load("gold") == {}


# --------------------------------------------------------------------------
# Degradation
# --------------------------------------------------------------------------


@pytest.mark.slow
def test_shared_factor_scores_are_published_as_a_cross_check(fitted, gold_observations):
    """Layer 0's reading travels with the state, on its own axis and clearly labelled.

    They must not be mistaken for the drivers: the factors are 0-100 percentiles
    and the drivers are percentage points, so the two are namespaced apart.
    """
    from findynamics.core.contracts.state import FactorState

    as_of = date(2026, 7, 30)
    factors = {
        name: FactorState(name=name, as_of=as_of, score=score, components={})
        for name, score in (("real_rate", 41.2), ("usd_strength", 66.0), ("liquidity", 58.5))
    }
    state = fitted.predict(world_from(gold_observations, factors=factors))
    components = state.components or {}

    assert components["factor_real_rate"] == 41.2
    assert components["factor_usd_strength"] == 66.0
    assert components["factor_liquidity"] == 58.5
    # The engine's own real-rate driver is a rate, not a percentile.
    assert components["real_rate"] != components["factor_real_rate"]


@pytest.mark.slow
def test_missing_shared_factors_are_simply_absent(fitted, gold_observations):
    """A factor Layer 0 could not score must not appear as a zero."""
    state = fitted.predict(world_from(gold_observations))
    components = state.components or {}
    assert not [key for key in components if key.startswith("factor_")]


@pytest.mark.slow
def test_the_absent_equity_rii_is_reported_rather_than_hidden(fitted, gold_observations):
    """The snapshot has no published equity output, which is the first-run case."""
    state = fitted.predict(world_from(gold_observations))
    signal = next(s for s in state.signals if s.name == "equity_rii_absent")
    assert signal.direction == 0
    assert "instability index" in (signal.note or "")


@pytest.mark.slow
def test_a_corrupt_artifact_degrades_to_no_state(gold_engine, gold_observations, artifacts):
    artifacts.save("gold", {"regime": {"params": "not a vector"}, "model_version": MODEL_VERSION})
    with pytest.raises(StateUnavailable):
        gold_engine.predict(world_from(gold_observations))
    # But the per-date outputs still publish: that is why they are separate methods.
    assert gold_engine.outputs(world_from(gold_observations))


def test_the_vocabulary_rejects_a_name_it_does_not_own():
    with pytest.raises(ValueError, match="unknown gold regime"):
        regime_code("bull_expansion")
    with pytest.raises(ValueError, match="unknown gold regime"):
        posterior_metric("crisis")


def test_contract_violations_are_caught_at_construction():
    with pytest.raises(ContractError):
        AssetState(
            asset="gold",
            as_of=date(2026, 7, 30),
            regime="hedge_bid",
            expected_return=0.05,
            risk_score=140.0,
            confidence=0.5,
            signals=(),
            model_version=MODEL_VERSION,
        )
