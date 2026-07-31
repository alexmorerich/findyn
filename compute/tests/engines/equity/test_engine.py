"""The engine wrapper: what it publishes, and what it refuses to publish."""

from __future__ import annotations

from datetime import date

import pytest

from findynamics.core.engine import StateUnavailable
from findynamics.core.registry import ENGINES
from findynamics.engines.equity.domain import CHART_METRICS, KINEMATIC_FEATURES
from findynamics.engines.equity.engine import (
    ALL_ROLES,
    ARTIFACT_NAME,
    MODEL_VERSION_BASE,
    EquityEngine,
)
from tests.engines.equity.conftest import (
    BACKFILL,
    DEEP_HISTORY,
    PRIMARY,
    REGIME_PROXY,
    world_from,
)


def test_the_engine_registers_itself():
    assert ENGINES["equity"] is EquityEngine


def test_it_declares_every_configured_price_series(equity_engine):
    """All four, not just the resolved ones: whether a role is *available* is a
    question about D1, and a series the run never loads can never be available."""
    assert set(equity_engine.required_series()) == {
        PRIMARY,
        REGIME_PROXY,
        BACKFILL,
        DEEP_HISTORY,
    }


def test_the_model_version_names_the_calibration_series(equity_engine, equity_observations):
    analysis = equity_engine.analyze(world_from(equity_observations))
    assert analysis.model_version == f"{MODEL_VERSION_BASE}+cal.fred_nasdaq100"


# --- predict: the deliberate refusal ---------------------------------------


def test_predict_declines_rather_than_inventing_a_regime(equity_engine, equity_observations):
    """An AssetState needs a regime and there is no honest source of one until
    the HMM lands. Declining is the answer; a placeholder would be published to
    a dashboard and read as a market view."""
    with pytest.raises(StateUnavailable, match="sub-milestone B"):
        equity_engine.predict(world_from(equity_observations))


def test_the_refusal_says_what_it_did_manage_to_compute(equity_engine, equity_observations):
    with pytest.raises(StateUnavailable) as caught:
        equity_engine.predict(world_from(equity_observations))
    assert "features are published" in str(caught.value)


def test_state_unavailable_is_not_an_ordinary_failure():
    """The job layer branches on this type, so the distinction is load-bearing."""
    assert issubclass(StateUnavailable, RuntimeError)
    assert not issubclass(StateUnavailable, ValueError)


# --- what it does publish ---------------------------------------------------


def test_outputs_carry_the_charted_metrics_in_index_units(equity_engine, equity_observations):
    world = world_from(equity_observations)
    rows = equity_engine.outputs(world)
    assert rows

    published = {row.metric for row in rows}
    assert published == set(CHART_METRICS)

    # price_close and price_filtered are index points here, not logs: an S&P
    # close in the thousands, not a log level near nine.
    closes = [row.value for row in rows if row.metric == "price_close"]
    filtered = [row.value for row in rows if row.metric == "price_filtered"]
    assert min(closes) > 100 and min(filtered) > 100


def test_derived_features_carry_the_model_inputs_in_model_units(equity_engine, equity_observations):
    rows = equity_engine.derived_features(world_from(equity_observations))
    assert rows

    published = {row.feature for row in rows}
    assert set(KINEMATIC_FEATURES) <= published
    assert {"momentum_1m", "momentum_3m", "momentum_12m"} <= published

    # The feature store keeps logs — this is what the model saw.
    logs = [row.value for row in rows if row.feature == "price_filtered"]
    assert 5.0 < min(logs) < max(logs) < 12.0


def test_every_feature_row_carries_its_own_model_version(equity_engine, equity_observations):
    """It is part of the table's key, so it cannot come off the run envelope."""
    rows = equity_engine.derived_features(world_from(equity_observations))
    versions = {row.model_version for row in rows}
    assert versions == {f"{MODEL_VERSION_BASE}+cal.fred_nasdaq100"}


def test_published_rows_are_finite_and_within_the_history_window(
    equity_engine, equity_observations
):
    import math

    world = world_from(equity_observations)
    rows = [*equity_engine.outputs(world), *equity_engine.derived_features(world)]
    assert all(math.isfinite(row.value) for row in rows)

    oldest = min(row.as_of for row in rows)
    newest = max(row.as_of for row in rows)
    assert (newest - oldest).days <= equity_engine.history_days + 5


def test_the_jerk_lamp_travels_as_a_code_with_its_label(equity_engine, equity_observations):
    """engine_output stores REALs, so the label rides in meta."""
    rows = [
        r for r in equity_engine.outputs(world_from(equity_observations)) if r.metric == "jerk_lamp"
    ]
    assert rows
    assert {r.value for r in rows} <= {0.0, 1.0, 2.0}
    assert all(r.meta and r.meta["lamp"] in {"calm", "elevated", "extreme"} for r in rows)


def test_nothing_is_published_past_the_information_set(equity_engine, equity_observations):
    cutoff = date(2023, 6, 30)
    world = world_from(equity_observations, cutoff)
    rows = [*equity_engine.outputs(world), *equity_engine.derived_features(world)]
    assert rows
    assert max(row.as_of for row in rows) <= cutoff


# --- fit --------------------------------------------------------------------


def test_fit_freezes_parameters_for_every_resolved_role(
    equity_engine, equity_observations, artifacts
):
    equity_engine.fit(world_from(equity_observations))
    document = artifacts.load(ARTIFACT_NAME)

    assert set(document["series"]) == {
        "fred_sp500",
        "fred_nasdaq100",
        "shiller_nominal_price",
    }
    for body in document["series"].values():
        assert body["ffd"]["d"] >= 0.0
        assert body["kalman"]["irregular"] > 0.0


def test_the_frozen_d_differs_between_series(equity_engine, equity_observations, artifacts):
    """The whole reason ``d`` is frozen per series rather than per engine."""
    equity_engine.fit(world_from(equity_observations))
    chosen = {
        slug: body["ffd"]["d"] for slug, body in artifacts.load(ARTIFACT_NAME)["series"].items()
    }
    assert len(set(chosen.values())) > 1, chosen


def test_the_artifact_records_which_series_supplied_the_parameters(
    equity_engine, equity_observations, artifacts
):
    equity_engine.fit(world_from(equity_observations))
    roles = artifacts.load(ARTIFACT_NAME)["roles"]
    assert roles["calibration"] == REGIME_PROXY
    assert roles["calibration_is_proxy"] is True


def test_a_daily_run_reuses_frozen_parameters(equity_engine, equity_observations):
    equity_engine.fit(world_from(equity_observations))
    equity_engine._cache = None
    analysis = equity_engine.analyze(world_from(equity_observations))
    assert analysis.publication.used_frozen_params


def test_a_missing_artifact_is_not_fatal(equity_engine, equity_observations):
    """A daily run must not stop because a refit has not happened yet."""
    analysis = equity_engine.analyze(world_from(equity_observations))
    assert not analysis.publication.used_frozen_params
    assert len(analysis.publication.frame) > 2000


def test_the_daily_run_only_pays_for_the_publication_path(equity_engine, equity_observations):
    """The calibration path is 10k observations of MLE; a run that will not use
    it should not compute it."""
    world = world_from(equity_observations)
    assert set(equity_engine.analyze(world).features) == {"publication"}
    equity_engine._cache = None
    assert set(equity_engine.analyze(world, roles=ALL_ROLES).features) == set(ALL_ROLES)


def test_analysis_is_memoized_per_world_and_role_set(equity_engine, equity_observations):
    world = world_from(equity_observations)
    assert equity_engine.analyze(world) is equity_engine.analyze(world)


def test_an_information_set_with_no_prices_declines_rather_than_crashes(equity_engine):
    import pandas as pd

    from findynamics.core.contracts.state import WorldState
    from findynamics.data.accessor import PandasPITAccessor

    empty = pd.DataFrame(
        columns=["series_id", "obs_date", "release_date", "revision_date", "value"]
    )
    world = WorldState(
        as_of=date(2026, 7, 30),
        factors={},
        series=PandasPITAccessor(empty, date(2026, 7, 30)),
    )
    # Thin, not broken: the job layer must not record this as an engine failure.
    with pytest.raises(StateUnavailable, match="publication series"):
        equity_engine.analyze(world)

    # And the publishing methods degrade to empty rather than propagating.
    assert equity_engine.outputs(world) == ()
    assert equity_engine.derived_features(world) == ()
