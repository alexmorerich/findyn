"""The engine wrapper: what it publishes, and what it refuses to publish."""

from __future__ import annotations

import re
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
    SNAPSHOT_AS_OF,
    world_from,
)


def test_the_engine_registers_itself():
    assert ENGINES["equity"] is EquityEngine


def test_it_declares_every_configured_price_series(equity_engine):
    """All four, not just the resolved ones: whether a role is *available* is a
    question about D1, and a series the run never loads can never be available."""
    assert {
        PRIMARY,
        REGIME_PROXY,
        BACKFILL,
        DEEP_HISTORY,
    } <= set(equity_engine.required_series())


def test_it_declares_the_instability_inputs_it_can_do_without(equity_engine):
    """M4 added six non-price series, every one of them optional.

    They feed the RII's credit and liquidity components and the crash
    decomposition's fragility score. All are declared so the run can load them
    and none is required, because a composite that vanishes when one FRED series
    is late is worse than one that names what it was built from — which is what
    `RiiResult.missing` and `transmission_inputs` exist to do.
    """
    required = set(equity_engine.required_series())
    assert {"FRED:BAMLH0A0HYM2", "FRED:NFCI", "FRED:T10Y3M"} <= required
    # The money engine's short rate arrives as data, not as an import: §5's
    # layering forbids equity from importing rates or money at all.
    assert "ENGINE:money.short_rate" in required


def test_the_model_version_names_the_calibration_series(equity_engine, equity_observations):
    analysis = equity_engine.analyze(world_from(equity_observations))
    assert analysis.model_version == f"{MODEL_VERSION_BASE}+cal.yahoo_gspc"


# --- predict: the deliberate refusal ---------------------------------------


def test_predict_declines_before_the_first_refit(equity_engine, equity_observations):
    """An AssetState needs a regime and the only honest source is the fitted HMM.

    A fresh deployment has no stored fit, and a placeholder regime on a dashboard
    is read as a market view. So the engine declines — and says which job would
    fix it, because "no state" with no cause is an outage report, not an answer.
    """
    with pytest.raises(StateUnavailable, match="monthly_refit"):
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
    assert versions == {f"{MODEL_VERSION_BASE}+cal.yahoo_gspc"}


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
        "yahoo_gspc",
        "shiller_nominal_price",
    }
    for body in document["series"].values():
        assert body["ffd"]["d"] >= 0.0
        assert body["kalman"]["irregular"] > 0.0


def test_the_artifact_carries_no_wall_clock_timestamp(
    equity_engine, equity_observations, artifacts
):
    """Issue #6: an artifact must be a function of its inputs, not of the clock.

    The store compares bytes and refuses a changed payload under the same
    ``(model_version, fit date)``. A ``fitted_at`` recording when the container
    happened to run made every refit differ by construction, so a second run on
    one date was rejected as a conflict — while adding nothing, because the fit
    date is already the key and R2 records the write time as object metadata.

    Asserted structurally rather than by name: any field whose value looks like a
    wall-clock timestamp would reintroduce the same failure under a new spelling.
    ``as_of`` is a plain date and is exactly what should be here.
    """
    equity_engine.fit(world_from(equity_observations))
    document = artifacts.load(ARTIFACT_NAME)

    assert "fitted_at" not in document
    assert document["as_of"] == SNAPSHOT_AS_OF.isoformat()

    stamped = [
        key
        for key, value in document.items()
        if isinstance(value, str)
        and re.fullmatch(r"\d{4}-\d{2}-\d{2}T[\d:.]+(?:[+-][\d:]+|Z)?", value)
    ]
    assert not stamped, f"wall-clock timestamp(s) back in the artifact: {stamped}"


def test_two_fits_on_one_information_set_agree_on_what_they_publish(
    config, artifacts, tmp_path, equity_observations
):
    """The invariant worth holding: *semantic* equivalence, not byte equality.

    Two fits of the same specification on the same information set differ in
    their raw parameters — measured on the real snapshot, 128 float fields with a
    maximum relative difference of 3.3e-08, because multithreaded BLAS does not
    fix its reduction order. Every one of those differences is inert by the time
    it reaches a consumer: the regime posteriors agree to 4.7e-10 across 24,364
    dates, no date's label changes, and the published state is identical field
    for field.

    So this asserts what a reader of the dashboard would notice, which is
    nothing. Byte equality is available too — pin the BLAS thread count — but it
    is a stronger promise than the system needs and a slower one to keep.
    """
    from findynamics.core.artifacts import ArtifactStore

    world = world_from(equity_observations)
    published = []
    for name in ("first", "second"):
        store = ArtifactStore(tmp_path / name)
        engine = EquityEngine(config, store)
        engine.fit(world)
        engine._cache = None
        published.append(engine.predict(world))

    first, second = published
    assert first.regime == second.regime
    assert first.risk_score == second.risk_score
    assert first.confidence == second.confidence
    assert first.expected_return == second.expected_return
    assert (first.components or {}) == (second.components or {})
    assert [(s.name, s.value, s.direction) for s in first.signals] == [
        (s.name, s.value, s.direction) for s in second.signals
    ]


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
    assert roles["calibration"] == BACKFILL
    assert roles["calibration_is_proxy"] is False


def test_the_refit_stores_the_tail_and_the_daily_run_inherits_it(
    equity_engine, equity_observations, artifacts
):
    """§4's p_shock must survive a run that has no deep history — and one did not.

    `DAILY_ROLES` excludes `deep_history`: it is a 155-year monthly rebuild for
    an estimate that only moves when a new crash happens. So a nightly run can
    never *fit* the tail, and for one full deploy cycle it did not have one —
    every published `p_shock` was the flagged 1−exp(−0.25) base rate, identical
    on all 1254 dates, while `tail_fitted` said so in a field nobody was reading.

    The fit stores it; the daily run loads it. Both halves are asserted here
    because either alone is silently useless.
    """
    world = world_from(equity_observations)
    equity_engine.fit(world)

    stored = artifacts.load(ARTIFACT_NAME).get("tail")
    assert stored is not None, "the monthly refit must persist the tail fit"
    assert stored["series_id"] == DEEP_HISTORY
    assert stored["periods_per_year"] == 12.0, "the deep record is monthly; the units matter"

    equity_engine._cache = None
    analysis = equity_engine.analyze(world)  # DAILY_ROLES — no deep history

    assert "deep_history" not in analysis.features
    view = analysis.instability
    assert view is not None
    assert view.tail is not None, "the daily run must inherit the refit's tail"
    assert view.tail.series_id == DEEP_HISTORY
    # And the factor it feeds must actually vary, which is the symptom that
    # would have caught this: a constant p_shock is the fallback wearing a hat.
    assert view.crash["p_shock"].nunique() > 1


def test_without_a_stored_tail_p_shock_says_it_is_a_fallback(equity_engine, equity_observations):
    """No fit yet: publish the base rate, flagged, rather than nothing or zero."""
    analysis = equity_engine.analyze(world_from(equity_observations))
    view = analysis.instability
    if view is None:  # no regime model without a fit — nothing to assert
        return
    assert view.tail is None
    assert view.factors[12].detail["tail_fitted"] == 0.0


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


def test_the_daily_run_skips_the_deep_history_path(equity_engine, equity_observations):
    """The 1871+ monthly path is a backtest input, not a daily one.

    Calibration *is* computed daily from M4 onward: the RII scores each component
    as an expanding percentile, and against ten years of publication history a
    2020 reading would be ranked against a window that starts in 2016. The
    century-long calibration series is what makes the percentile mean anything.
    Deep history stays out — it is monthly, and mixing frequencies into one
    percentile is a different bug.
    """
    world = world_from(equity_observations)
    assert set(equity_engine.analyze(world).features) == {"publication", "calibration"}
    assert "deep_history" not in equity_engine.analyze(world).features
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
