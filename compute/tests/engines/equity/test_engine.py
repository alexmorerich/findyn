"""The engine wrapper: what it publishes, and what it refuses to publish."""

from __future__ import annotations

import math
from datetime import date, timedelta

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


def test_the_model_version_names_both_inputs(equity_engine, equity_observations):
    """The fitted series *and* the filtered series, because both are choices.

    ``cal.`` has always been there: a parameter fitted on the NASDAQ is not a
    parameter fitted on the S&P. ``pub.`` is there for the same reason plus one
    more — whether the backfill is spliced in front of the primary series is
    decided partly by the data, so the same code can publish a decade or a
    century. Sharing a version, the two would upsert over each other on every
    date they have in common.
    """
    analysis = equity_engine.analyze(world_from(equity_observations))
    assert analysis.model_version == (
        f"{MODEL_VERSION_BASE}+pub.fred_sp500_yahoo_gspc+cal.yahoo_gspc"
    )


def test_an_unspliced_run_keeps_the_shorter_version_string(equity_engine, equity_observations):
    """No ``pub.`` tag when nothing was spliced — the suffix marks a real choice.

    Dropping the backfill is the configuration where the publication path is the
    ten years FRED licences, and its version must stay distinguishable from the
    century's rather than merely being a different string.
    """
    without_backfill = equity_observations[equity_observations["series_id"] != BACKFILL]
    analysis = equity_engine.analyze(world_from(without_backfill))

    assert analysis.publication.series.series_id == PRIMARY
    assert analysis.model_version == f"{MODEL_VERSION_BASE}+cal.fred_nasdaq100"


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
    assert versions == {f"{MODEL_VERSION_BASE}+pub.fred_sp500_yahoo_gspc+cal.yahoo_gspc"}


def test_published_rows_are_finite_and_within_the_history_window(
    equity_engine, equity_observations
):
    world = world_from(equity_observations)
    rows = [*equity_engine.outputs(world), *equity_engine.derived_features(world)]
    assert all(math.isfinite(row.value) for row in rows)

    oldest = min(row.as_of for row in rows)
    newest = max(row.as_of for row in rows)
    assert (newest - oldest).days <= equity_engine.history_days + 5


# --- the century --------------------------------------------------------------


@pytest.fixture(scope="module")
def full_history_outputs(config_module, equity_observations):
    """Every ``engine_output`` row a ``--full-history`` run would publish.

    Module-scoped: the run filters a century twice and the assertions below are
    all about the same set of rows.
    """
    from findynamics.core.artifacts import ArtifactStore

    engine = EquityEngine(config_module, ArtifactStore(directory=None))
    engine.full_history = True
    return engine.outputs(world_from(equity_observations))


@pytest.fixture(scope="module")
def config_module():
    from findynamics.core.config import load_series_config

    return load_series_config()


def test_velocity_is_published_for_the_whole_daily_record(full_history_outputs):
    """The acceptance criterion: a century of trend states, not a decade.

    The counts are floors rather than equalities because the record grows by a
    row a day and the ten-year FRED window slides. What they pin down is the
    order of magnitude — 24,700 sessions since 1927 against the ~2,500 FRED
    licences — so this cannot be satisfied by the publication series alone
    however the calendar moves.
    """
    velocity = sorted(row.as_of for row in full_history_outputs if row.metric == "velocity")

    assert len(velocity) > 20000
    assert velocity[0] < date(1930, 1, 1)
    assert velocity[-1] > date(2026, 1, 1)


def test_acceleration_reaches_as_far_back_as_velocity(full_history_outputs):
    """Both are the same state estimate read at two orders; a chart of one
    against the other must not be comparing different spans."""
    spans = {
        metric: sorted(row.as_of for row in full_history_outputs if row.metric == metric)
        for metric in ("velocity", "acceleration")
    }
    assert spans["acceleration"][0] - spans["velocity"][0] <= timedelta(days=5)
    assert len(spans["acceleration"]) > 20000


def test_the_early_high_volatility_years_do_not_blow_the_estimates_up(full_history_outputs):
    """1929-32 is the stress case the extension exists to reach, and it is also
    where a filter that had not settled would diverge.

    The bounds are read as economics, not as tolerances. An annualized log drift
    of ±1 is a trend of +172%/-63% a year sustained by the *filter*, which is not
    a thing markets do — the 1929-32 collapse, the worst in the record, reaches
    -0.34. Anything past 1 is the diffuse prior leaking through, which is exactly
    what the burn-in exists to remove.
    """
    early = [
        row
        for row in full_history_outputs
        if row.metric in {"velocity", "acceleration"} and row.as_of < date(1940, 1, 1)
    ]
    assert early, "no pre-1940 rows were published at all"

    velocity = [row.value for row in early if row.metric == "velocity"]
    acceleration = [row.value for row in early if row.metric == "acceleration"]

    assert all(math.isfinite(v) for v in velocity)
    assert max(abs(v) for v in velocity) < 1.0
    # Acceleration is a difference of annualized rates scaled again by 252, so it
    # is legitimately an order of magnitude larger; the start-up spike it is
    # being checked against was 359.
    assert max(abs(a) for a in acceleration) < 50.0


def test_the_jerk_baseline_is_no_longer_short(equity_engine, equity_observations):
    """§8.3 asks for a ten-year expanding baseline and used to be refused it.

    The publication series held ten years in total, so the baseline degraded to
    half of that and every published state carried `jerk_baseline_is_short`. With
    the record spliced back to 1927 the spec's window is simply available.
    """
    analysis = equity_engine.analyze(world_from(equity_observations))
    diagnostics = analysis.publication.diagnostics
    assert diagnostics["jerk_baseline_is_short"] == 0.0
    assert diagnostics["jerk_baseline_periods"] == 2520.0


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

    # The publication path is keyed on the *spliced* identity, not on the primary
    # series: `d` and the Kalman variances were searched over a century of closes
    # and applying them to the ten years FRED alone holds would impose the wrong
    # memory. compute_features refuses that outright — the guard is the key.
    assert set(document["series"]) == {
        "fred_sp500_yahoo_gspc",
        "yahoo_gspc",
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
