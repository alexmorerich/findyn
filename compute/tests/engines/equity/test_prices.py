"""Role resolution: fixed precedence, pure in what D1 holds.

The acceptance criterion this file exists for: "``prices.py`` role resolution is
tested as a pure function of D1 contents". The failure it guards against is
subtle and expensive — a lower-precedence role arriving later, quietly moving the
window a model was fitted on without anything changing in code or config.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from findynamics.core.config import load_series_config
from findynamics.engines.equity import prices as prices_mod
from findynamics.engines.equity.prices import (
    CALIBRATION_PRECEDENCE,
    PriceRoleError,
    resolve,
    resolve_from,
)
from tests.engines.equity.conftest import (
    BACKFILL,
    DEEP_HISTORY,
    PRIMARY,
    REGIME_PROXY,
    SNAPSHOT_AS_OF,
    world_from,
)

ENOUGH = 5000


def counts(**kwargs: int) -> dict[str, int]:
    """Observation counts keyed by series id, defaulting everything to absent."""
    base = dict.fromkeys((PRIMARY, REGIME_PROXY, BACKFILL, DEEP_HISTORY), 0)
    base.update(kwargs)
    return base


@pytest.fixture
def config():
    return load_series_config()


def test_publication_is_always_the_primary_series(config):
    """No fallback, deliberately: a state labelled equity describes the S&P."""
    roles = resolve(counts(**{PRIMARY: ENOUGH, REGIME_PROXY: ENOUGH}), config)
    assert roles.publication.series_id == PRIMARY
    assert roles.publication.source_role == "primary"


def test_backfill_outranks_the_proxy_whenever_it_is_present(config):
    """The precedence is fixed, not "whichever has more rows today"."""
    roles = resolve(
        counts(**{PRIMARY: ENOUGH, BACKFILL: ENOUGH, REGIME_PROXY: 10 * ENOUGH}), config
    )
    assert roles.calibration.series_id == BACKFILL
    assert roles.calibration.source_role == "backfill"
    # Same index as the publication series, so nothing fitted on it is a proxy.
    assert not roles.calibration_is_proxy


def test_the_proxy_carries_calibration_when_the_backfill_is_absent(config):
    roles = resolve(counts(**{PRIMARY: ENOUGH, REGIME_PROXY: ENOUGH}), config)
    assert roles.calibration.series_id == REGIME_PROXY
    assert roles.calibration_is_proxy, "NASDAQ100 is not the published index"


def test_ingesting_the_lower_precedence_role_cannot_move_the_window(config):
    """The regression this module is really about.

    Once the backfill is in D1 it owns the calibration role. A later NASDAQ100
    ingest — a routine daily run — must not change which series a fitted model
    was trained on, or every parameter silently starts describing a different
    index under an unchanged model version.
    """
    before = resolve(counts(**{PRIMARY: ENOUGH, BACKFILL: ENOUGH}), config)
    after = resolve(
        counts(**{PRIMARY: ENOUGH, BACKFILL: ENOUGH, REGIME_PROXY: 10 * ENOUGH}), config
    )
    assert before.calibration.series_id == after.calibration.series_id
    assert before.tag == after.tag


def test_resolution_is_a_pure_function_of_the_counts(config):
    """Same counts in, same assignment out — no clock, no config state, no I/O."""
    given = counts(**{PRIMARY: ENOUGH, REGIME_PROXY: ENOUGH, DEEP_HISTORY: 1800})
    first = resolve(given, config)
    second = resolve(dict(given), config)
    assert first == second


def test_a_half_finished_backfill_does_not_win_the_precedence(config):
    """Below min_observations a role is absent. A partial ingest is not a
    training set, and letting one win is exactly the silent window move above."""
    roles = resolve(
        counts(**{PRIMARY: ENOUGH, BACKFILL: 40, REGIME_PROXY: ENOUGH}),
        config,
        min_observations=250,
    )
    assert roles.calibration.series_id == REGIME_PROXY


def test_the_threshold_is_the_only_thing_availability_depends_on(config):
    """Just over the line counts; just under does not."""
    over = resolve(counts(**{PRIMARY: ENOUGH, BACKFILL: 250}), config, min_observations=250)
    under = resolve(counts(**{PRIMARY: ENOUGH, BACKFILL: 249}), config, min_observations=250)
    assert over.calibration.series_id == BACKFILL
    assert under.calibration.series_id != BACKFILL


def test_no_publication_series_is_an_error_not_a_fallback(config):
    with pytest.raises(PriceRoleError, match="publication series"):
        resolve(counts(**{REGIME_PROXY: ENOUGH}), config)


def test_calibration_degrades_to_the_publication_series_loudly(config, caplog):
    """With no long series at all the engine still runs, and says what it lost."""
    with caplog.at_level("WARNING"):
        roles = resolve(counts(**{PRIMARY: ENOUGH}), config)
    assert roles.calibration.series_id == PRIMARY
    assert not roles.calibration_is_proxy
    assert "no crisis episodes in sample" in caplog.text


def test_the_tag_names_the_series_the_parameters_came_from(config):
    """A reader must never have to guess whether a number came from the S&P."""
    proxy = resolve(counts(**{PRIMARY: ENOUGH, REGIME_PROXY: ENOUGH}), config)
    real = resolve(counts(**{PRIMARY: ENOUGH, BACKFILL: ENOUGH}), config)

    assert proxy.tag == "cal.fred_nasdaq100"
    assert real.tag == "cal.yahoo_gspc"
    # Different tags mean different model versions, which means different
    # asset_state rows — the two fits cannot be confused for one another.
    assert proxy.tag != real.tag


def test_deep_history_is_optional(config):
    roles = resolve(counts(**{PRIMARY: ENOUGH, REGIME_PROXY: ENOUGH}), config)
    assert roles.deep_history is None
    assert "deep_history_obs" not in roles.as_components()


def test_precedence_order_is_the_documented_one():
    """The order is the contract; a reordering is a model change."""
    assert CALIBRATION_PRECEDENCE == ("backfill", "regime_proxy")


# --- what a refit is allowed to fit on (issue #6) ---------------------------


def test_a_complete_information_set_reports_nothing_unresolved(config):
    roles = resolve(
        counts(**{PRIMARY: ENOUGH, BACKFILL: ENOUGH, DEEP_HISTORY: ENOUGH, REGIME_PROXY: ENOUGH}),
        config,
    )
    assert roles.unresolved == ()


def test_a_missing_deep_history_is_reported_as_unresolved(config):
    """The case that made this necessary.

    `deep_history` feeds the `tail` block but not `model_version`, so a refit
    that lost it wrote a *different* artifact under the *same* key — which the
    storage layer can only report as a 409 on the next run.
    """
    roles = resolve(counts(**{PRIMARY: ENOUGH, BACKFILL: ENOUGH, REGIME_PROXY: ENOUGH}), config)
    assert roles.deep_history is None
    assert roles.unresolved == ("deep_history",)


def test_a_proxy_left_unused_by_precedence_is_not_missing(config):
    """The false alarm this has to avoid.

    `backfill` outranks `regime_proxy`, so in the shipped configuration the proxy
    is never consulted. Reporting it as unresolved would fail every refit for a
    series the fit does not use — the red pipeline for a non-event that issue #6
    complains about, reintroduced from the other direction.
    """
    roles = resolve(counts(**{PRIMARY: ENOUGH, BACKFILL: ENOUGH, DEEP_HISTORY: ENOUGH}), config)
    assert roles.calibration.source_role == "backfill"
    assert roles.unresolved == ()


def test_losing_every_calibration_candidate_reports_both(config):
    """Precedence excuses a lower-ranked role only while something outranks it."""
    roles = resolve(counts(**{PRIMARY: ENOUGH, DEEP_HISTORY: ENOUGH}), config)
    assert set(roles.unresolved) == {"backfill", "regime_proxy"}


def test_the_drivers_are_never_reported_as_unresolved(config):
    """`engines.equity.series` configures more than price records.

    `credit_spread`, `risk_free` and the rest feed the instability view, never
    :func:`resolve` and never a fitted artifact. A refit that failed because NFCI
    was briefly unavailable would be refusing to fit over a series it does not
    fit on.
    """
    roles = resolve(
        counts(**{PRIMARY: ENOUGH, BACKFILL: ENOUGH, DEEP_HISTORY: ENOUGH, REGIME_PROXY: ENOUGH}),
        config,
    )
    configured = set(prices_mod.configured_roles(config))
    assert {"credit_spread", "risk_free"} <= configured, "fixture assumes drivers are configured"
    assert set(roles.unresolved) <= set(prices_mod.PRICE_ROLES)


def test_the_shipped_snapshot_leaves_nothing_unresolved(equity_observations, config):
    """The configuration a production refit actually runs on."""
    roles = resolve_from(world_from(equity_observations).series, config)
    assert roles.unresolved == ()


# --- against the real snapshot ---------------------------------------------


def test_the_shipped_snapshot_resolves_to_the_documented_split(equity_observations, config):
    """The backfill role is filled, so calibration is the S&P itself.

    This is the configuration the precedence was written for and it arrived
    without a code change: a daily S&P source appeared, and because `backfill`
    outranks `regime_proxy` it took the calibration role and dropped the proxy
    caveat with it. The NASDAQ path is still exercised — see
    ``test_the_proxy_carries_calibration_when_the_backfill_is_absent`` — because
    it is what the engine falls back to.
    """
    world = world_from(equity_observations)
    roles = resolve_from(world.series, config)

    assert roles.publication.series_id == PRIMARY
    assert roles.calibration.series_id == BACKFILL
    assert roles.calibration.source_role == "backfill"
    assert not roles.calibration_is_proxy, (
        "YAHOO:^GSPC and FRED:SP500 are two vendors' copies of one index; "
        "calling that a proxy would attach a caveat the data does not warrant"
    )
    assert roles.deep_history is not None
    assert roles.deep_history.series_id == DEEP_HISTORY


def test_counts_come_from_the_information_set_not_from_config(equity_observations, config):
    """Every configured role is counted from what D1 actually holds."""
    world = world_from(equity_observations)
    tallied = prices_mod.observation_counts(
        world.series, [PRIMARY, REGIME_PROXY, BACKFILL, DEEP_HISTORY]
    )
    assert tallied[PRIMARY] == 2512
    assert tallied[BACKFILL] > 24000, "the S&P backfill reaches back to 1927"
    assert tallied[DEEP_HISTORY] > 1800


def test_an_earlier_cutoff_sees_fewer_observations(equity_observations, config):
    """Purity is in D1 contents, and the information set is part of D1 contents."""
    early = prices_mod.observation_counts(
        world_from(equity_observations, date(2018, 1, 2)).series, [PRIMARY]
    )
    late = prices_mod.observation_counts(
        world_from(equity_observations, SNAPSHOT_AS_OF).series, [PRIMARY]
    )
    assert 0 < early[PRIMARY] < late[PRIMARY]


def test_frequencies_annualize_differently(config):
    """Daily and monthly paths must not share an annualization factor."""
    roles = resolve(counts(**{PRIMARY: ENOUGH, REGIME_PROXY: ENOUGH, DEEP_HISTORY: 1800}), config)
    assert roles.publication.periods_per_year == 252.0
    assert roles.deep_history is not None
    assert roles.deep_history.periods_per_year == 12.0


def test_non_positive_closes_are_dropped_before_the_log(equity_observations, config):
    """A zero from a bad ingest would become -inf through the whole pipeline."""
    from tests.engines.equity.conftest import price_rows

    poisoned = pd.concat(
        [
            equity_observations,
            pd.DataFrame(price_rows(PRIMARY, {date(2026, 7, 30): 0.0})),
        ],
        ignore_index=True,
    )
    world = world_from(poisoned, date(2026, 8, 5))
    roles = resolve_from(world.series, config)
    path = prices_mod.price_path(world.series, roles.publication)
    assert (path > 0).all()


# --- the splice -------------------------------------------------------------


def test_the_backfill_takes_the_extension_role(config):
    """Same-index, same-frequency, so it can go in front of the publication series."""
    roles = resolve(counts(**{PRIMARY: ENOUGH, BACKFILL: ENOUGH}), config)
    assert roles.extension is not None
    assert roles.extension.series_id == BACKFILL
    assert roles.publication_input.series_id == f"{PRIMARY}+{BACKFILL}"


def test_the_proxy_is_never_spliced_in(config):
    """The failure this role separation exists for.

    ``regime_proxy`` is admissible as a *fitting* series with a caveat attached,
    and inadmissible as history for the published index under any caveat: a 1990
    velocity taken from the NASDAQ, published under a label that says S&P, is not
    something a footnote repairs.
    """
    roles = resolve(counts(**{PRIMARY: ENOUGH, REGIME_PROXY: 10 * ENOUGH}), config)
    assert roles.extension is None
    assert roles.publication_input.series_id == PRIMARY


def test_a_half_finished_backfill_does_not_extend_the_record_either(config):
    """The same threshold that governs calibration governs the splice.

    Prepending forty rows of a partial ingest would move the start of the
    published record — and therefore the filter's whole start-up — on the basis
    of how far an unfinished job happened to get.
    """
    roles = resolve(counts(**{PRIMARY: ENOUGH, BACKFILL: 40}), config, min_observations=250)
    assert roles.extension is None


def test_the_spliced_path_is_the_union_of_both_records(equity_observations, config):
    """The acceptance criterion: a century of closes under one identity."""
    world = world_from(equity_observations)
    roles = resolve_from(world.series, config)
    path, series = prices_mod.publication_path(world.series, roles)

    assert path.index[0].year == 1927
    assert len(path) > 24000
    assert series.series_id == f"{PRIMARY}+{BACKFILL}"
    assert series.observations == len(path)
    assert path.index.is_monotonic_increasing and not path.index.has_duplicates


def test_the_primary_vendor_owns_every_date_it_covers(equity_observations, config):
    """The splice may only *lengthen* the record, never restate it.

    Both vendors carry the whole of 2016-2026. If the extension were allowed to
    win a shared date, a figure already published from FRED would change vendor
    retroactively — the same class of silent restatement the point-in-time layer
    exists to prevent.
    """
    world = world_from(equity_observations)
    roles = resolve_from(world.series, config)
    path, _ = prices_mod.publication_path(world.series, roles)
    primary = prices_mod.price_path(world.series, roles.publication)

    shared = path.index.intersection(primary.index)
    assert len(shared) == len(primary)
    pd.testing.assert_series_equal(path.loc[shared], primary.loc[shared], check_names=False)


def test_a_disagreeing_vendor_is_refused_rather_than_spliced(equity_observations, config, caplog):
    """The guard, planted.

    Two vendors' copies of one index agree to rounding. A series that does not is
    something else — a different index, another currency, a rebasing — and
    joining it on would publish one market's history under another's name. The
    refusal falls back to the publication series alone, so the run still
    publishes; what it must not do is publish quietly.
    """
    rescaled = equity_observations.copy()
    backfill = rescaled["series_id"] == BACKFILL
    rescaled.loc[backfill, "value"] = rescaled.loc[backfill, "value"] * 1.4

    world = world_from(rescaled)
    roles = resolve_from(world.series, config)
    assert roles.extension is not None, "the role still resolves; only the join is refused"

    with caplog.at_level("ERROR"):
        path, series = prices_mod.publication_path(world.series, roles)

    assert series.series_id == PRIMARY
    assert path.index[0].year >= 2016
    assert "refusing to splice" in caplog.text


def test_records_that_never_overlap_cannot_be_checked_and_are_refused(config):
    """No shared dates is a refusal, not a pass.

    The agreement measurement is the only evidence that the two vendors carry the
    same index. Where there is none, splicing would be an assertion about the
    data taken purely from a role name in a yaml file.
    """
    early = pd.Series(
        [10.0, 11.0, 12.0], index=pd.to_datetime(["1990-01-02", "1990-01-03", "1990-01-04"])
    )
    late = pd.Series(
        [20.0, 21.0, 22.0], index=pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-06"])
    )
    overlap, disagreement = prices_mod.splice_disagreement(late, early)
    assert overlap == 0
    assert disagreement == float("inf")


def test_a_monthly_backfill_is_not_spliced_onto_a_daily_series(config, caplog):
    """Frequencies are not interchangeable: velocity is a rate per observation.

    Joining a monthly record to a daily one would change what an observation
    means partway through the series, and every annualized figure before the seam
    would be wrong by a factor of twenty-one. Nothing in the role name says
    otherwise, so the frequencies are compared rather than assumed.
    """
    with caplog.at_level("WARNING"):
        roles = resolve(counts(**{PRIMARY: ENOUGH, BACKFILL: ENOUGH}), _monthly_backfill(config))

    assert roles.extension is None
    assert roles.publication_input.series_id == PRIMARY
    assert "cannot be spliced" in caplog.text


def _monthly_backfill(config):
    """The shipped config with the backfill role re-declared as monthly."""
    from dataclasses import replace as replace_field

    engine = config.engines["equity"]
    series = dict(engine.series)
    series["backfill"] = replace_field(series["backfill"], frequency="monthly")
    engines = dict(config.engines)
    engines["equity"] = replace_field(engine, series=series)
    return replace_field(config, engines=engines)
