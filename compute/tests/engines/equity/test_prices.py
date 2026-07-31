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
