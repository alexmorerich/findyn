"""DERIVED:* series — arithmetic over other series, and their release dates.

The arithmetic is the easy half and barely worth testing. What these pin is the
**point-in-time composition**, because getting it wrong produces a series that
looks perfect: right shape, right magnitude, plausible history, and a release
date early enough to license lookahead that no downstream check can detect.

The rule under test: a derived observation is knowable only once its *slowest*
input has been released. Not on its own observation date plus a configured lag —
`series.yaml` gives DERIVED:EXCESS_CAPE_YIELD a 30-day lag, and honouring that
literally would publish January's value on 31 January while it depends on a CAPE
print that lands weeks later.
"""

from __future__ import annotations

import json
from datetime import date

import pytest

from findynamics.data.providers.base import NotFoundError, Observation
from findynamics.data.providers.derived import RECIPES, DerivedProvider

BASE = "https://findyn.test"


class FakeUpstream:
    """Stands in for FRED/Shiller: canned observations, and a record of asks."""

    id = "fake"
    requires_api_key = False

    def __init__(self, series: dict[str, list[Observation]]) -> None:
        self.series = series
        self.requested: list[str] = []

    def available_series(self) -> list[str]:
        return sorted(self.series)

    def fetch_metadata(self, series_id: str):  # noqa: ANN201
        raise NotImplementedError

    def fetch_observations(self, series_id, *, start=None, end=None):  # noqa: ANN001, ANN201
        self.requested.append(series_id)
        return list(self.series.get(series_id, []))


def obs(obs_date: str, release: str | None, value: float) -> Observation:
    """One canned upstream observation. The series id is irrelevant to the
    composition under test, so it is not a parameter."""
    return Observation(
        series_id="X",
        provider="fake",
        frequency="daily",
        unit="percent",
        observation_date=date.fromisoformat(obs_date),
        release_date=date.fromisoformat(release) if release else date.fromisoformat(obs_date),
        value=value,
    )


def provider(series: dict[str, list[Observation]]) -> DerivedProvider:
    """A DerivedProvider whose inputs come from one canned upstream.

    The real one resolves each input's provider through series.yaml and builds
    it from the registry; here one fake serves them all, which is the seam the
    injectable builder exists to give.
    """
    upstream = FakeUpstream(series)
    return DerivedProvider(build=lambda _provider_id: upstream)


# ---------------------------------------------------------------------------
# the composition rule
# ---------------------------------------------------------------------------


def test_the_release_date_is_the_slowest_input_not_a_configured_lag() -> None:
    """The reason this module exists rather than the arithmetic living in a factor.

    CAPE for January is released on 20 February. The rate legs for the same
    period were released the next business day. The excess CAPE yield for
    January is therefore knowable on **20 February** — not on 31 January, which
    is what `publication_lag_days: 30` would have produced, and not on the rate
    legs' own date.
    """
    p = provider(
        {
            "SHILLER:CAPE": [obs("2026-01-31", "2026-02-20", 30.0)],
            "FRED:DGS10": [obs("2026-01-30", "2026-01-31", 4.2)],
            "FRED:T10YIE": [obs("2026-01-30", "2026-01-31", 2.3)],
        }
    )

    [observation] = p.fetch_observations("DERIVED:EXCESS_CAPE_YIELD")

    assert observation.observation_date == date(2026, 1, 31)
    assert observation.release_date == date(2026, 2, 20)
    # 1/30 = 3.333%, real rate = 4.2 - 2.3 = 1.9%
    assert observation.value == pytest.approx(3.3333 - 1.9, abs=1e-3)


def test_a_fast_driver_still_waits_for_a_slow_leg() -> None:
    """The maximum runs in both directions, which a `driver wins` shortcut would miss."""
    p = provider(
        {
            "FRED:BAMLH0A0HYM2": [obs("2026-03-02", "2026-03-03", 7.8)],
            "FRED:BAMLC0A0CM": [obs("2026-03-02", "2026-03-09", 1.2)],
        }
    )

    [observation] = p.fetch_observations("DERIVED:HY_IG_DIFFERENTIAL")
    assert observation.release_date == date(2026, 3, 9)
    assert observation.value == pytest.approx(6.6)


def test_values_pair_by_period_and_the_release_date_is_computed_after() -> None:
    """The two questions, kept apart.

    *Which values pair?* January's CAPE with January's rate — the excess CAPE
    yield is a statement about one period. Pairing on release dates instead would
    hand January's valuation a mid-February rate, because that is what had been
    published by the time the CAPE print landed. Economically wrong, and
    invisible in the output.

    *When is the pair knowable?* Once the slowest of them was released.
    """
    p = provider(
        {
            "SHILLER:CAPE": [obs("2026-01-31", "2026-02-20", 25.0)],
            "FRED:DGS10": [
                obs("2026-01-30", "2026-01-31", 4.0),  # January: the right period
                obs("2026-02-19", "2026-02-20", 9.9),  # February: must not be used
            ],
            "FRED:T10YIE": [obs("2026-01-30", "2026-01-31", 2.0)],
        }
    )

    [observation] = p.fetch_observations("DERIVED:EXCESS_CAPE_YIELD")
    # 1/25 = 4%, real = 4.0 - 2.0 = 2.0 -> 2.0. February's leg would give -5.9.
    assert observation.value == pytest.approx(4.0 - 2.0, abs=1e-3)
    # Knowable when CAPE landed, which is later than either rate leg.
    assert observation.release_date == date(2026, 2, 20)


def test_periods_with_no_input_are_dropped_not_back_filled() -> None:
    """Trims the early history to where every input actually existed.

    The alternative — carrying the first available rate backwards — invents a
    breakeven for 1990, when TIPS did not trade.
    """
    p = provider(
        {
            "SHILLER:CAPE": [
                obs("2025-11-30", "2025-12-20", 30.0),  # predates the rate legs
                obs("2026-01-31", "2026-02-20", 30.0),
            ],
            "FRED:DGS10": [obs("2026-01-02", "2026-01-03", 4.2)],
            "FRED:T10YIE": [obs("2026-01-02", "2026-01-03", 2.3)],
        }
    )

    observations = p.fetch_observations("DERIVED:EXCESS_CAPE_YIELD")
    assert [o.observation_date for o in observations] == [date(2026, 1, 31)]


def test_every_input_carries_a_release_date_by_construction() -> None:
    """The composition needs one per input, and the type already guarantees it.

    An earlier version of this provider parsed inputs out of JSON and had to
    check for a missing release date itself. Taking `Observation` instead moves
    the guarantee into the type: it refuses to construct without one, and
    refuses a release date that precedes its own observation date. One rule,
    enforced where the data enters the system rather than at each consumer.
    """
    with pytest.raises(ValueError, match="license lookahead"):
        Observation(
            series_id="X",
            provider="fake",
            frequency="daily",
            unit="percent",
            observation_date=date(2026, 3, 2),
            release_date=date(2026, 3, 1),
            value=1.0,
        )


# ---------------------------------------------------------------------------
# refusing to publish something that is not what it says it is
# ---------------------------------------------------------------------------


def test_a_missing_input_is_an_error_not_a_shorter_series() -> None:
    """An excess CAPE yield without its rate legs is an earnings yield.

    Publishing it under the derived name would be a different quantity wearing
    the right label — which no consumer could detect, because the numbers are
    individually plausible.
    """
    p = provider({"SHILLER:CAPE": [obs("2026-01-31", "2026-02-20", 30.0)]})
    with pytest.raises(NotFoundError, match="FRED:DGS10"):
        p.fetch_observations("DERIVED:EXCESS_CAPE_YIELD")


def test_an_unknown_derived_series_names_what_it_can_build() -> None:
    p = provider({})
    with pytest.raises(NotFoundError, match="DERIVED:EXCESS_CAPE_YIELD"):
        p.fetch_observations("DERIVED:NOT_A_THING")


def test_an_input_missing_from_series_yaml_is_named(config) -> None:
    """Inputs are resolved through config, not by parsing the id prefix.

    A recipe naming something nobody configured has to say so — otherwise the
    provider fetches nothing and the derived series is simply absent, which is
    the failure mode this whole task existed to remove.
    """
    from findynamics.data.providers.derived import RECIPES, Recipe

    bogus = Recipe(
        series_id="DERIVED:BOGUS",
        frequency="daily",
        unit="percent",
        title="t",
        notes="n",
        inputs=("FRED:NOT_CONFIGURED_ANYWHERE",),
        combine=lambda values: values[0],
    )
    p = DerivedProvider(build=lambda _id: FakeUpstream({}), config=config)
    with (
        _TemporaryRecipe(RECIPES, "DERIVED:BOGUS", bogus),
        pytest.raises(NotFoundError, match="FRED:NOT_CONFIGURED_ANYWHERE"),
    ):
        p.fetch_observations("DERIVED:BOGUS")


class _TemporaryRecipe:
    """Add one recipe for the duration of a test, then take it away."""

    def __init__(self, table: dict, key: str, value: object) -> None:
        self.table, self.key, self.value = table, key, value

    def __enter__(self) -> None:
        self.table[self.key] = self.value

    def __exit__(self, *exc: object) -> None:
        self.table.pop(self.key, None)


def test_a_non_invertible_cape_is_skipped_rather_than_producing_an_infinity() -> None:
    p = provider(
        {
            "SHILLER:CAPE": [
                obs("2026-01-31", "2026-02-20", 0.0),
                obs("2026-02-28", "2026-03-20", 25.0),
            ],
            "FRED:DGS10": [obs("2026-01-02", "2026-01-03", 4.0)],
            "FRED:T10YIE": [obs("2026-01-02", "2026-01-03", 2.0)],
        }
    )
    observations = p.fetch_observations("DERIVED:EXCESS_CAPE_YIELD")
    assert [o.observation_date for o in observations] == [date(2026, 2, 28)]


# ---------------------------------------------------------------------------
# metadata and units
# ---------------------------------------------------------------------------


def test_metadata_declares_the_frequency_the_recipe_produces() -> None:
    p = provider({})
    assert p.fetch_metadata("DERIVED:EXCESS_CAPE_YIELD").frequency == "monthly"
    assert p.fetch_metadata("DERIVED:HY_IG_DIFFERENTIAL").frequency == "daily"


def test_the_yield_leg_is_in_percent_like_the_rate_legs() -> None:
    """Units. 1/CAPE is a fraction and the rates are percentages; mixing them
    scales the factor by 100 and it still looks like a plausible series."""
    p = provider(
        {
            "SHILLER:CAPE": [obs("2026-01-31", "2026-02-20", 20.0)],
            "FRED:DGS10": [obs("2026-01-02", "2026-01-03", 0.0)],
            "FRED:T10YIE": [obs("2026-01-02", "2026-01-03", 0.0)],
        }
    )
    # 1/20 = 5%, not 0.05.
    assert p.fetch_observations("DERIVED:EXCESS_CAPE_YIELD")[0].value == pytest.approx(5.0)


def test_every_recipe_names_inputs_that_series_yaml_ingests() -> None:
    """A recipe pointing at a series nobody fetches is a provider that always
    raises — the exact failure mode this whole task exists to remove."""
    from findynamics.core.config import load_series_config

    configured = {spec.id for spec in load_series_config().all_series()}
    for recipe in RECIPES.values():
        missing = [name for name in recipe.inputs if name not in configured]
        assert not missing, f"{recipe.series_id} needs uningested {missing}"


def test_an_upstream_failure_propagates_rather_than_yielding_a_short_series() -> None:
    """A derived series built from a provider that errored is not shorter — it is
    wrong, and it would publish under a name promising otherwise."""
    from findynamics.data.providers.base import ProviderError

    class BrokenUpstream(FakeUpstream):
        def fetch_observations(self, series_id, *, start=None, end=None):  # noqa: ANN001, ANN201
            raise ProviderError("fake", "upstream is down", retryable=True)

    p = DerivedProvider(build=lambda _id: BrokenUpstream({}))
    with pytest.raises(ProviderError, match="upstream is down"):
        p.fetch_observations("DERIVED:HY_IG_DIFFERENTIAL")


def test_the_recipes_are_serializable_documentation() -> None:
    """The table is the spec for what these series are; it has to be readable."""
    for recipe in RECIPES.values():
        assert recipe.notes
        assert json.dumps({"id": recipe.series_id, "inputs": list(recipe.inputs)})


# ---------------------------------------------------------------------------
# the other half: a provider that cannot be built must stop the run
# ---------------------------------------------------------------------------


def test_an_unimplemented_provider_stops_the_run_instead_of_being_logged(config) -> None:
    """The failure that hid DERIVED:* for months, now impossible to miss.

    §14.2 says a provider *failure* costs its series and not the run — the API
    is down, the key expired, and losing that input is the right price. A
    provider that cannot be *built* is a different animal: series.yaml names
    something no code implements, every run degrades identically, and retrying
    changes nothing. Continuing there published valuation and risk-appetite
    scores computed from fewer inputs than they claimed, with nothing in the
    output saying so.
    """
    import dataclasses

    from findynamics.data.providers.base import ProviderError
    from findynamics.data.store import load_observations

    real = next(spec for spec in config.all_series() if spec.provider == "fred")
    broken = dataclasses.replace(real, provider="not_a_provider")
    patched = dataclasses.replace(
        config,
        factors={
            **config.factors,
            "valuation": dataclasses.replace(config.factors["valuation"], series=[broken]),
        },
    )

    with pytest.raises(ProviderError, match="configuration error"):
        load_observations([broken.id], config=patched)


def test_derived_is_registered_and_keyless() -> None:
    """It reads the serving plane, not a vendor, so it needs a location rather
    than a credential — the same shape as the engine_output provider."""
    from findynamics.data.providers import KEYLESS_PROVIDERS, NETWORK_PROVIDERS, build_provider

    assert "derived" in NETWORK_PROVIDERS
    assert "derived" in KEYLESS_PROVIDERS

    built = build_provider("derived", env={"FINDYN_API_URL": BASE})
    assert built.id == "derived"
    assert built.requires_api_key is False


def test_it_reports_available_because_its_inputs_carry_their_own_credentials() -> None:
    """Unlike engine_output, `derived` needs no location and no key of its own.

    Each input is fetched through its own provider, under that source's
    credentials and quota. Reporting it unavailable when FINDYN_API_URL is unset
    — which an earlier version did — described a dependency it does not have.
    """
    from findynamics.data.providers import available_providers

    assert available_providers({})["derived"] is True
