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

from findynamics.data.providers.base import NotFoundError, ParseError
from findynamics.data.providers.derived import RECIPES, DerivedProvider

BASE = "https://findyn.test"


class FakeResponse:
    def __init__(self, payload: object) -> None:
        self._payload = payload

    def json(self) -> object:
        if isinstance(self._payload, str):
            raise ValueError("not JSON")
        return self._payload


class FakeTransport:
    """Serves canned observations per series id, and records what was asked for."""

    def __init__(self, series: dict[str, list[dict]]) -> None:
        self.series = series
        self.requested: list[str] = []

    def get(self, url: str, *, params=None, cache_ttl=None):  # noqa: ANN001
        from urllib.parse import unquote

        series_id = unquote(url.rsplit("/", 1)[-1])
        self.requested.append(series_id)
        rows = self.series.get(series_id)
        if rows is None:
            return FakeResponse({"data": {"observations": []}})
        return FakeResponse({"data": {"observations": rows}})


def obs(obs_date: str, release: str | None, value: float) -> dict:
    return {"obs_date": obs_date, "release_date": release, "value": value}


def provider(series: dict[str, list[dict]]) -> DerivedProvider:
    return DerivedProvider(FakeTransport(series), base_url=BASE)


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


def test_an_observation_with_no_release_date_is_refused() -> None:
    """It cannot be composed: the maximum would silently ignore it and the result
    would claim to be knowable earlier than it was."""
    p = provider(
        {
            "FRED:BAMLH0A0HYM2": [obs("2026-03-02", None, 7.8)],
            "FRED:BAMLC0A0CM": [obs("2026-03-02", "2026-03-03", 1.2)],
        }
    )
    with pytest.raises(ParseError, match="release_date"):
        p.fetch_observations("DERIVED:HY_IG_DIFFERENTIAL")


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


def test_without_a_serving_plane_it_says_so_rather_than_returning_nothing() -> None:
    p = DerivedProvider(FakeTransport({}), base_url=None)
    with pytest.raises(NotFoundError, match="FINDYN_API_URL"):
        p.fetch_observations("DERIVED:HY_IG_DIFFERENTIAL")


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


def test_a_malformed_envelope_is_a_parse_error_not_an_empty_series() -> None:
    class BadTransport:
        def get(self, url, *, params=None, cache_ttl=None):  # noqa: ANN001
            return FakeResponse({"nope": True})

    p = DerivedProvider(BadTransport(), base_url=BASE)
    with pytest.raises(ParseError, match="envelope"):
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


def test_availability_tracks_whether_there_is_a_plane_to_read_from() -> None:
    from findynamics.data.providers import available_providers

    assert available_providers({"FINDYN_API_URL": BASE})["derived"] is True
    assert available_providers({})["derived"] is False
