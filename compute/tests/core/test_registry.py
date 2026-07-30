"""Engine and provider registry contracts (03-contracts.md §3).

The registry is how the daily job finds models without importing them. Its two
failure modes are both silent in production — one engine overwriting another's
key, or a configured engine that no deployed code implements — so both are
pinned here.
"""

from __future__ import annotations

from datetime import date

import pytest

from findynamics.core import registry
from findynamics.core.config import EngineConfig, SeriesConfig
from findynamics.core.contracts.state import AssetState, WorldState
from findynamics.core.engine import AssetEngine
from findynamics.core.registry import (
    ENGINES,
    RegistryError,
    enabled_engines,
    engine_class,
    get_engine,
    provider_factory,
    register_engine,
    register_provider,
    registered_engines,
    registered_providers,
)


@pytest.fixture(autouse=True)
def isolated_registries():
    """Registries are module-level; empty them for the test and restore after.

    Emptying matters as much as restoring: real engines register themselves at
    import time, so a test asserting on registry contents would otherwise depend
    on which other test module happened to run first.
    """
    engines = dict(registry.ENGINES)
    providers = dict(registry.PROVIDERS)
    registry.ENGINES.clear()
    yield
    registry.ENGINES.clear()
    registry.ENGINES.update(engines)
    registry.PROVIDERS.clear()
    registry.PROVIDERS.update(providers)


@pytest.fixture
def pit_accessor():
    import pandas as pd

    from findynamics.data.accessor import PandasPITAccessor

    frame = pd.DataFrame(
        {
            "series_id": ["FRED:DGS10"],
            "obs_date": ["2026-07-01"],
            "release_date": ["2026-07-02"],
            "value": [4.25],
        }
    )
    return PandasPITAccessor(frame, date(2026, 7, 29))


def make_engine(
    name: str, version: str = "0.1.0", *, experimental: bool = False
) -> type[AssetEngine]:
    """A minimal concrete engine — the smallest thing the ABC accepts."""

    def predict(self, world: WorldState) -> AssetState:
        return AssetState(
            asset=name,
            as_of=world.as_of,
            regime="flat",
            expected_return=None,
            risk_score=0.0,
            confidence=0.5,
            signals=(),
            model_version=version,
        )

    return type(
        f"{name.capitalize()}Engine",
        (AssetEngine,),
        {
            "name": name,
            "version": version,
            "experimental": experimental,
            "required_series": lambda self: (),
            "fit": lambda self, world: None,
            "predict": predict,
        },
    )


def config_with(**enabled: bool) -> SeriesConfig:
    return SeriesConfig(
        spec_version="1.0",
        info_set="t-1",
        history_start="1871-01-01",
        factors={},
        engines={name: EngineConfig(name=name, enabled=flag) for name, flag in enabled.items()},
        realtime_cache_ttl_seconds=900,
        realtime_cache_series=(),
    )


# --------------------------------------------------------------------------
# Engines
# --------------------------------------------------------------------------


def test_register_and_look_up_an_engine():
    cls = register_engine(make_engine("rates"))
    assert ENGINES["rates"] is cls
    assert engine_class("rates") is cls
    assert isinstance(get_engine("rates"), AssetEngine)
    assert registered_engines() == ("rates",)


def test_registered_engines_come_back_in_asset_order():
    for name in ("gold", "money", "rates"):
        register_engine(make_engine(name))
    # ASSETS order, not registration order — a run must not depend on import order.
    assert registered_engines() == ("money", "rates", "gold")


def test_duplicate_registration_is_rejected():
    register_engine(make_engine("rates"))
    with pytest.raises(RegistryError, match="already registered"):
        register_engine(make_engine("rates"))


def test_re_registering_the_same_class_is_idempotent():
    """Module re-import must not explode; only a *different* class is a clash."""
    cls = make_engine("rates")
    register_engine(cls)
    assert register_engine(cls) is cls


def test_engine_outside_the_asset_vocabulary_is_rejected():
    with pytest.raises(RegistryError, match="not one of"):
        register_engine(make_engine("commodities"))


def test_engine_without_a_name_is_rejected():
    class Nameless(AssetEngine):
        version = "0.1.0"

    with pytest.raises(RegistryError, match="'name'"):
        register_engine(Nameless)


def test_engine_without_a_version_is_rejected():
    class Unversioned(AssetEngine):
        name = "gold"

    with pytest.raises(RegistryError, match="'version'"):
        register_engine(Unversioned)


def test_unknown_engine_lookup_names_what_is_registered():
    register_engine(make_engine("rates"))
    with pytest.raises(RegistryError, match=r"unknown engine 'gold'.*\['rates'\]"):
        engine_class("gold")


def test_enabled_engines_returns_only_the_enabled_ones():
    register_engine(make_engine("rates"))
    register_engine(make_engine("gold"))
    engines = enabled_engines(config_with(rates=True, gold=False))
    assert [e.name for e in engines] == ["rates"]


def test_enabled_engines_are_instances_not_classes():
    register_engine(make_engine("rates"))
    (engine,) = enabled_engines(config_with(rates=True))
    assert isinstance(engine, AssetEngine)


def test_enabled_engines_follow_asset_order():
    for name in ("crypto", "money", "rates"):
        register_engine(make_engine(name))
    engines = enabled_engines(config_with(crypto=True, money=True, rates=True))
    assert [e.name for e in engines] == ["money", "rates", "crypto"]


def test_engine_enabled_without_an_implementation_is_an_error():
    """Silently skipping it would publish a stale state under a live asset."""
    with pytest.raises(RegistryError, match="enabled in config but no implementation"):
        enabled_engines(config_with(rates=True))


def test_load_engines_reports_what_config_enabled():
    """The job layer names no engine; it asks config and gets packages imported."""
    from findynamics.engines import load_engines

    # Idempotent by design — several jobs call it in one process.
    assert load_engines() == load_engines() == ("rates",)


def test_importing_an_engine_package_is_what_registers_it():
    """Discovery is by name; the import *is* the registration (03-contracts.md §3).

    Run in a fresh interpreter: registration happens at import time, and this
    process imported the engines long ago. Clearing the table and re-importing
    would prove nothing, because the module cache makes the second import a
    no-op.
    """
    import subprocess
    import sys

    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            "from findynamics.core.registry import ENGINES;"
            "assert ENGINES == {}, ENGINES;"
            "from findynamics.engines import load_engines;"
            "load_engines();"
            "print(','.join(sorted(ENGINES)))",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert probe.returncode == 0, probe.stderr
    # P1 ships rates. Anything else appearing here is scope creep.
    assert probe.stdout.strip() == "rates"


def test_a_registered_engine_can_run_end_to_end(pit_accessor):
    """The ABC is satisfiable: register → look up → predict a valid AssetState."""
    register_engine(make_engine("rates", "0.2.0"))
    world = WorldState(as_of=pit_accessor.as_of, factors={}, series=pit_accessor)
    state = get_engine("rates").predict(world)
    assert state.asset == "rates"
    assert state.model_version == "0.2.0"
    assert state.as_of == pit_accessor.as_of


# --------------------------------------------------------------------------
# Providers
# --------------------------------------------------------------------------


def test_the_data_layer_populated_the_provider_registry():
    """Discovery by name is the point; an empty table means nobody imported it."""
    import findynamics.data.providers  # noqa: F401  (import triggers registration)

    assert set(registered_providers()) == {"fred", "shiller", "stooq", "bls", "bea"}


def test_mock_is_deliberately_not_discoverable_by_name():
    """Synthetic data must stay reachable only through build_provider(allow_mock)."""
    import findynamics.data.providers  # noqa: F401

    assert "mock" not in registered_providers()
    with pytest.raises(RegistryError):
        provider_factory("mock")


def test_register_and_look_up_a_provider():
    def factory(transport, api_key=None):
        return ("built", transport, api_key)

    register_provider("nasdaq", factory)
    assert provider_factory("nasdaq") is factory


def test_duplicate_provider_registration_is_rejected():
    register_provider("nasdaq", lambda transport: None)
    with pytest.raises(RegistryError, match="already registered"):
        register_provider("nasdaq", lambda transport: None)


def test_unnamed_provider_is_rejected():
    with pytest.raises(RegistryError, match="non-empty"):
        register_provider("", lambda transport: None)


def test_unknown_provider_lookup_is_an_error():
    with pytest.raises(RegistryError, match="unknown provider"):
        provider_factory("bloomberg")
