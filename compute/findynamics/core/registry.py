"""The two name→implementation registries (``03-contracts.md`` §3).

Engines and providers are both discovered *by name*, never by import: the job
layer asks for ``'rates'`` and gets whatever registered under that key. That is
what lets a new engine ship without editing job code (§8) and what stops the
portfolio layer from reaching into an engine's internals.

``core`` may not import ``data`` or ``engines`` (§3 rule 1), so this module holds
only the tables and the lookup rules. Population happens on the way up:
:mod:`findynamics.data.providers.registry` registers the provider factories, and
each engine package registers itself with :func:`register_engine` at import time.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any, TypeVar

from findynamics.core.config import SeriesConfig
from findynamics.core.contracts.vocab import ASSETS
from findynamics.core.engine import AssetEngine


class RegistryError(LookupError):
    """Raised for an unknown, duplicate or malformed registration."""


# --------------------------------------------------------------------------
# Engines
# --------------------------------------------------------------------------

ENGINES: dict[str, type[AssetEngine]] = {}

E = TypeVar("E", bound=type[AssetEngine])


def register_engine(cls: E) -> E:
    """Class decorator: publish an engine under its ``name``.

    Duplicate registration is an error rather than an overwrite — two engines
    silently sharing a key would mean the daily run publishes one model's state
    under the other's name.
    """
    name = getattr(cls, "name", None)
    if not isinstance(name, str) or not name:
        raise RegistryError(f"{cls.__name__} must declare a non-empty class attribute 'name'")
    if name not in ASSETS:
        raise RegistryError(f"{cls.__name__}.name={name!r} is not one of {list(ASSETS)}")
    version = getattr(cls, "version", None)
    if not isinstance(version, str) or not version:
        raise RegistryError(f"{cls.__name__} must declare a non-empty class attribute 'version'")
    existing = ENGINES.get(name)
    if existing is not None and existing is not cls:
        raise RegistryError(
            f"engine {name!r} is already registered by {existing.__module__}.{existing.__name__}"
        )
    ENGINES[name] = cls
    return cls


def engine_class(name: str) -> type[AssetEngine]:
    """The registered class for ``name``."""
    try:
        return ENGINES[name]
    except KeyError:
        raise RegistryError(f"unknown engine {name!r}; registered: {sorted(ENGINES)}") from None


def get_engine(name: str) -> AssetEngine:
    """Instantiate the engine registered under ``name``."""
    return engine_class(name)()


def registered_engines() -> tuple[str, ...]:
    """Registered engine names, in ASSETS order."""
    return tuple(name for name in ASSETS if name in ENGINES)


def enabled_engines(config: SeriesConfig) -> list[AssetEngine]:
    """Instances of every engine that is both registered and enabled in config.

    An engine enabled in config but not registered is a configuration error, not
    a silent no-op: it means the operator asked for a model that the deployed
    code cannot produce.
    """
    engines: list[AssetEngine] = []
    for name in config.enabled_engine_names():
        if name not in ENGINES:
            raise RegistryError(
                f"engine {name!r} is enabled in config but no implementation is registered"
            )
        engines.append(get_engine(name))
    return engines


def iter_engines() -> Iterator[tuple[str, type[AssetEngine]]]:
    for name in registered_engines():
        yield name, ENGINES[name]


# --------------------------------------------------------------------------
# Providers
# --------------------------------------------------------------------------

#: A provider factory receives the already-protected transport (and an api_key
#: where the source needs one) and returns a ``data.providers.base.Provider``.
#: Typed loosely on purpose: ``core`` cannot name the ``data`` types.
ProviderFactory = Callable[..., Any]

PROVIDERS: dict[str, ProviderFactory] = {}


def register_provider(name: str, factory: ProviderFactory) -> ProviderFactory:
    """Publish a provider factory under ``name``."""
    if not name:
        raise RegistryError("provider name must be non-empty")
    existing = PROVIDERS.get(name)
    if existing is not None and existing is not factory:
        raise RegistryError(f"provider {name!r} is already registered")
    PROVIDERS[name] = factory
    return factory


def provider_factory(name: str) -> ProviderFactory:
    """The registered factory for ``name``."""
    try:
        return PROVIDERS[name]
    except KeyError:
        raise RegistryError(f"unknown provider {name!r}; registered: {sorted(PROVIDERS)}") from None


def registered_providers() -> tuple[str, ...]:
    return tuple(sorted(PROVIDERS))
