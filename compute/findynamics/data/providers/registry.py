"""Provider construction.

Jobs ask for a provider by id and get one with the protection layer already
wired. Per-provider limiter settings live here rather than inside each adapter,
so the quota policy for the whole system is readable in one place.

The name→class table is published into :mod:`findynamics.core.registry` at
import time, so provider discovery is by name like engine discovery
(``03-contracts.md`` §3). The construction code stays here because ``core`` may
not import ``data`` (``01-target-architecture.md`` §3 rule 1).
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from findynamics.core.registry import RegistryError, provider_factory, register_provider
from findynamics.data.providers.base import Provider, ProviderError
from findynamics.data.providers.bea import BeaProvider
from findynamics.data.providers.bls import BlsProvider
from findynamics.data.providers.derived import DerivedProvider
from findynamics.data.providers.fred import FredProvider
from findynamics.data.providers.lbma import LbmaProvider
from findynamics.data.providers.mock import MockProvider
from findynamics.data.providers.published import PublishedOutputProvider, resolve_api_base
from findynamics.data.providers.resilience import (
    Cache,
    CircuitBreaker,
    Fetcher,
    FileCache,
    MemoryCache,
    RateLimiter,
    RetryPolicy,
    Transport,
    httpx_fetcher,
)
from findynamics.data.providers.shiller import ShillerProvider
from findynamics.data.providers.stooq import StooqProvider
from findynamics.data.providers.yahoo import YahooProvider


@dataclass(frozen=True)
class Quota:
    """Documented request policy for one provider."""

    capacity: int
    refill_per_second: float
    min_interval: float
    failure_threshold: int
    cooldown: float
    max_attempts: int
    note: str


#: Chosen from each provider's published limits. Where a provider documents no
#: limit, the settings are still conservative: these are free public services.
QUOTAS: Mapping[str, Quota] = {
    "fred": Quota(20, 2.0, 0.05, 5, 60.0, 3, "FRED allows 120 req/min"),
    "shiller": Quota(2, 0.1, 1.0, 3, 300.0, 3, "one static workbook, cached for a day"),
    "lbma": Quota(2, 0.1, 1.0, 3, 300.0, 3, "one static JSON per metal, cached for a day"),
    "stooq": Quota(5, 0.5, 1.0, 2, 600.0, 2, "bot-filtered; fail fast and fall back"),
    "bls": Quota(5, 0.05, 1.0, 4, 120.0, 3, "BLS v2 allows 500 req/day with a key"),
    "bea": Quota(10, 0.5, 0.5, 4, 120.0, 3, "BEA allows 100 req/min, 100MB/day"),
    "engine_output": Quota(20, 5.0, 0.0, 3, 30.0, 2, "our own Worker; no published quota"),
    # Same target as engine_output — our own Worker — but one derived series
    # fans out into several input reads, so the burst allowance is larger.
    "derived": Quota(30, 8.0, 0.0, 3, 30.0, 2, "our own Worker; several reads per series"),
    # Undocumented and IP-throttled: a 429 is routine rather than exceptional,
    # so this is the most conservative pacing of any source here.
    "yahoo": Quota(2, 0.2, 2.0, 3, 300.0, 3, "unofficial; throttles hard by source IP"),
    "mock": Quota(1000, 1000.0, 0.0, 100, 1.0, 1, "no network"),
}

_DEFAULT_QUOTA = Quota(10, 1.0, 0.1, 5, 60.0, 3, "default")

KEY_ENV = {
    "fred": "FRED_API_KEY",
    "bls": "BLS_API_KEY",
    "bea": "BEA_API_KEY",
}

#: Providers that reach the network and therefore need a transport.
NETWORK_PROVIDERS = frozenset(
    {"fred", "shiller", "stooq", "bls", "bea", "engine_output", "yahoo", "derived", "lbma"}
)

#: Providers usable with no credential at all. ``engine_output`` needs no key but
#: does need to be told where the serving plane is; when it is not, it reports no
#: observations and the consuming engine degrades (see its module docstring).
KEYLESS_PROVIDERS = frozenset({"shiller", "stooq", "engine_output", "yahoo", "derived", "lbma"})

#: The network adapters, published by name. ``mock`` is deliberately absent:
#: reaching synthetic data must stay a deliberate act through ``build_provider``
#: rather than something a generic registry lookup can hand out.
for _name, _cls in (
    ("fred", FredProvider),
    ("shiller", ShillerProvider),
    ("stooq", StooqProvider),
    ("bls", BlsProvider),
    ("bea", BeaProvider),
    ("engine_output", PublishedOutputProvider),
    ("derived", DerivedProvider),
    ("yahoo", YahooProvider),
    ("lbma", LbmaProvider),
):
    register_provider(_name, _cls)


class UnknownProviderError(ProviderError):
    def __init__(self, provider_id: str) -> None:
        super().__init__("registry", f"unknown provider {provider_id!r}", retryable=False)


def build_transport(
    provider_id: str,
    *,
    fetcher: Fetcher | None = None,
    cache: Cache | None = None,
    cache_dir: Path | None = None,
) -> Transport:
    quota = QUOTAS.get(provider_id, _DEFAULT_QUOTA)
    if cache is None:
        cache = FileCache(cache_dir) if cache_dir is not None else MemoryCache()
    return Transport(
        provider_id,
        fetcher or httpx_fetcher(),
        rate_limiter=RateLimiter(
            capacity=quota.capacity,
            refill_per_second=quota.refill_per_second,
            min_interval=quota.min_interval,
        ),
        breaker=CircuitBreaker(
            provider_id,
            failure_threshold=quota.failure_threshold,
            cooldown=quota.cooldown,
        ),
        retry=RetryPolicy(max_attempts=quota.max_attempts),
        cache=cache,
    )


def build_provider(
    provider_id: str,
    *,
    fetcher: Fetcher | None = None,
    cache: Cache | None = None,
    cache_dir: Path | None = None,
    env: Mapping[str, str] | None = None,
    allow_mock: bool = False,
) -> Provider:
    """Construct a provider with its protection layer.

    ``allow_mock`` is required for the synthetic provider. Fabricated numbers are
    indistinguishable from measurements once they are in the database, so
    reaching them has to be a deliberate act rather than a default.
    """
    environment = env if env is not None else os.environ

    if provider_id == "mock":
        if not allow_mock:
            raise ProviderError(
                "registry",
                "the mock provider produces synthetic data and must be requested "
                "explicitly with allow_mock=True",
                retryable=False,
            )
        return MockProvider()

    if provider_id not in NETWORK_PROVIDERS:
        raise UnknownProviderError(provider_id)

    transport = build_transport(provider_id, fetcher=fetcher, cache=cache, cache_dir=cache_dir)

    try:
        factory = provider_factory(provider_id)
    except RegistryError:
        raise UnknownProviderError(provider_id) from None

    if provider_id in KEY_ENV:
        return factory(transport, api_key=environment.get(KEY_ENV[provider_id]))
    # Published outputs and derived series are configured by location rather
    # than by credential: both read the serving plane rather than a vendor.
    if provider_id in {"engine_output", "derived"}:
        return factory(transport, base_url=resolve_api_base(environment))
    return factory(transport)


def available_providers(env: Mapping[str, str] | None = None) -> dict[str, bool]:
    """Map provider id -> whether it can currently return real data."""
    environment = env if env is not None else os.environ
    status: dict[str, bool] = {}
    for provider_id in sorted(NETWORK_PROVIDERS):
        if provider_id in {"engine_output", "derived"}:
            # Keyless, but useless without somewhere to read from.
            status[provider_id] = resolve_api_base(environment) is not None
        elif provider_id in KEYLESS_PROVIDERS:
            status[provider_id] = True
        else:
            status[provider_id] = bool(environment.get(KEY_ENV.get(provider_id, ""), ""))
    return status
