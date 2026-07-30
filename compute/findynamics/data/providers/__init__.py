"""Provider adapters.

Every source is reduced to the canonical model in :mod:`findynamics.data.providers.base`
before it reaches a compute job, and every network call goes through the
protection layer in :mod:`findynamics.data.providers.resilience`.
"""

from findynamics.data.providers.base import (
    FetchResult,
    Observation,
    Provider,
    ProviderError,
    SeriesMetadata,
    synthesize_release_date,
)
from findynamics.data.providers.published import PublishedOutputProvider, resolve_api_base
from findynamics.data.providers.registry import (
    KEYLESS_PROVIDERS,
    NETWORK_PROVIDERS,
    QUOTAS,
    available_providers,
    build_provider,
    build_transport,
)

__all__ = [
    "KEYLESS_PROVIDERS",
    "NETWORK_PROVIDERS",
    "QUOTAS",
    "FetchResult",
    "Observation",
    "Provider",
    "ProviderError",
    "PublishedOutputProvider",
    "SeriesMetadata",
    "available_providers",
    "build_provider",
    "build_transport",
    "resolve_api_base",
    "synthesize_release_date",
]
