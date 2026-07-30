"""Provider protection layer.

    Provider → RateLimiter → CircuitBreaker → Retry → live call
                     ↘ Cache ↙

A fresh cache hit short-circuits everything. On a live call the rate limiter
paces requests, the breaker refuses to hammer a source that is already failing,
and retry handles the transient middle ground. When every attempt fails the
cache is consulted a second time — this time accepting *stale* entries — so a
provider outage degrades the data's freshness rather than the whole run
(FINDYN_V1_SPEC.md §14.2).

Time and sleep are injected so the tests exercise real backoff and breaker
transitions without spending wall-clock seconds on them.
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol

from findynamics.data.providers.base import ProviderError, RateLimitError

log = logging.getLogger("findynamics.data.providers.resilience")

Clock = Callable[[], float]
Sleeper = Callable[[float], None]


@dataclass(frozen=True)
class Response:
    status_code: int
    content: bytes
    headers: Mapping[str, str] = field(default_factory=dict)

    @property
    def text(self) -> str:
        return self.content.decode("utf-8", errors="replace")

    def json(self) -> Any:
        return json.loads(self.text)


class Fetcher(Protocol):
    """Minimal HTTP surface, so tests can substitute a callable for the network."""

    def __call__(
        self,
        url: str,
        *,
        params: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
        timeout: float = 30.0,
    ) -> Response: ...


# --------------------------------------------------------------------------
# Rate limiting
# --------------------------------------------------------------------------


class RateLimiter:
    """Token bucket with a floor on the interval between calls.

    Two knobs because providers cap two different things: Alpha Vantage limits
    daily volume, BLS limits burst rate. ``capacity``/``refill_per_second``
    handle volume; ``min_interval`` handles burst.
    """

    def __init__(
        self,
        *,
        capacity: int = 10,
        refill_per_second: float = 1.0,
        min_interval: float = 0.0,
        clock: Clock = time.monotonic,
        sleeper: Sleeper = time.sleep,
    ) -> None:
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        self.capacity = capacity
        self.refill_per_second = refill_per_second
        self.min_interval = min_interval
        self._clock = clock
        self._sleep = sleeper
        self._tokens = float(capacity)
        self._last_refill = clock()
        self._last_call: float | None = None

    def _refill(self) -> None:
        now = self._clock()
        elapsed = max(0.0, now - self._last_refill)
        self._tokens = min(self.capacity, self._tokens + elapsed * self.refill_per_second)
        self._last_refill = now

    def acquire(self) -> float:
        """Block until a call is permitted. Returns seconds waited."""
        waited = 0.0

        if self._last_call is not None and self.min_interval > 0:
            gap = self._clock() - self._last_call
            if gap < self.min_interval:
                delay = self.min_interval - gap
                self._sleep(delay)
                waited += delay

        self._refill()
        if self._tokens < 1.0 and self.refill_per_second > 0:
            delay = (1.0 - self._tokens) / self.refill_per_second
            self._sleep(delay)
            waited += delay
            self._refill()

        self._tokens = max(0.0, self._tokens - 1.0)
        self._last_call = self._clock()
        return waited


# --------------------------------------------------------------------------
# Circuit breaker
# --------------------------------------------------------------------------


class CircuitState(StrEnum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitOpenError(ProviderError):
    """The breaker refused the call outright."""

    def __init__(self, provider: str, retry_in: float) -> None:
        super().__init__(provider, f"circuit open, retrying in {retry_in:.1f}s", retryable=False)
        self.retry_in = retry_in


class CircuitBreaker:
    """Standard three-state breaker.

    After ``failure_threshold`` consecutive failures the circuit opens and calls
    are refused for ``cooldown``. The next call after that is allowed through as
    a probe (half-open); ``success_threshold`` consecutive probe successes close
    it again, and a single probe failure re-opens it.
    """

    def __init__(
        self,
        provider: str,
        *,
        failure_threshold: int = 5,
        cooldown: float = 60.0,
        success_threshold: int = 1,
        clock: Clock = time.monotonic,
    ) -> None:
        self.provider = provider
        self.failure_threshold = failure_threshold
        self.cooldown = cooldown
        self.success_threshold = success_threshold
        self._clock = clock
        self._state = CircuitState.CLOSED
        self._failures = 0
        self._successes = 0
        self._opened_at: float | None = None

    @property
    def state(self) -> CircuitState:
        # Lazily promote OPEN -> HALF_OPEN so callers see the real state without
        # needing a background timer.
        if (
            self._state is CircuitState.OPEN
            and self._opened_at is not None
            and self._clock() - self._opened_at >= self.cooldown
        ):
            self._state = CircuitState.HALF_OPEN
            self._successes = 0
        return self._state

    @property
    def failure_count(self) -> int:
        return self._failures

    def before_call(self) -> None:
        if self.state is CircuitState.OPEN:
            elapsed = self._clock() - (self._opened_at or self._clock())
            raise CircuitOpenError(self.provider, max(0.0, self.cooldown - elapsed))

    def record_success(self) -> None:
        if self.state is CircuitState.HALF_OPEN:
            self._successes += 1
            if self._successes >= self.success_threshold:
                self._reset()
        else:
            self._reset()

    def record_failure(self) -> None:
        if self.state is CircuitState.HALF_OPEN:
            self._trip()
            return
        self._failures += 1
        if self._failures >= self.failure_threshold:
            self._trip()

    def _trip(self) -> None:
        self._state = CircuitState.OPEN
        self._opened_at = self._clock()
        self._successes = 0
        log.warning("circuit opened for %s after %d failures", self.provider, self._failures)

    def _reset(self) -> None:
        self._state = CircuitState.CLOSED
        self._failures = 0
        self._successes = 0
        self._opened_at = None


# --------------------------------------------------------------------------
# Retry
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class RetryPolicy:
    """Exponential backoff with full jitter."""

    max_attempts: int = 3
    base_delay: float = 0.5
    max_delay: float = 30.0
    jitter: bool = True

    def delay_for(self, attempt: int, rng: random.Random) -> float:
        """Backoff before ``attempt`` (1-based); attempt 1 never waits."""
        if attempt <= 1:
            return 0.0
        ceiling = min(self.max_delay, self.base_delay * (2 ** (attempt - 2)))
        # Full jitter: sampling in [0, ceiling] avoids synchronised retry storms
        # when several series fail against the same provider at once.
        return rng.uniform(0.0, ceiling) if self.jitter else ceiling


# --------------------------------------------------------------------------
# Cache
# --------------------------------------------------------------------------


@dataclass
class CacheEntry:
    value: bytes
    stored_at: float
    ttl: float

    def is_fresh(self, now: float) -> bool:
        return (now - self.stored_at) < self.ttl


class Cache(Protocol):
    def get(self, key: str, *, allow_stale: bool = False) -> CacheEntry | None: ...
    def set(self, key: str, value: bytes, ttl: float) -> None: ...


class MemoryCache:
    """Process-local cache. Default for tests and single-shot jobs."""

    def __init__(self, clock: Clock = time.monotonic) -> None:
        self._clock = clock
        self._entries: dict[str, CacheEntry] = {}

    def get(self, key: str, *, allow_stale: bool = False) -> CacheEntry | None:
        entry = self._entries.get(key)
        if entry is None:
            return None
        if entry.is_fresh(self._clock()) or allow_stale:
            return entry
        return None

    def set(self, key: str, value: bytes, ttl: float) -> None:
        self._entries[key] = CacheEntry(value=value, stored_at=self._clock(), ttl=ttl)


class FileCache:
    """Disk cache, so a rerun of a backfill does not re-download 1.6MB of Shiller."""

    def __init__(self, directory: Path, clock: Clock = time.time) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self._clock = clock

    def _paths(self, key: str) -> tuple[Path, Path]:
        digest = hashlib.sha256(key.encode()).hexdigest()[:32]
        return self.directory / f"{digest}.bin", self.directory / f"{digest}.json"

    def get(self, key: str, *, allow_stale: bool = False) -> CacheEntry | None:
        blob, meta = self._paths(key)
        if not blob.exists() or not meta.exists():
            return None
        try:
            info = json.loads(meta.read_text())
            entry = CacheEntry(
                value=blob.read_bytes(), stored_at=float(info["stored_at"]), ttl=float(info["ttl"])
            )
        except (OSError, ValueError, KeyError):
            return None
        if entry.is_fresh(self._clock()) or allow_stale:
            return entry
        return None

    def set(self, key: str, value: bytes, ttl: float) -> None:
        blob, meta = self._paths(key)
        blob.write_bytes(value)
        meta.write_text(json.dumps({"stored_at": self._clock(), "ttl": ttl, "key": key}))


# --------------------------------------------------------------------------
# Transport
# --------------------------------------------------------------------------


@dataclass
class TransportStats:
    calls: int = 0
    cache_hits: int = 0
    stale_fallbacks: int = 0
    retries: int = 0
    failures: int = 0


class Transport:
    """Composes the protection layers around one provider's HTTP calls."""

    def __init__(
        self,
        provider: str,
        fetcher: Fetcher,
        *,
        rate_limiter: RateLimiter | None = None,
        breaker: CircuitBreaker | None = None,
        retry: RetryPolicy | None = None,
        cache: Cache | None = None,
        default_ttl: float = 3600.0,
        clock: Clock = time.monotonic,
        sleeper: Sleeper = time.sleep,
        rng: random.Random | None = None,
    ) -> None:
        self.provider = provider
        self.fetcher = fetcher
        self.rate_limiter = rate_limiter or RateLimiter(clock=clock, sleeper=sleeper)
        self.breaker = breaker or CircuitBreaker(provider, clock=clock)
        self.retry = retry or RetryPolicy()
        self.cache = cache if cache is not None else MemoryCache(clock=clock)
        self.default_ttl = default_ttl
        self.stats = TransportStats()
        self._clock = clock
        self._sleep = sleeper
        self._rng = rng or random.Random(0)

    def get(
        self,
        url: str,
        *,
        params: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
        cache_ttl: float | None = None,
        timeout: float = 30.0,
    ) -> Response:
        key = self._cache_key(url, params)
        ttl = self.default_ttl if cache_ttl is None else cache_ttl

        if ttl > 0:
            hit = self.cache.get(key)
            if hit is not None:
                self.stats.cache_hits += 1
                return Response(
                    status_code=200, content=hit.value, headers={"x-findyn-cache": "hit"}
                )

        try:
            response = self._call_with_retry(url, params, headers, timeout)
        except ProviderError as err:
            # Last resort: serve whatever we have rather than failing the run.
            stale = self.cache.get(key, allow_stale=True)
            if stale is not None:
                self.stats.stale_fallbacks += 1
                log.warning("%s: serving stale cache for %s (%s)", self.provider, url, err)
                return Response(
                    status_code=200, content=stale.value, headers={"x-findyn-cache": "stale"}
                )
            self.stats.failures += 1
            raise

        if ttl > 0:
            self.cache.set(key, response.content, ttl)
        return response

    def _call_with_retry(
        self,
        url: str,
        params: Mapping[str, Any] | None,
        headers: Mapping[str, str] | None,
        timeout: float,
    ) -> Response:
        last: ProviderError | None = None

        for attempt in range(1, self.retry.max_attempts + 1):
            delay = self.retry.delay_for(attempt, self._rng)
            if delay > 0:
                self.stats.retries += 1
                self._sleep(delay)

            # Checked inside the loop: a long backoff can be enough for an
            # earlier failure to trip the breaker from another series' calls.
            self.breaker.before_call()
            self.rate_limiter.acquire()
            self.stats.calls += 1

            try:
                response = self.fetcher(url, params=params, headers=headers, timeout=timeout)
            except ProviderError as err:
                last = err
                self.breaker.record_failure()
                if not err.retryable:
                    raise
            except Exception as err:  # network/socket errors surface as generic exceptions
                last = ProviderError(self.provider, f"transport error: {err}", retryable=True)
                self.breaker.record_failure()
            else:
                error = self._classify(response)
                if error is None:
                    self.breaker.record_success()
                    return response
                last = error
                self.breaker.record_failure()
                if not error.retryable:
                    raise error

        assert last is not None
        raise last

    def _classify(self, response: Response) -> ProviderError | None:
        status = response.status_code
        if 200 <= status < 300:
            return None
        if status == 429:
            retry_after = response.headers.get("retry-after")
            return RateLimitError(
                self.provider,
                "rate limited (HTTP 429)",
                retry_after=float(retry_after) if retry_after and retry_after.isdigit() else None,
            )
        if status in (401, 403):
            from findynamics.data.providers.base import AuthError

            return AuthError(self.provider, f"rejected with HTTP {status}")
        if status == 404:
            from findynamics.data.providers.base import NotFoundError

            return NotFoundError(self.provider, "HTTP 404")
        if 500 <= status < 600:
            return ProviderError(self.provider, f"upstream error HTTP {status}", retryable=True)
        return ProviderError(self.provider, f"unexpected HTTP {status}", retryable=False)

    def _cache_key(self, url: str, params: Mapping[str, Any] | None) -> str:
        if not params:
            return url
        ordered = "&".join(f"{k}={params[k]}" for k in sorted(params))
        return f"{url}?{ordered}"


def httpx_fetcher(
    user_agent: str = "FinDyn/1.0 (+https://github.com/alexmorerich/findyn)",
) -> Fetcher:
    """Real network fetcher. Imported lazily so tests need no httpx at import time."""

    import httpx

    def fetch(
        url: str,
        *,
        params: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
        timeout: float = 30.0,
    ) -> Response:
        merged = {"user-agent": user_agent, **(headers or {})}
        response = httpx.get(
            url, params=params, headers=merged, timeout=timeout, follow_redirects=True
        )
        return Response(
            status_code=response.status_code,
            content=response.content,
            headers={k.lower(): v for k, v in response.headers.items()},
        )

    return fetch
