"""Rate limiter, circuit breaker, retry, cache and their composition."""

from __future__ import annotations

import random

import pytest

from findynamics.data.providers.base import AuthError, ProviderError
from findynamics.data.providers.resilience import (
    CircuitBreaker,
    CircuitOpenError,
    CircuitState,
    MemoryCache,
    RateLimiter,
    RetryPolicy,
    Transport,
)
from tests.conftest import FakeFetcher, ok

# --------------------------------------------------------------------------
# Rate limiter
# --------------------------------------------------------------------------


def test_burst_up_to_capacity_costs_nothing(clock, sleeper):
    limiter = RateLimiter(capacity=3, refill_per_second=1.0, clock=clock, sleeper=sleeper)
    for _ in range(3):
        assert limiter.acquire() == 0.0
    assert sleeper.total == 0.0


def test_exhausted_bucket_waits_for_a_refill(clock, sleeper):
    limiter = RateLimiter(capacity=2, refill_per_second=2.0, clock=clock, sleeper=sleeper)
    limiter.acquire()
    limiter.acquire()
    waited = limiter.acquire()
    assert waited == pytest.approx(0.5, abs=1e-6)  # one token at 2/sec


def test_min_interval_paces_consecutive_calls(clock, sleeper):
    limiter = RateLimiter(
        capacity=100, refill_per_second=100.0, min_interval=1.5, clock=clock, sleeper=sleeper
    )
    limiter.acquire()
    assert limiter.acquire() == pytest.approx(1.5, abs=1e-6)


def test_tokens_refill_as_time_passes(clock, sleeper):
    limiter = RateLimiter(capacity=2, refill_per_second=1.0, clock=clock, sleeper=sleeper)
    limiter.acquire()
    limiter.acquire()
    clock.advance(5.0)
    assert limiter.acquire() == 0.0


def test_capacity_must_be_positive():
    with pytest.raises(ValueError):
        RateLimiter(capacity=0)


# --------------------------------------------------------------------------
# Circuit breaker
# --------------------------------------------------------------------------


def test_breaker_opens_after_threshold(clock):
    breaker = CircuitBreaker("x", failure_threshold=3, cooldown=60.0, clock=clock)
    for _ in range(2):
        breaker.record_failure()
    assert breaker.state is CircuitState.CLOSED
    breaker.record_failure()
    assert breaker.state is CircuitState.OPEN


def test_open_breaker_refuses_calls(clock):
    breaker = CircuitBreaker("x", failure_threshold=1, cooldown=60.0, clock=clock)
    breaker.record_failure()
    with pytest.raises(CircuitOpenError) as excinfo:
        breaker.before_call()
    assert excinfo.value.retry_in == pytest.approx(60.0)
    # Refusal is not itself retryable — backing off would just re-refuse.
    assert excinfo.value.retryable is False


def test_breaker_half_opens_after_cooldown(clock):
    breaker = CircuitBreaker("x", failure_threshold=1, cooldown=30.0, clock=clock)
    breaker.record_failure()
    clock.advance(29.0)
    assert breaker.state is CircuitState.OPEN
    clock.advance(2.0)
    assert breaker.state is CircuitState.HALF_OPEN
    breaker.before_call()  # probe permitted


def test_successful_probe_closes_the_breaker(clock):
    breaker = CircuitBreaker("x", failure_threshold=1, cooldown=10.0, clock=clock)
    breaker.record_failure()
    clock.advance(11.0)
    breaker.record_success()
    assert breaker.state is CircuitState.CLOSED
    assert breaker.failure_count == 0


def test_failed_probe_reopens_immediately(clock):
    breaker = CircuitBreaker("x", failure_threshold=5, cooldown=10.0, clock=clock)
    for _ in range(5):
        breaker.record_failure()
    clock.advance(11.0)
    assert breaker.state is CircuitState.HALF_OPEN
    breaker.record_failure()
    assert breaker.state is CircuitState.OPEN


def test_success_resets_the_failure_count(clock):
    breaker = CircuitBreaker("x", failure_threshold=3, clock=clock)
    breaker.record_failure()
    breaker.record_failure()
    breaker.record_success()
    breaker.record_failure()
    assert breaker.state is CircuitState.CLOSED


# --------------------------------------------------------------------------
# Retry policy
# --------------------------------------------------------------------------


def test_first_attempt_never_waits():
    assert RetryPolicy().delay_for(1, random.Random(0)) == 0.0


def test_backoff_ceiling_doubles_then_caps():
    policy = RetryPolicy(base_delay=1.0, max_delay=4.0, jitter=False)
    rng = random.Random(0)
    assert [policy.delay_for(n, rng) for n in (2, 3, 4, 5)] == [1.0, 2.0, 4.0, 4.0]


def test_jitter_stays_within_the_ceiling():
    policy = RetryPolicy(base_delay=1.0, max_delay=8.0, jitter=True)
    rng = random.Random(1234)
    for attempt in range(2, 8):
        ceiling = min(8.0, 1.0 * 2 ** (attempt - 2))
        for _ in range(50):
            assert 0.0 <= policy.delay_for(attempt, rng) <= ceiling


# --------------------------------------------------------------------------
# Cache
# --------------------------------------------------------------------------


def test_memory_cache_expires_but_keeps_the_value_for_stale_reads(clock):
    cache = MemoryCache(clock=clock)
    cache.set("k", b"v", ttl=10.0)
    assert cache.get("k") is not None
    clock.advance(11.0)
    assert cache.get("k") is None
    stale = cache.get("k", allow_stale=True)
    assert stale is not None and stale.value == b"v"


def test_file_cache_round_trips(tmp_path, clock):
    from findynamics.data.providers.resilience import FileCache

    cache = FileCache(tmp_path, clock=clock)
    cache.set("https://example.test/a", b"payload", ttl=60.0)
    entry = cache.get("https://example.test/a")
    assert entry is not None and entry.value == b"payload"
    assert cache.get("https://example.test/other") is None


# --------------------------------------------------------------------------
# Transport composition
# --------------------------------------------------------------------------


def _transport(fetcher, clock, sleeper, **kwargs) -> Transport:
    defaults = {
        "rate_limiter": RateLimiter(
            capacity=100, refill_per_second=100.0, clock=clock, sleeper=sleeper
        ),
        "breaker": CircuitBreaker("test", failure_threshold=3, cooldown=60.0, clock=clock),
        "retry": RetryPolicy(max_attempts=3, base_delay=1.0, jitter=False),
        "cache": MemoryCache(clock=clock),
        "clock": clock,
        "sleeper": sleeper,
    }
    return Transport("test", fetcher, **{**defaults, **kwargs})


def test_transport_returns_a_successful_response(clock, sleeper):
    transport = _transport(FakeFetcher(ok("hello")), clock, sleeper)
    assert transport.get("https://example.test/a").text == "hello"


def test_transport_serves_a_fresh_cache_hit_without_calling_out(clock, sleeper):
    fetcher = FakeFetcher(ok("first"))
    transport = _transport(fetcher, clock, sleeper)
    transport.get("https://example.test/a", cache_ttl=100)
    second = transport.get("https://example.test/a", cache_ttl=100)
    assert fetcher.call_count == 1
    assert second.headers["x-findyn-cache"] == "hit"


def test_transport_retries_a_server_error_then_succeeds(clock, sleeper):
    fetcher = FakeFetcher([ok("", 503), ok("", 503), ok("recovered")])
    transport = _transport(fetcher, clock, sleeper)
    assert transport.get("https://example.test/a").text == "recovered"
    assert fetcher.call_count == 3
    assert sleeper.delays == [1.0, 2.0]


def test_transport_does_not_retry_an_auth_failure(clock, sleeper):
    fetcher = FakeFetcher(ok("", 401))
    transport = _transport(fetcher, clock, sleeper)
    with pytest.raises(AuthError):
        transport.get("https://example.test/a")
    assert fetcher.call_count == 1


def test_transport_falls_back_to_stale_cache_when_the_source_dies(clock, sleeper):
    fetcher = FakeFetcher([ok("good"), ok("", 500)])
    transport = _transport(fetcher, clock, sleeper)

    transport.get("https://example.test/a", cache_ttl=10)
    clock.advance(50.0)  # cached copy is now stale

    response = transport.get("https://example.test/a", cache_ttl=10)
    assert response.text == "good"
    assert response.headers["x-findyn-cache"] == "stale"
    assert transport.stats.stale_fallbacks == 1


def test_transport_raises_when_there_is_no_cache_to_fall_back_to(clock, sleeper):
    transport = _transport(FakeFetcher(ok("", 500)), clock, sleeper)
    with pytest.raises(ProviderError):
        transport.get("https://example.test/a")
    assert transport.stats.failures == 1


def test_repeated_failures_open_the_circuit(clock, sleeper):
    fetcher = FakeFetcher(ok("", 500))
    transport = _transport(
        fetcher, clock, sleeper, retry=RetryPolicy(max_attempts=3, base_delay=0.0, jitter=False)
    )

    with pytest.raises(ProviderError):
        transport.get("https://example.test/a")
    assert transport.breaker.state is CircuitState.OPEN

    calls_before = fetcher.call_count
    with pytest.raises(ProviderError):
        transport.get("https://example.test/b")
    # The open circuit refuses without reaching the network.
    assert fetcher.call_count == calls_before


def test_transport_treats_transport_exceptions_as_retryable(clock, sleeper):
    fetcher = FakeFetcher([ConnectionError("reset by peer"), ok("recovered")])
    transport = _transport(fetcher, clock, sleeper)
    assert transport.get("https://example.test/a").text == "recovered"


def test_rate_limit_response_is_retried(clock, sleeper):
    fetcher = FakeFetcher([ok("", 429), ok("fine")])
    transport = _transport(fetcher, clock, sleeper)
    assert transport.get("https://example.test/a").text == "fine"


def test_cache_key_is_insensitive_to_parameter_order(clock, sleeper):
    fetcher = FakeFetcher(ok("v"))
    transport = _transport(fetcher, clock, sleeper)
    transport.get("https://example.test/a", params={"b": 2, "a": 1}, cache_ttl=100)
    transport.get("https://example.test/a", params={"a": 1, "b": 2}, cache_ttl=100)
    assert fetcher.call_count == 1
