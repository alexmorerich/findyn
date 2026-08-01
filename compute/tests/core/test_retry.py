"""Retry policy and the transient/permanent distinction.

The distinction is the whole value of this module. Retrying a dropped socket
saves a night's charts; retrying a rejected signature turns one clear failure
into five slow ones and still fails. Both directions are pinned here.

Nothing sleeps: ``retry_call`` takes its sleep function, so the backoff schedule
is asserted by inspection rather than by waiting for it.
"""

from __future__ import annotations

import random

import httpx
import pytest

from findynamics.core.retry import (
    NETWORK_POLICY,
    RetryPolicy,
    is_transient_http,
    retry_call,
)


class Recorder:
    """Records requested delays instead of sleeping them."""

    def __init__(self) -> None:
        self.delays: list[float] = []

    def __call__(self, seconds: float) -> None:
        self.delays.append(seconds)


def always_retry(_: BaseException) -> bool:
    return True


# --------------------------------------------------------------------------
# The policy
# --------------------------------------------------------------------------


def test_the_first_attempt_never_waits():
    policy = RetryPolicy()
    assert policy.delay_for(1, random.Random(0)) == 0.0


def test_backoff_grows_and_is_capped():
    policy = RetryPolicy(base_delay=1.0, max_delay=8.0, jitter=False)
    rng = random.Random(0)
    delays = [policy.delay_for(n, rng) for n in range(2, 8)]
    assert delays == [1.0, 2.0, 4.0, 8.0, 8.0, 8.0]


def test_full_jitter_stays_inside_the_ceiling():
    """Jitter samples [0, ceiling] — it must never exceed the ceiling."""
    policy = RetryPolicy(base_delay=1.0, max_delay=8.0, jitter=True)
    rng = random.Random(7)
    for attempt in range(2, 10):
        ceiling = min(8.0, 1.0 * (2 ** (attempt - 2)))
        assert 0.0 <= policy.delay_for(attempt, rng) <= ceiling


def test_the_policy_is_one_class_across_the_layers():
    """``data`` re-exports ``core``'s; two backoffs would be free to drift."""
    from findynamics.data.providers.resilience import RetryPolicy as Reexported

    assert Reexported is RetryPolicy


# --------------------------------------------------------------------------
# retry_call
# --------------------------------------------------------------------------


def test_a_call_that_works_is_not_retried():
    calls = []
    sleeper = Recorder()

    result = retry_call(
        lambda: calls.append(1) or "ok",
        retry_on=always_retry,
        description="probe",
        sleep=sleeper,
    )
    assert result == "ok"
    assert len(calls) == 1
    assert sleeper.delays == []


def test_a_transient_failure_is_retried_until_it_succeeds():
    attempts = {"n": 0}
    sleeper = Recorder()

    def flaky() -> str:
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise httpx.ConnectError("half-open socket")
        return "recovered"

    result = retry_call(
        flaky,
        policy=RetryPolicy(max_attempts=5, base_delay=1.0, jitter=False),
        retry_on=is_transient_http,
        description="probe",
        sleep=sleeper,
    )
    assert result == "recovered"
    assert attempts["n"] == 3
    assert sleeper.delays == [1.0, 2.0]


def test_the_last_failure_is_raised_when_attempts_run_out():
    attempts = {"n": 0}
    sleeper = Recorder()

    def always_fails() -> None:
        attempts["n"] += 1
        raise httpx.ConnectError("still dead")

    with pytest.raises(httpx.ConnectError, match="still dead"):
        retry_call(
            always_fails,
            policy=RetryPolicy(max_attempts=3, base_delay=1.0, jitter=False),
            retry_on=is_transient_http,
            description="probe",
            sleep=sleeper,
        )
    # Three attempts, two waits between them — never a wait after the last.
    assert attempts["n"] == 3
    assert sleeper.delays == [1.0, 2.0]


def test_a_zero_delay_does_not_call_sleep_at_all():
    """A configured base_delay of 0 means "no backoff", not "sleep(0) twice"."""
    sleeper = Recorder()

    with pytest.raises(httpx.ConnectError):
        retry_call(
            lambda: (_ for _ in ()).throw(httpx.ConnectError("dead")),
            policy=RetryPolicy(max_attempts=3, base_delay=0.0, jitter=False),
            retry_on=is_transient_http,
            description="probe",
            sleep=sleeper,
        )
    assert sleeper.delays == []


def test_a_permanent_failure_is_raised_immediately_without_sleeping():
    """The expensive mistake this predicate exists to prevent."""
    attempts = {"n": 0}
    sleeper = Recorder()

    def rejected() -> None:
        attempts["n"] += 1
        raise httpx.HTTPStatusError(
            "unauthorized",
            request=httpx.Request("POST", "https://example.test/admin/v1/results"),
            response=httpx.Response(401),
        )

    with pytest.raises(httpx.HTTPStatusError):
        retry_call(rejected, retry_on=is_transient_http, description="probe", sleep=sleeper)

    assert attempts["n"] == 1, "a rejected signature was retried"
    assert sleeper.delays == []


# --------------------------------------------------------------------------
# What counts as transient
# --------------------------------------------------------------------------


def response_error(status: int) -> httpx.HTTPStatusError:
    return httpx.HTTPStatusError(
        f"status {status}",
        request=httpx.Request("GET", "https://example.test/"),
        response=httpx.Response(status),
    )


@pytest.mark.parametrize(
    "err",
    [
        httpx.ConnectError("refused"),
        httpx.ReadTimeout("slow"),
        httpx.RemoteProtocolError("EOF in violation of protocol"),
        response_error(500),
        response_error(503),
        # A 4xx that literally means "try again later".
        response_error(429),
    ],
)
def test_transient_failures_are_retried(err):
    assert is_transient_http(err)


@pytest.mark.parametrize(
    "err",
    [
        response_error(400),
        response_error(401),
        response_error(404),
        # The artifact conflict: this version was fitted on this date with
        # different bytes, and waiting does not change that.
        response_error(409),
        response_error(422),
        ValueError("not an HTTP failure at all"),
    ],
)
def test_permanent_failures_are_not_retried(err):
    assert not is_transient_http(err)


def test_the_network_policy_is_more_patient_than_the_provider_default():
    """These run once a night against our own Worker; there is no quota to spare."""
    assert NETWORK_POLICY.max_attempts > RetryPolicy().max_attempts
