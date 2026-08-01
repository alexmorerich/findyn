"""Retrying a transient failure, and knowing which failures those are.

Three places in this system talk to a network they do not control: the provider
transport, the artifact store, and the write-back. The transport has had backoff
since M0; the other two did not, and P4 is what made that expensive. A run
publishing FinGold dropped TLS connections roughly nine requests into a
seventeen-chunk write-back, twice, and on a third run the artifact GET failed
once — which cost FinEquity its fitted model for that run and published a state
that said, correctly and uselessly, that no regime model was stored.

Both failures were a half-open socket and nothing more. Neither deserved to
reach a model.

**Why the policy lives in ``core``.** :class:`RetryPolicy` was originally in
``data.providers.resilience``, which is the right home for a *provider*
protection layer and the wrong one for a rule three layers need — ``core`` may
not import ``data`` (``01-target-architecture.md`` §3 rule 1), so the artifact
store could not have reached it. Rather than write a second exponential backoff
with slightly different constants and let the two drift, the policy moved down
here and ``resilience`` re-exports it. Every existing import still resolves.

**What is retryable is a decision, not a default.** Retrying the wrong failure is
worse than not retrying: a rejected HMAC signature, a validation error, or an
artifact conflict will fail identically every time, and retrying burns the
Worker's budget to arrive at the same answer more slowly. So the caller supplies
the predicate, and the two callers here supply narrow ones.
"""

from __future__ import annotations

import logging
import random
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeVar

log = logging.getLogger("findynamics.core.retry")

T = TypeVar("T")


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


#: The write-back and artifact policies. Longer and more patient than the
#: provider default, because these run once a night against our own Worker: a
#: daily run that gives up on a chunk leaves a visible hole in a chart until
#: tomorrow, and there is no quota to protect by failing fast.
NETWORK_POLICY = RetryPolicy(max_attempts=5, base_delay=1.0, max_delay=20.0)


def retry_call(
    operation: Callable[[], T],
    *,
    policy: RetryPolicy = NETWORK_POLICY,
    retry_on: Callable[[BaseException], bool],
    description: str,
    sleep: Callable[[float], None] = time.sleep,
    rng: random.Random | None = None,
) -> T:
    """Call ``operation``, retrying while ``retry_on`` says the failure is transient.

    Re-raises the last exception when the attempts run out, and re-raises
    immediately — without sleeping — for anything ``retry_on`` rejects. The
    distinction is the point: a 409 from the artifact store means the fit already
    exists with different bytes, and no amount of waiting changes that.
    """
    generator = rng or random.Random()
    last: BaseException | None = None

    for attempt in range(1, policy.max_attempts + 1):
        if attempt > 1:
            delay = policy.delay_for(attempt, generator)
            if delay:
                sleep(delay)
        try:
            return operation()
        except BaseException as err:  # noqa: BLE001 — re-raised below either way
            if not retry_on(err):
                raise
            last = err
            log.warning(
                "%s failed (%s: %s); attempt %d of %d",
                description,
                type(err).__name__,
                err,
                attempt,
                policy.max_attempts,
            )

    assert last is not None  # unreachable: the loop either returns or sets `last`
    log.error("%s failed after %d attempts; giving up", description, policy.max_attempts)
    raise last


def is_transient_http(err: BaseException) -> bool:
    """True for a failure that a second identical request might survive.

    Connection and timeout errors, and the server-side statuses. Explicitly
    **not** 4xx: a bad signature, a malformed payload and an artifact conflict
    are all deterministic, and retrying them turns one clear failure into five
    slow ones. 429 is the exception — it is a 4xx that literally means "try
    again later".
    """
    import httpx

    if isinstance(err, httpx.TimeoutException | httpx.NetworkError | httpx.RemoteProtocolError):
        return True
    if isinstance(err, httpx.HTTPStatusError):
        status = err.response.status_code
        return status == 429 or status >= 500
    return False


__all__ = [
    "NETWORK_POLICY",
    "RetryPolicy",
    "is_transient_http",
    "retry_call",
]
