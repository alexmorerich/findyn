"""HMAC signing for the compute → serving door.

One implementation, used by both the write-back and the artifact client, because
the signature has to be byte-identical to
``serving/src/admin/hmac.ts::verifyHmac`` and two copies of that would eventually
stop being identical in a way that only shows up as a 401 at 3am.

Lives in ``core`` rather than ``jobs`` because it is a wire-protocol fact, not a
job-runner detail: :mod:`findynamics.core.artifacts` needs it too, and ``core``
may not import from the app layer.
"""

from __future__ import annotations

import hashlib
import hmac
import time


def sign(secret: str, body: str, timestamp: int | None = None) -> tuple[str, str]:
    """Return ``(timestamp, hex signature)`` over ``{timestamp}.{body}``.

    An empty ``body`` is signed exactly like any other — a GET carries no
    payload, and the timestamp alone is still what makes the request fresh.
    """
    stamp = str(timestamp if timestamp is not None else int(time.time()))
    mac = hmac.new(secret.encode(), f"{stamp}.{body}".encode(), hashlib.sha256)
    return stamp, mac.hexdigest()


def headers(secret: str, body: str, timestamp: int | None = None) -> dict[str, str]:
    """Signature headers for one request."""
    stamp, signature = sign(secret, body, timestamp)
    return {"x-findyn-timestamp": stamp, "x-findyn-signature": signature}


__all__ = ["headers", "sign"]
