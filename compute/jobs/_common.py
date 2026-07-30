"""Shared plumbing for compute jobs: argument parsing, logging, HMAC write-back."""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import logging
import os
import time
from collections.abc import Iterator
from typing import Any

import httpx

LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"


def configure_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format=LOG_FORMAT,
        datefmt="%Y-%m-%dT%H:%M:%S",
    )


def base_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--as-of",
        help="Information-set cutoff (YYYY-MM-DD). Defaults to today, UTC.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute but do not write back to the serving plane.",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser


def sign_payload(secret: str, body: str, timestamp: int | None = None) -> tuple[str, str]:
    """Return (timestamp, hex signature) over ``{timestamp}.{body}``.

    Must stay byte-identical to serving/src/admin/hmac.ts::verifyHmac.
    """
    ts = str(timestamp if timestamp is not None else int(time.time()))
    mac = hmac.new(secret.encode(), f"{ts}.{body}".encode(), hashlib.sha256)
    return ts, mac.hexdigest()


def write_back(payload: dict[str, Any], *, dry_run: bool = False) -> None:
    """POST results to the serving plane's HMAC-authenticated admin endpoint (§6)."""
    log = logging.getLogger("findynamics.writeback")
    body = json.dumps(payload, separators=(",", ":"), sort_keys=True)

    if dry_run:
        log.info("dry run: withholding %d byte payload", len(body))
        return

    endpoint = os.environ.get("FINDYN_ADMIN_URL")
    secret = os.environ.get("ADMIN_HMAC_SECRET")
    if not endpoint or not secret:
        raise RuntimeError("FINDYN_ADMIN_URL and ADMIN_HMAC_SECRET must be set")

    timestamp, signature = sign_payload(secret, body)
    response = httpx.post(
        endpoint,
        content=body,
        headers={
            "content-type": "application/json",
            "x-findyn-timestamp": timestamp,
            "x-findyn-signature": signature,
        },
        timeout=60.0,
    )
    response.raise_for_status()
    log.info("wrote back %d bytes -> %s", len(body), response.status_code)


def chunk_on(
    payload: dict[str, Any],
    key: str,
    batch_size: int,
    *,
    carry: tuple[str, ...] = ("model_version", "generated_at", "as_of"),
) -> Iterator[dict[str, Any]]:
    """Split ``payload`` into requests the serving plane can absorb, on one array.

    The Worker upserts in D1 batches inside a single request, so a payload large
    enough eventually runs out of CPU partway through and leaves a partial write
    with no record of where it stopped. Every chunk is independently idempotent —
    all these tables upsert on their primary key — so a failure costs one chunk
    and a re-run repairs it.

    The first chunk carries the whole payload; later chunks carry only ``carry``
    plus their slice. Everything else in the envelope (the states, the factor
    scores) is small and per-run rather than per-row, so repeating it in every
    chunk would just re-upsert the same handful of rows.

    Yields the payload unchanged when ``key`` is absent or already small enough,
    which keeps the single-request case exactly as it was.
    """
    rows = payload.get(key) or []
    if len(rows) <= batch_size:
        yield payload
        return

    head = {k: v for k, v in payload.items() if k != key}
    for start in range(0, len(rows), batch_size):
        chunk = dict(head) if start == 0 else {k: payload[k] for k in carry if k in payload}
        chunk[key] = rows[start : start + batch_size]
        yield chunk


def not_yet(milestone: str, section: str) -> int:
    """Exit path for a job whose milestone has not landed."""
    logging.getLogger("findynamics.jobs").error(
        "not implemented: delivered in milestone %s (FINDYN_V1_SPEC.md %s)", milestone, section
    )
    return 2
