"""Shared plumbing for compute jobs: argument parsing, logging, HMAC write-back."""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import logging
import os
import time
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


def not_yet(milestone: str, section: str) -> int:
    """Exit path for a job whose milestone has not landed."""
    logging.getLogger("findynamics.jobs").error(
        "not implemented: delivered in milestone %s (FINDYN_V1_SPEC.md %s)", milestone, section
    )
    return 2
