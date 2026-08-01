"""Shared plumbing for compute jobs: argument parsing, logging, HMAC write-back."""

from __future__ import annotations

import argparse
import json
import logging
import os
from collections.abc import Iterator
from typing import Any
from urllib.parse import quote

import httpx

from findynamics.core.signing import sign

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


#: Re-exported from core so the write-back and the artifact client cannot drift
#: apart. Must stay byte-identical to serving/src/admin/hmac.ts::verifyHmac.
sign_payload = sign


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


def archive_simulation(
    document: dict[str, Any],
    *,
    asset: str,
    as_of: str,
    dry_run: bool = False,
) -> None:
    """PUT one Monte Carlo run's per-path outcomes to R2 (§11).

    Never fatal. The archive is for offline analysis; a daily run that published
    its state and its quantiles has done its job, and losing tonight's path
    outcomes to a network blip is not worth failing the run and losing those
    too. The failure is logged at WARNING and the run continues — which is a
    deliberate asymmetry with `write_back`, where a failure means the published
    numbers are missing and the run really has failed.
    """
    log = logging.getLogger("findynamics.simulations")
    body = json.dumps(document, separators=(",", ":"), sort_keys=True)

    if dry_run:
        log.info("dry run: withholding %d byte simulation archive", len(body))
        return

    endpoint = os.environ.get("FINDYN_ADMIN_URL")
    secret = os.environ.get("ADMIN_HMAC_SECRET")
    if not endpoint or not secret:
        log.warning("no admin credentials; skipping the simulation archive")
        return

    base = endpoint.rstrip("/")
    if base.endswith("/results"):
        base = base[: -len("/results")]
    url = f"{base}/simulations/{quote(asset, safe='')}/{quote(as_of, safe='')}"

    timestamp, signature = sign_payload(secret, body)
    try:
        response = httpx.put(
            url,
            content=body,
            headers={
                "content-type": "application/json",
                "x-findyn-timestamp": timestamp,
                "x-findyn-signature": signature,
            },
            timeout=120.0,
        )
        response.raise_for_status()
    except httpx.HTTPError as err:
        log.warning("simulation archive failed (%s); the run continues", err)
        return
    log.info(
        "archived %d bytes of %s paths for %s -> %s", len(body), asset, as_of, response.status_code
    )


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
