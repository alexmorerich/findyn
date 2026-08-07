"""Shared plumbing for compute jobs: argument parsing, logging, HMAC write-back."""

from __future__ import annotations

import argparse
import json
import logging
import os
from collections.abc import Iterator
from datetime import date, timedelta
from typing import Any
from urllib.parse import quote

import httpx

from findynamics.core.retry import is_transient_http, retry_call
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


def refit_cutoff(run_date: date) -> date:
    """The information set a refit fits on: the end of the last completed month.

    A refit's cutoff is **derived from the calendar, not from the clock**, and
    that is the whole point. Fitting "as of now" made the artifact a function of
    when the container happened to start, which is how two refits on 2026-08-01 —
    a manual dispatch at 01:13 and the scheduled run at 04:00, five and a half
    hours and several new observations apart — produced different parameters
    under the same ``model_version`` on the same calendar date, and the second
    one 409'd against the first (issue #6).

    Deriving the cutoff instead makes every run within a month see the same
    information set, so they produce byte-identical artifacts and land on the
    existing idempotent path in ``putArtifact`` rather than on the conflict. It
    also makes "the August fit" a well-defined object: a thing that can be
    reproduced later from the same inputs, rather than a thing that depended on a
    scheduler's timing.

    The cost is deliberate and worth stating plainly: the fit is a few days stale
    by construction. A refit run on 15 August ignores the first fortnight of
    August. For a model refitted monthly on an expanding window of decades, that
    is a rounding error against the reproducibility it buys — but it *is* a real
    choice, not a free one.

    Idempotency is exact only as far as the sources are. A provider that silently
    restates an old observation without a vintage — NFCI is the one this repo
    knows about — can still move the bytes between two runs of the same month.
    That is a data-fidelity problem rather than a scheduling one, and it now fails
    loudly as a conflict instead of being absorbed by a moving cutoff.
    """
    return run_date.replace(day=1) - timedelta(days=1)


def write_back(payload: dict[str, Any], *, dry_run: bool = False) -> None:
    """POST results to the serving plane's HMAC-authenticated admin endpoint (§6).

    Retries transient failures. A daily run splits into a dozen or more signed
    requests, and a connection dropped on the seventh leaves the night's charts
    with a hole in them until tomorrow — which is what happened publishing P4,
    twice, on the same network that then completed every chunk on a retry.

    The signature is computed once and reused across attempts on purpose: it
    covers ``{timestamp}.{body}`` and the body does not change, so re-signing
    would only move the timestamp and risk drifting outside the server's window
    on a long backoff. ``httpx.post`` opens and closes a connection per call,
    which is what a retry after a half-open socket needs.
    """
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

    def post() -> httpx.Response:
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
        # Inside the retried operation, so a 5xx is retried and a 4xx — a bad
        # signature, a rejected payload — fails immediately and says why.
        response.raise_for_status()
        return response

    response = retry_call(
        post,
        retry_on=is_transient_http,
        description=f"write-back of {len(body)} bytes",
    )
    log.info("wrote back %d bytes -> %s", len(body), response.status_code)


def archive_simulation(
    document: dict[str, Any],
    *,
    asset: str,
    dry_run: bool = False,
) -> None:
    """PUT one Monte Carlo run's per-path outcomes to R2 (§11).

    The date comes from the document, not from the caller. Those are two
    different dates and they are routinely different: the job runs on 2026-08-01
    and simulates the information set as of 2026-07-31, the last date the market
    published. Passing the run date filed the archive under a day the simulation
    was not about — the serving side rejected the mismatch, which is what that
    check is for, and the fix is to stop having two sources for one fact.

    Never fatal. The archive is for offline analysis; a daily run that published
    its state and its quantiles has done its job, and losing tonight's path
    outcomes to a network blip is not worth failing the run and losing those
    too. The failure is logged at WARNING and the run continues — which is a
    deliberate asymmetry with `write_back`, where a failure means the published
    numbers are missing and the run really has failed.
    """
    log = logging.getLogger("findynamics.simulations")

    as_of = str(document.get("as_of") or "").strip()
    if not as_of:
        log.warning("simulation archive for %s has no as_of; skipping", asset)
        return

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
