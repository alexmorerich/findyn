"""Read every enabled engine's fitted parameters back out of storage.

The step this replaces claimed to publish fitted parameters and did not. It
uploaded ``compute/artifacts/`` as a build artifact, but a production refit sets
``FINDYN_ADMIN_URL`` and ``ADMIN_HMAC_SECRET``, and with both present
:func:`~findynamics.core.artifacts.build_artifact_store` returns the **R2**
store — so the local directory is never written. With ``if-no-files-found:
warn`` the step then found nothing, said so in a line nobody reads, and passed.
Measured on the 2026-08-08 refit: zero artifacts uploaded, job green.

That is worse than having no step at all, because the job *looks* like it
verifies its own output. So this reads the fit back from wherever it was
actually stored and checks that what came out is the thing that went in.

What it asserts, and why each one is a real failure mode rather than a
formality:

``model_version``
    The daily run loads by name and trusts the version it finds. A refit that
    stored under one version while the engine publishes another is invisible
    until a state appears with parameters from a model nobody fitted.
``as_of``
    The artifact is addressed by (model_version, fit date). A stored date that
    disagrees with the run's cutoff means the address is wrong, and the next
    refit of the same month will not collide with it the way it should.
no wall clock
    ``fitted_at`` made every document differ from every other by construction,
    which is what turned an idempotent re-write into a 409 (issue #6). It is
    gone; this is the standing check that it stays gone in the *stored* bytes
    rather than only in the code that builds them.

Exits non-zero on the first engine that fails, so a refit that silently stored
nothing usable turns the job red instead of green.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

from findynamics.core.artifacts import build_artifact_store
from findynamics.core.config import load_series_config
from findynamics.core.registry import enabled_engines
from findynamics.engines import load_engines
from jobs._common import configure_logging

log = logging.getLogger("findynamics.jobs.verify_artifacts")

#: Keys whose presence would mean a wall clock is back in the stored document.
#:
#: Checked by name because that is how it got in the first time — a field added
#: for debugging that nothing downstream read. Deliberately excludes ``as_of``
#: and ``fitted_as_of``: those are the *information set* the fit describes, which
#: is derived from the calendar and is the same on every run of a month. It is
#: the wall clock that cannot be, not the date.
CLOCK_KEYS: tuple[str, ...] = ("fitted_at", "generated_at", "created_at", "timestamp")

#: Field naming is per engine — equity writes ``as_of``, gold and rates write
#: ``fitted_as_of``. Both mean the same thing and either satisfies the check;
#: requiring one spelling would fail three engines for a naming difference.
FIT_DATE_KEYS: tuple[str, ...] = ("as_of", "fitted_as_of")


def check(name: str, document: dict[str, Any], *, as_of: str | None) -> list[str]:
    """Problems with one stored artifact. Empty means it is usable.

    An empty document is **not** a problem: ``money`` fits nothing on purpose —
    the numeraire is arithmetic on observed rates and the liquidity thresholds
    are configuration — so it has no parameters to persist and never writes any.
    Failing it here would make the step cry wolf every month, which is how a
    check stops being read.
    """
    problems: list[str] = []

    if not document:
        log.info("%s: stores no fitted parameters, by design", name)
        return problems

    if not document.get("model_version"):
        problems.append(f"{name}: stored artifact carries no model_version")

    stored = next((document[key] for key in FIT_DATE_KEYS if document.get(key)), None)
    if not stored:
        problems.append(
            f"{name}: stored artifact carries no fit date ({' or '.join(FIT_DATE_KEYS)}); "
            "it is addressed by (model_version, fit date), so it has half an address"
        )
    elif as_of is not None and stored != as_of:
        problems.append(
            f"{name}: stored fit date {stored!r} is not the run's cutoff {as_of!r}; "
            "the artifact is filed under a date it does not describe"
        )

    present = [key for key in CLOCK_KEYS if key in document]
    if present:
        problems.append(
            f"{name}: stored artifact carries wall-clock field(s) {', '.join(present)}. "
            "Two refits of one information set must produce the same bytes or the "
            "second conflicts with the first (issue #6)"
        )

    return problems


def run(*, config_path: Path | None = None, as_of: str | None = None) -> int:
    config = load_series_config(config_path)
    load_engines(config)
    store = build_artifact_store()
    engines = enabled_engines(config, artifacts=store)

    if not engines:
        log.error("no engines are enabled; there is nothing to verify")
        return 2

    failures: list[str] = []
    for engine in engines:
        name = getattr(engine, "name", "?")
        artifact = getattr(engine, "ARTIFACT_NAME", None) or name
        try:
            document = store.load(artifact)
        except Exception as err:  # a store that cannot be read is the failure
            failures.append(f"{name}: could not read stored parameters ({err})")
            continue

        problems = check(name, document, as_of=as_of)
        if problems:
            failures.extend(problems)
            continue

        if document:
            log.info(
                "%s: stored parameters read back — %s fitted through %s",
                name,
                document.get("model_version"),
                next((document[k] for k in FIT_DATE_KEYS if document.get(k)), "?"),
            )

    for failure in failures:
        log.error("%s", failure)
    return 1 if failures else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, help="path to series.yaml (defaults to the shipped one)"
    )
    parser.add_argument(
        "--as-of",
        help=(
            "The cutoff the refit fitted on. When given, the stored as_of must match "
            "it — that pair is the artifact's address."
        ),
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)
    configure_logging(args.verbose)
    return run(config_path=args.config, as_of=args.as_of)


if __name__ == "__main__":
    sys.exit(main())
