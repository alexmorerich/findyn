"""FinDynamics compute job: monthly refit.

Calls ``fit`` on every enabled engine over an expanding window ending at the
information-set cutoff (§14.1 rule 4). For FinRates that is Nelson-Siegel lambda
selection: chosen once over the grid, written to the artifact store, and frozen
until the next refit — a daily run must never move it, because a moving lambda
silently rebases every factor history the engine has published.

Separate from ``daily`` on purpose. Refitting is expensive and its output is a
parameter set, not a state; running it on the daily cadence would make yesterday
and today incomparable for no gain.

**The cutoff is derived from the calendar, not read off the clock.** Whatever day
this job runs, it fits on the last completed month
(:func:`jobs._common.refit_cutoff`), so every run within a month sees the same
information set and produces the same artifact. Fitting "as of now" instead made
the parameters a function of when the container started — two runs on 2026-08-01
disagreed and the second was rejected as a conflict (issue #6). ``--no-pin``
restores the old behaviour for a deliberate one-off; nothing scheduled should use
it.
"""

from __future__ import annotations

import logging
import sys
from datetime import date
from pathlib import Path

from findynamics.core.config import load_series_config
from jobs._common import base_parser, configure_logging, refit_cutoff
from jobs.daily import build_world, parse_as_of

log = logging.getLogger("findynamics.jobs.monthly_refit")


def run(
    as_of: date,
    *,
    config_path: Path | None = None,
    cache_dir: Path | None = None,
    dry_run: bool = False,
    pin: bool = True,
) -> int:
    config = load_series_config(config_path)

    # `as_of` is when the run happens; `cutoff` is what it is allowed to see. The
    # two are deliberately different, and the artifact is keyed on the second.
    cutoff = refit_cutoff(as_of) if pin else as_of
    if pin:
        log.info("run date %s; fitting on the last completed month, through %s", as_of, cutoff)
    else:
        log.warning(
            "--no-pin: fitting through the run date %s. The artifact is keyed on this "
            "date, so a second run today against changed data will conflict (issue #6).",
            cutoff,
        )

    world, engines = build_world(cutoff, config, cache_dir=cache_dir)

    if not engines:
        log.error("no engines are enabled; nothing to refit")
        return 2

    if dry_run:
        log.info(
            "dry run: would refit %s over data through %s",
            ", ".join(e.name for e in engines),
            cutoff,
        )
        return 0

    failed: list[str] = []
    for engine in engines:
        try:
            engine.fit(world)
            log.info("refit %s (%s)", engine.name, engine.version)
        except Exception as err:
            failed.append(engine.name)
            log.exception("refit failed for %s: %s", engine.name, err)

    if failed:
        log.error("refit failed for: %s", ", ".join(failed))
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = base_parser(__doc__ or "monthly_refit")
    parser.add_argument(
        "--config", type=Path, help="path to series.yaml (defaults to the shipped one)"
    )
    parser.add_argument("--cache-dir", type=Path, help="HTTP response cache directory")
    parser.add_argument(
        "--no-pin",
        action="store_true",
        help=(
            "Fit through the run date instead of the last completed month. For a "
            "deliberate one-off only: the artifact is keyed on the fit date, so two "
            "runs on one day against changed data conflict (issue #6)."
        ),
    )
    args = parser.parse_args(argv)
    configure_logging(args.verbose)

    return run(
        parse_as_of(args.as_of),
        config_path=args.config,
        cache_dir=args.cache_dir,
        dry_run=args.dry_run,
        pin=not args.no_pin,
    )


if __name__ == "__main__":
    sys.exit(main())
