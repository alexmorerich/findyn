"""FinDynamics compute job: monthly refit.

Calls ``fit`` on every enabled engine over an expanding window ending at the
information-set cutoff (§14.1 rule 4). For FinRates that is Nelson-Siegel lambda
selection: chosen once over the grid, written to the artifact store, and frozen
until the next refit — a daily run must never move it, because a moving lambda
silently rebases every factor history the engine has published.

Separate from ``daily`` on purpose. Refitting is expensive and its output is a
parameter set, not a state; running it on the daily cadence would make yesterday
and today incomparable for no gain.
"""

from __future__ import annotations

import logging
import sys
from datetime import date
from pathlib import Path

from findynamics.core.config import load_series_config
from jobs._common import base_parser, configure_logging
from jobs.daily import build_world, parse_as_of

log = logging.getLogger("findynamics.jobs.monthly_refit")


def run(
    as_of: date,
    *,
    config_path: Path | None = None,
    cache_dir: Path | None = None,
    dry_run: bool = False,
) -> int:
    config = load_series_config(config_path)
    world, engines = build_world(as_of, config, cache_dir=cache_dir)

    if not engines:
        log.error("no engines are enabled; nothing to refit")
        return 2

    if dry_run:
        log.info(
            "dry run: would refit %s over data through %s",
            ", ".join(e.name for e in engines),
            as_of,
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
    args = parser.parse_args(argv)
    configure_logging(args.verbose)

    return run(
        parse_as_of(args.as_of),
        config_path=args.config,
        cache_dir=args.cache_dir,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    sys.exit(main())
