"""Ingestion job: fetch series from a provider, validate, write back.

Handles both the one-off historical backfill and incremental top-ups — the
difference is only ``--start``. Every batch passes through the quality engine
before it is sent, and a batch with errors is withheld unless ``--force`` is
given, because a bad figure is far more expensive to remove from a
point-in-time store than to never write.

FINDYN_V1_SPEC.md §6 (ingestion strategy), M1-A.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import sys
from collections.abc import Iterator
from datetime import date, datetime
from pathlib import Path
from typing import Any

from findynamics.core.config import SeriesSpec, get_series_config
from findynamics.data.providers import build_provider
from findynamics.data.providers.base import Provider, ProviderError
from findynamics.data.providers.registry import KEYLESS_PROVIDERS
from findynamics.data.providers.stooq import StooqFileProvider
from findynamics.data.quality import DataQualityReport, QualityPolicy, check_series
from findynamics.data.vintages import repair_pre_archive_releases
from jobs._common import base_parser, chunk_on, configure_logging, write_back

log = logging.getLogger("findynamics.jobs.backfill")

#: Observations per write-back request. The serving plane upserts in D1 batches
#: of 900, so this is a handful of batches per request — well inside a Worker's
#: CPU budget, and small enough that a retry is cheap.
DEFAULT_BATCH_SIZE = 5000


def parse_date(value: str | None) -> date | None:
    if not value:
        return None
    return datetime.strptime(value, "%Y-%m-%d").date()


def _parse_start(spec: SeriesSpec | None, series_id: str) -> date | None:
    """The configured ``start`` for one series, or ``None``.

    ``core.config`` already rejected an unparseable value at load, so anything
    that reaches here is a valid date; the guard is for a spec built in a test
    rather than loaded from yaml.
    """
    if spec is None or spec.start is None:
        return None
    try:
        return date.fromisoformat(spec.start)
    except ValueError:  # pragma: no cover - config load validates this
        log.warning("%s: ignoring unparseable configured start %r", series_id, spec.start)
        return None


def _later(*days: date | None) -> date | None:
    """The latest of the given dates, ignoring ``None``."""
    known = [day for day in days if day is not None]
    return max(known) if known else None


def configured_series(provider_id: str) -> list[str]:
    """Every series in series.yaml served by ``provider_id``, factors and engines alike.

    Deduplicated and sorted: a series can appear under both a factor and an
    engine (DGS10 does), and fetching it twice would just burn quota.
    """
    config = get_series_config()
    return sorted({spec.id for spec in config.all_series() if spec.provider == provider_id})


def ingest_series(
    provider: Provider,
    series_id: str,
    *,
    start: date | None,
    end: date | None,
    policy: QualityPolicy | None = None,
    spec: SeriesSpec | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], DataQualityReport]:
    """Fetch, repair release dates, validate.

    Returns (metadata wire dict, observation wire dicts, quality report).

    The repair happens before the quality check and before the write-back, so
    what lands in D1 is what a point-in-time query needs. Doing it on the read
    side instead would leave the stored table quietly wrong.
    """
    result = provider.fetch(series_id, start=start, end=end)
    observations = (
        repair_pre_archive_releases(result.observations, spec)
        if spec is not None
        else result.observations
    )
    report = check_series(result.metadata, observations, policy=policy)

    # Anomalies are attached to the rows they describe rather than blocking the
    # series. `check_series` decided which observations look extreme; this is
    # where that judgement becomes something the store and its consumers can see.
    flagged = report.anomalous_dates()
    if flagged:
        observations = [
            dataclasses.replace(o, quality_flag=flagged[o.observation_date.isoformat()])
            if o.observation_date.isoformat() in flagged
            else o
            for o in observations
        ]
    return result.metadata.to_wire(), [o.to_wire() for o in observations], report


def run(
    provider_id: str,
    series_ids: list[str],
    *,
    start: date | None,
    end: date | None,
    dry_run: bool,
    force: bool,
    cache_dir: Path | None,
    out: Path | None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    from_file: Path | None = None,
) -> int:
    if from_file is not None:
        if provider_id != "stooq":
            log.error("--from-file is only implemented for --provider stooq, got %r", provider_id)
            return 2
        if not from_file.is_file():
            log.error("--from-file: no such file: %s", from_file)
            return 2
        provider = StooqFileProvider(from_file)
    else:
        provider = build_provider(provider_id, cache_dir=cache_dir)
    # Config first, catalogue second. FRED hosts 800k series and exposes no
    # catalogue, so "everything this provider serves us" is a question only
    # series.yaml can answer — and answering it there is what lets a new engine's
    # series be backfilled by editing yaml rather than this file.
    targets = series_ids or configured_series(provider_id) or provider.available_series()
    if not targets:
        log.error(
            "%s has no series in series.yaml and exposes no default catalogue; "
            "name them explicitly with --series",
            provider_id,
        )
        return 2

    # One file holds one series, and which one is not inferable. The id must be
    # named rather than defaulted: a file downloaded by hand carries no
    # identifier a program can read, so any default is a guess, and a wrong guess
    # writes one instrument's prices under another's name — silent, and expensive
    # to unpick from a PIT store.
    #
    # Defaulting was survivable while stooq had no configured series to fall back
    # to and the check below caught the seven-symbol catalogue. P5 configured
    # STOOQ:BTCUSD, which made `configured_series` resolve to exactly one id —
    # so an ^SPX download with no --series would have passed the count check and
    # been ingested as bitcoin.
    if from_file is not None and not series_ids:
        log.error(
            "--from-file needs an explicit --series: a hand-downloaded file carries "
            "no id, so the series it holds cannot be inferred (resolved by default "
            "to %s, which is a guess)",
            ", ".join(targets),
        )
        return 2
    if from_file is not None and len(targets) != 1:
        log.error(
            "--from-file ingests one series from one file, but %d resolved (%s); "
            "name exactly one with --series",
            len(targets),
            ", ".join(targets),
        )
        return 2

    # The spec carries the publication lag, which is what a release date is
    # synthesized from where the vintage archive cannot speak (data/vintages.py),
    # and now an optional per-series `start`.
    specs = {spec.id: spec for spec in get_series_config().all_series()}

    metadata: list[dict[str, Any]] = []
    observations: list[dict[str, Any]] = []
    quality: list[dict[str, Any]] = []
    ingestion: list[dict[str, Any]] = []
    withheld = 0
    failed = 0

    for series_id in targets:
        spec = specs.get(series_id)
        # A configured `start` bounds what this series is asked for; an explicit
        # --start bounds the whole run. The later of the two wins, so the CLI can
        # narrow a backfill but cannot widen it into history the source does not
        # have — asking Yahoo for bitcoin in 2009 returns nothing useful, and
        # asking blockchain.info returns a run of zeros.
        series_start = _later(start, _parse_start(spec, series_id))
        try:
            meta, rows, report = ingest_series(
                provider, series_id, start=series_start, end=end, spec=spec
            )
        except ProviderError as err:
            failed += 1
            log.error("%s: %s", series_id, err)
            ingestion.append(
                {
                    "source": provider_id,
                    "series_id": series_id,
                    "status": "failed",
                    "rows_written": 0,
                    "error": str(err),
                }
            )
            continue

        quality.append(report.to_wire())
        log.info("%s", report.summary())
        for finding in report.errors:
            log.error("  error  %s: %s", finding.code, finding.message)
        for finding in report.warnings:
            log.warning("  warn   %s: %s", finding.code, finding.message)

        if report.errors and not force:
            withheld += 1
            ingestion.append(
                {
                    "source": provider_id,
                    "series_id": series_id,
                    "status": "failed",
                    "rows_written": 0,
                    "error": f"withheld: {len(report.errors)} quality error(s)",
                }
            )
            continue

        metadata.append(meta)
        observations.extend(rows)
        ingestion.append(
            {
                "source": provider_id,
                "series_id": series_id,
                "status": "degraded" if report.warnings else "ok",
                "rows_written": len(rows),
                "error": None,
            }
        )

    payload: dict[str, Any] = {
        "model_version": "m1a",
        "generated_at": datetime.now().astimezone().isoformat(),
        "metadata": metadata,
        "observations": observations,
        "quality": quality,
        "ingestion": ingestion,
    }

    log.info(
        "%s: %d series ok, %d observations, %d withheld, %d failed",
        provider_id,
        len(metadata),
        len(observations),
        withheld,
        failed,
    )

    if out is not None:
        out.write_text(json.dumps(payload, indent=2))
        log.info("wrote payload to %s (%d bytes)", out, out.stat().st_size)

    for chunk in chunk_payload(payload, batch_size):
        write_back(chunk, dry_run=dry_run)

    # Retrieving nothing usable is a failure; partial success is not.
    return 1 if (failed and not metadata) else 0


def chunk_payload(payload: dict[str, Any], batch_size: int) -> Iterator[dict[str, Any]]:
    """Split a backfill into requests the serving plane can actually absorb.

    A full historical backfill is hundreds of thousands of observations. Sent as
    one request that is tens of megabytes and hundreds of D1 batches, and the
    Worker runs out of CPU somewhere in the middle — leaving a partial write
    with no record of where it stopped.

    Chunking by observations keeps each request bounded. Metadata, quality and
    ingestion rows ride along with the first chunk: they are per-series, not
    per-observation, so they are small and repeating them would just re-upsert.
    Every chunk is independently idempotent, so a failure costs one chunk and a
    re-run fixes it.

    The mechanics live in :func:`jobs._common.chunk_on`, because the daily job hit
    the same wall on ``engine_output`` once a second engine shipped.
    """
    return chunk_on(
        payload,
        "observations",
        batch_size,
        carry=("model_version", "generated_at"),
    )


def main(argv: list[str] | None = None) -> int:
    parser = base_parser(__doc__ or "backfill")
    parser.add_argument(
        "--provider",
        default="shiller",
        help=f"provider id (no key required: {', '.join(sorted(KEYLESS_PROVIDERS))})",
    )
    parser.add_argument(
        "--series",
        nargs="*",
        default=[],
        help="series ids; defaults to every series series.yaml maps to this provider",
    )
    parser.add_argument("--start", help="earliest observation date, YYYY-MM-DD")
    parser.add_argument("--end", help="latest observation date, YYYY-MM-DD")
    parser.add_argument(
        "--force",
        action="store_true",
        help="write series even when the quality engine reports errors",
    )
    parser.add_argument(
        "--from-file",
        help=(
            "ingest a CSV already on disk instead of fetching (stooq only), for "
            "history its bot filter puts out of automated reach. Needs an explicit "
            "--series, since series.yaml points the equity backfill at Yahoo: "
            "--provider stooq --series STOOQ:^SPX --from-file ~/Downloads/^spx_d.csv"
        ),
    )
    parser.add_argument("--cache-dir", default=".cache", help="on-disk response cache")
    parser.add_argument("--out", help="also write the write-back payload to this file")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="observations per write-back request",
    )

    args = parser.parse_args(argv)
    configure_logging(args.verbose)

    return run(
        args.provider,
        args.series,
        start=parse_date(args.start),
        end=parse_date(args.end),
        dry_run=args.dry_run,
        force=args.force,
        cache_dir=Path(args.cache_dir) if args.cache_dir else None,
        out=Path(args.out) if args.out else None,
        batch_size=args.batch_size,
        from_file=Path(args.from_file).expanduser() if args.from_file else None,
    )


if __name__ == "__main__":
    sys.exit(main())
