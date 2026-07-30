"""Ingestion job: fetch series from a provider, validate, write back.

Handles both the one-off historical backfill and incremental top-ups — the
difference is only ``--start``. Every batch passes through the quality engine
before it is sent, and a batch with errors is withheld unless ``--force`` is
given, because a bad figure is far more expensive to remove from a
point-in-time store than to never write.

FINDYN_V1_SPEC.md §6 (ingestion strategy), M1-A.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Any

from findynamics.data.providers import build_provider
from findynamics.data.providers.base import Provider, ProviderError
from findynamics.data.providers.registry import KEYLESS_PROVIDERS
from findynamics.data.quality import DataQualityReport, QualityPolicy, check_series
from jobs._common import base_parser, configure_logging, write_back

log = logging.getLogger("findynamics.jobs.backfill")


def parse_date(value: str | None) -> date | None:
    if not value:
        return None
    return datetime.strptime(value, "%Y-%m-%d").date()


def ingest_series(
    provider: Provider,
    series_id: str,
    *,
    start: date | None,
    end: date | None,
    policy: QualityPolicy | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], DataQualityReport]:
    """Fetch and validate one series.

    Returns (metadata wire dict, observation wire dicts, quality report).
    """
    result = provider.fetch(series_id, start=start, end=end)
    report = check_series(result.metadata, result.observations, policy=policy)
    return result.metadata.to_wire(), [o.to_wire() for o in result.observations], report


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
) -> int:
    provider = build_provider(provider_id, cache_dir=cache_dir)
    targets = series_ids or provider.available_series()
    if not targets:
        log.error(
            "%s exposes no default catalogue; name the series explicitly with --series",
            provider_id,
        )
        return 2

    metadata: list[dict[str, Any]] = []
    observations: list[dict[str, Any]] = []
    quality: list[dict[str, Any]] = []
    ingestion: list[dict[str, Any]] = []
    withheld = 0
    failed = 0

    for series_id in targets:
        try:
            meta, rows, report = ingest_series(provider, series_id, start=start, end=end)
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

    write_back(payload, dry_run=dry_run)

    # Retrieving nothing usable is a failure; partial success is not.
    return 1 if (failed and not metadata) else 0


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
        help="series ids; defaults to the provider's whole catalogue",
    )
    parser.add_argument("--start", help="earliest observation date, YYYY-MM-DD")
    parser.add_argument("--end", help="latest observation date, YYYY-MM-DD")
    parser.add_argument(
        "--force",
        action="store_true",
        help="write series even when the quality engine reports errors",
    )
    parser.add_argument("--cache-dir", default=".cache", help="on-disk response cache")
    parser.add_argument("--out", help="also write the write-back payload to this file")

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
    )


if __name__ == "__main__":
    sys.exit(main())
