"""Repairing release dates the vintage archive cannot actually speak to.

ALFRED records vintages from the day it started tracking a series, and stamps
everything older with that date. For DGS10 that is 2005-12-20, though the series
begins in 1962. It happens a second way too: DGS20 and DGS30 were discontinued
and later revived, and their pre-gap history was loaded into the archive in
2020 — so a 1988 twenty-year quote comes back claiming it became knowable
thirty-two years later.

Taken literally, either artifact says the historical Treasury curve was unknown
until decades after it traded, which makes a point-in-time query before that
date return nothing and a backtest over the 2000 inversion impossible.

Two independent tells that a stamp is bookkeeping rather than publication:

**Bulk seeding.** The release equals the series' earliest recorded vintage, that
same vintage covers more than one period, and the period predates it. One period
sharing its release with nothing else is an ordinary first print, not a seed —
which is why the count matters.

**Retroactive loading.** Some *later* period of the same series carries an
*earlier* release. Publication is monotone in practice; nobody issues March's
figure before January's. A violation means the older row was added afterwards.

For a row with either tell, the release date falls back to the conservative
synthesized lag (§5.2) — the same estimate used for sources that publish no
vintage at all — capped by the tightest proof available: the earliest release
recorded for this period or any later one. Whatever the lag says, the figure was
certainly knowable by then.

Everything else is left exactly as reported. The archive is the better authority
wherever it has an opinion, and overriding a genuine publication lag with a
configured guess is how lookahead gets manufactured.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from datetime import date

from findynamics.core.config import SeriesSpec
from findynamics.data.providers.base import Observation, synthesize_release_date

log = logging.getLogger("findynamics.data.vintages")


def archive_epoch(observations: Iterable[Observation]) -> date | None:
    """Earliest release date present — the archive's own start for this series."""
    return min((o.release_date for o in observations), default=None)


def _release_floor(rows: list[Observation]) -> dict[date, date]:
    """Earliest release recorded for each period *or any later one*.

    Read right to left as a running minimum. This is the tightest upper bound on
    when a period became knowable that the archive actually proves: if a later
    period was public by some date, this one was too.
    """
    floors: dict[date, date] = {}
    running: date | None = None
    for observation in sorted(rows, key=lambda o: o.observation_date, reverse=True):
        running = (
            observation.release_date if running is None else min(running, observation.release_date)
        )
        floors[observation.observation_date] = running
    return floors


def repair_pre_archive_releases(
    observations: Iterable[Observation],
    spec: SeriesSpec,
) -> list[Observation]:
    """Replace archive-artifact release dates with the configured publication lag.

    Returns a new list; inputs are frozen dataclasses and are not mutated.
    """
    rows = list(observations)
    if not rows:
        return rows

    epoch = archive_epoch(rows)
    if epoch is None:
        return rows

    # A seed vintage covers many periods at once; a genuine first print covers
    # exactly one. Without this count, the first observation of every series
    # looks like a seed, because its period always predates its release.
    seeded = sum(1 for o in rows if o.release_date == epoch) > 1
    floors = _release_floor(rows)

    repaired: list[Observation] = []
    touched = 0
    for observation in rows:
        floor = floors[observation.observation_date]
        bulk_seeded = (
            seeded and observation.release_date == epoch and observation.observation_date < epoch
        )
        loaded_retroactively = floor < observation.release_date

        if not (bulk_seeded or loaded_retroactively):
            repaired.append(observation)
            continue

        estimated = synthesize_release_date(
            observation.observation_date,
            spec.frequency,
            spec.publication_lag_days,
        )
        release = min(estimated, floor)
        revision = observation.revision_date
        repaired.append(
            Observation(
                series_id=observation.series_id,
                provider=observation.provider,
                frequency=observation.frequency,
                unit=observation.unit,
                observation_date=observation.observation_date,
                release_date=release,
                # A revision cannot precede the release it revises. Where the
                # revision was part of the same artifact, it moves with the
                # release; a genuinely later revision keeps its own date.
                revision_date=(
                    release
                    if revision is None or revision <= observation.release_date
                    else revision
                ),
                value=observation.value,
            )
        )
        touched += 1

    if touched:
        log.info(
            "%s: %d of %d observation(s) carried an archive-artifact release date "
            "(epoch %s); release dates synthesized from the configured %d-day lag",
            spec.id,
            touched,
            len(rows),
            epoch,
            spec.publication_lag_days,
        )
    return repaired
