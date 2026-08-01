"""Assembling the observation frame a run computes from.

The compute plane owns provider access — that is the premise the serving plane's
write-back door is built on — so a job's inputs come from the providers, not from
a read-back of D1. That also keeps vintage fidelity: FRED's ALFRED endpoint
returns every figure ever issued for a period, and re-reading our own store would
give back only what we happened to have ingested.

Responses are cached on disk (``FileCache``), so the second run of a day is cheap
and a 25-year replay does not re-download 25 years.

The output is the long frame :mod:`findynamics.data.pit` expects — one row per
figure, with ``release_date`` and ``revision_date`` kept apart. Nothing here
filters by date: that is the accessor's job, and doing it in two places is how
the two definitions drift.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping
from datetime import date
from pathlib import Path

import pandas as pd

from findynamics.core.config import SeriesConfig, SeriesSpec, get_series_config
from findynamics.data.providers import build_provider
from findynamics.data.providers.base import Observation, Provider, ProviderError
from findynamics.data.providers.registry import UnknownProviderError
from findynamics.data.vintages import repair_pre_archive_releases

log = logging.getLogger("findynamics.data.store")

COLUMNS = ("series_id", "obs_date", "release_date", "revision_date", "value")

#: Written under compute/ and gitignored — cached HTTP responses, not data.
DEFAULT_CACHE_DIR = Path(__file__).resolve().parents[2] / ".cache"


def empty_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=list(COLUMNS))


def observations_to_frame(observations: Iterable[Observation]) -> pd.DataFrame:
    """Provider observations as the long frame the PIT layer reads."""
    rows = [
        {
            "series_id": o.series_id,
            "obs_date": o.observation_date,
            "release_date": o.release_date,
            "revision_date": o.revision_date or o.release_date,
            "value": float(o.value),
        }
        for o in observations
    ]
    if not rows:
        return empty_frame()
    frame = pd.DataFrame(rows)
    for column in ("obs_date", "release_date", "revision_date"):
        frame[column] = pd.to_datetime(frame[column]).dt.normalize()
    return frame


def resolve_specs(
    series_ids: Iterable[str], config: SeriesConfig | None = None
) -> list[SeriesSpec]:
    """Look up which provider serves each id, from config alone.

    A series nobody configured is skipped with a warning rather than guessed at:
    inferring the provider from an id prefix would work right up until it did
    not, and a silently wrong provider is a silently wrong release date.
    """
    resolved = config or get_series_config()
    by_id = {spec.id: spec for spec in resolved.all_series()}

    specs: list[SeriesSpec] = []
    for series_id in dict.fromkeys(series_ids):
        spec = by_id.get(series_id)
        if spec is None:
            log.warning("series %s is not in series.yaml; skipping", series_id)
            continue
        specs.append(spec)
    return specs


def load_observations(
    series_ids: Iterable[str],
    *,
    config: SeriesConfig | None = None,
    cache_dir: Path | None = None,
    env: Mapping[str, str] | None = None,
    start: date | None = None,
    end: date | None = None,
) -> pd.DataFrame:
    """Fetch every requested series and concatenate into one long frame.

    A provider failure costs its series, not the run (§14.2): the frame comes
    back short, the factor or engine that needed it degrades, and the failure is
    logged. That is the whole reason ``pit_join`` tolerates missing ids.

    **A provider that cannot be built is the exception, and raises.** That is a
    configuration error rather than a runtime fault: ``series.yaml`` names
    something no code implements, so every run degrades in exactly the same way
    and no amount of retrying or waiting will change it. Treating it like a
    network blip is how ``DERIVED:EXCESS_CAPE_YIELD`` and
    ``DERIVED:HY_IG_DIFFERENTIAL`` went unfetched for months while the valuation
    and risk-appetite factors kept publishing scores that looked complete.
    """
    specs = resolve_specs(series_ids, config)
    if not specs:
        return empty_frame()

    directory = cache_dir if cache_dir is not None else DEFAULT_CACHE_DIR
    providers: dict[str, Provider] = {}
    frames: list[pd.DataFrame] = []

    for spec in specs:
        provider = providers.get(spec.provider)
        if provider is None:
            try:
                provider = build_provider(spec.provider, cache_dir=directory, env=env)
            except UnknownProviderError as err:
                # NOT the §14.2 case, and conflating the two cost this system two
                # equity factor inputs for months. A provider that FAILS is a
                # runtime fault — the API is down, the key expired — and losing
                # its series is the right price. A provider that cannot be BUILT
                # is a configuration error: series.yaml names something no code
                # implements, every run degrades identically, and no amount of
                # waiting fixes it. Continuing there means a factor publishes a
                # score computed from fewer inputs than it claims, with nothing
                # in the output saying so.
                raise ProviderError(
                    "store",
                    f"{spec.id} is configured with provider {spec.provider!r}, which is not "
                    f"implemented ({err}). This is a configuration error, not a transient "
                    "failure: fix series.yaml or register the provider. Refusing to publish "
                    "factors built from a silently reduced input set.",
                    retryable=False,
                ) from err
            except ProviderError as err:
                log.error("cannot build provider %s: %s", spec.provider, err)
                continue
            providers[spec.provider] = provider

        try:
            result = provider.fetch(spec.id, start=start, end=end)
        except ProviderError as err:
            log.error("%s: %s", spec.id, err)
            continue

        frame = observations_to_frame(repair_pre_archive_releases(result.observations, spec))
        if frame.empty:
            log.warning("%s returned no observations", spec.id)
            continue
        frames.append(frame)
        log.info("%s: %d observations", spec.id, len(frame))

    if not frames:
        return empty_frame()
    return pd.concat(frames, ignore_index=True)


def required_series_ids(
    config: SeriesConfig,
    engine_series: Iterable[str] = (),
) -> tuple[str, ...]:
    """Every id a run needs: the shared factor inputs plus the engines' own."""
    factor_ids = {s.id for spec in config.factors.values() for s in spec.series}
    return tuple(sorted(factor_ids | set(engine_series)))
