"""Another engine's published output, read back as a series.

FinMoney needs FinRates' fitted curve to discount anything past a year, and the
independence rule forbids it from importing ``engines.rates``
(``01-target-architecture.md`` §3 rule 2, enforced by ``lint-imports``). Nor can
it simply be handed the rates engine's in-memory result: engines run in registry
order, so the producer may not have run yet in this process, and a hidden
ordering dependency between two supposedly independent engines is exactly what
the rule exists to prevent.

So the coupling is made explicit and turned into *data*. FinRates publishes
``ns_level`` and friends to ``engine_output``; this adapter reads them back out
of the serving plane under an ``ENGINE:rates.ns_level`` series id, and from there
they are indistinguishable from a FRED series — same frame, same ``pit_join``,
same release-date filter. A consumer therefore cannot see an output that had not
been published at its information-set cutoff, which is the property that would
have been lost by passing the object directly.

This is the one place the compute plane reads its own store back. Ordinary
observations deliberately come from the providers instead (see
:mod:`findynamics.data.store`) because re-reading D1 would lose vintage
fidelity — but an engine output *has* no upstream source to prefer. The serving
plane is its only home.

A missing endpoint, an unset URL or an engine that has never run all degrade to
"no observations" rather than raising, so the first-ever run of the system
works: money publishes with a flat short-rate curve, and the NS curve appears on
the next run once rates has written something.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from datetime import date, datetime, timedelta

from findynamics.core.contracts.vocab import (
    ENGINE_SERIES_PREFIX,
    engine_series_id,
    parse_engine_series_id,
)
from findynamics.data.providers.base import (
    NotFoundError,
    Observation,
    ParseError,
    Provider,
    SeriesMetadata,
)
from findynamics.data.providers.resilience import Transport

log = logging.getLogger("findynamics.data.providers.published")

#: Preferred configuration: the serving plane's origin, no path.
API_URL_ENV = "FINDYN_API_URL"

#: Fallback: the write-back endpoint the jobs already need, minus its path. One
#: secret fewer to configure, and the two can never point at different planes.
ADMIN_URL_ENV = "FINDYN_ADMIN_URL"

#: Newest N points per metric. Five years of trading days with headroom — the
#: same window the rates engine republishes, so asking for more returns nothing.
DEFAULT_LIMIT = 2600

#: Cache TTL. A metric's history changes once a day, after the daily run.
CACHE_TTL = 900


def resolve_api_base(env: Mapping[str, str]) -> str | None:
    """Origin of the serving plane, or ``None`` when nothing is configured.

    ``FINDYN_ADMIN_URL`` is a full endpoint (``https://host/admin/v1/results``),
    so the origin is everything before ``/admin``. Deriving it rather than asking
    for a second variable means an operator cannot configure a compute job that
    writes to one deployment and reads from another.
    """
    configured = (env.get(API_URL_ENV) or "").strip()
    if configured:
        return configured.rstrip("/")

    admin = (env.get(ADMIN_URL_ENV) or "").strip()
    if not admin:
        return None
    marker = "/admin/"
    head, sep, _ = admin.partition(marker)
    return (head if sep else admin).rstrip("/") or None


class PublishedOutputProvider(Provider):
    """Reads ``engine_output`` rows back through the public read API."""

    id = "engine_output"
    requires_api_key = False

    def __init__(
        self,
        transport: Transport,
        *,
        base_url: str | None,
        limit: int = DEFAULT_LIMIT,
    ) -> None:
        self.transport = transport
        self.base_url = base_url.rstrip("/") if base_url else None
        self.limit = limit

    def available_series(self) -> list[str]:
        # Which metrics exist is whatever the engines have published; the useful
        # set is what series.yaml names, exactly as for FRED.
        return []

    @staticmethod
    def _split(series_id: str) -> tuple[str, str]:
        try:
            parsed = parse_engine_series_id(series_id)
        except ValueError as err:
            raise NotFoundError("engine_output", str(err)) from None
        if parsed is None:
            raise NotFoundError(
                "engine_output",
                f"{series_id!r} is not a published-output id; expected "
                f"{ENGINE_SERIES_PREFIX}<asset>.<metric>",
            )
        return parsed

    def _require_base(self) -> str:
        if not self.base_url:
            raise NotFoundError(
                self.id,
                f"neither {API_URL_ENV} nor {ADMIN_URL_ENV} is set, so there is no "
                "serving plane to read published engine outputs from",
            )
        return self.base_url

    def fetch_metadata(self, series_id: str) -> SeriesMetadata:
        asset, metric = self._split(series_id)
        return SeriesMetadata(
            series_id=series_id,
            provider=self.id,
            title=f"FinDynamics {asset} engine output: {metric}",
            frequency="daily",
            unit="model",
            notes=(
                "Published by the "
                f"{asset} engine and read back through /api/v1/assets/{asset}/history."
            ),
        )

    def fetch_observations(
        self,
        series_id: str,
        *,
        start: date | None = None,
        end: date | None = None,
    ) -> list[Observation]:
        asset, metric = self._split(series_id)
        base = self._require_base()

        params: dict[str, str] = {"metric": metric, "limit": str(self.limit)}
        if start is not None:
            params["from"] = start.isoformat()
        if end is not None:
            params["to"] = end.isoformat()

        response = self.transport.get(
            f"{base}/api/v1/assets/{asset}/history",
            params=params,
            cache_ttl=CACHE_TTL,
        )
        try:
            payload = response.json()
        except ValueError as err:
            raise ParseError(self.id, f"non-JSON response for {series_id}: {err}") from err
        if not isinstance(payload, dict):
            raise ParseError(self.id, f"unexpected payload type for {series_id}")

        data = payload.get("data")
        if not isinstance(data, dict):
            raise ParseError(self.id, f"{series_id}: response is not a FinDyn envelope")

        points = data.get("points")
        if points is None:
            raise ParseError(self.id, f"{series_id}: response carries no points array")

        return self._to_observations(series_id, points)

    def _to_observations(self, series_id: str, points: object) -> list[Observation]:
        if not isinstance(points, list):
            raise ParseError(self.id, f"{series_id}: points is not an array")

        observations: list[Observation] = []
        for point in points:
            if not isinstance(point, dict):
                continue
            obs_date = _parse_date(point.get("as_of"))
            value = _parse_float(point.get("value"))
            if obs_date is None or value is None:
                continue
            observations.append(
                Observation(
                    series_id=series_id,
                    provider=self.id,
                    frequency="daily",
                    unit="model",
                    observation_date=obs_date,
                    # `written_at` is the run that published this row — a real
                    # vintage, not a synthesized lag, and the only date at which
                    # a consumer could honestly have seen the value. Where the
                    # serving plane does not report it, fall back to the
                    # conservative next-day estimate: the daily job publishes a
                    # date's output on that date at the earliest, and a lag that
                    # is too long costs freshness while one that is too short
                    # manufactures lookahead.
                    release_date=_release_for(obs_date, point.get("written_at")),
                    value=value,
                )
            )

        observations.sort(key=lambda o: o.observation_date)
        log.debug("%s: %d published point(s)", series_id, len(observations))
        return observations


def _release_for(obs_date: date, written_at: object) -> date:
    written = _parse_date(written_at)
    if written is None:
        return obs_date + timedelta(days=1)
    # A clock skew that puts publication before the date it describes would trip
    # the Observation invariant; clamping keeps the frame loadable.
    return max(written, obs_date)


def _parse_date(value: object) -> date | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        # Accepts both `2026-07-29` and the ISO timestamp `written_at` carries.
        return datetime.fromisoformat(value.replace("Z", "+00:00")).date()
    except ValueError:
        return None


def _parse_float(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    result = float(value)
    return result if result == result and abs(result) != float("inf") else None


__all__ = [
    "ADMIN_URL_ENV",
    "API_URL_ENV",
    "DEFAULT_LIMIT",
    "PublishedOutputProvider",
    "engine_series_id",
    "resolve_api_base",
]
