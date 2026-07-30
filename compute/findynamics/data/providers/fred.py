"""Federal Reserve Economic Data (FRED / ALFRED).

FRED is the only source here that publishes real vintages, and the adapter uses
them: requesting the full realtime range returns every figure ever issued for a
period, each stamped with the date it became available. That gives a true
``release_date`` (the first issue for a period) and ``revision_date`` (this
particular issue) instead of a synthesized lag — which matters because a
revision can move a number by more than the signal being measured.

FINDYN_V1_SPEC.md §5.1 source 1.
"""

from __future__ import annotations

import logging
from datetime import date, datetime

from findynamics.data.providers.base import (
    AuthError,
    NotFoundError,
    Observation,
    ParseError,
    Provider,
    SeriesMetadata,
)
from findynamics.data.providers.resilience import Transport

log = logging.getLogger("findynamics.data.providers.fred")

BASE_URL = "https://api.stlouisfed.org/fred"

#: Widest range FRED accepts — asks for every vintage rather than today's view.
REALTIME_START = "1776-07-04"
REALTIME_END = "9999-12-31"

#: FRED rejects an observations request whose real-time window spans more than
#: this many vintage dates. The daily Treasury series blow straight through it —
#: DGS10 has over 5000 — so the window is split into chunks that each stay under.
MAX_VINTAGE_DATES = 2000

#: Page size for the vintagedates endpoint (its documented maximum).
VINTAGE_PAGE = 10000

_FREQUENCY = {
    "D": "daily",
    "W": "weekly",
    "BW": "weekly",
    "M": "monthly",
    "Q": "quarterly",
    "SA": "quarterly",
    "A": "quarterly",
}

PREFIX = "FRED:"


def _dedupe_vintages(
    parsed: list[tuple[date, date, float]],
) -> list[tuple[date, date, float]]:
    """Collapse repeated issues of an unchanged figure.

    Keeps the first issue and every issue that actually moved the number. A
    "revision" that reprints the same value carries no information and, left in,
    would make the vintage count grow with how the fetch happened to be split.
    """
    by_period: dict[date, list[tuple[date, float]]] = {}
    for obs_date, issued, value in parsed:
        by_period.setdefault(obs_date, []).append((issued, value))

    out: list[tuple[date, date, float]] = []
    for obs_date, issues in by_period.items():
        previous: float | None = None
        for issued, value in sorted(set(issues)):
            if previous is None or value != previous:
                out.append((obs_date, issued, value))
                previous = value
    return out


class FredProvider(Provider):
    id = "fred"
    requires_api_key = True

    def __init__(
        self,
        transport: Transport,
        *,
        api_key: str | None,
        base_url: str = BASE_URL,
        use_vintages: bool = True,
    ) -> None:
        self.transport = transport
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.use_vintages = use_vintages

    def available_series(self) -> list[str]:
        # FRED hosts >800k series; the useful set is whatever series.yaml names,
        # so discovery is the config's job rather than a hard-coded list here.
        return []

    def _require_key(self) -> str:
        if not self.api_key:
            raise AuthError(self.id, "FRED_API_KEY is not set")
        return self.api_key

    @staticmethod
    def _bare(series_id: str) -> str:
        return series_id[len(PREFIX) :] if series_id.startswith(PREFIX) else series_id

    def _get(self, path: str, params: dict[str, str]) -> dict:
        response = self.transport.get(
            f"{self.base_url}/{path}",
            params={**params, "api_key": self._require_key(), "file_type": "json"},
            cache_ttl=3600,
        )
        try:
            payload = response.json()
        except ValueError as err:
            raise ParseError(self.id, f"non-JSON response from {path}: {err}") from err
        if not isinstance(payload, dict):
            raise ParseError(self.id, f"unexpected payload type from {path}")
        return payload

    def fetch_metadata(self, series_id: str) -> SeriesMetadata:
        payload = self._get("series", {"series_id": self._bare(series_id)})
        entries = payload.get("seriess") or []
        if not entries:
            raise NotFoundError(self.id, f"no metadata for {series_id}")
        entry = entries[0]

        short = str(entry.get("frequency_short", "M")).upper()
        frequency = _FREQUENCY.get(short)
        if frequency is None:
            raise ParseError(self.id, f"{series_id}: unsupported frequency {short!r}")

        def parse(value: str | None) -> date | None:
            if not value:
                return None
            try:
                return datetime.strptime(value, "%Y-%m-%d").date()
            except ValueError:
                return None

        return SeriesMetadata(
            series_id=series_id,
            provider=self.id,
            title=str(entry.get("title", series_id)),
            frequency=frequency,  # type: ignore[arg-type]
            unit=str(entry.get("units_short") or entry.get("units") or "unknown"),
            seasonal_adjustment=str(entry.get("seasonal_adjustment_short") or "") or None,
            first_observation=parse(entry.get("observation_start")),
            last_observation=parse(entry.get("observation_end")),
            notes=(str(entry.get("notes")) or None) if entry.get("notes") else None,
        )

    def vintage_dates(self, series_id: str) -> list[date]:
        """Every date on which this series was issued or reissued.

        Needed before the observations call, because the size of the real-time
        window is what FRED rejects on — and it will not tell us how wide a
        window is safe without being asked.
        """
        collected: list[date] = []
        offset = 0
        while True:
            payload = self._get(
                "series/vintagedates",
                {
                    "series_id": self._bare(series_id),
                    "realtime_start": REALTIME_START,
                    "realtime_end": REALTIME_END,
                    "limit": str(VINTAGE_PAGE),
                    "offset": str(offset),
                },
            )
            page = payload.get("vintage_dates") or []
            for raw in page:
                try:
                    collected.append(datetime.strptime(str(raw), "%Y-%m-%d").date())
                except ValueError:
                    continue
            if len(page) < VINTAGE_PAGE:
                break
            offset += VINTAGE_PAGE
        return sorted(set(collected))

    @staticmethod
    def _vintage_windows(vintages: list[date]) -> list[tuple[str, str]]:
        """Real-time windows that each cover at most :data:`MAX_VINTAGE_DATES`.

        Boundaries are actual vintage dates rather than a calendar split, so no
        window can accidentally contain more vintages than intended however
        unevenly a series was revised.

        Windows overlap at their endpoints by one vintage. That is harmless:
        merging is by (period, issue date), so a duplicated row collapses.
        """
        if not vintages:
            return [(REALTIME_START, REALTIME_END)]
        if len(vintages) <= MAX_VINTAGE_DATES:
            return [(REALTIME_START, REALTIME_END)]

        windows: list[tuple[str, str]] = []
        for i in range(0, len(vintages), MAX_VINTAGE_DATES):
            chunk = vintages[i : i + MAX_VINTAGE_DATES]
            first = REALTIME_START if i == 0 else chunk[0].isoformat()
            last = REALTIME_END if i + MAX_VINTAGE_DATES >= len(vintages) else chunk[-1].isoformat()
            windows.append((first, last))
        return windows

    def fetch_observations(
        self,
        series_id: str,
        *,
        start: date | None = None,
        end: date | None = None,
    ) -> list[Observation]:
        metadata = self.fetch_metadata(series_id)

        base: dict[str, str] = {"series_id": self._bare(series_id)}
        if start is not None:
            base["observation_start"] = start.isoformat()
        if end is not None:
            base["observation_end"] = end.isoformat()

        windows: list[tuple[str, str] | None]
        if self.use_vintages:
            windows = list(self._vintage_windows(self.vintage_dates(series_id)))
            if len(windows) > 1:
                log.info("%s: %d vintage windows", series_id, len(windows))
        else:
            windows = [None]

        parsed: list[tuple[date, date, float]] = []
        for window in windows:
            params = dict(base)
            if window is not None:
                params["realtime_start"], params["realtime_end"] = window
            payload = self._get("series/observations", params)
            rows = payload.get("observations")
            if rows is None:
                raise ParseError(self.id, f"{series_id}: response has no observations array")
            parsed.extend(self._parse_rows(rows))

        # Splitting the window clamps each chunk's realtime_start to the chunk
        # boundary, so an unchanged figure reappears looking like a fresh issue.
        # Deduplicating on (period, issue) and then dropping repeats of the same
        # value keeps only the issues that actually changed something.
        parsed = _dedupe_vintages(parsed)

        # First issue per period is the release date; each row keeps its own
        # issue date as the revision date. Taking the minimum across all windows
        # is what makes the chunked fetch equivalent to one unsplit request.
        first_issue: dict[date, date] = {}
        for obs_date, issued, _ in parsed:
            if obs_date not in first_issue or issued < first_issue[obs_date]:
                first_issue[obs_date] = issued

        observations = [
            Observation(
                series_id=series_id,
                provider=self.id,
                frequency=metadata.frequency,
                unit=metadata.unit,
                observation_date=obs_date,
                release_date=first_issue[obs_date],
                revision_date=issued,
                value=value,
            )
            for obs_date, issued, value in parsed
        ]
        observations.sort(key=lambda o: (o.observation_date, o.revision_date or o.release_date))
        return observations

    def _parse_rows(self, rows: list[dict]) -> list[tuple[date, date, float]]:
        parsed: list[tuple[date, date, float]] = []
        for row in rows:
            raw_value = row.get("value")
            # FRED encodes "no data for this period" as a single period.
            if raw_value in (None, ".", ""):
                continue
            try:
                obs_date = datetime.strptime(row["date"], "%Y-%m-%d").date()
                issued = datetime.strptime(
                    row.get("realtime_start") or row["date"], "%Y-%m-%d"
                ).date()
                value = float(raw_value)
            except (KeyError, ValueError):
                continue
            # A vintage stamp earlier than the period itself is a source quirk,
            # not a genuine early publication; clamp so the invariant holds.
            parsed.append((obs_date, max(issued, obs_date), value))
        return parsed
