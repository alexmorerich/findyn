"""LBMA precious metal benchmark prices (the London fix), daily from 1968.

The gold price FinGold is specified on is the **LBMA Gold Price PM** — the
afternoon London benchmark, which is what gold contracts settle against.
``docs/redesign`` and the P4 brief both name it as ``FRED:GOLDPMGBD228NLBM``,
and that series no longer exists: FRED delisted the two LBMA gold series
(``GOLDAMGBD228NLBM`` / ``GOLDPMGBD228NLBM``) along with the rest of the ICE
Benchmark Administration set. Asking FRED for either now returns
``400 The series does not exist`` — verified 2026-08-01, with a working key.

So the benchmark is read from the body that sets it. LBMA publishes the whole
history as static JSON beside the price page, no key and no crawl:

    [{"is_cms_locked": 0, "d": "1968-04-01", "v": [37.7, 15.68, null]}, ...]

``v`` is positional — ``[USD, GBP, EUR]`` — and only USD is taken. The EUR leg is
``null`` for everything before 1999 for the obvious reason, which is a good
reminder that the array is not a mapping and its length says nothing about which
currencies a given row actually carries.

**Vintages: there are none, and there is nothing to reconstruct.** The fix is a
benchmark, not a statistic: it is set at 15:00 London, published immediately and
never revised. Release dates are therefore synthesized from
:data:`PUBLICATION_LAG_DAYS` the way §5.2 prescribes for a source that exposes no
archive — one day, which is conservative rather than exact (the afternoon fix is
public the same afternoon) and agrees with the ``publication_lag_days: 1`` the
series carries in ``series.yaml``.

FINDYN_V1_SPEC.md §5.1 lists no precious-metals source, because v1 was an equity
spec. This adapter is the P4 addition, and it is deliberately shaped like
:mod:`findynamics.data.providers.shiller`: one static document, fetched once
through the transport cache and sliced in memory.
"""

from __future__ import annotations

import logging
from datetime import date, datetime
from typing import Any

from findynamics.data.providers.base import (
    Observation,
    ParseError,
    Provider,
    SeriesMetadata,
    synthesize_release_date,
)
from findynamics.data.providers.resilience import Transport

log = logging.getLogger("findynamics.data.providers.lbma")

BASE_URL = "https://prices.lbma.org.uk/json"

#: The fix is public the afternoon it is set; one day is the conservative read
#: and the one the ``t-1`` information set already assumes.
PUBLICATION_LAG_DAYS = 1

#: Position of the USD leg in each row's ``v`` array.
USD = 0

#: series_id -> (document, title, unit)
_SERIES: dict[str, tuple[str, str, str]] = {
    "LBMA:GOLD_PM": ("gold_pm", "LBMA Gold Price PM (London afternoon fix)", "usd_per_troy_ounce"),
    "LBMA:GOLD_AM": ("gold_am", "LBMA Gold Price AM (London morning fix)", "usd_per_troy_ounce"),
}

#: Below this many rows the document is not the history it claims to be. The
#: gold series holds ~14,600 rows over fifty-eight years, so a response with a
#: few hundred is a truncated or placeholder document rather than a short series
#: — and it would silently move the start of every expanding window fitted on it.
MIN_ROWS = 1000


class LbmaProvider(Provider):
    id = "lbma"
    requires_api_key = False

    def __init__(self, transport: Transport, *, base_url: str = BASE_URL) -> None:
        self.transport = transport
        self.base_url = base_url.rstrip("/")

    def available_series(self) -> list[str]:
        return sorted(_SERIES)

    def _document(self, series_id: str) -> list[dict[str, Any]]:
        self._require_known_series(series_id)
        name, _, _ = _SERIES[series_id]
        # Cached for a day: it is one static file per metal, rewritten once a
        # session, and every series on it is sliced from the same fetch.
        response = self.transport.get(f"{self.base_url}/{name}.json", cache_ttl=86400)
        try:
            payload = response.json()
        except ValueError as err:
            raise ParseError(self.id, f"non-JSON response for {series_id}: {err}") from err
        if not isinstance(payload, list):
            raise ParseError(
                self.id,
                f"{series_id}: expected a JSON array of daily fixes, got {type(payload).__name__}",
            )
        if len(payload) < MIN_ROWS:
            raise ParseError(
                self.id,
                f"{series_id}: document carries {len(payload)} rows, fewer than the "
                f"{MIN_ROWS} a real history has — refusing to truncate the series silently",
            )
        return payload

    @staticmethod
    def _row(row: Any) -> tuple[date, float] | None:
        """One ``{"d": ..., "v": [...]}`` entry, or ``None`` if it carries no USD fix.

        A missing fix is an absent observation rather than a failure: the fix is
        not set on London bank holidays, and the feed carries such days with a
        null leg instead of omitting them.
        """
        if not isinstance(row, dict):
            return None
        values = row.get("v")
        if not isinstance(values, list) or len(values) <= USD:
            return None
        raw = values[USD]
        if raw is None:
            return None
        try:
            observation_date = datetime.strptime(str(row["d"]), "%Y-%m-%d").date()
            value = float(raw)
        except (KeyError, TypeError, ValueError):
            return None
        # A fix of zero is not a price; the feed uses it nowhere, and admitting
        # one would put a -inf log return into the jump detector.
        return (observation_date, value) if value > 0.0 else None

    def fetch_metadata(self, series_id: str) -> SeriesMetadata:
        self._require_known_series(series_id)
        _, title, unit = _SERIES[series_id]
        return SeriesMetadata(
            series_id=series_id,
            provider=self.id,
            title=title,
            frequency="daily",
            unit=unit,
            notes=(
                "London Bullion Market Association benchmark, administered by ICE "
                "Benchmark Administration. Published once per session and never "
                "revised, so release dates are synthesized from a one-day lag."
            ),
        )

    def fetch_observations(
        self,
        series_id: str,
        *,
        start: date | None = None,
        end: date | None = None,
    ) -> list[Observation]:
        _, _, unit = _SERIES[series_id]

        parsed = [
            parsed_row for parsed_row in map(self._row, self._document(series_id)) if parsed_row
        ]
        if not parsed:
            raise ParseError(self.id, f"{series_id}: document held no usable fixes")

        observations = [
            Observation(
                series_id=series_id,
                provider=self.id,
                frequency="daily",
                unit=unit,
                observation_date=observation_date,
                release_date=synthesize_release_date(
                    observation_date, "daily", PUBLICATION_LAG_DAYS
                ),
                # No archive, so nothing to revise against: the release date is
                # the only date this figure ever had.
                revision_date=synthesize_release_date(
                    observation_date, "daily", PUBLICATION_LAG_DAYS
                ),
                value=value,
            )
            for observation_date, value in parsed
            if (start is None or observation_date >= start)
            and (end is None or observation_date <= end)
        ]
        observations.sort(key=lambda o: o.observation_date)
        log.info("%s: %d fixes", series_id, len(observations))
        return observations
