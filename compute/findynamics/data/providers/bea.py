"""Bureau of Economic Analysis API (NIPA tables).

Supplies corporate profits and other national-accounts aggregates feeding the
Earnings force. BEA publishes quarterly with a long lag and revises heavily, but
exposes no vintage through this API, so release dates are synthesized from the
end of the quarter (§5.2) — deliberately generous, because a corporate-profits
figure that arrives two months late but is treated as same-quarter knowledge is
exactly the kind of leak point-in-time discipline exists to prevent.

FINDYN_V1_SPEC.md §5.1 source 4.
"""

from __future__ import annotations

import logging
from datetime import date

from findynamics.data.providers.base import (
    AuthError,
    NotFoundError,
    Observation,
    ParseError,
    Provider,
    ProviderError,
    SeriesMetadata,
    synthesize_release_date,
)
from findynamics.data.providers.resilience import Transport

log = logging.getLogger("findynamics.data.providers.bea")

BASE_URL = "https://apps.bea.gov/api/data"

PREFIX = "BEA:"

#: Publication lag for NIPA quarterly releases (advance estimate ~1 month after
#: quarter end; later vintages revise it, which this API does not expose).
PUBLICATION_LAG_DAYS = 60

#: series_id -> (title, unit, NIPA table, line number, frequency)
_SERIES: dict[str, tuple[str, str, str, int, str]] = {
    "BEA:CORPORATE_PROFITS": (
        "Corporate Profits with IVA and CCAdj",
        "usd_billions",
        "T11200",
        13,
        "Q",
    ),
    "BEA:GDP": ("Gross Domestic Product", "usd_billions", "T10105", 1, "Q"),
}

_QUARTER_START_MONTH = {"Q1": 1, "Q2": 4, "Q3": 7, "Q4": 10}


class BeaProvider(Provider):
    id = "bea"
    requires_api_key = True

    def __init__(
        self,
        transport: Transport,
        *,
        api_key: str | None,
        base_url: str = BASE_URL,
        publication_lag_days: int = PUBLICATION_LAG_DAYS,
    ) -> None:
        self.transport = transport
        self.api_key = api_key
        self.base_url = base_url
        self.publication_lag_days = publication_lag_days

    def available_series(self) -> list[str]:
        return sorted(_SERIES)

    def _require_key(self) -> str:
        if not self.api_key:
            raise AuthError(self.id, "BEA_API_KEY is not set")
        return self.api_key

    def fetch_metadata(self, series_id: str) -> SeriesMetadata:
        self._require_known_series(series_id)
        title, unit, table, line, _ = _SERIES[series_id]
        return SeriesMetadata(
            series_id=series_id,
            provider=self.id,
            title=title,
            frequency="quarterly",
            unit=unit,
            seasonal_adjustment="seasonally_adjusted_annual_rate",
            notes=(
                f"BEA NIPA table {table}, line {line}. No vintage data is exposed; "
                f"release dates are synthesized as quarter end + {self.publication_lag_days} days."
            ),
        )

    @staticmethod
    def _parse_period(raw: str) -> date | None:
        """BEA encodes quarters as ``2024Q1``."""
        if len(raw) != 6 or raw[4] != "Q":
            return None
        quarter = raw[4:]
        month = _QUARTER_START_MONTH.get(quarter)
        if month is None:
            return None
        try:
            return date(int(raw[:4]), month, 1)
        except ValueError:
            return None

    def fetch_observations(
        self,
        series_id: str,
        *,
        start: date | None = None,
        end: date | None = None,
    ) -> list[Observation]:
        self._require_known_series(series_id)
        _, unit, table, line, frequency_code = _SERIES[series_id]

        response = self.transport.get(
            self.base_url,
            params={
                "UserID": self._require_key(),
                "method": "GetData",
                "datasetname": "NIPA",
                "TableName": table,
                "Frequency": frequency_code,
                "Year": "ALL",
                "ResultFormat": "JSON",
            },
            cache_ttl=6 * 3600,
        )
        try:
            payload = response.json()
        except ValueError as err:
            raise ParseError(self.id, f"non-JSON response: {err}") from err

        results = (payload.get("BEAAPI") or {}).get("Results") or {}
        # BEA reports errors inside a 200 response.
        if "Error" in results or "Error" in (payload.get("BEAAPI") or {}):
            detail = results.get("Error") or payload["BEAAPI"]["Error"]
            raise ProviderError(self.id, f"API error: {detail}", retryable=False)

        rows = results.get("Data")
        if not rows:
            raise NotFoundError(self.id, f"no data rows for {series_id} (table {table})")

        observations: list[Observation] = []
        for row in rows:
            if str(row.get("LineNumber")) != str(line):
                continue
            obs_date = self._parse_period(str(row.get("TimePeriod", "")))
            if obs_date is None:
                continue
            try:
                value = float(str(row.get("DataValue", "")).replace(",", ""))
            except ValueError:
                continue
            release = synthesize_release_date(obs_date, "quarterly", self.publication_lag_days)
            observations.append(
                Observation(
                    series_id=series_id,
                    provider=self.id,
                    frequency="quarterly",
                    unit=unit,
                    observation_date=obs_date,
                    release_date=release,
                    revision_date=release,
                    value=value,
                )
            )

        if not observations:
            raise NotFoundError(self.id, f"table {table} has no line {line}")

        if start is not None:
            observations = [o for o in observations if o.observation_date >= start]
        if end is not None:
            observations = [o for o in observations if o.observation_date <= end]
        observations.sort(key=lambda o: o.observation_date)
        return observations
