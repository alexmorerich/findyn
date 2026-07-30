"""Bureau of Labor Statistics public API v2.

BLS publishes no vintage information, so release dates are synthesized from the
end of the period plus a conservative lag (§5.2). The registration key raises
the daily quota; without one the API allows a small number of anonymous calls,
so the adapter treats a missing key as degraded rather than fatal.

FINDYN_V1_SPEC.md §5.1 source 3.
"""

from __future__ import annotations

import logging
from datetime import date

from findynamics.data.providers.base import (
    NotFoundError,
    Observation,
    ParseError,
    Provider,
    ProviderError,
    SeriesMetadata,
    synthesize_release_date,
)
from findynamics.data.providers.resilience import Transport

log = logging.getLogger("findynamics.data.providers.bls")

BASE_URL = "https://api.bls.gov/publicAPI/v2/timeseries/data"

PREFIX = "BLS:"

#: Default lag: BLS releases reference-month data early the following month.
PUBLICATION_LAG_DAYS = 7

_KNOWN: dict[str, tuple[str, str]] = {
    "BLS:CES0500000003": ("Average Hourly Earnings, Total Private", "usd_per_hour"),
    "BLS:LNS14000000": ("Unemployment Rate (16 years and over)", "percent"),
    "BLS:CUUR0000SA0": ("CPI-U, All Items, U.S. City Average", "index"),
}

#: BLS period codes: M01-M12 monthly, Q01-Q04 quarterly, M13/Q05 are annual averages.
_MONTH_FROM_PERIOD = {f"M{m:02d}": m for m in range(1, 13)}
_QUARTER_START_MONTH = {"Q01": 1, "Q02": 4, "Q03": 7, "Q04": 10}


class BlsProvider(Provider):
    id = "bls"
    requires_api_key = False  # optional key; raises quota rather than gating access

    def __init__(
        self,
        transport: Transport,
        *,
        api_key: str | None = None,
        base_url: str = BASE_URL,
        publication_lag_days: int = PUBLICATION_LAG_DAYS,
    ) -> None:
        self.transport = transport
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.publication_lag_days = publication_lag_days

    def available_series(self) -> list[str]:
        return sorted(_KNOWN)

    @staticmethod
    def _bare(series_id: str) -> str:
        return series_id[len(PREFIX) :] if series_id.startswith(PREFIX) else series_id

    def fetch_metadata(self, series_id: str) -> SeriesMetadata:
        title, unit = _KNOWN.get(series_id, (series_id, "unknown"))
        return SeriesMetadata(
            series_id=series_id,
            provider=self.id,
            title=title,
            frequency="monthly",
            unit=unit,
            notes=(
                "BLS public API v2. No vintage data is published; release dates are "
                f"synthesized as period end + {self.publication_lag_days} days."
            ),
        )

    def fetch_observations(
        self,
        series_id: str,
        *,
        start: date | None = None,
        end: date | None = None,
    ) -> list[Observation]:
        params: dict[str, str] = {}
        if start is not None:
            params["startyear"] = str(start.year)
        if end is not None:
            params["endyear"] = str(end.year)
        if self.api_key:
            params["registrationkey"] = self.api_key

        response = self.transport.get(
            f"{self.base_url}/{self._bare(series_id)}", params=params, cache_ttl=3600
        )
        try:
            payload = response.json()
        except ValueError as err:
            raise ParseError(self.id, f"non-JSON response: {err}") from err

        status = payload.get("status")
        if status != "REQUEST_SUCCEEDED":
            messages = "; ".join(payload.get("message") or []) or str(status)
            # BLS reports quota exhaustion in the body with HTTP 200, so the
            # transport's status-code classification never sees it.
            if "threshold" in messages.lower() or "limit" in messages.lower():
                raise ProviderError(self.id, f"quota: {messages}", retryable=True)
            raise ProviderError(self.id, f"request failed: {messages}", retryable=False)

        series_list = (payload.get("Results") or {}).get("series") or []
        if not series_list:
            raise NotFoundError(self.id, f"no data returned for {series_id}")

        metadata = self.fetch_metadata(series_id)
        observations: list[Observation] = []

        for row in series_list[0].get("data") or []:
            period = str(row.get("period", ""))
            try:
                year = int(row["year"])
                value = float(row["value"])
            except (KeyError, TypeError, ValueError):
                continue

            if period in _MONTH_FROM_PERIOD:
                obs_date = date(year, _MONTH_FROM_PERIOD[period], 1)
                frequency = "monthly"
            elif period in _QUARTER_START_MONTH:
                obs_date = date(year, _QUARTER_START_MONTH[period], 1)
                frequency = "quarterly"
            else:
                # M13/Q05 are annual averages, not a period in our model.
                continue

            release = synthesize_release_date(obs_date, frequency, self.publication_lag_days)
            observations.append(
                Observation(
                    series_id=series_id,
                    provider=self.id,
                    frequency=frequency,  # type: ignore[arg-type]
                    unit=metadata.unit,
                    observation_date=obs_date,
                    release_date=release,
                    revision_date=release,
                    value=value,
                )
            )

        if start is not None:
            observations = [o for o in observations if o.observation_date >= start]
        if end is not None:
            observations = [o for o in observations if o.observation_date <= end]
        observations.sort(key=lambda o: o.observation_date)
        return observations
