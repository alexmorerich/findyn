"""Deterministic synthetic provider.

Exists so the resilience layer, quality engine and job wiring can be exercised
without a network. Its output is fabricated, and fabricated numbers that reach a
dashboard are indistinguishable from measurements — so every series id is
prefixed ``MOCK:`` and :func:`findynamics.data.providers.registry.build_provider` refuses
to construct it unless a caller explicitly opts in.
"""

from __future__ import annotations

import math
from datetime import date, timedelta

from findynamics.data.providers.base import (
    Observation,
    Provider,
    SeriesMetadata,
    synthesize_release_date,
)

MOCK_PREFIX = "MOCK:"

_SERIES: dict[str, tuple[str, str, str, int, float, float]] = {
    # series_id: (title, frequency, unit, publication_lag_days, level, amplitude)
    "MOCK:CPI": ("Mock Consumer Price Index", "monthly", "index", 14, 300.0, 3.0),
    "MOCK:POLICY_RATE": ("Mock Policy Rate", "monthly", "percent", 1, 4.0, 0.5),
    "MOCK:INDEX": ("Mock Equity Index", "daily", "index", 0, 5000.0, 120.0),
}


def _advance(day: date, frequency: str) -> date:
    if frequency == "daily":
        return day + timedelta(days=1)
    if frequency == "weekly":
        return day + timedelta(days=7)
    if frequency == "quarterly":
        month = day.month + 3
        return date(day.year + (month - 1) // 12, (month - 1) % 12 + 1, 1)
    month = day.month + 1
    return date(day.year + (month - 1) // 12, (month - 1) % 12 + 1, 1)


class MockProvider(Provider):
    id = "mock"
    requires_api_key = False

    def __init__(
        self,
        *,
        start: date = date(2020, 1, 1),
        end: date | None = None,
    ) -> None:
        self.start = start
        self.end = end or date(2026, 1, 1)

    def available_series(self) -> list[str]:
        return sorted(_SERIES)

    def fetch_metadata(self, series_id: str) -> SeriesMetadata:
        self._require_known_series(series_id)
        title, frequency, unit, _, _, _ = _SERIES[series_id]
        return SeriesMetadata(
            series_id=series_id,
            provider=self.id,
            title=title,
            frequency=frequency,  # type: ignore[arg-type]
            unit=unit,
            notes="SYNTHETIC DATA — deterministic fixture, not a measurement of anything.",
        )

    def fetch_observations(
        self,
        series_id: str,
        *,
        start: date | None = None,
        end: date | None = None,
    ) -> list[Observation]:
        self._require_known_series(series_id)
        title, frequency, unit, lag, level, amplitude = _SERIES[series_id]

        observations: list[Observation] = []
        cursor = self.start
        step = 0
        while cursor <= self.end:
            # Smooth deterministic wave plus drift — no RNG, so fixtures are stable.
            value = level + amplitude * math.sin(step / 12.0) + step * 0.05
            observations.append(
                Observation(
                    series_id=series_id,
                    provider=self.id,
                    frequency=frequency,  # type: ignore[arg-type]
                    unit=unit,
                    observation_date=cursor,
                    release_date=synthesize_release_date(cursor, frequency, lag),
                    revision_date=synthesize_release_date(cursor, frequency, lag),
                    value=round(value, 6),
                )
            )
            cursor = _advance(cursor, frequency)
            step += 1

        if start is not None:
            observations = [o for o in observations if o.observation_date >= start]
        if end is not None:
            observations = [o for o in observations if o.observation_date <= end]
        return observations
