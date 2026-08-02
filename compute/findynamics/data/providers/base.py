"""Provider abstraction and the canonical financial data model.

Every external source — FRED, Shiller, Stooq, BLS, BEA — is reduced to the same
two shapes before it reaches anything else: :class:`SeriesMetadata` describes a
series, :class:`Observation` is one normalized data point. Compute jobs consume
only these, so no provider-specific quirk (a float-encoded date, a nested JSON
envelope, a footnote row) can leak upward.

FINDYN_V1_SPEC.md §5.1.
"""

from __future__ import annotations

import calendar
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Literal

Frequency = Literal["daily", "weekly", "monthly", "quarterly"]

FREQUENCIES: frozenset[str] = frozenset({"daily", "weekly", "monthly", "quarterly"})

#: Nominal spacing between observations, used by the quality engine to spot gaps.
FREQUENCY_DAYS: dict[str, int] = {
    "daily": 1,
    "weekly": 7,
    "monthly": 31,
    "quarterly": 92,
}


class ProviderError(Exception):
    """Base class for every provider failure.

    ``retryable`` decides whether the resilience layer should back off and try
    again or give up immediately. Retrying a bad API key just burns quota.

    ``status_code`` carries the HTTP status where one exists, so an adapter can
    react to a specific upstream rejection (FRED answers 400, not 404, for a
    series that exists in FRED but has no ALFRED vintage archive).
    """

    def __init__(
        self,
        provider: str,
        message: str,
        *,
        retryable: bool = True,
        status_code: int | None = None,
    ) -> None:
        super().__init__(f"[{provider}] {message}")
        self.provider = provider
        self.retryable = retryable
        self.status_code = status_code


class RateLimitError(ProviderError):
    """Upstream signalled quota exhaustion (HTTP 429 or a documented quota message)."""

    def __init__(self, provider: str, message: str, *, retry_after: float | None = None) -> None:
        super().__init__(provider, message, retryable=True)
        self.retry_after = retry_after


class AuthError(ProviderError):
    """Missing or rejected credentials. Never retried."""

    def __init__(self, provider: str, message: str) -> None:
        super().__init__(provider, message, retryable=False)


class NotFoundError(ProviderError):
    """The series does not exist at this provider. Never retried."""

    def __init__(self, provider: str, message: str) -> None:
        super().__init__(provider, message, retryable=False)


class BotChallengeError(ProviderError):
    """The endpoint served an interstitial challenge instead of data.

    Distinct from a transport error: the request succeeded, the payload is just
    not the data we asked for. Retrying from the same host will not help, so the
    circuit should open and a fallback source take over.
    """

    def __init__(self, provider: str, message: str) -> None:
        super().__init__(provider, message, retryable=False)


class ParseError(ProviderError):
    """The payload arrived but did not have the expected shape."""

    def __init__(self, provider: str, message: str) -> None:
        super().__init__(provider, message, retryable=False)


@dataclass(frozen=True)
class SeriesMetadata:
    """Descriptive half of the canonical model."""

    series_id: str
    provider: str
    title: str
    frequency: Frequency
    unit: str
    seasonal_adjustment: str | None = None
    first_observation: date | None = None
    last_observation: date | None = None
    notes: str | None = None

    def to_wire(self) -> dict[str, object]:
        return {
            "series_id": self.series_id,
            "provider": self.provider,
            "title": self.title,
            "frequency": self.frequency,
            "unit": self.unit,
            "seasonal_adjustment": self.seasonal_adjustment,
            "first_observation": self.first_observation.isoformat()
            if self.first_observation
            else None,
            "last_observation": self.last_observation.isoformat()
            if self.last_observation
            else None,
            "notes": self.notes,
        }


@dataclass(frozen=True)
class Observation:
    """One normalized data point.

    Three dates, deliberately kept apart:

    ``observation_date``
        the period the value describes.
    ``release_date``
        when a figure for that period first became public. This is the field
        point-in-time logic filters on (§14.1).
    ``revision_date``
        when *this particular* figure was issued. Equals ``release_date`` for an
        original print; later for a revision. Keeping both means a revision does
        not retroactively change when the period became observable.
    """

    series_id: str
    provider: str
    frequency: Frequency
    unit: str
    observation_date: date
    release_date: date
    value: float
    revision_date: date | None = None
    #: Set when the quality engine flagged THIS observation — currently only
    #: ``abnormal_jump``. Carried rather than acted on: the value is stored and
    #: served either way, and a consumer decides whether to exclude or
    #: down-weight it. The alternative, dropping the row, would delete March 2020
    #: and September 2008 from a system built to study them.
    quality_flag: str | None = None

    def __post_init__(self) -> None:
        if self.release_date < self.observation_date:
            raise ValueError(
                f"{self.series_id}: release_date {self.release_date} precedes "
                f"observation_date {self.observation_date} — that would license lookahead"
            )
        if self.revision_date is not None and self.revision_date < self.release_date:
            raise ValueError(
                f"{self.series_id}: revision_date {self.revision_date} precedes "
                f"release_date {self.release_date}"
            )

    def to_wire(self) -> dict[str, object]:
        """Shape accepted by the serving plane's admin write-back endpoint."""
        return {
            "series_id": self.series_id,
            "obs_date": self.observation_date.isoformat(),
            "release_date": self.release_date.isoformat(),
            "revision_date": self.revision_date.isoformat() if self.revision_date else None,
            "value": float(self.value),
            "source": self.provider,
            "quality_flag": self.quality_flag,
        }


@dataclass
class FetchResult:
    """What a provider hands back for one series."""

    metadata: SeriesMetadata
    observations: list[Observation] = field(default_factory=list)
    #: True when the data came from cache because the live source was unavailable.
    from_cache: bool = False


def month_end(day: date) -> date:
    """Last calendar day of ``day``'s month."""
    return day.replace(day=calendar.monthrange(day.year, day.month)[1])


def period_end(observation_date: date, frequency: str) -> date:
    """Last day covered by an observation period.

    A monthly figure dated 2025-01-01 describes all of January, so it cannot be
    published before 2025-01-31 no matter how short the stated lag.
    """
    if frequency == "monthly":
        return month_end(observation_date)
    if frequency == "quarterly":
        quarter_end_month = ((observation_date.month - 1) // 3) * 3 + 3
        return month_end(observation_date.replace(month=quarter_end_month, day=1))
    if frequency == "weekly":
        return observation_date + timedelta(days=6)
    return observation_date


def synthesize_release_date(
    observation_date: date, frequency: str, publication_lag_days: int
) -> date:
    """Conservative release date for sources that publish no vintage (§5.2).

    Measured from the end of the period, not its start. Biased late on purpose:
    over-stating the lag costs a little responsiveness, under-stating it
    manufactures lookahead.
    """
    if publication_lag_days < 0:
        raise ValueError("publication_lag_days must be >= 0 (negative lag is lookahead)")
    return period_end(observation_date, frequency) + timedelta(days=publication_lag_days)


class Provider(ABC):
    """Interface every source adapter implements.

    Deliberately narrow. Anything a caller might want to special-case per
    provider — auth, pagination, retries, response format — belongs behind it.
    """

    #: Stable identifier, matching the values allowed in compute/config/series.yaml.
    id: str = "abstract"
    #: Whether the adapter needs a credential to return real data.
    requires_api_key: bool = False

    @abstractmethod
    def available_series(self) -> list[str]:
        """Series ids this adapter can serve without further discovery."""

    @abstractmethod
    def fetch_metadata(self, series_id: str) -> SeriesMetadata:
        """Describe one series."""

    @abstractmethod
    def fetch_observations(
        self,
        series_id: str,
        *,
        start: date | None = None,
        end: date | None = None,
    ) -> list[Observation]:
        """Return normalized observations, ascending by ``observation_date``."""

    def fetch(
        self,
        series_id: str,
        *,
        start: date | None = None,
        end: date | None = None,
    ) -> FetchResult:
        """Metadata and observations together — the usual entry point for jobs."""
        observations = self.fetch_observations(series_id, start=start, end=end)
        metadata = self.fetch_metadata(series_id)
        if observations:
            metadata = SeriesMetadata(
                **{
                    **metadata.__dict__,
                    "first_observation": observations[0].observation_date,
                    "last_observation": observations[-1].observation_date,
                }
            )
        return FetchResult(metadata=metadata, observations=observations)

    def _require_known_series(self, series_id: str) -> None:
        if series_id not in self.available_series():
            raise NotFoundError(
                self.id,
                f"unknown series {series_id!r}; available: {', '.join(self.available_series())}",
            )
