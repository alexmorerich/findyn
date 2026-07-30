"""The point-in-time accessor interface.

FINDYN_V1_SPEC.md §14.1 rule 2 makes ``pit_join`` the only sanctioned way to read
``macro_series``. This protocol is how that rule becomes structural rather than
cultural: an engine is handed a ``PITAccessor`` inside its ``WorldState`` and has
no other route to data, so lookahead is impossible to write by accident.

The protocol lives in ``core`` and the pandas-backed implementation lives in
:mod:`findynamics.data.accessor` — ``core`` may not import ``data`` (§3 rule 1),
and structural typing means it does not have to.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import date
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class PITAccessor(Protocol):
    """Read-only view of the observation store, clamped to an information set."""

    @property
    def as_of(self) -> date:
        """The information-set cutoff. Nothing released after this is visible."""
        ...

    def latest(self, series_ids: Iterable[str] | None = None) -> Any:
        """Latest knowable observation per series as of :attr:`as_of`.

        Returns the frame produced by ``data.pit.pit_join`` — indexed by
        ``series_id`` with ``obs_date``, ``release_date``, ``value`` and
        ``staleness_days``. Series with no knowable observation are omitted
        rather than raising, so a provider outage degrades the feature set
        instead of the run (§14.2).
        """
        ...

    def value(self, series_id: str) -> float | None:
        """Latest knowable value for one series, or ``None`` if unavailable."""
        ...
