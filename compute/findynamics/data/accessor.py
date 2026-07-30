"""The concrete point-in-time accessor handed to engines.

A thin wrapper over :func:`findynamics.data.pit.pit_join` that binds one
observation frame to one ``as_of`` cutoff. Binding the cutoff at construction is
the whole point: an engine cannot pass a different date, so it cannot see the
future even by mistake (FINDYN_V1_SPEC.md §14.1 rule 2).

Implements :class:`findynamics.core.contracts.pit.PITAccessor` structurally —
``core`` may not import ``data``, so the protocol is satisfied by shape.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from datetime import date, datetime

import pandas as pd

from findynamics.data.pit import pit_join


class PandasPITAccessor:
    """PIT view over a long-format observation frame."""

    def __init__(
        self,
        observations: pd.DataFrame,
        as_of: str | date | datetime | pd.Timestamp,
    ) -> None:
        self._observations = observations
        self._as_of = pd.Timestamp(as_of).normalize().date()

    @property
    def as_of(self) -> date:
        return self._as_of

    def latest(self, series_ids: Iterable[str] | None = None) -> pd.DataFrame:
        """Latest knowable observation per series as of :attr:`as_of`."""
        return pit_join(self._observations, self._as_of, series_ids=series_ids)

    def value(self, series_id: str) -> float | None:
        """Latest knowable value for one series, or ``None`` if unavailable.

        A missing series degrades the feature set rather than the run (§14.2), so
        an absent id and a non-finite value are both reported as ``None``.
        """
        frame = self.latest([series_id])
        if frame.empty or series_id not in frame.index:
            return None
        value = float(frame.at[series_id, "value"])
        return value if math.isfinite(value) else None

    def __repr__(self) -> str:
        return f"PandasPITAccessor(as_of={self._as_of.isoformat()}, rows={len(self._observations)})"
