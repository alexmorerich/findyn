"""Point-in-time joins — the no-lookahead choke point.

FINDYN_V1_SPEC.md §14.1 rule 2: this module is the ONLY sanctioned way to read
``macro_series`` for feature computation. Reading that table anywhere else
bypasses the release-date filter and silently contaminates every downstream
backtest, because a macro value carries the date it *describes* (``obs_date``)
and the date it first became *knowable* (``release_date``), and those can differ
by two months.

The join is deliberately pure: it takes an observation frame and returns what
was knowable at ``as_of``. Fetching from D1 is a separate concern (M1/M2) so
that this logic stays trivially testable.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import date, datetime

import pandas as pd

REQUIRED_COLUMNS = ("series_id", "obs_date", "release_date", "value")


class LookaheadError(AssertionError):
    """Raised when data that could not have been known at ``as_of`` is present."""


def _to_date(value: str | date | datetime | pd.Timestamp) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    return ts.normalize()


def _validate(observations: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in REQUIRED_COLUMNS if c not in observations.columns]
    if missing:
        raise ValueError(f"observations is missing required columns: {missing}")

    frame = observations.copy()
    frame["obs_date"] = pd.to_datetime(frame["obs_date"]).dt.normalize()
    frame["release_date"] = pd.to_datetime(frame["release_date"]).dt.normalize()

    # A value released before the period it describes is a data error, not a scoop.
    early = frame["release_date"] < frame["obs_date"]
    if early.any():
        offenders = frame.loc[early, ["series_id", "obs_date", "release_date"]].head(5)
        raise LookaheadError(
            "release_date precedes obs_date for "
            f"{int(early.sum())} row(s); first offenders:\n{offenders.to_string(index=False)}"
        )

    frame["knowable_at"] = _knowable_at(frame)
    return frame


def _knowable_at(frame: pd.DataFrame) -> pd.Series:
    """When each individual figure became knowable.

    ``release_date`` says when the *period* first became observable and is
    therefore constant across revisions of that period; ``revision_date`` says
    when *this* figure was issued. Filtering on release_date alone would admit a
    later revision the moment the original print was published — a value nobody
    could have seen. Sources that publish no vintage carry no revision_date, and
    for them the two dates coincide.
    """
    released = frame["release_date"]
    if "revision_date" not in frame.columns:
        return released
    revised = pd.to_datetime(frame["revision_date"]).dt.normalize()
    return revised.fillna(released).where(lambda s: s >= released, released)


def pit_join(
    observations: pd.DataFrame,
    as_of: str | date | datetime | pd.Timestamp,
    *,
    series_ids: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Return the latest observation per series that was knowable at ``as_of``.

    Args:
        observations: long-format frame with columns ``series_id``, ``obs_date``,
            ``release_date``, ``value`` and optionally ``revision_date``/``vintage``.
        as_of: the information-set cutoff. Figures not yet knowable at that date
            are dropped outright — they did not exist yet.
        series_ids: optional restriction; series absent from the data are simply
            not returned rather than raising, so a provider outage degrades the
            feature set instead of the run (§14.2).

    Returns:
        One row per surviving series, indexed by ``series_id``, carrying the
        observation's ``obs_date``, ``release_date``, ``value`` and the
        ``staleness_days`` between ``obs_date`` and ``as_of``.
    """
    cutoff = _to_date(as_of)
    knowable = _knowable(observations, cutoff, series_ids)
    if knowable.empty:
        return pd.DataFrame(
            columns=["obs_date", "release_date", "value", "staleness_days"]
        ).rename_axis("series_id")

    # Newest period first; among revisions of the same period, the newest vintage.
    ordered = knowable.sort_values(
        ["series_id", "obs_date", "knowable_at"], ascending=[True, False, False]
    )
    latest = ordered.groupby("series_id", sort=True).head(1).set_index("series_id")

    result = latest[["obs_date", "release_date", "value"]].copy()
    result["staleness_days"] = (cutoff - result["obs_date"]).dt.days

    assert_no_lookahead(result, cutoff)
    return result


def _knowable(
    observations: pd.DataFrame,
    cutoff: pd.Timestamp,
    series_ids: Iterable[str] | None,
) -> pd.DataFrame:
    """Validated rows that had been published on or before ``cutoff``."""
    frame = _validate(observations)
    if series_ids is not None:
        frame = frame[frame["series_id"].isin(set(series_ids))]
    return frame[frame["knowable_at"] <= cutoff]


def pit_history(
    observations: pd.DataFrame,
    as_of: str | date | datetime | pd.Timestamp,
    *,
    series_ids: Iterable[str] | None = None,
    start: str | date | datetime | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """The whole knowable history, not just its last point.

    :func:`pit_join` answers "what is the current value"; a model that fits an
    expanding window needs "what did the series look like, as seen from
    ``as_of``". Same filter, same vintage rule — one row per (series, period),
    carrying the newest figure that had been published by the cutoff.

    Returns a long frame sorted by ``series_id`` then ``obs_date`` ascending,
    with columns ``series_id``, ``obs_date``, ``release_date`` and ``value``.
    Empty input yields an empty frame of that shape rather than raising.
    """
    cutoff = _to_date(as_of)
    knowable = _knowable(observations, cutoff, series_ids)
    if start is not None:
        knowable = knowable[knowable["obs_date"] >= _to_date(start)]

    columns = ["series_id", "obs_date", "release_date", "value"]
    if knowable.empty:
        return pd.DataFrame(columns=columns)

    ordered = knowable.sort_values(
        ["series_id", "obs_date", "knowable_at"], ascending=[True, True, False]
    )
    newest = ordered.groupby(["series_id", "obs_date"], sort=True).head(1)

    result = newest[columns].sort_values(["series_id", "obs_date"]).reset_index(drop=True)
    assert_no_lookahead(result, cutoff)
    return result


def assert_no_lookahead(frame: pd.DataFrame, as_of: str | date | datetime | pd.Timestamp) -> None:
    """Fail loudly if any row could not have been known at ``as_of``.

    Called on the way out of :func:`pit_join` and reusable as a guard anywhere a
    frame crosses into model-fitting code.
    """
    cutoff = _to_date(as_of)
    if "release_date" not in frame.columns:
        raise ValueError("frame has no release_date column to check")

    released = pd.to_datetime(frame["release_date"]).dt.normalize()
    violations = frame[released > cutoff]
    if not violations.empty:
        raise LookaheadError(
            f"{len(violations)} row(s) have release_date after as_of={cutoff.date()}:\n"
            f"{violations.head(5).to_string()}"
        )


def synthesize_release_date(
    obs_date: str | date | datetime | pd.Timestamp, publication_lag_days: int
) -> pd.Timestamp:
    """Conservative release date for sources that expose no vintage (§5.2).

    Deliberately biased late: over-stating the lag costs a little responsiveness,
    under-stating it manufactures lookahead.
    """
    if publication_lag_days < 0:
        raise ValueError("publication_lag_days must be >= 0 (negative lag is lookahead)")
    return _to_date(obs_date) + pd.Timedelta(days=publication_lag_days)
