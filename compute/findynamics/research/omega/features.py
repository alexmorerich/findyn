"""The observable block ``Z_t`` that Ω is inferred from.

Nine candidate columns over six series that are **already declared** in
``config/series.yaml`` — adding a data dependency is out of scope for this
track. The full list, its publication lags and its availability in the committed
fixture are in ``docs/research/kk-omega-design.md`` §3.

Every column is a scale-free causal transform, never a level
--------------------------------------------------------------

A PC1 taken over raw levels is dominated by whichever column has the largest
variance in its own units, which on this block is the index itself — Ω would be
a reparameterized S&P and nothing more. So the columns are volatilities,
absolute returns, drawdown depth, a differenced correlation, yield *changes* and
two levels that are already spreads or indices rather than prices. Windows are
declared in months in ``omega.yaml`` and converted against the series' own
``periods_per_year``, which is the convention ``features/kinematics.py`` uses so
that one config number means one span on a daily and on a monthly path.

Two columns deliberately differ from their RII namesakes
--------------------------------------------------------

``rii.py`` takes credit velocity as ``|Δspread|`` and says why: "instability is
about the size of the change in trend, either way". ``rate_level_chg`` and
``credit_velocity`` here are **signed**, and that is not an oversight. The RII
asks *how unstable is this*, a magnitude question. Ω asks *where in latent state
space is this*, and a widening spread and a narrowing one are different places —
collapsing them would fold the calm half of the coordinate onto the stressed
half before the PCA ever sees it. Anything reading both must not assume the two
``credit_velocity`` columns are the same quantity; they are not.

Release dates, not observation dates
------------------------------------

Each column is computed on its **own observation index**, so rolling windows and
``diff(n)`` count real observations rather than calendar gaps. It is then placed
on the trading calendar at the date it became **knowable** — its
``release_date`` — and forward-filled.

That distinction is worth the machinery. ``PITAccessor.wide()`` indexes by
``obs_date``, and the accessor's release filter is global: it guarantees
``release_date <= as_of`` for the whole frame, not ``release_date <= t`` at each
historical row *t*. For a price series that is immaterial — ``YAHOO:^GSPC``
carries a zero lag, so the two indices coincide. For ``FRED:NFCI`` it is not: its
median lag in the committed fixture is **13 days**, so an obs-date-aligned
liquidity column would put a reading into the feature frame nine trading days
before anyone could have read it. Every historical Ω built on it would be
slightly, invisibly, too well informed.

This is not re-implementing a lag — ``publication_lag_days`` is never consulted
here. It is using the real ``release_date`` the accessor already carries on
every row.

A column built from **two** series takes the **later** of the two release dates,
because it is not knowable until both prints are out. ``dispersion`` is the one
such column, and getting it wrong was a real bug in this phase: it was stamped
with the price's release date alone, and since ``YAHOO:^GSPC`` carries a zero lag
where ``FRED:NASDAQ100`` carries one day, every run published a correlation on
its last row computed from a print that had not been released. One day of
lookahead, on one column, found by the truncation test in
``tests/research/omega/test_leakage.py`` and by nothing else.

Forward-filling stops at ``max_staleness_days``. A weekly series has to survive
the days between its releases, which is the whole point of the fill; a series
that stopped publishing eighteen months ago must not go on contributing its last
value forever, which is the point of the limit.

Availability is checked, never resolved by ``dropna``
-----------------------------------------------------

``FRED:BAMLH0A0HYM2`` exists in the committed fixture from **2023-08-01** — 787
observations against 6,649 for the rates columns. The obvious implementation
builds the frame, drops incomplete rows and fits; that silently starts the
research window in August 2023, produces an Ω with no 2008 and no 2020 in it,
and reports nothing unusual. Every number downstream would then be a statement
about thirty-five months of a bull market.

So :func:`available_columns` tests each column against **both** an absolute
floor and a coverage fraction of the window being fitted. The fraction is the
one that matters: 787 rows clears any sane absolute floor and is still 12% of
the primary window. A column that fails either is dropped and named in
``OmegaSpec.dropped``, never imputed and never zero-filled — zero on a stress
axis means "maximally calm" and absence means nothing of the kind. That is the
rule ``compute_rii`` already applies to its seven components, reused rather than
reinvented, because two implementations of "what to do about a missing input"
eventually disagree.

The window start is a config value. A column never decides it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from findynamics.core.contracts.pit import PITAccessor
from findynamics.engines.equity.prices import PERIODS_PER_YEAR
from findynamics.engines.equity.regime.design import realized_vol
from findynamics.engines.equity.rii import vol_of_vol
from findynamics.research.omega.config import OmegaConfig
from findynamics.research.omega.domain import FEATURE_COLUMNS

log = logging.getLogger("findynamics.research.omega.features")


class OmegaFeatureError(ValueError):
    """Raised when the feature block cannot be built from the data given."""


#: Config key -> the role that key plays. Named here rather than hard-coded at
#: each use so that re-pointing Ω at a different curve slope is a yaml edit,
#: which is the rule ``config/series.yaml`` exists to enforce.
SERIES_ROLES: tuple[str, ...] = (
    "price",
    "dispersion_reference",
    "rate_level",
    "curve_slope",
    "liquidity",
    "credit_spread",
)


@dataclass(frozen=True)
class FeatureParams:
    """Everything configurable about ``Z_t``, from ``omega.yaml``."""

    #: Role -> series id. Only ``price`` is required; a role left unconfigured
    #: means the columns built from it come back empty and are dropped by the
    #: availability check, which is the same path a series outage takes.
    series: dict[str, str]
    #: Observation cadence of the price series, for the months→periods maths.
    frequency: str = "daily"
    #: Earliest date the feature frame is published for. Warm-up is computed
    #: from data *before* this and then discarded, so a rolling window is never
    #: short at the start of the window.
    start: date | None = None
    realized_vol_months: float = 1.0
    vol_of_vol_months: float = 3.0
    abs_return_months: float = 1.0
    dispersion_months: float = 3.0
    dispersion_change_months: float = 1.0
    rate_change_months: float = 1.0
    credit_change_months: float = 1.0
    #: How long a released value may go on standing in for the current one.
    max_staleness_days: int = 21
    #: Candidate columns. Every entry must be a member of ``FEATURE_COLUMNS``.
    columns: tuple[str, ...] = FEATURE_COLUMNS
    #: Absolute floor on a column's knowable rows within the fit window.
    min_column_observations: int = 252
    #: And its share of that window. This is the one that catches the HY OAS.
    min_column_coverage: float = 0.60

    @property
    def periods_per_year(self) -> float:
        try:
            return PERIODS_PER_YEAR[self.frequency]
        except KeyError:
            raise OmegaFeatureError(
                f"unknown frequency {self.frequency!r}; expected one of {sorted(PERIODS_PER_YEAR)}"
            ) from None

    def periods(self, months: float, floor: int = 2) -> int:
        """Months converted to observations of this series' own cadence."""
        return max(int(round(months * self.periods_per_year / 12.0)), floor)

    @classmethod
    def from_config(cls, config: OmegaConfig) -> FeatureParams:
        block = config.block("features")
        series_block = config.block("series")

        unknown_roles = sorted(set(series_block) - set(SERIES_ROLES))
        if unknown_roles:
            raise OmegaFeatureError(
                f"series: unknown role(s) {unknown_roles}; expected {list(SERIES_ROLES)}"
            )
        if not series_block.get("price"):
            raise OmegaFeatureError("series.price is required — Ω has no spine without a price")
        series = {role: str(sid) for role, sid in series_block.items() if sid}

        columns = tuple(block.get("columns") or FEATURE_COLUMNS)
        unknown_columns = sorted(set(columns) - set(FEATURE_COLUMNS))
        if unknown_columns:
            raise OmegaFeatureError(
                f"features.columns: unknown column(s) {unknown_columns}; expected a subset of "
                f"{list(FEATURE_COLUMNS)}"
            )
        if not columns:
            raise OmegaFeatureError("features.columns must name at least one column")

        start_raw = block.get("start")
        start = date.fromisoformat(str(start_raw)) if start_raw else None

        coverage = float(block.get("min_column_coverage", 0.60))
        if not 0.0 < coverage <= 1.0:
            raise OmegaFeatureError(
                f"features.min_column_coverage must be within (0, 1], got {coverage}"
            )
        staleness = int(block.get("max_staleness_days", 21))
        if staleness < 0:
            raise OmegaFeatureError(
                f"features.max_staleness_days must be >= 0, got {staleness} "
                "(a negative staleness budget would be lookahead)"
            )

        # Every window must be positive. `periods()` floors its result at two
        # observations, so a zero or negative month count would not raise — it
        # would silently become a two-day window and the config would be
        # describing a transform that is not running.
        windows = {
            name: float(block.get(name, default))
            for name, default in (
                ("realized_vol_months", 1.0),
                ("vol_of_vol_months", 3.0),
                ("abs_return_months", 1.0),
                ("dispersion_months", 3.0),
                ("dispersion_change_months", 1.0),
                ("rate_change_months", 1.0),
                ("credit_change_months", 1.0),
            )
        }
        for name, months in sorted(windows.items()):
            if months <= 0.0:
                raise OmegaFeatureError(
                    f"features.{name} must be > 0, got {months}; `periods()` floors at two "
                    "observations, so this would run a two-day window under a config that "
                    "claims otherwise"
                )

        return cls(
            series=series,
            frequency=str(block.get("frequency", "daily")),
            start=start,
            **windows,
            max_staleness_days=staleness,
            columns=tuple(str(c) for c in columns),
            min_column_observations=int(block.get("min_column_observations", 252)),
            min_column_coverage=coverage,
        )


def _series_views(long: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Series id -> its knowable history, indexed by ``obs_date``, ascending.

    ``pit_history`` already returns one row per (series, obs_date) carrying the
    newest vintage published by the cutoff, so there is nothing left to
    de-duplicate here.
    """
    if long.empty:
        return {}
    views: dict[str, pd.DataFrame] = {}
    for series_id, group in long.groupby("series_id", sort=True):
        frame = group.set_index("obs_date").sort_index()
        views[str(series_id)] = frame[["release_date", "value"]]
    return views


def _to_calendar(
    values: pd.Series,
    releases: pd.Series,
    calendar: pd.DatetimeIndex,
    *,
    max_staleness_days: int,
) -> pd.Series:
    """Place an obs-date-indexed column on ``calendar`` by its release dates.

    A value becomes visible on the day it was published and stands until the
    next publication or until it goes stale, whichever comes first. Forward only
    — nothing here can move a value backwards in time.
    """
    usable = pd.DataFrame({"value": values, "release": releases.reindex(values.index)})
    usable = usable[np.isfinite(usable["value"].to_numpy(dtype=float))].dropna(subset=["release"])
    if usable.empty:
        return pd.Series(np.nan, index=calendar, dtype=float)

    # Stable sort on the release date alone: rows arrive in obs_date order, so
    # `last()` inside a release date is the newest observation published that
    # day. A batch backfill releases many obs_dates at once and the newest of
    # them is the one a reader would have seen.
    ordered = usable.sort_values("release", kind="stable")
    by_release = ordered.groupby("release")["value"].last().sort_index()

    union = by_release.index.union(calendar)
    filled = by_release.reindex(union).ffill().reindex(calendar)

    stamps = pd.Series(by_release.index, index=by_release.index, dtype="datetime64[ns]")
    last_release = stamps.reindex(union).ffill().reindex(calendar)

    age_days = (calendar.to_numpy() - last_release.to_numpy()) / np.timedelta64(1, "D")
    # NaT before the first release propagates to NaN here, which is the right
    # answer: there was no knowable value at all, not a stale one.
    with np.errstate(invalid="ignore"):
        stale = ~np.isfinite(age_days) | (age_days > float(max_staleness_days))
    return filled.mask(pd.Series(stale, index=calendar)).astype(float)


def _log(values: pd.Series) -> pd.Series:
    """Natural log, with non-positive prints treated as absent rather than -inf."""
    positive = values.where(values > 0.0)
    return np.log(positive)


def build_feature_frame(accessor: PITAccessor, config: OmegaConfig) -> pd.DataFrame:
    """The causal feature block ``Z_t``, one column per configured candidate.

    Read through the PIT accessor and nothing else: the accessor binds the
    information-set cutoff, so no transform here can widen it. Columns whose
    source series the accessor cannot supply come back all-NaN rather than
    missing, so :func:`available_columns` sees them and names them in
    ``OmegaSpec.dropped`` instead of the frame quietly changing width.
    """
    params = FeatureParams.from_config(config)
    ids = sorted(set(params.series.values()))
    views = _series_views(accessor.history(ids))

    price_id = params.series["price"]
    price_view = views.get(price_id)
    if price_view is None or price_view.empty:
        raise OmegaFeatureError(
            f"{price_id}: no knowable observations as of {accessor.as_of}; Ω has no spine"
        )

    log_price = _log(price_view["value"]).dropna()
    if len(log_price) < 2:
        raise OmegaFeatureError(
            f"{price_id}: {len(log_price)} usable close(s) as of {accessor.as_of}; "
            "a return needs two"
        )
    price_releases = price_view["release_date"]
    returns = log_price.diff()
    ppy = params.periods_per_year

    #: The trading calendar every column is published on: the dates the price
    #: itself became knowable. Not `obs_date` — see the module docstring.
    calendar = pd.DatetimeIndex(
        sorted(set(price_releases.reindex(log_price.index).dropna())), name="obs_date"
    )

    raw: dict[str, tuple[pd.Series, pd.Series]] = {}

    if "realized_vol" in params.columns:
        raw["realized_vol"] = (
            realized_vol(returns, periods_per_year=ppy, months=params.realized_vol_months),
            price_releases,
        )
    if "vol_of_vol" in params.columns:
        inner = realized_vol(returns, periods_per_year=ppy, months=params.realized_vol_months)
        raw["vol_of_vol"] = (
            vol_of_vol(inner, periods_per_year=ppy, months=params.vol_of_vol_months),
            price_releases,
        )
    if "abs_return" in params.columns:
        # EWMA rather than a box window: an absolute return is spiky, and the
        # exponential kernel is the one that puts most of its weight on the
        # recent past without a hard edge falling out of the window. `adjust`
        # is False so the value at t depends only on the recursion, which makes
        # a truncated run and a full run agree exactly on the shared dates.
        span = params.periods(params.abs_return_months)
        raw["abs_return"] = (returns.abs().ewm(span=span, adjust=False).mean(), price_releases)
    if "drawdown" in params.columns:
        # Depth below the running maximum, in log points and non-negative, so a
        # larger number is a deeper drawdown. `cummax` is the expanding maximum
        # and sees only rows 0..t; a `rolling(...).max()` would be a different
        # (and shorter-memoried) statement, and a global max would be lookahead.
        raw["drawdown"] = (log_price.cummax() - log_price, price_releases)

    if "dispersion" in params.columns:
        raw["dispersion"] = _dispersion(views, params, returns, price_releases)

    for column, role, months, differenced in (
        ("rate_level_chg", "rate_level", params.rate_change_months, True),
        ("curve_slope", "curve_slope", None, False),
        ("liquidity_stress", "liquidity", None, False),
        ("credit_velocity", "credit_spread", params.credit_change_months, True),
    ):
        if column not in params.columns:
            continue
        raw[column] = _from_role(views, params, role, months=months, differenced=differenced)

    columns: dict[str, pd.Series] = {
        name: _to_calendar(values, releases, calendar, max_staleness_days=params.max_staleness_days)
        for name, (values, releases) in raw.items()
    }

    frame = pd.DataFrame(columns, index=calendar)
    # Ordered by the vocabulary, not by construction order, so two runs that
    # built the same columns produce byte-identical CSVs.
    frame = frame[[c for c in FEATURE_COLUMNS if c in frame.columns]]
    frame.index.name = "obs_date"

    if params.start is not None:
        frame = frame.loc[pd.Timestamp(params.start) :]

    if frame.empty:
        raise OmegaFeatureError(
            f"the feature block is empty as of {accessor.as_of}"
            + (f" for a window starting {params.start}" if params.start else "")
        )

    log.info(
        "omega features: %d rows %s → %s over %d candidate column(s); knowable rows per column: %s",
        len(frame),
        frame.index[0].date(),
        frame.index[-1].date(),
        len(frame.columns),
        ", ".join(f"{c}={int(frame[c].notna().sum())}" for c in frame.columns),
    )
    return frame


def _from_role(
    views: dict[str, pd.DataFrame],
    params: FeatureParams,
    role: str,
    *,
    months: float | None,
    differenced: bool,
) -> tuple[pd.Series, pd.Series]:
    """One column read straight off a configured series, optionally differenced.

    A role that is not configured, or a series the accessor could not supply,
    yields an empty pair. That becomes an all-NaN column, which the availability
    check drops and names — the same path a live provider outage would take.
    """
    series_id = params.series.get(role)
    view = views.get(series_id) if series_id else None
    if view is None or view.empty:
        empty = pd.Series(dtype=float)
        return empty, empty

    values = view["value"].astype(float)
    if differenced and months is not None:
        # Signed, unlike `rii.rate_of_change` — see the module docstring.
        values = values.diff(params.periods(months))
    return values, view["release_date"]


def _dispersion(
    views: dict[str, pd.DataFrame],
    params: FeatureParams,
    returns: pd.Series,
    price_releases: pd.Series,
) -> tuple[pd.Series, pd.Series]:
    """Change in the rolling correlation between the price and its reference.

    The *level* of the correlation is close to constant across decades; its
    movement is not, and a breakdown in co-movement is the thing worth carrying
    into a latent-state coordinate. Differenced for that reason, following the
    same logic ``rii.correlation_breakdown`` uses when it publishes the
    deviation from a baseline rather than the correlation itself.
    """
    reference_id = params.series.get("dispersion_reference")
    view = views.get(reference_id) if reference_id else None
    if view is None or view.empty:
        empty = pd.Series(dtype=float)
        return empty, empty

    reference_returns = _log(view["value"]).dropna().diff()
    # sort=True explicitly: the two indices rarely share a calendar exactly, and
    # pandas is deprecating the implicit sort — an unsorted union would put the
    # rolling correlation out of date order. `rii.correlation_breakdown` makes
    # the same call for the same reason.
    aligned = pd.concat([returns, reference_returns], axis=1, sort=True).dropna()
    window = params.periods(params.dispersion_months)
    if len(aligned) < window:
        empty = pd.Series(dtype=float)
        return empty, empty

    rolling = aligned.iloc[:, 0].rolling(window=window, min_periods=window).corr(aligned.iloc[:, 1])
    change = rolling.diff(params.periods(params.dispersion_change_months))

    # The release date of a two-series column is the LATER of the two
    # contributing releases: a correlation between today's S&P return and
    # today's NASDAQ return is not knowable until both prints are out, and
    # ``FRED:NASDAQ100`` carries a one-day lag where ``YAHOO:^GSPC`` carries
    # none. Stamping this column with the price's release date alone was a real
    # bug, and the KK1 truncation test is what found it: the last row of a run
    # truncated at T disagreed with the same row seen from later, because the
    # truncated run had correctly not yet seen the reference print and the
    # comparison run had.
    #
    # np.maximum rather than DataFrame.max(axis=1): the latter skips NaN by
    # default, which would silently fall back to whichever leg *was* published
    # and reintroduce the same bug for any date the other leg is missing.
    price_leg = price_releases.reindex(change.index).to_numpy()
    reference_leg = view["release_date"].reindex(change.index).to_numpy()
    releases = pd.Series(np.maximum(price_leg, reference_leg), index=change.index)
    return change, releases


def available_columns(
    frame: pd.DataFrame,
    min_observations: int,
    min_coverage: float = 0.0,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """``(surviving, dropped)`` column names, both sorted.

    Sorted rather than in insertion order because the surviving set becomes
    ``OmegaSpec.columns`` and is dotted with a loading vector; a set or a dict
    iteration order would make the projection depend on how the frame happened
    to be assembled.

    Both tests apply. The absolute floor rejects a column that is short in any
    terms; the coverage fraction rejects one that is long enough on its own and
    still describes only a corner of the window — which is exactly the shape of
    ``FRED:BAMLH0A0HYM2`` against the 2000-2026 primary window, at 787 rows and
    12% coverage.
    """
    if not len(frame.columns):
        return (), ()
    rows = len(frame)
    counts = frame.notna().sum()

    surviving: list[str] = []
    dropped: list[str] = []
    for column in frame.columns:
        count = int(counts[column])
        coverage = (count / rows) if rows else 0.0
        if count >= min_observations and coverage >= min_coverage:
            surviving.append(str(column))
        else:
            dropped.append(str(column))

    if dropped:
        log.info(
            "omega features: dropping %s over %d row(s) — below %d observations or %.0f%% "
            "coverage; the surviving block is %s",
            ", ".join(f"{c} ({int(counts[c])} rows)" for c in sorted(dropped)),
            rows,
            min_observations,
            min_coverage * 100.0,
            ", ".join(sorted(surviving)) or "empty",
        )
    return tuple(sorted(surviving)), tuple(sorted(dropped))


__all__ = [
    "SERIES_ROLES",
    "FeatureParams",
    "OmegaFeatureError",
    "available_columns",
    "build_feature_frame",
]
