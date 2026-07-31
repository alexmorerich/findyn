"""Which price series plays which part — the one place roles become series.

``series.yaml`` names its equity series by *where the data comes from*
(``primary``, ``backfill``, ``regime_proxy``, ``deep_history``). The rest of this
engine names them by *what they are for*:

===============  =====================================================
publication      the index the published state describes — always ``primary``
calibration      the daily series long enough to fit a regime model on
deep_history     the monthly 1871+ series, the only basis for tail work
===============  =====================================================

Two vocabularies on purpose, and this module is the seam. Nothing downstream
reads a ``series.yaml`` role, so re-pointing a role at a different source is a
yaml edit plus a re-fit, not a code change.

Why ``calibration`` is not simply ``primary``
---------------------------------------------

``FRED:SP500`` is licence-capped to a rolling ~10-year window: no 2000, no 2008,
only the tail of 2020. Fitting a five-state model whose worst state is meant to
mean *crisis* on a window containing one drawdown does not produce a crisis
regime, it produces an outlier detector. So the fit runs on whatever daily series
actually reaches back through the crises, and the fitted model is then applied to
the publication series' features.

Precedence, not availability-of-the-day
---------------------------------------

``CALIBRATION_PRECEDENCE`` is a fixed order, and :func:`resolve` is a pure
function of the observation counts handed to it. That combination is the point:
if ``STOOQ:^SPX`` is present it is the calibration series **always**, and
ingesting ``FRED:NASDAQ100`` afterwards cannot quietly move the training window
underneath a fitted model. The resolution is stamped into ``model_version``
(:attr:`PriceRoles.tag`), so a state published against one calibration series
cannot be confused with a state published against another — they are different
model versions and they occupy different rows.

A role with a handful of rows is treated as absent (``min_observations``): a
half-finished backfill is not a training set, and letting one win the precedence
would be exactly the silent window move this design exists to prevent.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import pandas as pd

from findynamics.core.config import SeriesConfig, SeriesSpec
from findynamics.core.contracts.pit import PITAccessor

log = logging.getLogger("findynamics.engines.equity.prices")

#: The engine's own role names. Everything downstream speaks only these.
ENGINE_ROLES: tuple[str, ...] = ("publication", "calibration", "deep_history")

#: The published state describes this index and no other. There is no fallback:
#: a state labelled "equity" that silently described the NASDAQ would be a lie
#: no confidence penalty could repair.
PUBLICATION_ROLE = "primary"

#: Fixed precedence for the daily fitting series. ``backfill`` is genuine daily
#: S&P history where the endpoint answers; ``regime_proxy`` is the fallback and
#: is a *different index* — higher vol, tech-heavy — which every consumer of a
#: fitted parameter has to be told about (:attr:`PriceRoles.calibration_is_proxy`).
CALIBRATION_PRECEDENCE: tuple[str, ...] = ("backfill", "regime_proxy")

#: Monthly, 1871+. The only series that is the S&P across every episode the
#: backtest cares about, and the only admissible basis for the spec's §4
#: "1871+ drawdowns".
DEEP_HISTORY_ROLE = "deep_history"

#: Source roles that carry the **same index** as the publication series.
#:
#: Index identity is not derivable from the series id — ``STOOQ:^SPX`` and
#: ``FRED:SP500`` are two vendors' copies of one index, while ``FRED:NASDAQ100``
#: is a different market with different volatility. Comparing ids would label the
#: best available case (real daily S&P history back to 1928) as a proxy and
#: attach a caveat to every parameter fitted on it that the data does not
#: warrant. So the roles carry the semantics: ``backfill`` means "the same index,
#: further back", ``regime_proxy`` means "a stand-in, and say so".
SAME_INDEX_ROLES: frozenset[str] = frozenset({PUBLICATION_ROLE, "backfill"})

#: Below this many knowable observations a role is not a training set.
DEFAULT_MIN_OBSERVATIONS = 250


class PriceRoleError(ValueError):
    """Raised when the configured roles cannot produce a usable assignment."""


def _slug(series_id: str) -> str:
    """``FRED:NASDAQ100`` -> ``fred_nasdaq100`` — safe inside a model version."""
    return re.sub(r"[^a-z0-9]+", "_", series_id.lower()).strip("_")


@dataclass(frozen=True)
class PriceSeries:
    """One resolved series, carrying where it came from and how much there is."""

    #: Engine role: ``publication`` | ``calibration`` | ``deep_history``.
    role: str
    #: ``series.yaml`` role that supplied it.
    source_role: str
    series_id: str
    frequency: str
    #: Knowable observations at the information set this was resolved against.
    observations: int

    @property
    def slug(self) -> str:
        return _slug(self.series_id)

    @property
    def periods_per_year(self) -> float:
        """Annualization factor for a rate expressed per observation.

        Trading days for a daily series, calendar months for a monthly one. The
        kinematic features are per-observation slopes, and a velocity quoted
        without this would silently mean something different for the monthly
        deep-history path than for the daily one.
        """
        return PERIODS_PER_YEAR[self.frequency]


#: Observations per year by configured frequency. 252 is the conventional US
#: trading-day count; using 365 would understate an annualized daily slope by a
#: third and make the daily and monthly paths disagree about the same trend.
PERIODS_PER_YEAR: dict[str, float] = {
    "daily": 252.0,
    "weekly": 52.0,
    "monthly": 12.0,
    "quarterly": 4.0,
}


@dataclass(frozen=True)
class PriceRoles:
    """The resolved assignment for one information set."""

    publication: PriceSeries
    calibration: PriceSeries
    #: ``None`` when Shiller could not be fetched. The engine still publishes a
    #: kinematic state; only the EVT tail work in sub-milestone C needs this.
    deep_history: PriceSeries | None

    @property
    def calibration_is_proxy(self) -> bool:
        """True when the fitting series is a different *index*, not just a different id.

        Every fitted parameter inherits this. It is surfaced on the state, in the
        backtest report and in ``model_version`` rather than kept internal,
        because "the crisis probability was fitted on the NASDAQ" is a caveat a
        reader is entitled to without reading the source.

        Decided by :data:`SAME_INDEX_ROLES`, not by comparing series ids: the
        backfill is the same index from another vendor, and calling that a proxy
        would attach a warning to the one configuration that does not need one.
        """
        return self.calibration.source_role not in SAME_INDEX_ROLES

    @property
    def tag(self) -> str:
        """The suffix ``model_version`` carries, e.g. ``cal.fred_nasdaq100``.

        In the version rather than only in metadata so that two states fitted on
        different calibration series cannot collide: ``asset_state`` is keyed on
        (asset, as_of, model_version), so they land in different rows and a
        backtest of one never silently becomes a backtest of the other.
        """
        return f"cal.{self.calibration.slug}"

    def series_ids(self) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(
                series.series_id
                for series in (self.publication, self.calibration, self.deep_history)
                if series is not None
            )
        )

    def as_components(self) -> dict[str, float]:
        """Observation counts per engine role, for the explainability trace."""
        counts = {
            "publication_obs": float(self.publication.observations),
            "calibration_obs": float(self.calibration.observations),
            "calibration_is_proxy": float(self.calibration_is_proxy),
        }
        if self.deep_history is not None:
            counts["deep_history_obs"] = float(self.deep_history.observations)
        return counts

    def describe(self) -> dict[str, Any]:
        """Human-readable resolution, for artifacts and the backtest report."""
        return {
            "publication": self.publication.series_id,
            "calibration": self.calibration.series_id,
            "calibration_source_role": self.calibration.source_role,
            "calibration_is_proxy": self.calibration_is_proxy,
            "deep_history": None if self.deep_history is None else self.deep_history.series_id,
            "tag": self.tag,
        }


def configured_roles(config: SeriesConfig) -> dict[str, SeriesSpec]:
    """``engines.equity.series`` from series.yaml, keyed by its own role names."""
    return config.engine_series("equity")


def resolve(
    counts: Mapping[str, int],
    config: SeriesConfig,
    *,
    min_observations: int = DEFAULT_MIN_OBSERVATIONS,
) -> PriceRoles:
    """Assign engine roles from configured roles and observation counts.

    Pure: the same ``counts`` always produce the same assignment. That is what
    makes the choice testable and what stops it drifting between the fit run and
    the daily run — the two see the same D1 and therefore reach the same answer.

    ``counts`` maps series id to knowable observations. A configured role absent
    from the mapping, or below ``min_observations``, is treated as not there.
    """
    configured = configured_roles(config)
    if not configured:
        raise PriceRoleError(
            "series.yaml declares no engines.equity.series; the equity engine "
            "has no price backbone to run on"
        )

    def usable(source_role: str, engine_role: str) -> PriceSeries | None:
        spec = configured.get(source_role)
        if spec is None:
            return None
        observations = int(counts.get(spec.id, 0))
        if observations < min_observations:
            if observations:
                log.info(
                    "equity: %s (%s) has %d knowable observations, below the %d "
                    "needed to count as present",
                    source_role,
                    spec.id,
                    observations,
                    min_observations,
                )
            return None
        return PriceSeries(
            role=engine_role,
            source_role=source_role,
            series_id=spec.id,
            frequency=spec.frequency,
            observations=observations,
        )

    publication = usable(PUBLICATION_ROLE, "publication")
    if publication is None:
        spec = configured.get(PUBLICATION_ROLE)
        raise PriceRoleError(
            f"equity: the publication series ({'unconfigured' if spec is None else spec.id}) "
            f"has fewer than {min_observations} knowable observations; the engine "
            "cannot describe an index it cannot see"
        )

    calibration: PriceSeries | None = None
    for source_role in CALIBRATION_PRECEDENCE:
        calibration = usable(source_role, "calibration")
        if calibration is not None:
            break

    if calibration is None:
        # Falling back to the publication series is a deliberate, loud
        # degradation rather than a failure: a kinematic state is still worth
        # publishing, and sub-milestone B's fit will refuse this series on its
        # own terms rather than silently fitting a crisis state on ten years.
        log.warning(
            "equity: no calibration series is available (%s); falling back to the "
            "publication series — any regime model fitted on this window has no "
            "crisis episodes in sample",
            ", ".join(CALIBRATION_PRECEDENCE),
        )
        calibration = PriceSeries(
            role="calibration",
            source_role=PUBLICATION_ROLE,
            series_id=publication.series_id,
            frequency=publication.frequency,
            observations=publication.observations,
        )

    deep_history = usable(DEEP_HISTORY_ROLE, "deep_history")
    if deep_history is None:
        log.warning("equity: no deep-history series; the 1871+ tail fit is unavailable this run")

    roles = PriceRoles(publication=publication, calibration=calibration, deep_history=deep_history)
    log.info(
        "equity roles: publication=%s calibration=%s (%s%s) deep_history=%s",
        roles.publication.series_id,
        roles.calibration.series_id,
        roles.calibration.source_role,
        ", PROXY — not the published index" if roles.calibration_is_proxy else "",
        None if roles.deep_history is None else roles.deep_history.series_id,
    )
    return roles


def observation_counts(
    accessor: PITAccessor,
    series_ids: Iterable[str],
) -> dict[str, int]:
    """Knowable observations per series at the accessor's cutoff.

    Counts what the information set actually holds, which is what makes
    :func:`resolve` a function of D1 contents rather than of configuration
    optimism: a series declared in yaml but never ingested has a count of zero.
    """
    wanted = list(dict.fromkeys(series_ids))
    if not wanted:
        return {}
    frame = accessor.wide(wanted)
    if frame.empty:
        return dict.fromkeys(wanted, 0)
    return {
        series_id: int(frame[series_id].notna().sum()) if series_id in frame.columns else 0
        for series_id in wanted
    }


def resolve_from(
    accessor: PITAccessor,
    config: SeriesConfig,
    *,
    min_observations: int = DEFAULT_MIN_OBSERVATIONS,
) -> PriceRoles:
    """:func:`resolve` against what one information set can actually see."""
    configured = configured_roles(config)
    counts = observation_counts(accessor, (spec.id for spec in configured.values()))
    return resolve(counts, config, min_observations=min_observations)


def price_path(accessor: PITAccessor, series: PriceSeries) -> pd.Series:
    """The knowable close history of one resolved series, oldest first.

    Non-positive closes are dropped rather than repaired: every feature in this
    engine is built on log price, and a zero from a bad ingest would propagate as
    ``-inf`` through the Kalman filter into every downstream state.
    """
    frame = accessor.wide([series.series_id])
    if frame.empty or series.series_id not in frame.columns:
        return pd.Series(dtype="float64", name=series.series_id)

    path = frame[series.series_id].dropna().sort_index()
    positive = path[path > 0.0]
    dropped = len(path) - len(positive)
    if dropped:
        log.warning("equity: dropped %d non-positive close(s) from %s", dropped, series.series_id)
    positive.name = series.series_id
    return positive


__all__ = [
    "CALIBRATION_PRECEDENCE",
    "DEEP_HISTORY_ROLE",
    "DEFAULT_MIN_OBSERVATIONS",
    "ENGINE_ROLES",
    "PERIODS_PER_YEAR",
    "PUBLICATION_ROLE",
    "PriceRoleError",
    "PriceRoles",
    "PriceSeries",
    "SAME_INDEX_ROLES",
    "configured_roles",
    "observation_counts",
    "price_path",
    "resolve",
    "resolve_from",
]
