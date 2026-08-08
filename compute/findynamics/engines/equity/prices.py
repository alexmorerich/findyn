"""Which price series plays which part — the one place roles become series.

``series.yaml`` names its equity series by *where the data comes from*
(``primary``, ``backfill``, ``regime_proxy``, ``deep_history``). The rest of this
engine names them by *what they are for*:

===============  =====================================================
publication      the index the published state describes — always ``primary``
extension        the same index further back, spliced in front of it
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

Why the publication record is spliced
--------------------------------------

The same cap used to bound the *published* record too: velocity was a ten-year
line because the filter had ten years of price to run on, not because the S&P
began in 2016. ``backfill`` is declared to be the same index further back
(:data:`SAME_INDEX_ROLES`), which is exactly the licence needed to join the two
into one input vector — ``YAHOO:^GSPC`` before ``FRED:SP500`` starts, ``FRED``
from there on, so the vendor of record for a date never changes for a date that
already had one.

The join is validated rather than assumed. The two records overlap by their
whole ten years, so :func:`publication_path` measures the disagreement across
2,500 shared dates before splicing anything (median 2e-8, worst day 1.2e-3) and
declines the extension if the two are not the same series. Declining changes the
input, so it changes ``model_version`` too — a decade of velocity and a century
of it are different claims and must not land in the same row.

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
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import pandas as pd

from findynamics.core.config import SeriesConfig, SeriesSpec
from findynamics.core.contracts.pit import PITAccessor

log = logging.getLogger("findynamics.engines.equity.prices")

#: The engine's own role names. Everything downstream speaks only these.
#:
#: ``extension`` is the odd one out and deliberately so: the other three each get
#: their own feature path, while this one is *folded into* ``publication`` by
#: :func:`publication_path`. It is named as a role because it is a resolution
#: decision with its own precedence and its own failure modes, not because the
#: engine ever computes a velocity for it.
ENGINE_ROLES: tuple[str, ...] = ("publication", "extension", "calibration", "deep_history")

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

#: The role whose closes extend the publication record backwards.
#:
#: ``backfill`` and only ``backfill``, because :data:`SAME_INDEX_ROLES` is what
#: makes the splice legitimate rather than merely convenient. ``regime_proxy`` is
#: a different market, and prepending it would publish a velocity for 1990 that
#: describes the NASDAQ under a label that says S&P.
EXTENSION_ROLE = "backfill"

#: The source roles :func:`resolve` assigns, and the only ones it can report as
#: missing. ``engines.equity.series`` also configures the drivers the instability
#: view reads — ``credit_spread``, ``risk_free`` and the rest — which are not
#: price records, never reach :func:`resolve`, and do not reach a fitted artifact
#: either; a refit must not fail over one.
PRICE_ROLES: tuple[str, ...] = tuple(
    dict.fromkeys((PUBLICATION_ROLE, *CALIBRATION_PRECEDENCE, EXTENSION_ROLE, DEEP_HISTORY_ROLE))
)

#: Joins the two ids in a spliced series id. ``+`` rather than ``/`` or ``:``
#: because it survives :func:`_slug` as a single separator and reads as "and
#: then" in the ``model_version`` it ends up in.
SPLICE_JOIN = "+"

#: Shared dates the two records must have before their agreement can be judged.
#: Below this the comparison is anecdote, and an unvalidated splice is exactly
#: the silent error this module exists to prevent.
MIN_SPLICE_OVERLAP = 250

#: Median relative disagreement tolerated across the overlap.
#:
#: Two vendors' copies of one index differ only by rounding: over the 2,514 dates
#: ``FRED:SP500`` and ``YAHOO:^GSPC`` share, the median relative gap is 2e-8 and
#: the worst single day 1.2e-3. A tenth of a percent therefore sits four orders
#: of magnitude above the noise and still far below anything that could be a
#: different index, a different currency, or a rebasing.
MAX_SPLICE_DISAGREEMENT = 1e-3

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
    #: The same-index daily record spliced in front of the publication series, or
    #: ``None`` when no backfill is available. Resolution says whether one is
    #: *configured and present*; :func:`publication_path` decides whether the two
    #: records actually agree well enough to be joined.
    extension: PriceSeries | None = None
    #: Source roles that ``series.yaml`` configures and this information set could
    #: not supply — a provider that failed, or a series never ingested.
    #:
    #: A daily run degrades around these and publishes what it can; that is §14.2
    #: and it is right. A **refit** must not, which is why this is recorded rather
    #: than only logged: the artifact it writes is immutable and addressed by
    #: content, so a month fitted without deep history is a different artifact
    #: under the same key as the month fitted with it (issue #6).
    #:
    #: Excludes calibration roles that precedence legitimately left unused — a
    #: `regime_proxy` that went unconsulted because `backfill` outranked it is not
    #: missing, and failing a refit over it would be the false alarm this is
    #: meant to prevent.
    unresolved: tuple[str, ...] = ()

    @property
    def publication_input(self) -> PriceSeries:
        """The series identity the publication feature path is computed under.

        ``publication`` when nothing extends it; a composite naming both vendors
        when something does. Composite rather than "still FRED:SP500" because a
        1955 velocity did not come from a series that starts in 2016, and a
        reader tracing a number back to its source is entitled to both names.

        ``observations`` here is a floor — the union of two overlapping records
        is at least as long as the longer of them, and the exact count is a
        property of the dates, which only :func:`publication_path` has seen. It
        returns the same identity with the true count filled in.
        """
        if self.extension is None:
            return self.publication
        return PriceSeries(
            role="publication",
            source_role=f"{self.publication.source_role}{SPLICE_JOIN}{self.extension.source_role}",
            series_id=(f"{self.publication.series_id}{SPLICE_JOIN}{self.extension.series_id}"),
            frequency=self.publication.frequency,
            observations=max(self.publication.observations, self.extension.observations),
        )

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
                for series in (
                    self.publication,
                    self.extension,
                    self.calibration,
                    self.deep_history,
                )
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
        if self.extension is not None:
            counts["extension_obs"] = float(self.extension.observations)
        if self.deep_history is not None:
            counts["deep_history_obs"] = float(self.deep_history.observations)
        return counts

    def describe(self) -> dict[str, Any]:
        """Human-readable resolution, for artifacts and the backtest report."""
        return {
            "publication": self.publication.series_id,
            "extension": None if self.extension is None else self.extension.series_id,
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

    extension = usable(EXTENSION_ROLE, "extension")
    if extension is not None and extension.frequency != publication.frequency:
        # A monthly record cannot be prepended to a daily one: every kinematic
        # feature is a rate per observation, so the join would silently change
        # what an observation means partway through the series.
        log.warning(
            "equity: %s is %s and the publication series is %s; the records cannot "
            "be spliced and the published history stays as long as %s reaches",
            extension.series_id,
            extension.frequency,
            publication.frequency,
            publication.series_id,
        )
        extension = None

    # Which configured roles this information set could not supply. `primary` is
    # not among them: it raises above rather than resolving short. A calibration
    # candidate is only counted when nothing ahead of it in the precedence
    # answered either, so the ordinary case of `backfill` outranking
    # `regime_proxy` reports nothing missing.
    def present(source_role: str) -> bool:
        # Deliberately not `usable`, which logs: this is a second pass over the
        # same roles and re-running it would say everything twice.
        spec = configured.get(source_role)
        return spec is not None and int(counts.get(spec.id, 0)) >= min_observations

    calibration_gap = calibration.source_role == PUBLICATION_ROLE
    unresolved = tuple(
        source_role
        for source_role in PRICE_ROLES
        if source_role in configured
        and source_role != PUBLICATION_ROLE
        and (
            source_role not in CALIBRATION_PRECEDENCE
            or source_role == EXTENSION_ROLE
            or calibration_gap
        )
        and not present(source_role)
    )

    roles = PriceRoles(
        publication=publication,
        calibration=calibration,
        deep_history=deep_history,
        extension=extension,
        unresolved=unresolved,
    )
    log.info(
        "equity roles: publication=%s%s calibration=%s (%s%s) deep_history=%s",
        roles.publication.series_id,
        "" if extension is None else f" (+{extension.series_id} behind it)",
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


def splice_disagreement(base: pd.Series, extension: pd.Series) -> tuple[int, float]:
    """Shared dates between two records, and their median relative gap.

    The evidence that ``backfill`` really is the same index as ``primary``, taken
    from the data rather than from the role name. Median rather than maximum: one
    vendor's stale print on one day is a known fact of life about free feeds and
    should not veto a decade of otherwise identical closes, while a series that
    is genuinely something else is wrong on every date.

    Returns ``(0, inf)`` when the records do not overlap — which is a refusal,
    not a pass, because two records that never meet cannot be checked at all.
    """
    common = base.index.intersection(extension.index)
    if len(common) == 0:
        return 0, float("inf")
    left = base.loc[common].astype(float)
    right = extension.loc[common].astype(float)
    gap = ((left - right).abs() / left.abs().replace(0.0, np.nan)).dropna()
    if gap.empty:
        return len(common), float("inf")
    return len(common), float(gap.median())


def publication_path(
    accessor: PITAccessor,
    roles: PriceRoles,
    *,
    min_overlap: int = MIN_SPLICE_OVERLAP,
    max_disagreement: float = MAX_SPLICE_DISAGREEMENT,
) -> tuple[pd.Series, PriceSeries]:
    """The closes the publication feature path runs on, and what they are.

    The publication series wherever it reaches, the extension in front of it, and
    the identity that describes the result. Both are returned together because
    they must not be able to disagree: the series a value was computed from is
    what ``model_version`` names and what the frozen parameters are keyed on, so
    deriving one here and the other from the role resolution would let a spliced
    path be published under an unspliced version.

    **The primary vendor keeps every date it already covers.** The extension only
    supplies dates *before* the publication series begins, so this can lengthen
    the record and can never restate a figure that has already been published
    from FRED.

    The extension is declined — with the identity falling back to the
    publication series alone — when it adds nothing, when the two records do not
    overlap enough to be compared, or when they disagree past
    ``max_disagreement``. Every refusal is loud: silently publishing a decade
    where a century was expected is the failure this returns an identity for.
    """
    base = price_path(accessor, roles.publication)
    if roles.extension is None or base.empty:
        return base, roles.publication

    extension = price_path(accessor, roles.extension)
    if extension.empty:
        log.warning(
            "equity: %s has no knowable closes; the published record stays as long as %s reaches",
            roles.extension.series_id,
            roles.publication.series_id,
        )
        return base, roles.publication

    overlap, disagreement = splice_disagreement(base, extension)
    if overlap < min_overlap or disagreement > max_disagreement:
        log.error(
            "equity: refusing to splice %s in front of %s — they share %d date(s) "
            "(need %d) with a median relative gap of %.3g (limit %.3g). Two vendors' "
            "copies of one index agree to rounding; this pair does not, so joining "
            "them would publish one index's history under the other's name",
            roles.extension.series_id,
            roles.publication.series_id,
            overlap,
            min_overlap,
            disagreement,
            max_disagreement,
        )
        return base, roles.publication

    prefix = extension.loc[extension.index < base.index[0]]
    if prefix.empty:
        log.info(
            "equity: %s reaches no further back than %s; nothing to splice",
            roles.extension.series_id,
            roles.publication.series_id,
        )
        return base, roles.publication

    path = pd.concat([prefix, base]).sort_index()
    series = replace(roles.publication_input, observations=len(path))
    path.name = series.series_id
    log.info(
        "equity: publication path is %d observations %s → %s (%d from %s, %d from %s; "
        "the two agree to %.3g across %d shared dates)",
        len(path),
        path.index[0].date(),
        path.index[-1].date(),
        len(prefix),
        roles.extension.series_id,
        len(base),
        roles.publication.series_id,
        disagreement,
        overlap,
    )
    return path, series


__all__ = [
    "CALIBRATION_PRECEDENCE",
    "DEEP_HISTORY_ROLE",
    "DEFAULT_MIN_OBSERVATIONS",
    "ENGINE_ROLES",
    "EXTENSION_ROLE",
    "MAX_SPLICE_DISAGREEMENT",
    "MIN_SPLICE_OVERLAP",
    "PERIODS_PER_YEAR",
    "PUBLICATION_ROLE",
    "SPLICE_JOIN",
    "PriceRoleError",
    "PriceRoles",
    "PriceSeries",
    "SAME_INDEX_ROLES",
    "configured_roles",
    "observation_counts",
    "price_path",
    "publication_path",
    "resolve",
    "resolve_from",
    "splice_disagreement",
]
