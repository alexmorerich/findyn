"""Data quality engine.

Runs between a provider and the database. Its job is to make a bad fetch loud:
a truncated CSV, a units change, a decimal-point error, a revision that
contradicts what was already stored. None of those raise an exception at the
source — they arrive as perfectly well-formed numbers — and once written they
are indistinguishable from signal.

Severity is split deliberately. An *error* means the batch should not be
trusted; a *warning* means it is usable but something moved. Callers decide
policy; this module only reports.
"""

from __future__ import annotations

import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Literal

from findynamics.data.providers.base import FREQUENCY_DAYS, Observation, SeriesMetadata

Severity = Literal["warning", "error"]
Status = Literal["ok", "warning", "error"]


@dataclass(frozen=True)
class Finding:
    code: str
    message: str
    severity: Severity
    context: dict[str, Any] = field(default_factory=dict)

    def to_wire(self) -> dict[str, Any]:
        return {"code": self.code, "message": self.message, "context": self.context}


@dataclass
class DataQualityReport:
    series_id: str
    provider: str
    observations: int
    warnings: list[Finding] = field(default_factory=list)
    errors: list[Finding] = field(default_factory=list)
    checked_range: str | None = None
    #: Findings about individual observations rather than about the series.
    #:
    #: These do **not** block ingestion, and the distinction is the whole point.
    #: A unit mismatch or a coverage hole says the series cannot be trusted as a
    #: series — nothing built on it is safe. A single observation moving further
    #: than any threshold expected says something happened. In macro and market
    #: data, regime shifts and crises are not corruption; they are the signal
    #: this system exists to capture, and refusing the series that contains them
    #: throws away exactly the history the models need.
    #:
    #: So the observation is stored, flagged, and passed on with its flag. A
    #: consumer that wants to exclude or down-weight it can; one that wants
    #: March 2020 gets March 2020. What never happens is silent deletion.
    anomalies: list[Finding] = field(default_factory=list)

    @property
    def status(self) -> Status:
        if self.errors:
            return "error"
        if self.warnings or self.anomalies:
            return "warning"
        return "ok"

    @property
    def ok(self) -> bool:
        """Whether the SERIES may be ingested. Anomalies do not affect this."""
        return not self.errors

    def anomalous_dates(self) -> dict[str, str]:
        """Observation date -> the code that flagged it, for the write-back."""
        flagged: dict[str, str] = {}
        for finding in self.anomalies:
            observation_date = finding.context.get("observation_date")
            if isinstance(observation_date, str):
                flagged[observation_date] = finding.code
        return flagged

    def to_wire(self) -> dict[str, Any]:
        return {
            "series_id": self.series_id,
            "provider": self.provider,
            "status": self.status,
            "observations": self.observations,
            "warnings": [w.to_wire() for w in self.warnings],
            "errors": [e.to_wire() for e in self.errors],
            "anomalies": [a.to_wire() for a in self.anomalies],
            "checked_range": self.checked_range,
        }

    def summary(self) -> str:
        return (
            f"{self.series_id}: {self.status} "
            f"({self.observations} obs, {len(self.errors)} errors, {len(self.warnings)} warnings)"
        )


@dataclass(frozen=True)
class QualityPolicy:
    """Thresholds. Defaults are deliberately permissive for markets and strict
    for anything that should not move fast."""

    #: Period-over-period relative change above which a value is rejected.
    #: 2.0 = a tripling. Real macro series do not do this; decimal errors do.
    error_relative_jump: float = 2.0
    #: Robust z-score of log-change above which a value is flagged for review.
    warn_robust_z: float = 12.0
    #: Minimum observations before dispersion-based checks are meaningful.
    min_points_for_dispersion: int = 30
    #: Gap larger than frequency spacing × this factor is reported.
    gap_tolerance_factor: float = 3.0
    #: Fraction of expected observations below which coverage is flagged.
    min_coverage: float = 0.8
    #: Series whose values are legitimately allowed to be non-positive.
    allow_non_positive: bool = False
    #: A series whose smallest magnitude falls below this fraction of its own
    #: typical magnitude has been at or near zero, so percentage changes are not
    #: a stable description of it anywhere. See `_jump_check`.
    zero_crossing_fraction: float = 0.05
    #: Absolute move, as a multiple of the series' typical magnitude, that counts
    #: as a jump for a series judged on absolute change.
    error_absolute_jump_factor: float = 3.0


#: Largest plausible gap between consecutive observations, in days, before it
#: counts as missing data. Daily allows for long weekends and holidays.
_MAX_NORMAL_GAP = {"daily": 5, "weekly": 10, "monthly": 45, "quarterly": 130}


def _median_abs_deviation(values: list[float]) -> float:
    if not values:
        return 0.0
    med = statistics.median(values)
    return statistics.median([abs(v - med) for v in values])


def _robust_z(values: list[float]) -> list[float]:
    """Z-scores against median/MAD rather than mean/stdev.

    A single 300% spike inflates the standard deviation enough to hide itself;
    the median absolute deviation is unmoved by it.
    """
    if not values:
        return []
    med = statistics.median(values)
    mad = _median_abs_deviation(values)
    # 1.4826 makes MAD a consistent estimator of sigma for normal data.
    scale = mad * 1.4826
    if scale <= 0:
        return [0.0] * len(values)
    return [(v - med) / scale for v in values]


def _expected_observations(first: date, last: date, frequency: str) -> int:
    span_days = (last - first).days
    if span_days <= 0:
        return 1
    if frequency == "daily":
        # Business days, roughly: 5/7 of the calendar less ~9 holidays a year.
        return max(1, int(span_days * (5 / 7) - span_days / 365 * 9))
    return max(1, int(span_days / FREQUENCY_DAYS[frequency]) + 1)


def check_series(
    metadata: SeriesMetadata,
    observations: list[Observation],
    *,
    policy: QualityPolicy | None = None,
    known_values: dict[tuple[date, date], float] | None = None,
) -> DataQualityReport:
    """Validate one series.

    ``known_values`` maps (observation_date, revision_date) to what is already
    stored, so a provider silently changing a figure it already published under
    the same vintage is caught rather than upserted over.
    """
    policy = policy or QualityPolicy()
    report = DataQualityReport(
        series_id=metadata.series_id,
        provider=metadata.provider,
        observations=len(observations),
    )

    if not observations:
        report.errors.append(Finding("empty", "provider returned no observations", "error"))
        return report

    ordered = sorted(
        observations, key=lambda o: (o.observation_date, o.revision_date or o.release_date)
    )
    first, last = ordered[0].observation_date, ordered[-1].observation_date
    report.checked_range = f"{first.isoformat()}..{last.isoformat()}"

    _check_units(metadata, ordered, report)
    _check_duplicates(ordered, report)
    _check_revision_conflicts(ordered, report, known_values)
    _check_frequency(metadata, ordered, report, policy)
    _check_gaps(metadata, ordered, report, policy)
    _check_coverage(metadata, ordered, report, policy, first, last)
    _check_jumps(metadata, ordered, report, policy)
    _check_release_ordering(ordered, report)

    return report


def _check_units(
    metadata: SeriesMetadata, observations: list[Observation], report: DataQualityReport
) -> None:
    mismatched = {o.unit for o in observations if o.unit != metadata.unit}
    if mismatched:
        report.errors.append(
            Finding(
                "unit_mismatch",
                f"observations declare units {sorted(mismatched)} "
                f"but the series is {metadata.unit!r}",
                "error",
                {"expected": metadata.unit, "found": sorted(mismatched)},
            )
        )

    frequencies = {o.frequency for o in observations if o.frequency != metadata.frequency}
    if frequencies:
        report.errors.append(
            Finding(
                "frequency_mismatch",
                f"observations declare frequencies {sorted(frequencies)} but the series is "
                f"{metadata.frequency!r}",
                "error",
                {"expected": metadata.frequency, "found": sorted(frequencies)},
            )
        )


def _check_duplicates(observations: list[Observation], report: DataQualityReport) -> None:
    # A period legitimately appears many times with different vintages, so the
    # key includes the revision date; only an exact repeat is a duplicate.
    keys = Counter((o.observation_date, o.revision_date or o.release_date) for o in observations)
    duplicates = [key for key, n in keys.items() if n > 1]
    if duplicates:
        sample = [
            {"observation_date": d.isoformat(), "revision_date": r.isoformat()}
            for d, r in sorted(duplicates)[:5]
        ]
        report.errors.append(
            Finding(
                "duplicate_timestamp",
                f"{len(duplicates)} (observation_date, revision_date) pairs appear more than once",
                "error",
                {"count": len(duplicates), "sample": sample},
            )
        )


def _check_revision_conflicts(
    observations: list[Observation],
    report: DataQualityReport,
    known_values: dict[tuple[date, date], float] | None,
) -> None:
    # Within the batch: the same vintage carrying two different values.
    by_key: dict[tuple[date, date], set[float]] = defaultdict(set)
    for o in observations:
        by_key[(o.observation_date, o.revision_date or o.release_date)].add(round(o.value, 10))
    internal = {k: v for k, v in by_key.items() if len(v) > 1}
    if internal:
        sample = [
            {
                "observation_date": d.isoformat(),
                "revision_date": r.isoformat(),
                "values": sorted(values),
            }
            for (d, r), values in sorted(internal.items())[:5]
        ]
        report.errors.append(
            Finding(
                "revision_conflict",
                f"{len(internal)} vintages carry more than one value within this batch",
                "error",
                {"count": len(internal), "sample": sample},
            )
        )

    if not known_values:
        return

    # Against storage: a figure already published under this vintage has changed.
    changed = []
    for o in observations:
        key = (o.observation_date, o.revision_date or o.release_date)
        previous = known_values.get(key)
        if previous is not None and abs(previous - o.value) > 1e-9:
            changed.append(
                {
                    "observation_date": o.observation_date.isoformat(),
                    "revision_date": key[1].isoformat(),
                    "stored": previous,
                    "incoming": o.value,
                }
            )
    if changed:
        report.errors.append(
            Finding(
                "revision_conflict_stored",
                f"{len(changed)} vintages contradict values already stored under the same "
                "revision date",
                "error",
                {"count": len(changed), "sample": changed[:5]},
            )
        )


def _distinct_period_dates(observations: list[Observation]) -> list[date]:
    return sorted({o.observation_date for o in observations})


def _check_frequency(
    metadata: SeriesMetadata,
    observations: list[Observation],
    report: DataQualityReport,
    policy: QualityPolicy,
) -> None:
    dates = _distinct_period_dates(observations)
    if len(dates) < 3:
        return
    spacings = [(b - a).days for a, b in zip(dates, dates[1:], strict=False)]
    median_spacing = statistics.median(spacings)
    nominal = FREQUENCY_DAYS[metadata.frequency]

    # Generous bounds: monthly spacing ranges 28-31, daily 1-4 across weekends.
    lower, upper = nominal * 0.4, nominal * 2.0
    if not lower <= median_spacing <= upper:
        report.errors.append(
            Finding(
                "invalid_frequency",
                f"median spacing is {median_spacing:.0f} days, inconsistent with declared "
                f"{metadata.frequency} (~{nominal} days)",
                "error",
                {
                    "declared": metadata.frequency,
                    "median_spacing_days": median_spacing,
                    "expected_days": nominal,
                },
            )
        )


def _check_gaps(
    metadata: SeriesMetadata,
    observations: list[Observation],
    report: DataQualityReport,
    policy: QualityPolicy,
) -> None:
    dates = _distinct_period_dates(observations)
    if len(dates) < 2:
        return
    threshold = _MAX_NORMAL_GAP[metadata.frequency] * policy.gap_tolerance_factor
    gaps = [
        {"after": a.isoformat(), "before": b.isoformat(), "days": (b - a).days}
        for a, b in zip(dates, dates[1:], strict=False)
        if (b - a).days > threshold
    ]
    if gaps:
        largest = sorted(gaps, key=lambda g: -int(g["days"]))[:5]
        report.warnings.append(
            Finding(
                "missing_observations",
                f"{len(gaps)} gaps exceed {threshold:.0f} days",
                "warning",
                {"count": len(gaps), "largest": largest},
            )
        )


def _check_coverage(
    metadata: SeriesMetadata,
    observations: list[Observation],
    report: DataQualityReport,
    policy: QualityPolicy,
    first: date,
    last: date,
) -> None:
    actual = len(_distinct_period_dates(observations))
    expected = _expected_observations(first, last, metadata.frequency)
    ratio = actual / expected if expected else 1.0
    if ratio < policy.min_coverage:
        report.warnings.append(
            Finding(
                "sparse_coverage",
                f"{actual} periods present against ~{expected} expected "
                f"({ratio:.0%} coverage) over {first}..{last}",
                "warning",
                {"actual": actual, "expected": expected, "coverage": round(ratio, 4)},
            )
        )


def _check_jumps(
    metadata: SeriesMetadata,
    observations: list[Observation],
    report: DataQualityReport,
    policy: QualityPolicy,
) -> None:
    # One value per period — the newest vintage — so revisions are not mistaken
    # for period-over-period movement.
    latest: dict[date, Observation] = {}
    for o in observations:
        key = o.observation_date
        current = latest.get(key)
        if current is None or (o.revision_date or o.release_date) >= (
            current.revision_date or current.release_date
        ):
            latest[key] = o

    series = [latest[d] for d in sorted(latest)]
    if len(series) < 2:
        return

    # The series' own typical magnitude. The 75th percentile of |value| rather
    # than the median: for a series that spends years near zero — a policy rate
    # at the floor, a spread that sits inverted — the median IS near zero, and
    # using it would make the scale as uninformative as the values.
    magnitudes = sorted(abs(o.value) for o in series if o.value == o.value)
    scale = magnitudes[int(len(magnitudes) * 0.75)] if magnitudes else 0.0

    # Does this series live near zero, or cross it? Decided once for the whole
    # series rather than per observation, because that is the level the property
    # holds at: a quantity that changes sign has no stable percentage change
    # ANYWHERE, not just at the crossing. Judging it point by point still let
    # DGS1MO fail on 0.08 -> 0.51 while passing 0.07 -> 0.26, which is not a
    # distinction anyone could defend.
    crosses_zero = bool(magnitudes) and (
        any(o.value < 0 for o in series) or magnitudes[0] < scale * policy.zero_crossing_fraction
    )

    changes: list[tuple[Observation, float]] = []
    for prev, curr in zip(series, series[1:], strict=False):
        delta = curr.value - prev.value

        if crosses_zero:
            # Absolute change against the series' own scale. A genuine decimal
            # error still trips this; a rate moving from 2bp to 10bp does not.
            if scale > 0 and abs(delta) > scale * policy.error_absolute_jump_factor:
                report.anomalies.append(
                    Finding(
                        "abnormal_jump",
                        f"{curr.observation_date} moves {delta:+.4g} from {prev.value:.4g}, "
                        f"beyond {policy.error_absolute_jump_factor:g}x the series' typical "
                        f"magnitude ({scale:.4g}). This series reaches zero, so it is judged "
                        "on absolute change rather than percentage.",
                        "warning",
                        {
                            "observation_date": curr.observation_date.isoformat(),
                            "absolute_change": delta,
                            "series_scale": scale,
                        },
                    )
                )
            continue

        if prev.value == 0:
            continue
        changes.append((curr, delta / abs(prev.value)))

    if not changes:
        return

    for obs, rel in changes:
        if abs(rel) > policy.error_relative_jump:
            report.anomalies.append(
                Finding(
                    "abnormal_jump",
                    f"{obs.observation_date} moves {rel:+.1%} from the previous period, "
                    f"beyond the {policy.error_relative_jump:.0%} limit",
                    "warning",
                    {"observation_date": obs.observation_date.isoformat(), "relative_change": rel},
                )
            )

    if len(changes) >= policy.min_points_for_dispersion:
        magnitudes = [rel for _, rel in changes]
        for (obs, rel), z in zip(changes, _robust_z(magnitudes), strict=False):
            if abs(z) > policy.warn_robust_z and abs(rel) <= policy.error_relative_jump:
                report.warnings.append(
                    Finding(
                        "outlier_move",
                        f"{obs.observation_date} moves {rel:+.2%}, robust z={z:+.1f}",
                        "warning",
                        {
                            "observation_date": obs.observation_date.isoformat(),
                            "relative_change": rel,
                            "robust_z": round(z, 2),
                        },
                    )
                )

    if not policy.allow_non_positive:
        non_positive = [o for o in series if o.value <= 0]
        if non_positive and metadata.unit in {"index", "usd", "usd_billions", "ratio"}:
            report.warnings.append(
                Finding(
                    "non_positive_value",
                    f"{len(non_positive)} non-positive values in a {metadata.unit} series",
                    "warning",
                    {
                        "count": len(non_positive),
                        "sample": [o.observation_date.isoformat() for o in non_positive[:5]],
                    },
                )
            )


def _check_release_ordering(observations: list[Observation], report: DataQualityReport) -> None:
    """Later periods should not be released before earlier ones.

    An inversion usually means a synthesized lag was applied inconsistently, and
    it silently changes what a point-in-time join returns.
    """
    latest: dict[date, date] = {}
    for o in observations:
        if o.observation_date not in latest or o.release_date < latest[o.observation_date]:
            latest[o.observation_date] = o.release_date

    ordered = sorted(latest.items())
    inversions = [
        {
            "earlier_period": a.isoformat(),
            "later_period": b.isoformat(),
            "earlier_release": ra.isoformat(),
            "later_release": rb.isoformat(),
        }
        for (a, ra), (b, rb) in zip(ordered, ordered[1:], strict=False)
        if rb < ra
    ]
    if inversions:
        report.warnings.append(
            Finding(
                "release_date_inversion",
                f"{len(inversions)} later periods were released before an earlier period",
                "warning",
                {"count": len(inversions), "sample": inversions[:5]},
            )
        )
