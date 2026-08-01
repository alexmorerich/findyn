"""Data quality engine.

Each check is exercised against the failure it exists to catch, and against a
clean series to prove it does not fire on ordinary data.
"""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from findynamics.data.providers.base import Observation, SeriesMetadata
from findynamics.data.quality import QualityPolicy, check_series


def metadata(**overrides) -> SeriesMetadata:
    base = {
        "series_id": "FRED:CPIAUCSL",
        "provider": "fred",
        "title": "CPI",
        "frequency": "monthly",
        "unit": "index",
    }
    return SeriesMetadata(**{**base, **overrides})


def monthly(values: list[float], *, start=date(2020, 1, 1), unit="index", frequency="monthly"):
    """Observations one month apart, released 15 days after each period ends."""
    observations = []
    for i, value in enumerate(values):
        month = start.month + i
        obs_date = date(start.year + (month - 1) // 12, (month - 1) % 12 + 1, 1)
        next_month = obs_date.replace(day=28) + timedelta(days=4)
        release = next_month.replace(day=1) + timedelta(days=14)
        observations.append(
            Observation(
                series_id="FRED:CPIAUCSL",
                provider="fred",
                frequency=frequency,
                unit=unit,
                observation_date=obs_date,
                release_date=release,
                revision_date=release,
                value=value,
            )
        )
    return observations


def codes(findings) -> set[str]:
    return {f.code for f in findings}


# --------------------------------------------------------------------------


def test_a_clean_series_passes():
    report = check_series(metadata(), monthly([300.0 + i * 0.3 for i in range(36)]))
    assert report.status == "ok"
    assert report.errors == []
    assert report.warnings == []
    assert report.checked_range == "2020-01-01..2022-12-01"


def test_no_observations_is_an_error():
    report = check_series(metadata(), [])
    assert report.status == "error"
    assert "empty" in codes(report.errors)


def test_unit_mismatch_is_an_error():
    observations = monthly([300.0, 301.0, 302.0], unit="percent")
    report = check_series(metadata(unit="index"), observations)
    assert "unit_mismatch" in codes(report.errors)


def test_frequency_mismatch_between_metadata_and_rows_is_an_error():
    observations = monthly([300.0, 301.0, 302.0], frequency="daily")
    report = check_series(metadata(frequency="monthly"), observations)
    assert "frequency_mismatch" in codes(report.errors)


def test_duplicate_timestamps_are_an_error():
    observations = monthly([300.0, 301.0, 302.0])
    report = check_series(metadata(), [*observations, observations[1]])
    assert "duplicate_timestamp" in codes(report.errors)


def test_distinct_vintages_of_one_period_are_not_duplicates():
    """FRED legitimately returns several figures for the same period."""
    first, second = monthly([300.0, 301.0])
    revision = Observation(
        series_id=first.series_id,
        provider=first.provider,
        frequency=first.frequency,
        unit=first.unit,
        observation_date=first.observation_date,
        release_date=first.release_date,
        revision_date=first.release_date + timedelta(days=30),
        value=300.4,
    )
    report = check_series(metadata(), [first, revision, second])
    assert "duplicate_timestamp" not in codes(report.errors)
    assert "revision_conflict" not in codes(report.errors)


def test_two_values_under_one_vintage_is_a_revision_conflict():
    first, second = monthly([300.0, 301.0])
    contradicting = Observation(
        series_id=first.series_id,
        provider=first.provider,
        frequency=first.frequency,
        unit=first.unit,
        observation_date=first.observation_date,
        release_date=first.release_date,
        revision_date=first.revision_date,
        value=999.0,
    )
    report = check_series(metadata(), [first, contradicting, second])
    assert "revision_conflict" in codes(report.errors)


def test_contradicting_a_stored_vintage_is_an_error():
    observations = monthly([300.0, 301.0, 302.0])
    stored = {
        (observations[0].observation_date, observations[0].revision_date): 288.0,
    }
    report = check_series(metadata(), observations, known_values=stored)
    assert "revision_conflict_stored" in codes(report.errors)
    detail = next(e for e in report.errors if e.code == "revision_conflict_stored")
    assert detail.context["sample"][0]["stored"] == 288.0
    assert detail.context["sample"][0]["incoming"] == 300.0


def test_agreeing_with_a_stored_vintage_is_fine():
    observations = monthly([300.0, 301.0, 302.0])
    stored = {(observations[0].observation_date, observations[0].revision_date): 300.0}
    report = check_series(metadata(), observations, known_values=stored)
    assert "revision_conflict_stored" not in codes(report.errors)


def test_spacing_that_contradicts_the_declared_frequency_is_an_error():
    observations = monthly([300.0 + i for i in range(12)])
    report = check_series(metadata(frequency="daily"), observations)
    assert "invalid_frequency" in codes(report.errors)


def test_a_long_gap_is_reported_as_missing_observations():
    early = monthly([300.0, 301.0, 302.0], start=date(2020, 1, 1))
    late = monthly([310.0, 311.0, 312.0], start=date(2022, 6, 1))
    report = check_series(metadata(), early + late)
    assert "missing_observations" in codes(report.warnings)
    gap = next(w for w in report.warnings if w.code == "missing_observations")
    assert gap.context["largest"][0]["days"] > 45


def test_a_tripling_is_rejected():
    """The spec's example: CPI cannot suddenly jump 300%."""
    values = [300.0 + i * 0.2 for i in range(24)]
    values[12] = 1200.0
    report = check_series(metadata(), monthly(values))
    assert report.status == "error"
    assert "abnormal_jump" in codes(report.errors)


def test_an_ordinary_market_drawdown_is_not_rejected():
    # October 1987 was about -20%; a policy that rejects that is unusable.
    values = [3000.0 + i * 5 for i in range(40)]
    values[20] = values[19] * 0.80
    report = check_series(metadata(unit="index"), monthly(values, unit="index"))
    assert "abnormal_jump" not in codes(report.errors)


def test_a_moderate_outlier_is_a_warning_not_an_error():
    values = [300.0 + i * 0.1 for i in range(60)]
    values[30] = values[29] * 1.5  # +50%: unusual, below the 200% error bar
    report = check_series(metadata(), monthly(values))
    assert "abnormal_jump" not in codes(report.errors)
    assert "outlier_move" in codes(report.warnings)


def test_a_revision_is_not_mistaken_for_a_period_over_period_move():
    """Only the newest vintage per period feeds the jump check."""
    observations = monthly([300.0 + i * 0.2 for i in range(30)])
    stale_first_print = Observation(
        series_id=observations[10].series_id,
        provider=observations[10].provider,
        frequency=observations[10].frequency,
        unit=observations[10].unit,
        observation_date=observations[10].observation_date,
        release_date=observations[10].release_date,
        revision_date=observations[10].release_date - timedelta(days=0),
        value=observations[10].value,
    )
    report = check_series(metadata(), [*observations, stale_first_print])
    assert "abnormal_jump" not in codes(report.errors)


def test_release_dates_that_go_backwards_are_flagged():
    observations = monthly([300.0, 301.0, 302.0])
    # Individually valid — released after its own period starts — but ahead of
    # the previous period's release, which is a lag applied unevenly.
    early_release = date(2020, 3, 5)
    assert early_release < observations[1].release_date
    inverted = Observation(
        series_id=observations[2].series_id,
        provider=observations[2].provider,
        frequency=observations[2].frequency,
        unit=observations[2].unit,
        observation_date=observations[2].observation_date,
        release_date=early_release,
        revision_date=early_release,
        value=302.0,
    )
    report = check_series(metadata(), [observations[0], observations[1], inverted])
    assert "release_date_inversion" in codes(report.warnings)


def test_sparse_coverage_is_flagged():
    observations = [monthly([300.0], start=date(2020, 1, 1))[0]]
    observations += monthly([305.0], start=date(2021, 1, 1))
    observations += monthly([310.0], start=date(2022, 1, 1))
    report = check_series(metadata(), observations)
    assert "sparse_coverage" in codes(report.warnings)


def test_thresholds_are_configurable():
    values = [300.0 + i * 0.2 for i in range(24)]
    values[12] = 1200.0
    lenient = QualityPolicy(error_relative_jump=10.0)
    report = check_series(metadata(), monthly(values), policy=lenient)
    assert "abnormal_jump" not in codes(report.errors)


def test_report_serialises_for_the_write_back_endpoint():
    report = check_series(metadata(), monthly([300.0, 301.0, 302.0]))
    wire = report.to_wire()
    assert set(wire) == {
        "series_id",
        "provider",
        "status",
        "observations",
        "warnings",
        "errors",
        "checked_range",
    }
    assert wire["status"] in {"ok", "warning", "error"}
    assert isinstance(wire["warnings"], list)


@pytest.mark.parametrize(
    ("errors", "warnings", "expected"),
    [(0, 0, "ok"), (0, 1, "warning"), (1, 0, "error"), (1, 1, "error")],
)
def test_status_precedence(errors, warnings, expected):
    from findynamics.data.quality import DataQualityReport, Finding

    report = DataQualityReport(series_id="s", provider="p", observations=1)
    report.errors = [Finding("e", "m", "error")] * errors
    report.warnings = [Finding("w", "m", "warning")] * warnings
    assert report.status == expected


# ---------------------------------------------------------------------------
# jumps near zero — the check that withheld fifteen FRED series at once
# ---------------------------------------------------------------------------


def test_a_rate_moving_off_a_near_zero_floor_is_not_an_abnormal_jump():
    """The production failure this rule exists to prevent.

    `FRED:DGS1MO` went from 0.07% to 0.26% on 2008-09-18 — nineteen basis points,
    in the week Lehman failed, and completely correct. As a *relative* change it
    is +271%, which tripped the jump check and withheld the whole series.

    Fifteen FRED series were withheld this way in one backfill: the entire short
    end of the curve (1m, 3m, 6m, DTB3, SOFR), both financial-conditions indices
    (NFCI, STLFSI4), both term spreads (T10Y2Y, T10Y3M), TIPS real yields, and
    the reverse-repo balance. Every one of them can sit at or cross zero, and a
    percentage change is meaningless when its denominator is.
    """
    # Typical 1m bill around 1.5%, then the 2008 sequence near zero.
    values = [1.5] * 40 + [0.07, 0.26, 0.12, 0.03, 0.15]
    report = check_series(
        metadata(unit="percent", frequency="daily"),
        monthly(values, unit="percent", frequency="daily"),
        policy=QualityPolicy(allow_non_positive=True),
    )

    assert "abnormal_jump" not in codes(report.errors)


def test_a_spread_crossing_zero_is_not_an_abnormal_jump():
    """T10Y3M inverts. Crossing zero makes the denominator tiny, then negative."""
    values = [1.2] * 40 + [0.30, 0.05, -0.02, -0.40, 0.10]
    report = check_series(
        metadata(unit="percent", frequency="daily"),
        monthly(values, unit="percent", frequency="daily"),
        policy=QualityPolicy(allow_non_positive=True),
    )

    assert "abnormal_jump" not in codes(report.errors)


def test_a_real_decimal_error_near_zero_is_still_caught():
    """The relaxation must not become a hole.

    Falling back to an absolute test is not the same as switching the check off:
    a value that moves by many times the series' own typical magnitude is still
    wrong, whatever its predecessor was.
    """
    values = [1.5] * 40 + [0.07, 150.0]  # a hundredfold slip, from near zero
    report = check_series(
        metadata(unit="percent", frequency="daily"),
        monthly(values, unit="percent", frequency="daily"),
        policy=QualityPolicy(allow_non_positive=True),
    )

    assert "abnormal_jump" in codes(report.errors)


def test_a_decimal_error_on_a_large_series_is_still_caught_by_the_relative_test():
    """Series that never approach zero keep the original behaviour exactly."""
    values = [3000.0] * 40 + [30000.0]
    report = check_series(
        metadata(unit="index", frequency="daily"),
        monthly(values, unit="index", frequency="daily"),
        policy=QualityPolicy(),
    )

    assert "abnormal_jump" in codes(report.errors)
