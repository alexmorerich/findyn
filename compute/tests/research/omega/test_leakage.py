"""The truncation tests. The most valuable tests in the track.

A source grep catches lookahead-*shaped* code. These catch lookahead, including
the kind that arrives through a library, by computing the feature block and Ω
twice over different amounts of data and requiring the overlapping dates to be
**bit-identical**.

There are two different questions here and the suite asks them separately,
because a single test that conflated them would have to be loosened to pass and
would then prove neither.

**A. Does a transform read a row that comes after it?** Hold the information set
fixed and vary how much *data* exists: run over observations truncated at ``T``,
then over the whole record. Any difference on a date ``<= T`` means something
looked forward. :func:`test_the_feature_block_is_bit_identical_under_truncation`.

**B. Is a value published before its inputs were released?** Vary the
*information set*: run as of ``T``, then as of ``T + ~500`` trading days. A
difference here means a column was stamped with a release date earlier than the
prints it was computed from.
:func:`test_the_block_is_bit_identical_across_information_sets`.

Question B is run against a **revision-free** copy of the fixture, and the
reason is a finding rather than a convenience. The committed snapshot carries
real vintages: ``FRED:NASDAQ100`` has 1,195 observation dates published more
than once and **657 whose value actually changed**, plus a handful in
``FRED:DGS10`` and ``FRED:T10Y3M``. So two runs at different cutoffs differ for
two reasons — one saw more dates, and one saw a later vintage of the same date —
and only the first would be lookahead. ``first_vintage`` removes the second
cause so the assertion can be bit-identity and mean it.
:func:`test_an_earlier_information_set_sees_an_earlier_vintage` records the
revisions themselves, so that a future reader who runs the comparison on the raw
fixture and sees a difference knows what they are looking at.

Both are run at three cutoffs spanning a calm stretch, the 2008 crisis and a
recent window, because a window that reads one row ahead shows it most clearly
where the data is moving fastest.

What these do *not* assert
--------------------------

That Ω's history is frozen forever. Refitting on a longer window re-estimates
the mean, the scale and the loadings, so every historical value moves slightly.
That is the correct behaviour of an expanding-window estimator and
``features/kalman.py`` documents the same property for the Kalman variances.
:func:`test_refitting_on_more_data_does_move_the_history` pins the distinction
down so that nobody later "fixes" the expanding fit into a frozen one in pursuit
of a stricter-looking invariant.
"""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import pytest

from findynamics.engines.equity.features.kinematics import expanding_z
from findynamics.factors.compute import score_series
from tests.research.omega.conftest import (
    SNAPSHOT_AS_OF,
    TRUNCATION_CUTOFFS,
    accessor_at,
    first_vintage,
    observation_rows,
)

#: How far past the cutoff the "later" run sees. ~500 trading days.
LOOKAHEAD_WINDOW = timedelta(days=700)

CUTOFFS = sorted(TRUNCATION_CUTOFFS.items())


def _truncate(observations: pd.DataFrame, cutoff: date) -> pd.DataFrame:
    """Observations of ``cutoff`` or earlier — a shorter record, same vintages."""
    return observations[observations["obs_date"] <= pd.Timestamp(cutoff)]


# ---------------------------------------------------------------------------
# A. Does a transform read a row that comes after it?
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("label,cutoff", CUTOFFS)
def test_the_feature_block_is_bit_identical_under_truncation(
    omega_engine, omega_observations, label, cutoff
):
    """Every column of ``Z_t`` at every date ``<= T``, to the last bit."""
    short = omega_engine.features(accessor_at(_truncate(omega_observations, cutoff)))
    full = omega_engine.features(accessor_at(omega_observations)).loc[: short.index[-1]]

    assert len(short) > 500, f"{label}: too few rows for the test to mean anything"
    pd.testing.assert_index_equal(short.index, full.index)
    pd.testing.assert_frame_equal(short, full, check_exact=True)


@pytest.mark.parametrize("label,cutoff", CUTOFFS)
def test_omega_is_bit_identical_under_truncation(omega_engine, omega_observations, label, cutoff):
    """And the projection carries that property through to Ω itself."""
    short = omega_engine.features(accessor_at(_truncate(omega_observations, cutoff)))
    full = omega_engine.features(accessor_at(omega_observations)).loc[: short.index[-1]]

    spec = omega_engine.fit(short)
    assert omega_engine.fit(full).as_dict() == spec.as_dict()
    pd.testing.assert_series_equal(
        omega_engine.transform(short, spec),
        omega_engine.transform(full, spec),
        check_exact=True,
    )


@pytest.mark.parametrize("label,cutoff", CUTOFFS)
def test_a_future_spike_changes_nothing_before_it(omega_engine, omega_observations, label, cutoff):
    """A ten-thousand-fold print after ``T`` must be invisible on or before ``T``.

    The ordinary truncation test compares two runs over *similar* data. This one
    plants a value so extreme that any leak — a centred window, a full-sample
    scaler, a backwards fill — moves the earlier rows by an amount no float
    tolerance could hide.
    """
    spiked = pd.concat(
        [
            _truncate(omega_observations, cutoff),
            pd.DataFrame(
                observation_rows("YAHOO:^GSPC", {cutoff + timedelta(days=3): 9_999_999.0})
            ),
        ],
        ignore_index=True,
    )

    clean = omega_engine.features(accessor_at(_truncate(omega_observations, cutoff)))
    contaminated = omega_engine.features(accessor_at(spiked)).loc[: clean.index[-1]]

    pd.testing.assert_frame_equal(clean, contaminated, check_exact=True)

    spec = omega_engine.fit(clean)
    pd.testing.assert_series_equal(
        omega_engine.transform(clean, spec),
        omega_engine.transform(contaminated, spec),
        check_exact=True,
    )


# ---------------------------------------------------------------------------
# B. Is a value published before its inputs were released?
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("label,cutoff", CUTOFFS)
def test_the_block_is_bit_identical_across_information_sets(
    omega_engine, omega_observations, label, cutoff
):
    """A column may not carry a value its inputs had not published yet.

    This is the assertion that caught the ``dispersion`` release-date bug in
    KK1: the column combines the price with ``FRED:NASDAQ100``, and it was
    stamped with the price's release date alone. ``YAHOO:^GSPC`` carries a zero
    lag and the NASDAQ a one-day lag, so the last row of every run published a
    correlation computed from a print that had not been released — one day of
    lookahead, on one column, invisible to every other test in this suite.
    """
    revision_free = first_vintage(omega_observations)

    short = omega_engine.features(accessor_at(revision_free, cutoff))
    later = omega_engine.features(accessor_at(revision_free, cutoff + LOOKAHEAD_WINDOW)).loc[
        : short.index[-1]
    ]

    pd.testing.assert_index_equal(short.index, later.index)
    pd.testing.assert_frame_equal(short, later, check_exact=True)


@pytest.mark.parametrize("label,cutoff", CUTOFFS)
def test_omega_is_bit_identical_across_information_sets(
    omega_engine, omega_observations, label, cutoff
):
    revision_free = first_vintage(omega_observations)

    short = omega_engine.features(accessor_at(revision_free, cutoff))
    later = omega_engine.features(accessor_at(revision_free, cutoff + LOOKAHEAD_WINDOW)).loc[
        : short.index[-1]
    ]
    spec = omega_engine.fit(short)

    pd.testing.assert_series_equal(
        omega_engine.transform(short, spec),
        omega_engine.transform(later, spec),
        check_exact=True,
    )


def test_an_earlier_information_set_sees_an_earlier_vintage(omega_observations):
    """The revisions are real, and this is what they look like.

    Recorded as a test rather than a comment because the natural next move for
    someone extending this suite is to run the cross-information-set comparison
    on the raw fixture, watch it fail, and conclude there is a leak. There is
    not: FRED restated 657 ``FRED:NASDAQ100`` observations, and a 2024
    information set correctly sees the pre-restatement numbers.
    """
    revised = omega_observations.groupby(["series_id", "obs_date"])["value"].nunique().gt(1)
    by_series = revised[revised].reset_index().groupby("series_id").size()

    assert by_series["FRED:NASDAQ100"] == 657
    assert "YAHOO:^GSPC" not in by_series, "the price spine must not be revised"
    assert "FRED:NFCI" not in by_series

    # And the difference reaches the values a run would use.
    early = accessor_at(omega_observations, date(2024, 6, 28)).wide(["FRED:NASDAQ100"])
    late = accessor_at(omega_observations, SNAPSHOT_AS_OF).wide(["FRED:NASDAQ100"])
    common = early.index.intersection(late.index)
    changed = (early.loc[common, "FRED:NASDAQ100"] != late.loc[common, "FRED:NASDAQ100"]).sum()
    assert changed > 500


# ---------------------------------------------------------------------------
# The expanding statistics, isolated from the refit
# ---------------------------------------------------------------------------


def test_the_expanding_statistics_never_look_forward(omega_engine, omega_observations):
    """``score_series`` and ``expanding_z`` at date *t* see rows ``0..t`` only.

    Asserted on the Ω path itself rather than on a synthetic series, and
    isolated from the refit: one Ω is computed, then the expanding statistics
    are taken over a prefix of it and over all of it. Any difference on the
    prefix means the percentile or the z-score consulted the future — which is
    the one way ``omega_regime`` could be leaking without any of the tests above
    noticing, because both run inside ``omega_dynamics`` rather than inside a
    transform.
    """
    block = omega_engine.features(accessor_at(omega_observations))
    omega = omega_engine.transform(block, omega_engine.fit(block)).dropna()
    prefix = omega.iloc[:3_000]

    pd.testing.assert_series_equal(
        score_series(prefix, 1), score_series(omega, 1).iloc[:3_000], check_exact=True
    )
    pd.testing.assert_series_equal(
        expanding_z(prefix, 2520), expanding_z(omega, 2520).iloc[:3_000], check_exact=True
    )


def test_refitting_on_more_data_does_move_the_history(omega_engine, omega_observations):
    """The expanding-window property, asserted so nobody removes it.

    A spec fitted through 2008 and one fitted through 2010 describe different
    windows and therefore different means, scales and loadings, so Ω's history
    shifts between them. That is not lookahead — it is what an expanding-window
    estimator *is*, and ``features/kalman.py`` documents the identical property
    for the Kalman variances.

    Recording it as a test matters because the alternative reading — "the
    history should be frozen" — leads to freezing the first fit forever, which
    would make every later Ω a projection onto axes chosen from 2000-2008.
    """
    early = omega_engine.features(accessor_at(omega_observations, date(2008, 10, 31)))
    late = omega_engine.features(accessor_at(omega_observations, date(2010, 10, 1)))

    early_spec = omega_engine.fit(early)
    late_spec = omega_engine.fit(late)

    assert early_spec.loadings != late_spec.loadings
    assert early_spec.sign_reference == late_spec.sign_reference

    a = omega_engine.transform(early, early_spec)
    b = omega_engine.transform(late.loc[early.index], late_spec)
    assert not a.equals(b)
    # Different, but describing the same thing: the two agree on the shape.
    correlation = float(a.dropna().corr(b.dropna()))
    assert correlation > 0.95, f"a two-year-later refit reorients Ω (rho={correlation:.3f})"
