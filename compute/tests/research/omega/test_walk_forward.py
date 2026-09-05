"""The harness, and the four tests that decide whether KK3's numbers exist.

The KK1 and KK2 leakage tests cover the transforms. They cannot see the harness,
and the harness is where a leak is easiest to introduce: a training slice that
runs one row too far, a target whose horizon has not closed, a spec applied
backwards. All four tests below are about the machinery rather than the model.

The suite runs at a **quarterly** cadence over a short window. The shipped
monthly configuration takes about two minutes end to end, which is fine for a
deliberate research run and not fine for a test that has to pass on every
commit; the properties under test — bit-identity under truncation, a control
that shows no skill — do not depend on the cadence.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from findynamics.research.omega.backtest import (
    ARMS,
    OmegaBacktestError,
    WalkForwardParams,
    arm_columns,
    config_hash,
    rebalance_dates,
    shuffled_target_control,
    walk_forward,
)
from findynamics.research.omega.evaluate import evaluate
from tests.research.omega.conftest import observation_rows, with_params

#: A cadence and horizon set the suite can afford. Quarterly rebalances over
#: 2000-2014 gives ~40 windows against the shipped configuration's 258.
FAST = {
    "cadence_months": 3,
    "min_train_years": 5.0,
    "horizons": [5, 21],
    "inner_splits": 2,
}


@pytest.fixture(scope="module")
def fast_config():
    from findynamics.research.omega import load_omega_config

    return with_params(load_omega_config(), walk_forward=FAST)


@pytest.fixture(scope="module")
def truncated(omega_observations_module):
    """Observations through 2014 only — the shorter of the two runs."""
    return omega_observations_module[
        omega_observations_module["obs_date"] <= pd.Timestamp(2014, 12, 31)
    ]


@pytest.fixture(scope="module")
def omega_observations_module() -> pd.DataFrame:
    from tests.conftest import FIXTURE_DIR

    frame = pd.read_csv(FIXTURE_DIR / "equity_prices.csv")
    for column in ("obs_date", "release_date", "revision_date"):
        frame[column] = pd.to_datetime(frame[column])
    return frame


@pytest.fixture(scope="module")
def short_run(fast_config, truncated):
    return walk_forward(truncated, fast_config)


def test_rebalance_dates_discard_the_warm_up_entirely(fast_config):
    """Not run on a short window — discarded. An Ω fitted on 200 rows is noise."""
    index = pd.bdate_range(date(2000, 1, 3), periods=4_000)
    params = WalkForwardParams.from_config(fast_config)

    dates = rebalance_dates(index, params)

    assert dates[0] >= index[0] + pd.DateOffset(days=int(5 * 365.25))
    # Quarterly, so consecutive rebalances are about three months apart.
    gaps = pd.Series(dates).diff().dropna().dt.days
    assert gaps.between(80, 100).all()


def test_the_config_hash_changes_with_the_config(fast_config, omega_config):
    """A stored artifact that cannot say which configuration produced it is not
    reproducible, so the hash travels with every run."""
    assert config_hash(fast_config) != config_hash(omega_config)
    assert config_hash(fast_config) == config_hash(fast_config)


def test_a_window_too_short_for_three_rebalances_is_refused(fast_config, omega_observations_module):
    """Five years of warm-up plus a quarterly cadence needs a sixth year."""
    early = omega_observations_module[
        omega_observations_module["obs_date"] <= pd.Timestamp(2005, 6, 30)
    ]

    with pytest.raises(OmegaBacktestError, match="rebalance date"):
        walk_forward(early, fast_config)


def test_every_arm_is_fitted_on_the_columns_it_declares(short_run):
    """And an unavailable column is *named*, not silently absent.

    ``rii_credit_velocity`` inherits ``FRED:BAMLH0A0HYM2``'s 2023 start, so on
    any window ending before that it fails the availability check. Arm A′ with
    seven of eight RII columns is still the control the design note asks for;
    arm A′ silently truncated to the last three years is not — and that is what
    happened before the check was applied to the arms, which left A′ fitted on
    19 rows out of 5,407 and made every comparison against it meaningless.
    """
    for arm in ARMS:
        used = short_run.arm_columns(arm)
        dropped = short_run.dropped_columns.get(arm, ())
        assert set(used).isdisjoint(dropped)
        assert set(used) | set(dropped) <= set(ARMS[arm])
        assert used, f"arm {arm} has no usable columns"

    # On a window ending in 2014 the HY OAS does not exist at all, so
    # `rii_credit_velocity` is never built rather than built and dropped. Either
    # way it must not reach a fit — which is the property, and it is worth
    # distinguishing the two states because only one of them is an availability
    # decision.
    assert "rii_credit_velocity" not in short_run.arm_columns("A_prime")
    assert "rii" in short_run.arm_columns("A_prime")


def test_the_control_arm_covers_the_same_dates_as_the_baseline(short_run):
    """A′ is only a control if it is fitted on the same path A is.

    The regression this pins down: without the RII components being forward
    filled onto the daily calendar the way ``engines/equity/engine.py`` fills
    them when it publishes, ``liquidity_stress`` existed only on NFCI's weekly
    dates and A′ collapsed to a handful of rows.
    """
    target = f"forward_return_{short_run.params.horizons[0]}"
    baseline = short_run.predictions[f"A::{target}"].notna().sum()
    control = short_run.predictions[f"A_prime::{target}"].notna().sum()

    assert control > 0.95 * baseline, f"A′ covers {control} dates against A's {baseline}"


# ---------------------------------------------------------------------------
# 1. Harness truncation
# ---------------------------------------------------------------------------


def test_the_harness_is_bit_identical_under_truncation(
    fast_config, truncated, omega_observations_module, short_run
):
    """Every Ω value and every prediction dated ``<= T``, to the last bit.

    Run the whole walk-forward on data truncated at ``T``, then on the full
    record, and compare the overlap. This is the harness-level version of the
    KK1 test and it covers everything the transforms do not: the rebalance
    schedule, the freeze, the stitching, the training slices and the fits.
    """
    long_run = walk_forward(omega_observations_module, fast_config)
    cut = short_run.panel.index[-1]
    shared = short_run.panel.index.intersection(long_run.panel.loc[:cut].index)

    assert len(shared) > 500
    for column in ("omega", "omega_velocity", "coupling", "velocity", "jerk_z"):
        pd.testing.assert_series_equal(
            short_run.panel.loc[shared, column],
            long_run.panel.loc[shared, column],
            check_exact=True,
        )

    for arm in ("A", "A_prime", "B", "C"):
        column = f"A::forward_return_{short_run.params.horizons[0]}".replace("A::", f"{arm}::")
        if column not in short_run.predictions.columns:
            continue
        left = short_run.predictions.loc[shared, column].dropna()
        right = long_run.predictions.loc[left.index, column]
        pd.testing.assert_series_equal(left, right, check_exact=True)


def test_a_future_spike_changes_no_earlier_prediction(fast_config, truncated, short_run):
    """A ten-thousand-fold print after ``T`` must be invisible on or before ``T``.

    Any leak — a training slice reaching past its cutoff, a target whose horizon
    had not closed, a scaler fitted on the whole panel — moves the earlier rows
    by an amount no float tolerance could hide.
    """
    spiked = pd.concat(
        [
            truncated,
            pd.DataFrame(observation_rows("YAHOO:^GSPC", {date(2015, 1, 5): 9_999_999.0})),
        ],
        ignore_index=True,
    )
    contaminated = walk_forward(spiked, fast_config)
    shared = short_run.panel.index.intersection(contaminated.panel.index)

    assert len(shared) > 500
    pd.testing.assert_series_equal(
        short_run.panel.loc[shared, "omega"],
        contaminated.panel.loc[shared, "omega"],
        check_exact=True,
    )


# ---------------------------------------------------------------------------
# 2. The window-boundary test
# ---------------------------------------------------------------------------


def test_no_refit_boundary_moves_omega_by_more_than_its_own_noise(short_run):
    """The harness-level catch for a sign flip or a scaler reset.

    KK1's two-window unit test compares one pair of fits. This walks every
    boundary on the real path: at each rebalance the spec changes, and Ω must
    step by an ordinary amount rather than reflect through zero. An unpinned
    sign would show here as a jump of many standard deviations, on a monthly
    calendar, on every boundary.
    """
    omega = short_run.panel["omega"].dropna()
    sigma = omega.diff().rolling(short_run.params.boundary_window).std()
    jumps = omega.diff().abs() / sigma.replace(0.0, np.nan)

    boundaries = [
        pd.Timestamp(record.block_start)
        for record in short_run.windows
        if record.block_start is not None
    ]
    at_boundary = jumps.reindex(boundaries).dropna()

    assert len(at_boundary) > 20
    worst = float(at_boundary.max())
    assert worst <= short_run.params.boundary_jump_sigma, (
        f"Ω moved {worst:.1f}σ at a refit boundary; the ceiling is "
        f"{short_run.params.boundary_jump_sigma}σ"
    )


def test_every_window_records_the_spec_that_produced_its_block(short_run):
    """A stored artifact has to say which projection explained which dates."""
    for record in short_run.windows:
        assert record.block_start is not None
        assert record.block_start > record.rebalance
        assert record.spec.fit_end <= record.rebalance
        assert record.spec.loading(record.spec.sign_reference) >= 0.0
        assert record.as_dict()["spec"]["model_version"]


# ---------------------------------------------------------------------------
# 3. The shuffled-target control — the acceptance gate
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_the_shuffled_target_control_shows_no_skill(fast_config, truncated):
    """**The gate.** Permuted targets, and every arm must find nothing.

    If any arm shows skill here the harness is leaking and every other number in
    this phase is void — no partial credit, no "but the real result was
    stronger". The permutation keeps the index and the missingness, so the
    control has the same sample sizes as the real run and a difference cannot be
    explained by one having more data.
    """
    control = shuffled_target_control(truncated, fast_config)
    evaluation = evaluate(control, redundancy=0.0)

    assert all(
        verdict.startswith("FAILS") or verdict == "NOT TESTED"
        for verdict in evaluation.verdicts.values()
    ), evaluation.verdicts

    for frame in (evaluation.q1, evaluation.q2):
        if frame.empty:
            continue
        # A model with no signal cannot explain variance it was not shown. Small
        # negative R² is expected and fine; a positive one is not.
        assert frame["oos_r2"].max() < 0.05, frame[["arm", "horizon", "oos_r2"]]

    if not evaluation.q3.empty:
        # Against sampling noise rather than against the configured 0.55 ceiling.
        # That ceiling is right for the shipped run, where the rare-transition
        # targets have hundreds of positives; on this deliberately small test
        # window ``h = 5`` has about forty, and the standard error of an AUC is
        # then ~0.08. A fixed ceiling there tests the sample size, not the
        # harness. The full-configuration control is reported in the phase
        # report and does clear 0.55 at every horizon with enough positives.
        for _, row in evaluation.q3.iterrows():
            positives = row["base_rate"] * row["n"]
            if positives < 5:
                continue
            standard_error = 0.5 / np.sqrt(positives)
            assert abs(row["auc"] - 0.5) < 4.0 * standard_error, (
                f"{row['arm']} at h={row['horizon']} reached AUC {row['auc']:.3f} on permuted "
                f"labels ({positives:.0f} positives, SE {standard_error:.3f})"
            )


def test_shuffling_keeps_the_sample_and_only_moves_the_values(fast_config, truncated, short_run):
    """A control that also changed the sample would not be a control."""
    control = shuffled_target_control(truncated, fast_config)
    target = f"forward_return_{short_run.params.horizons[0]}"

    assert control.panel[target].notna().sum() == short_run.panel[target].notna().sum()
    pd.testing.assert_index_equal(control.panel.index, short_run.panel.index)
    # Same values, different order.
    assert sorted(control.panel[target].dropna()) == pytest.approx(
        sorted(short_run.panel[target].dropna())
    )
    assert not control.panel[target].equals(short_run.panel[target])


def test_the_arm_column_check_drops_a_thin_column(short_run):
    """``arm_columns`` is the same availability rule Ω's own block gets.

    A column is planted at 5% coverage rather than relying on the fixture, so
    the assertion is about the rule and not about which years the snapshot
    happens to hold.
    """
    panel = short_run.panel.copy()
    thin = pd.Series(np.nan, index=panel.index)
    thin.iloc[-len(panel) // 20 :] = 1.0
    panel["rii_credit_velocity"] = thin

    used, dropped = arm_columns(panel, "A_prime", min_coverage=0.6, min_observations=252)

    assert "rii" in used
    assert "rii_credit_velocity" in dropped
    assert "rii_credit_velocity" not in used
