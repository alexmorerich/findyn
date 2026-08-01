"""The regime model, and the acceptance criterion P4 is graded on.

The headline test is :func:`test_reference_windows_classify_correctly`, and it is
slow on purpose: it refits the Markov chain five times, once per window, on data
that ends before the window it is about to grade. A faster test that fitted once
over the whole history would pass more easily and prove less — see
``findynamics/engines/gold/backtest.py`` for why.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from findynamics.engines.gold import backtest as backtest_mod
from findynamics.engines.gold import drivers as drivers_mod
from findynamics.engines.gold import regime as regime_mod
from findynamics.engines.gold.domain import GOLD_REGIMES
from findynamics.engines.gold.regime import RegimeRules, RegimeUnavailable
from tests.engines.gold.conftest import SERIES_IDS, wide_frame

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")


@pytest.fixture(scope="module")
def monthly(request) -> pd.DataFrame:
    """The month-end driver panel over the whole committed snapshot."""
    observations = request.getfixturevalue("gold_observations")
    panel = drivers_mod.build_panel(wide_frame(observations), SERIES_IDS, drivers_mod.DriverRules())
    return panel.monthly


# --------------------------------------------------------------------------
# The acceptance criterion
# --------------------------------------------------------------------------


@pytest.mark.slow
def test_reference_windows_classify_correctly(monthly):
    """P4 acceptance: five windows, five fits, none of which saw its own window.

    2008H2, 2011 and COVID must read as gold being bid for protection; the 2013
    taper tantrum and the 2022 tightening cycle must read as a rate headwind. The
    last two are the ones that matter — a model that calls everything a crisis
    passes the first three and is worthless.
    """
    results = backtest_mod.walk_forward(monthly, RegimeRules())
    failures = [r for r in results if not r.ok]
    assert not failures, backtest_mod.summarize(results)


@pytest.mark.slow
def test_the_fit_never_sees_its_own_window(monthly):
    """The property the whole backtest rests on, asserted rather than assumed."""
    for window in backtest_mod.REFERENCE_WINDOWS:
        result = backtest_mod.evaluate_window(monthly, window, RegimeRules())
        assert pd.Timestamp(result.fitted_through) < pd.Timestamp(window.start), (
            f"{window.name}: fitted through {result.fitted_through}, which is inside "
            f"the window starting {window.start}"
        )


@pytest.mark.slow
def test_the_driver_gates_are_what_classify_the_rate_shocks(monthly):
    """Turning Block 2 off must break exactly the two carry windows.

    This is the evidence for the claim in regime.py's docstring — that a Markov
    chain on returns alone cannot see a rate headwind — rather than a restatement
    of it. If this test ever starts passing with the gates off, the docstring is
    wrong and this module's design should be revisited.

    Ungated is a genuine three-state model, not a stub: the chain's own filtered
    states, mapped by variance rank and drift
    (:func:`regime.markov_only_posterior`). It is allowed to assign
    ``carry_headwind`` and simply does not, which is the whole point. The
    crisis windows are checked too, and must still pass — the failure has to be
    specific to the rate shocks, or the ungated model is merely broken and
    proves nothing about which axis the gates supply.
    """
    ungated = RegimeRules(driver_gates=False)
    results = backtest_mod.walk_forward(monthly, ungated)
    by_name = {r.window.name: r for r in results}

    carry_windows = [
        w.name for w in backtest_mod.REFERENCE_WINDOWS if "carry_headwind" in w.accepted
    ]
    crisis_windows = [w.name for w in backtest_mod.REFERENCE_WINDOWS if w.name not in carry_windows]

    assert all(not by_name[name].ok for name in carry_windows), (
        "the raw Markov posterior classified a rate-shock window correctly; "
        "regime.py's justification for the driver gates needs revisiting:\n"
        + backtest_mod.summarize(results)
    )
    assert all(by_name[name].ok for name in crisis_windows), (
        "the ungated model failed a crisis window too, so it is simply broken and "
        "says nothing about what the gates contribute:\n" + backtest_mod.summarize(results)
    )
    # And it really can express the label it never chooses.
    assert any(r.mean_posterior["carry_headwind"] > 0.0 for r in results)


# --------------------------------------------------------------------------
# The chain
# --------------------------------------------------------------------------


@pytest.mark.slow
def test_fit_is_reproducible(monthly):
    """A refit on identical data must land in the same place.

    Not a nicety: ``search_reps`` draws random starting values, and without the
    configured seed the replay test would be measuring the optimizer rather than
    the model.
    """
    first = regime_mod.fit(monthly, RegimeRules())
    second = regime_mod.fit(monthly, RegimeRules())
    assert first.params == pytest.approx(second.params, rel=1e-9)
    assert first.violent_state == second.violent_state


def test_fit_declines_on_short_history(monthly):
    with pytest.raises(RegimeUnavailable, match="floor"):
        regime_mod.fit(monthly.iloc[:60], RegimeRules())


@pytest.mark.slow
def test_the_violent_state_is_the_widest_one(monthly):
    """The chain's contribution is identified by variance, so that must hold."""
    model_fit = regime_mod.fit(monthly, RegimeRules())

    assert len(model_fit.variances) == model_fit.k_regimes
    assert len(model_fit.intercepts) == model_fit.k_regimes
    assert int(np.argmax(model_fit.variances)) == model_fit.violent_state
    # And it is genuinely the violent one, not a hair above the others.
    widest = max(model_fit.variances)
    others = sorted(model_fit.variances)[:-1]
    assert widest > 2.0 * max(others)


@pytest.mark.slow
def test_posterior_is_a_distribution(monthly):
    model_fit = regime_mod.fit(monthly, RegimeRules())
    view = regime_mod.posterior(monthly, model_fit, RegimeRules())

    assert list(view.posterior.columns) == list(GOLD_REGIMES)
    assert (view.posterior >= 0).all().all()
    assert (view.posterior <= 1).all().all()
    assert view.posterior.sum(axis=1).round(9).eq(1.0).all()


@pytest.mark.slow
def test_posterior_is_causal(monthly):
    """A month's posterior must not move when later months arrive.

    Filtered, not smoothed. Under smoothing every value in this comparison would
    change, which is exactly why the module uses one and not the other.
    """
    model_fit = regime_mod.fit(monthly.loc[:"2010-12"], RegimeRules())

    full = regime_mod.posterior(monthly, model_fit, RegimeRules()).posterior
    truncated = regime_mod.posterior(monthly.loc[:"2015-12"], model_fit, RegimeRules()).posterior

    shared = truncated.index
    pd.testing.assert_frame_equal(full.loc[shared], truncated)


# --------------------------------------------------------------------------
# The artifact round-trip
# --------------------------------------------------------------------------


@pytest.mark.slow
def test_fit_round_trips_through_its_payload(monthly):
    """The fit has to survive JSON: monthly_refit and daily are different runs."""
    import json

    original = regime_mod.fit(monthly, RegimeRules())
    restored = regime_mod.MarkovFit.from_payload(json.loads(json.dumps(original.to_payload())))

    assert restored is not None
    assert restored.params == pytest.approx(original.params, rel=1e-12)
    assert restored.exog_columns == original.exog_columns
    assert restored.violent_state == original.violent_state
    assert restored.fitted_through == original.fitted_through


def test_an_unusable_payload_degrades_rather_than_raising():
    """A daily run must not die because an artifact is from an older model."""
    assert regime_mod.MarkovFit.from_payload({}) is None
    assert regime_mod.MarkovFit.from_payload({"params": "not a list"}) is None
    assert regime_mod.MarkovFit.from_payload({"params": [1.0], "exog_columns": ["x"]}) is None


@pytest.mark.slow
def test_a_fit_from_a_different_specification_is_refused(monthly):
    """Parameters are positional; a mismatched vector must not be filtered with."""
    model_fit = regime_mod.fit(monthly, RegimeRules())
    wrong = regime_mod.MarkovFit(
        params=(*model_fit.params, 0.0),
        exog_columns=model_fit.exog_columns,
        k_regimes=model_fit.k_regimes,
        violent_state=model_fit.violent_state,
        n_observations=model_fit.n_observations,
        log_likelihood=model_fit.log_likelihood,
        fitted_through=model_fit.fitted_through,
    )
    with pytest.raises(RegimeUnavailable, match="different model"):
        regime_mod.posterior(monthly, wrong, RegimeRules())


def test_a_fit_naming_unknown_columns_is_refused(monthly):
    model_fit = regime_mod.MarkovFit(
        params=(0.1,) * 14,
        exog_columns=("z_nonexistent",),
        k_regimes=3,
        violent_state=2,
        n_observations=400,
        log_likelihood=-1.0,
        fitted_through=monthly.index[-1].date(),
    )
    with pytest.raises(RegimeUnavailable, match="missing"):
        regime_mod.posterior(monthly, model_fit, RegimeRules())


# --------------------------------------------------------------------------
# The gates
# --------------------------------------------------------------------------


def test_gates_are_monotone_in_their_drivers():
    index = pd.date_range("2000-01-31", periods=5, freq="ME")
    monthly = pd.DataFrame(
        {
            "z_stress": [-2.0, -1.0, 0.0, 1.0, 2.0],
            "z_real_rate_change_12m": [-2.0, -1.0, 0.0, 1.0, 2.0],
            "z_usd_trend": [-2.0, -1.0, 0.0, 1.0, 2.0],
        },
        index=index,
    )
    stress_gate, carry_gate = regime_mod.gates(monthly, RegimeRules())

    assert stress_gate.is_monotonic_increasing
    assert carry_gate.is_monotonic_increasing
    assert stress_gate.between(0.0, 1.0).all()
    assert carry_gate.between(0.0, 1.0).all()


def test_absent_drivers_leave_the_gates_neutral():
    """A missing driver must not read as a driver at its floor."""
    index = pd.date_range("2000-01-31", periods=3, freq="ME")
    monthly = pd.DataFrame({"z_stress": [np.nan] * 3}, index=index)
    stress_gate, carry_gate = regime_mod.gates(monthly, RegimeRules())

    rules = RegimeRules()
    expected_stress = 1.0 / (1.0 + np.exp(rules.stress_weight * rules.stress_offset))
    assert stress_gate.round(9).eq(round(expected_stress, 9)).all()
    assert (carry_gate < 0.5).all()


def test_config_rejects_an_impossible_regime_count():
    with pytest.raises(ValueError, match="k_regimes"):
        RegimeRules.from_params({"regime": {"k_regimes": 5}})


def test_config_rejects_a_non_mapping_block():
    with pytest.raises(ValueError, match="must be a mapping"):
        RegimeRules.from_params({"regime": [1, 2, 3]})
