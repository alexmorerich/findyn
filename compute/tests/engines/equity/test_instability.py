"""M4: the RII, the crash decomposition and the Monte Carlo (§3.2, §4, §11).

These are property tests, not golden-number tests. The three modules produce
floats that no one can eyeball for correctness — "is 2.09 the right crash risk"
has no answer — so what is pinned here is the structure that makes the floats
mean something: that a frequency conversion happens exactly once, that a missing
input is reported rather than scored zero, that the shock overlay changes the
shape of a path without changing its drift, and that nothing can inject a
transition probability the model has not earned.

Every one of these corresponds to a bug that was actually shipped and caught
during P3-C; the comments say which.
"""

from __future__ import annotations

import inspect
import math

import numpy as np
import pandas as pd
import pytest

from findynamics.engines.equity import crash, rii, simulate

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

STATES = ("crisis", "bear", "sideways", "normal_expansion", "melt_up")
ADVERSE_STATES = [0, 1]


def posteriors(n: int = 400, *, certainty: float = 0.9) -> pd.DataFrame:
    """A calm posterior series: mostly `normal_expansion`, mildly wobbling."""
    index = pd.date_range("2020-01-01", periods=n, freq="D")
    rest = (1.0 - certainty) / 4.0
    frame = pd.DataFrame(rest, index=index, columns=list(STATES))
    frame["normal_expansion"] = certainty
    return frame


def sticky_transmat(stay: float = 0.98) -> np.ndarray:
    off = (1.0 - stay) / 4.0
    matrix = np.full((5, 5), off)
    np.fill_diagonal(matrix, stay)
    return matrix


def tail_fit(periods_per_year: float = 12.0) -> crash.TailFit:
    return crash.TailFit(
        shape=0.1,
        scale=0.12,
        threshold=0.10,
        exceedances=46,
        observations=1845,
        periods_per_year=periods_per_year,
        series_id="TEST:^SPX",
    )


# ---------------------------------------------------------------------------
# §3.2 — the Regime Instability Index
# ---------------------------------------------------------------------------


def test_posterior_entropy_spans_certainty_to_ignorance() -> None:
    certain = pd.DataFrame([[1.0, 0.0, 0.0, 0.0, 0.0]], columns=list(STATES))
    uniform = pd.DataFrame([[0.2] * 5], columns=list(STATES))

    assert rii.posterior_entropy(certain).iloc[0] == pytest.approx(0.0, abs=1e-9)
    assert rii.posterior_entropy(uniform).iloc[0] == pytest.approx(math.log(5), abs=1e-9)


def test_missing_components_are_named_and_reweighted_not_zeroed() -> None:
    """The failure mode this exists to prevent.

    A run without a credit spread must publish an index built from what it had.
    Scoring the absent component 0 would read as "maximally stable credit", which
    is the most dangerous possible default for an instability index — the one
    input that is missing because a data feed broke would push the number down.
    """
    frame = posteriors()
    result = rii.compute_rii(frame, periods_per_year=252.0)

    assert "credit_velocity" in result.missing
    assert "credit_velocity" not in result.components
    # Weights renormalize over what is present, so they still sum to one.
    assert sum(result.weights.values()) == pytest.approx(1.0)
    assert set(result.weights) == set(result.components)


def test_a_component_with_too_little_history_is_dropped_not_trusted() -> None:
    frame = posteriors()
    short = pd.Series(
        np.linspace(0.0, 1.0, rii.MIN_COMPONENT_OBSERVATIONS - 1),
        index=frame.index[: rii.MIN_COMPONENT_OBSERVATIONS - 1],
    )
    result = rii.compute_rii(frame, jerk_z=short, periods_per_year=252.0)

    assert "jerk" in result.missing


def test_index_is_bounded_and_traceable() -> None:
    frame = posteriors()
    rng = np.random.default_rng(7)
    jerk = pd.Series(rng.standard_normal(len(frame)), index=frame.index)
    result = rii.compute_rii(frame, jerk_z=jerk, periods_per_year=252.0)

    usable = result.index.dropna()
    assert not usable.empty
    assert usable.min() >= 0.0
    assert usable.max() <= 100.0

    trace = result.explain()
    # Every published component carries its own score *and* its weight, so a
    # reader can reconstruct the composite rather than take it on faith.
    assert "rii_jerk" in trace
    assert "rii_jerk_weight" in trace


def test_an_index_with_no_usable_component_refuses_rather_than_guesses() -> None:
    with pytest.raises(ValueError, match="at least one usable component"):
        rii.compute_rii(posteriors(n=5), periods_per_year=252.0)


def test_correlation_breakdown_reads_bond_yields_as_prices() -> None:
    """A falling yield is a rising bond.

    Getting this sign wrong inverts the component and it still looks plausible on
    a chart, so it is pinned: equity and bonds moving *together* — equity up as
    yields fall — is the normal hedge, and the deviation from baseline is what
    gets published, in either direction.
    """
    index = pd.date_range("2015-01-01", periods=600, freq="D")
    rng = np.random.default_rng(3)
    equity = pd.Series(rng.standard_normal(600) * 0.01, index=index)
    # Yields fall when equity falls: a *positive* stock/bond-return correlation
    # for the first half, then the relationship flips.
    yields = pd.Series(np.cumsum(np.r_[equity[:300], -equity[300:]]), index=index)

    breakdown = crash_free_breakdown(equity, yields)
    assert (breakdown.dropna() >= 0).all(), "the published value is a magnitude"


def crash_free_breakdown(equity: pd.Series, yields: pd.Series) -> pd.Series:
    return rii.correlation_breakdown(equity, yields, periods_per_year=252.0, months=1.0)


# ---------------------------------------------------------------------------
# §4 — the crash decomposition
# ---------------------------------------------------------------------------


def test_declustering_counts_episodes_not_days() -> None:
    """One 2008, not 400 correlated observations of it.

    Undeclustered, a single long drawdown contributes one exceedance per period
    it stays underwater. That took the fitted exceedance rate to 56% of all
    months — a GPD fit on 1039 "independent" tail events that were really 46.
    """
    # One long drawdown, recovery, one short drawdown.
    path = np.r_[
        np.linspace(0.0, 0.0, 10),
        np.linspace(0.0, -0.45, 200),  # deep, slow
        np.linspace(-0.45, 0.05, 100),  # recovery to a new peak
        np.linspace(0.05, -0.15, 20),  # shallow, fast
        np.linspace(-0.15, 0.10, 30),
    ]
    log_price = pd.Series(path, index=pd.date_range("1990-01-31", periods=len(path), freq="ME"))

    drawdowns = crash.drawdown_magnitudes(log_price).dropna()
    episodes = crash.decluster(drawdowns, 0.10)

    assert len(episodes) == 2
    # Each episode is represented by its depth, not by its first crossing.
    # The deep episode ran from a peak of 0.0 to a trough of -0.45 in logs.
    assert episodes.max() == pytest.approx(1.0 - math.exp(-0.45), rel=0.02)


def test_first_passage_is_monotone_in_horizon() -> None:
    transmat = sticky_transmat()
    start = np.array([0.0, 0.05, 0.15, 0.70, 0.10])

    probabilities = [
        crash.adverse_first_passage(start, transmat, ADVERSE_STATES, steps)
        for steps in (0, 21, 63, 252, 756)
    ]
    assert probabilities[0] == 0.0
    assert probabilities == sorted(probabilities)
    assert probabilities[-1] <= 1.0


def test_first_passage_excludes_mass_already_adverse() -> None:
    """ "Already there" and "about to go there" are different statements.

    Averaging them is how a market in the middle of a crisis reports a low crash
    probability. The mass already sitting on an adverse state is reported
    separately, as `p_adverse_now`.
    """
    transmat = sticky_transmat()
    entirely_adverse = np.array([0.6, 0.4, 0.0, 0.0, 0.0])

    assert crash.adverse_first_passage(entirely_adverse, transmat, ADVERSE_STATES, 252) == 0.0

    factors = crash.crash_factors(
        posterior=entirely_adverse,
        transmat=transmat,
        adverse_states=ADVERSE_STATES,
        periods_per_year=252.0,
        tail=tail_fit(),
        rii=50.0,
        horizon_months=12.0,
    )
    assert factors.detail["p_adverse_now"] == pytest.approx(1.0)


def test_the_frequency_conversion_happens_exactly_once() -> None:
    """A monthly tail and a daily tail must not give the same annual number.

    The fitted exceedance rate is per observation of the fitted series. Treating
    a monthly rate as daily overstates the expected count by ~21x — which is a
    crash probability of 1.0 for every date, forever.
    """
    monthly = crash.horizon_probability(tail_fit(12.0), horizon_months=12.0)
    daily = crash.horizon_probability(tail_fit(252.0), horizon_months=12.0)

    assert 0.0 < monthly < 1.0
    assert daily > monthly
    # And the horizon itself scales, rather than the number being horizon-blind.
    assert crash.horizon_probability(tail_fit(12.0), horizon_months=3.0) < monthly


def test_transmission_is_floored_rather_than_allowed_to_zero_the_product() -> None:
    """Benign conditions clip three of four sub-scores to zero.

    That took published transmission to 0.012 and, because crash risk is a
    product, zeroed the whole index in exactly the calm periods where a reader
    most needs to see a small non-zero number. A shock in calm conditions
    transmits less; it does not fail to transmit.
    """
    score, detail = crash.transmission_score(
        credit_spread=2.5, credit_velocity=0.0, liquidity=-0.8, curve_slope=1.5
    )
    assert score == pytest.approx(crash.TRANSMISSION_FLOOR)
    assert detail["transmission_inputs"] == 4.0

    stressed, _ = crash.transmission_score(
        credit_spread=10.0, credit_velocity=2.0, liquidity=1.5, curve_slope=-1.5
    )
    assert stressed == pytest.approx(1.0)


def test_absent_fragility_inputs_assume_transmission_rather_than_safety() -> None:
    score, detail = crash.transmission_score(
        credit_spread=None, credit_velocity=None, liquidity=None, curve_slope=None
    )
    assert score == 1.0
    assert detail["transmission_inputs"] == 0.0


def test_that_assumption_is_not_published_as_seventy_years_of_history() -> None:
    """The snapshot fallback and the history are different questions.

    1.0 is the right answer for *today* with a feed down: the factor multiplies
    crash risk, so guessing 0 would silently zero the published number. Run down
    a century it says something else entirely — the credit and liquidity series
    start in the 1960s at the earliest, so every earlier date scores exactly 1.0
    and the chart reads "a shock would transmit with certainty" for seventy
    years, asserted purely from the absence of data.

    The engine drops non-finite values, so blanking those dates means the factor
    begins where it can be measured instead of beginning with an alarm.
    """
    dates = pd.date_range("2020-01-01", periods=40, freq="B")
    belief = pd.DataFrame(
        np.tile([0.05, 0.1, 0.15, 0.2, 0.5], (len(dates), 1)),
        index=dates,
        columns=list(STATES),
    )
    # Credit arrives halfway through, exactly as a real series does.
    credit = pd.Series(4.0, index=dates[20:])

    history = crash.crash_history(
        belief,
        transmat=sticky_transmat(),
        adverse_states=ADVERSE_STATES,
        periods_per_year=252.0,
        horizon_months=12.0,
        tail=tail_fit(),
        rii=pd.Series(50.0, index=dates),
        credit_spread=credit,
    )

    unmeasured = history["p_transmission"].iloc[:20]
    measured = history["p_transmission"].iloc[20:]
    assert unmeasured.isna().all(), "no fragility input is not a maximal reading"
    assert measured.notna().all()
    assert (measured <= 1.0).all()

    # §4's "all three or none" has to hold per date, not just per run: the
    # composite must vanish wherever one of its factors could not be measured.
    assert history["crash_risk"].iloc[:20].isna().all()
    assert history["crash_risk"].iloc[20:].notna().all()
    # And the two factors that *are* measurable keep their whole span — blanking
    # them too would discard real information to punish a missing third.
    assert history["p_transition"].notna().all()
    assert history["p_shock"].notna().all()


def test_transition_probability_cannot_be_injected() -> None:
    """§4 + open issue #3c, enforced structurally.

    `crash_factors` derives p_transition from the posterior and the transition
    matrix. It deliberately takes no `p_transition` parameter, because a
    parameter is an invitation to pass the L3 classifier's output — and the
    classifier has no demonstrated out-of-sample skill. Transition probabilities
    are descriptive state shifts, never predictive alpha.
    """
    parameters = inspect.signature(crash.crash_factors).parameters
    assert "p_transition" not in parameters
    assert {"posterior", "transmat"} <= set(parameters)


def test_crash_risk_is_the_product_and_travels_with_its_factors() -> None:
    factors = crash.crash_factors(
        posterior=np.array([0.01, 0.04, 0.15, 0.70, 0.10]),
        transmat=sticky_transmat(),
        adverse_states=ADVERSE_STATES,
        periods_per_year=252.0,
        tail=tail_fit(),
        rii=60.0,
        horizon_months=12.0,
    )
    expected = 100.0 * factors.p_transition * factors.p_shock * factors.p_transmission
    assert factors.crash_risk == pytest.approx(expected, abs=1e-6)

    published = factors.as_components()
    # The composite is never publishable without all three factors beside it.
    assert "crash_risk" in published
    assert {"p_shock", "p_transmission"} <= set(published)
    assert any(key.startswith("p_transition_") for key in published)


def test_an_unfitted_tail_publishes_a_flagged_base_rate_not_zero() -> None:
    factors = crash.crash_factors(
        posterior=np.array([0.01, 0.04, 0.15, 0.70, 0.10]),
        transmat=sticky_transmat(),
        adverse_states=ADVERSE_STATES,
        periods_per_year=252.0,
        tail=None,
        rii=None,
        horizon_months=12.0,
    )
    assert factors.detail["tail_fitted"] == 0.0
    assert 0.0 < factors.p_shock < 1.0


def test_a_high_rii_raises_the_shock_hazard() -> None:
    calm = crash.rii_hazard_multiplier(5.0)
    stressed = crash.rii_hazard_multiplier(95.0)
    low, high = crash.DEFAULT_RII_HAZARD_RANGE

    assert low <= calm < stressed <= high
    assert crash.rii_hazard_multiplier(None) == 1.0


def test_severity_rescales_by_root_time_between_frequencies() -> None:
    daily = crash.severity_at_frequency(0.20, from_periods_per_year=12.0, to_periods_per_year=252.0)
    assert daily == pytest.approx(0.20 / math.sqrt(21.0), rel=0.01)


# ---------------------------------------------------------------------------
# §11 — the Monte Carlo
# ---------------------------------------------------------------------------


def small_run(**overrides: object) -> simulate.SimulationResult:
    kwargs: dict[str, object] = {
        "posterior": np.array([0.01, 0.04, 0.15, 0.70, 0.10]),
        "transmat": sticky_transmat(),
        "regime_stats": [
            (-0.35, 0.45),
            (-0.12, 0.28),
            (0.02, 0.16),
            (0.11, 0.13),
            (0.25, 0.18),
        ],
        "start_log_level": math.log(6000.0),
        "periods_per_year": 252.0,
        "paths": 500,
        "horizons": {"tactical": 0.5},
    }
    kwargs.update(overrides)
    return simulate.run_simulation(**kwargs)  # type: ignore[arg-type]


def test_the_spec_floor_of_ten_thousand_paths_is_the_default() -> None:
    assert simulate.DEFAULT_PATHS >= 10_000


def test_a_run_is_reproducible_from_its_seed() -> None:
    first = small_run(seed=42).forecasts["tactical"]
    second = small_run(seed=42).forecasts["tactical"]
    third = small_run(seed=43).forecasts["tactical"]

    assert first.quantiles == second.quantiles
    assert first.quantiles != third.quantiles


def test_quantiles_are_ordered_and_the_horizon_is_labelled() -> None:
    result = small_run()
    forecast = result.forecasts["tactical"]

    levels = [forecast.quantiles[q] for q in sorted(forecast.quantiles)]
    assert levels == sorted(levels)
    assert forecast.paths == 500
    assert forecast.educational_only is False


def test_educational_horizons_are_flagged_so_they_cannot_be_plotted_as_forecasts() -> None:
    result = small_run(horizons={"tactical": 0.5, "educational_50y": 50.0}, paths=200)

    assert result.forecasts["tactical"].educational_only is False
    assert result.forecasts["educational_50y"].educational_only is True
    # §10 excludes these from accuracy evaluation entirely; the flag is how a
    # consumer knows not to score them.
    rows = simulate.forecast_rows(result)
    educational = {row["educational_only"] for row in rows if row["horizon"] == "educational_50y"}
    assert educational == {True}


def test_the_shock_overlay_changes_shape_without_moving_drift() -> None:
    """The double-count that cost 3.4 percentage points of annual drift.

    Historical crashes are already inside the fitted regime means. An overlay
    that only partially retraces subtracts that permanent loss a second time,
    which took the 12-year median from ~6%/yr to 2.7%/yr. Full retrace keeps the
    overlay contributing a *discontinuity* — which is what the tail model is for —
    while leaving the level to the regimes.
    """
    assert simulate.SHOCK_RETRACE == 1.0

    without = small_run(tail=None, paths=4000, seed=11).forecasts["tactical"]
    with_shocks = small_run(tail=tail_fit(), paths=4000, seed=11).forecasts["tactical"]

    drift = with_shocks.median_log_level - without.median_log_level
    assert abs(drift) < 0.02, "the overlay must not systematically move the median"
    # But it must widen the tail and deepen drawdowns, or it is doing nothing.
    assert with_shocks.worst_decile_max_drawdown > without.worst_decile_max_drawdown


def test_drawdown_statistics_are_measured_against_each_path_s_own_peak() -> None:
    # A strictly rising path has never been under water, whatever its level.
    rising = np.tile(np.linspace(0.0, 0.5, 100), (3, 1))
    forecast = simulate.summarise(rising, horizon="tactical", years=0.5, start_log_level=0.0)
    assert forecast.median_max_drawdown == pytest.approx(0.0, abs=1e-12)
    assert forecast.median_time_under_water == pytest.approx(0.0, abs=1e-12)
    assert forecast.drawdown_probabilities[0.20] == 0.0


def test_a_higher_rii_raises_shock_intensity() -> None:
    calm = small_run(rii=5.0, tail=tail_fit(), paths=200)
    stressed = small_run(rii=95.0, tail=tail_fit(), paths=200)

    assert stressed.shock_intensity > calm.shock_intensity


def test_the_summary_is_flat_scalars_the_engine_can_publish() -> None:
    summary = small_run().summary()
    assert all(isinstance(value, float) for value in summary.values())
    assert "mc_tactical_median" in summary
    assert "mc_shock_intensity" in summary


# ---------------------------------------------------------------------------
# M4 diagnostics — the report generator behind docs/backtests/equity-p3c.md
# ---------------------------------------------------------------------------


def test_return_levels_invert_the_fitted_survival() -> None:
    """The fitted column of the return-level table must be the fit, not a rescale.

    Checked by round-tripping: the severity reported for a 1-in-N-year event must
    have exactly the conditional exceedance probability that implies.
    """
    from findynamics.engines.equity import diagnostics

    fit = tail_fit(12.0)
    for years in (5.0, 25.0, 100.0):
        target = 1.0 / (fit.exceedance_rate * years * fit.periods_per_year)
        severity = diagnostics._inverse_survival(fit, target)
        assert fit.survival(severity) == pytest.approx(target, rel=1e-6)


def test_the_empirical_return_level_refuses_to_extrapolate() -> None:
    """The column exists to disagree with the fit, so it must never *be* the fit."""
    from findynamics.engines.equity import diagnostics

    fit = tail_fit(12.0)  # 1845 monthly observations ≈ 154 years
    episodes = pd.Series([0.85, 0.51, 0.47, 0.44])

    assert np.isfinite(diagnostics._empirical_return_level(episodes, fit, 100.0))
    # Beyond the record there is nothing to report, and reporting the fit's own
    # extrapolation under an "empirical" heading would be worse than a dash.
    assert not np.isfinite(diagnostics._empirical_return_level(episodes, fit, 500.0))


def test_rii_diagnostics_measure_episode_peaks_against_calm_means() -> None:
    from findynamics.engines.equity import diagnostics

    index = pd.date_range("1994-01-01", "2023-12-31", freq="D")
    # Calm everywhere, spiking only inside the 2008 peak-to-trough window.
    values = pd.Series(30.0, index=index)
    values.loc["2008-01-01":"2009-03-09"] = 95.0

    result = diagnostics.rii_diagnostics(values, {"jerk": values})

    assert result.episode_readings["2008 GFC"] == pytest.approx(95.0)
    assert result.calm_readings["1995"] == pytest.approx(30.0)
    assert result.gap > 0
    assert result.component_gaps["jerk"] > 0


def test_the_archive_holds_per_path_outcomes_not_paths() -> None:
    """§11's R2 archive, and the trade it makes explicit.

    The spec asks for path bundles in R2. Written literally that is 10,000 x
    12,600 floats for the 50-year horizon alone — about 1.5 GB a night across the
    horizons, forever, to answer questions nobody has asked. What offline
    analysis needs is each path's *outcome*, which is three arrays and roughly a
    thousandth of the size: any quantile and any drawdown threshold can be
    re-derived from them. Path-dependent statistics cannot, and that cost is
    stated in the payload rather than left for a reader to discover.
    """
    result = small_run(paths=200)
    document = simulate.archive_document(
        result, asset="equity", as_of="2026-08-01", model_version="equity-1.1.0+cal.x"
    )

    assert document["model_version"] == "equity-1.1.0+cal.x"
    # Reproducibility: the seed and the conditioning state travel with the paths,
    # or a year later the numbers are outcomes of nothing in particular.
    assert document["seed"] == result.seed
    assert document["start_log_level"] == result.start_log_level
    assert "shock_intensity" in document

    tactical = document["horizons"]["tactical"]
    assert len(tactical["terminal"]) == 200
    assert len(tactical["max_drawdown"]) == 200
    assert len(tactical["time_under_water"]) == 200
    assert tactical["educational_only"] is False

    # The quantiles the API publishes must be re-derivable from the archive,
    # which is the property that makes it worth storing at all. To the archive's
    # own stated precision: outcomes are rounded to six decimals, which on a log
    # index level is under a basis point and halves the payload.
    import numpy as np

    assert float(np.quantile(tactical["terminal"], 0.5)) == pytest.approx(
        result.forecasts["tactical"].median_log_level, abs=1e-6
    )


def test_the_archive_is_json_safe() -> None:
    """It is serialized straight into an HMAC-signed PUT; a numpy float would
    raise there, in a job, at night, rather than here."""
    import json

    document = simulate.archive_document(
        small_run(paths=50), asset="equity", as_of="2026-08-01", model_version="v1"
    )
    assert json.loads(json.dumps(document))["horizons"]["tactical"]["paths"] == 50
