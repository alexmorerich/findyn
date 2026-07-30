"""The rates engine end to end.

Two halves: exact assertions against synthetic curves whose factors we chose,
and a sanity backtest over the committed month-end snapshot of real history.
The second is the one that would catch a model that is internally consistent
and wrong about the world.
"""

from __future__ import annotations

from datetime import date

import pytest

from findynamics.core.contracts.state import AssetState
from findynamics.engines.rates.domain import RATE_REGIMES, regime_code
from findynamics.engines.rates.engine import (
    InsufficientCurveDataError,
    RatesEngine,
    carry_and_rolldown,
    par_bond_duration,
)
from findynamics.engines.rates.nelson_siegel import DEFAULT_LAMBDA, fit_curve
from tests.engines.rates.conftest import TENOR_IDS, synthetic_curve_frame, world_from


class TestParBondDuration:
    def test_matches_the_closed_form(self):
        # (1 - 1.04^-9) / 0.04
        assert par_bond_duration(4.0, 9.0) == pytest.approx(7.4353, abs=1e-3)

    def test_at_zero_yield_duration_is_the_maturity(self):
        """The expression is 0/0 there; the limit is n, not a NaN."""
        assert par_bond_duration(0.0, 10.0) == pytest.approx(10.0)

    def test_duration_falls_as_yields_rise(self):
        assert par_bond_duration(8.0, 10.0) < par_bond_duration(2.0, 10.0)

    def test_zero_maturity_has_no_duration(self):
        assert par_bond_duration(4.0, 0.0) == 0.0


class TestCarryAndRolldown:
    def test_on_an_upward_sloping_curve_rolldown_adds_to_carry(self):
        fit = fit_curve(*_curve(4.0, -2.0, 0.0))
        assert fit is not None
        carry, rolldown = carry_and_rolldown(fit, position_years=10.0, horizon_years=1.0)

        assert carry == pytest.approx(float(fit.yield_at(10.0)))
        assert rolldown > 0

    def test_on_an_inverted_curve_rolldown_subtracts(self):
        """The two legs have opposite signs here, which is why they stay apart."""
        fit = fit_curve(*_curve(4.0, 2.0, 0.0))
        assert fit is not None
        _, rolldown = carry_and_rolldown(fit, position_years=10.0, horizon_years=1.0)

        assert rolldown < 0

    def test_on_a_flat_curve_the_return_is_just_carry(self):
        fit = fit_curve(*_curve(3.0, 0.0, 0.0))
        assert fit is not None
        carry, rolldown = carry_and_rolldown(fit, position_years=10.0, horizon_years=1.0)

        assert carry == pytest.approx(3.0, abs=1e-6)
        assert rolldown == pytest.approx(0.0, abs=1e-6)


def _curve(b0: float, b1: float, b2: float):
    import numpy as np

    from findynamics.engines.rates.nelson_siegel import design_matrix

    maturities = np.array(list(TENOR_IDS.values()))
    return maturities, design_matrix(maturities, DEFAULT_LAMBDA) @ np.array([b0, b1, b2])


class TestPredict:
    def test_recovers_the_factors_it_was_given(self, monthly_engine):
        betas = [(4.0, -1.5, 2.0)] * 400
        observations = synthetic_curve_frame(betas)
        world = world_from(observations, date(2021, 6, 1))

        state = monthly_engine.predict(world)

        assert isinstance(state, AssetState)
        assert state.asset == "rates"
        assert state.components is not None
        assert state.components["ns_level"] == pytest.approx(4.0, abs=1e-6)
        assert state.components["ns_slope"] == pytest.approx(1.5, abs=1e-6)
        assert state.components["ns_curvature"] == pytest.approx(2.0, abs=1e-6)
        assert state.components["ns_rmse"] == pytest.approx(0.0, abs=1e-9)

    def test_as_of_is_the_newest_observation_not_the_cutoff(self, monthly_engine):
        """The cutoff is when we looked; as_of is what we are describing."""
        observations = synthetic_curve_frame([(4.0, -1.5, 2.0)] * 30, start=date(2020, 1, 1))
        world = world_from(observations, date(2020, 6, 1))

        state = monthly_engine.predict(world)

        assert world.as_of == date(2020, 6, 1)
        assert state.as_of == date(2020, 1, 30)

    def test_publishes_a_regime_from_the_vocabulary(self, monthly_engine):
        observations = synthetic_curve_frame([(4.0, -1.5, 2.0)] * 400)
        state = monthly_engine.predict(world_from(observations, date(2021, 6, 1)))
        assert state.regime in RATE_REGIMES

    def test_an_inverted_synthetic_curve_is_classified_inverted(self, monthly_engine):
        # b1 > 0 puts the short rate above the long end.
        observations = synthetic_curve_frame([(4.0, 2.0, 0.0)] * 400)
        state = monthly_engine.predict(world_from(observations, date(2021, 6, 1)))

        assert state.regime == "inverted"
        inversion = next(s for s in state.signals if s.name == "curve_inversion")
        assert inversion.value < 0
        assert inversion.direction == -1

    def test_expected_return_is_a_decimal_fraction(self, monthly_engine):
        observations = synthetic_curve_frame([(4.0, -1.5, 0.0)] * 400)
        state = monthly_engine.predict(world_from(observations, date(2021, 6, 1)))

        assert state.expected_return is not None
        # ~4% carry plus a little rolldown, expressed as 0.04-ish not 4.0.
        assert 0.02 < state.expected_return < 0.10
        assert state.components is not None
        assert state.expected_return == pytest.approx(
            (state.components["carry_pct"] + state.components["rolldown_pct"]) / 100.0, abs=1e-6
        )

    def test_scores_stay_inside_their_contracted_ranges(self, monthly_engine):
        observations = synthetic_curve_frame([(4.0, -1.5, 2.0)] * 400)
        state = monthly_engine.predict(world_from(observations, date(2021, 6, 1)))

        assert 0.0 <= state.risk_score <= 100.0
        assert 0.0 <= state.confidence <= 1.0

    def test_a_perfect_fit_on_a_full_cross_section_is_high_confidence(self, monthly_engine):
        observations = synthetic_curve_frame([(4.0, -1.5, 2.0)] * 400)
        state = monthly_engine.predict(world_from(observations, date(2021, 6, 1)))
        assert state.confidence == pytest.approx(1.0, abs=1e-6)

    def test_a_partial_cross_section_lowers_confidence(self, monthly_engine):
        """Six tenors fitted perfectly is a weaker statement than eleven."""
        subset = dict(list(TENOR_IDS.items())[:6])
        observations = synthetic_curve_frame([(4.0, -1.5, 2.0)] * 400, tenors=subset)
        state = monthly_engine.predict(world_from(observations, date(2021, 6, 1)))

        assert state.confidence == pytest.approx(6 / 11, abs=0.01)

    def test_signals_include_the_two_the_phase_requires(self, monthly_engine):
        observations = synthetic_curve_frame(
            [(4.0 + i * 0.002, -1.5 + i * 0.002, 0.0) for i in range(400)]
        )
        state = monthly_engine.predict(world_from(observations, date(2021, 6, 1)))

        names = {s.name for s in state.signals}
        assert {"curve_inversion", "term_premium_trend"} <= names
        assert all(s.direction in (-1, 0, 1) for s in state.signals)

    def test_stamps_the_model_version(self, monthly_engine):
        observations = synthetic_curve_frame([(4.0, -1.5, 2.0)] * 400)
        state = monthly_engine.predict(world_from(observations, date(2021, 6, 1)))
        assert state.model_version == monthly_engine.version

    def test_no_curve_at_all_is_an_explicit_failure(self, monthly_engine):
        """Silence beats publishing a state built from nothing."""
        observations = synthetic_curve_frame([(4.0, -1.5, 2.0)] * 10)
        # Cutoff before anything was released.
        with pytest.raises(InsufficientCurveDataError, match="no usable yield curve"):
            monthly_engine.predict(world_from(observations, date(2019, 1, 1)))


class TestOutputs:
    def test_publishes_the_four_factors_and_the_regime_per_date(self, monthly_engine):
        observations = synthetic_curve_frame([(4.0, -1.5, 2.0)] * 60)
        world = world_from(observations, date(2020, 6, 1))

        rows = monthly_engine.outputs(world)

        metrics = {row.metric for row in rows}
        assert metrics == {"ns_level", "ns_slope", "ns_curvature", "ns_rmse", "regime_code"}
        assert all(row.asset == "rates" for row in rows)

    def test_regime_rows_carry_the_label_in_meta(self, monthly_engine):
        """engine_output stores REALs, so the name travels alongside the code."""
        observations = synthetic_curve_frame([(4.0, 2.0, 0.0)] * 60)
        rows = monthly_engine.outputs(world_from(observations, date(2020, 6, 1)))

        coded = [r for r in rows if r.metric == "regime_code"]
        assert coded
        for row in coded:
            label = (row.meta or {}).get("regime")
            assert label in RATE_REGIMES
            assert row.value == float(regime_code(label))

    def test_every_output_date_is_within_the_information_set(self, monthly_engine):
        observations = synthetic_curve_frame([(4.0, -1.5, 2.0)] * 200)
        cutoff = date(2020, 3, 1)
        rows = monthly_engine.outputs(world_from(observations, cutoff))

        assert rows
        assert max(row.as_of for row in rows) <= cutoff

    def test_no_curve_yields_no_rows_rather_than_an_error(self, monthly_engine):
        observations = synthetic_curve_frame([(4.0, -1.5, 2.0)] * 10)
        assert monthly_engine.outputs(world_from(observations, date(2019, 1, 1))) == ()


class TestFit:
    def test_selects_and_persists_lambda(self, monthly_engine, artifacts):
        true_lambda = 0.4
        observations = synthetic_curve_frame([(4.0, -1.5, 2.0)] * 30, lambda_=true_lambda)

        monthly_engine.fit(world_from(observations, date(2020, 3, 1)))

        stored = artifacts.load("rates")
        assert stored["lambda"] == pytest.approx(true_lambda)
        assert stored["training_curves"] > 0
        assert "lambda_grid_rmse" in stored

    def test_predict_uses_the_persisted_lambda_afterwards(self, monthly_engine, artifacts):
        observations = synthetic_curve_frame([(4.0, -1.5, 2.0)] * 400, lambda_=0.4)
        assert monthly_engine.lambda_ == pytest.approx(DEFAULT_LAMBDA)

        monthly_engine.fit(world_from(observations, date(2021, 6, 1)))

        assert monthly_engine.lambda_ == pytest.approx(0.4)
        state = monthly_engine.predict(world_from(observations, date(2021, 6, 1)))
        assert state.components is not None
        assert state.components["ns_lambda"] == pytest.approx(0.4)
        # Refitting recovers the generating factors exactly; the default did not.
        assert state.components["ns_rmse"] == pytest.approx(0.0, abs=1e-9)

    def test_daily_prediction_never_moves_lambda(self, monthly_engine):
        """A moving lambda would silently rebase every published factor history."""
        observations = synthetic_curve_frame([(4.0, -1.5, 2.0)] * 400, lambda_=0.4)
        before = monthly_engine.lambda_

        monthly_engine.predict(world_from(observations, date(2021, 6, 1)))

        assert monthly_engine.lambda_ == before


class TestRequiredSeries:
    def test_declares_every_tenor_and_the_breakeven(self, monthly_engine):
        required = monthly_engine.required_series()
        assert set(TENOR_IDS) <= set(required)
        assert "FRED:T10YIE" in required

    def test_names_no_series_that_config_does_not(self, monthly_engine, config):
        configured = {spec.id for spec in config.all_series()}
        assert set(monthly_engine.required_series()) <= configured


class TestRegistration:
    def test_registers_itself_under_its_name(self):
        from findynamics.core.registry import get_engine

        assert isinstance(get_engine("rates"), RatesEngine)

    def test_is_enabled_in_the_shipped_config(self, config):
        assert config.is_enabled("rates")


@pytest.fixture(scope="module")
def labelled(treasury_observations):
    """Regimes and factors over the whole committed snapshot, computed once."""
    import tempfile
    from pathlib import Path

    from findynamics.core.artifacts import ArtifactStore
    from findynamics.core.config import load_series_config
    from tests.engines.rates.conftest import _with_params

    config = load_series_config()
    with tempfile.TemporaryDirectory() as tmp:
        params = dict(config.engines["rates"].params)
        params["regime"] = {**(params.get("regime") or {}), "trend_days": 12}
        engine = RatesEngine(_with_params(config, params), ArtifactStore(Path(tmp)))

        analysis = engine.analyze(world_from(treasury_observations, date(2026, 7, 30)))
        assert analysis is not None
        return analysis.regimes.dropna(), analysis.factors


class TestSanityBacktestAgainstRealHistory:
    """The curve really did invert in 2000, 2006-07, 2019 and 2022.

    A model that gets these wrong is wrong about the single most-studied feature
    of the yield curve, however clean its unit tests are.
    """

    @pytest.mark.parametrize(
        ("label", "start", "end"),
        [
            ("2000 inversion", "2000-04-01", "2000-12-31"),
            ("2006-07 inversion", "2006-08-01", "2007-05-31"),
            ("2019 inversion", "2019-06-01", "2019-09-30"),
            ("2022-23 inversion", "2022-11-01", "2023-12-31"),
        ],
    )
    def test_known_inversions_are_labelled_inverted(self, labelled, label, start, end):
        regimes, _ = labelled
        window = regimes.loc[start:end]

        assert len(window) > 0, f"{label}: the fixture covers no dates in this window"
        inverted = (window == "inverted").mean()
        assert inverted >= 0.6, (
            f"{label}: only {inverted:.0%} of {len(window)} months classified inverted; "
            f"got {sorted(set(window))}"
        )

    @pytest.mark.parametrize(
        ("label", "start", "end"),
        [
            ("mid-1990s expansion", "1993-01-01", "1994-12-31"),
            ("post-GFC recovery", "2010-01-01", "2013-12-31"),
            ("post-COVID reopening", "2020-06-01", "2021-06-30"),
        ],
    )
    def test_steeply_positive_periods_are_never_called_inverted(self, labelled, label, start, end):
        regimes, _ = labelled
        window = regimes.loc[start:end]

        assert len(window) > 0
        assert "inverted" not in set(window), f"{label}: {sorted(set(window))}"

    def test_the_spread_sign_agrees_with_the_label_everywhere(self, labelled):
        """The classifier must not disagree with its own input."""
        regimes, factors = labelled
        joined = factors.loc[regimes.index]

        inverted = joined.loc[regimes == "inverted", "spread"]
        assert (inverted < 0).all()

        not_inverted = joined.loc[regimes != "inverted", "spread"]
        assert (not_inverted >= 0).all()

    def test_every_label_is_in_the_vocabulary(self, labelled):
        regimes, _ = labelled
        assert set(regimes) <= set(RATE_REGIMES)

    def test_the_snapshot_covers_the_history_the_assertions_need(self, labelled):
        regimes, _ = labelled
        assert regimes.index.min().year <= 1988
        assert regimes.index.max().year >= 2026
