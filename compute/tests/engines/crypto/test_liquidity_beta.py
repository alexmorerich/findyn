"""The expanding liquidity regression.

Asserted against synthetic data with a **known** beta, which is the one thing
real data cannot offer here: nobody publishes bitcoin's true sensitivity to the
money supply, so an estimator tested only against history can be scored on
plausibility and never on correctness.

The real snapshot appears once at the end, for the finding it produces rather
than for a threshold it clears — see ``test_the_real_relationship_is_weak``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from findynamics.engines.crypto import liquidity_beta as beta_mod
from findynamics.engines.crypto.liquidity_beta import BetaRules
from tests.engines.crypto.conftest import (
    CENTRAL_BANK_ASSETS,
    M2,
    known_beta_series,
)

RULES = BetaRules()


class TestTheEstimatorRecoversAKnownBeta:
    @pytest.mark.parametrize("beta", [1.0, 4.0, 8.0, -2.0])
    def test_a_noiseless_relationship_is_recovered_exactly(self, beta):
        returns, level = known_beta_series(beta=beta)
        result = beta_mod.estimate(returns, level, RULES)

        assert result.latest() == pytest.approx(beta, abs=1e-9)
        # A deterministic relationship explains all of the variance.
        assert result.latest_r_squared() == pytest.approx(1.0, abs=1e-9)

    def test_a_flat_return_series_gives_a_zero_beta_and_an_undefined_r_squared(self):
        """The degenerate case, and the answer is not 1.0.

        With beta exactly 0 and no noise the returns are constant, so there is no
        variance to explain and R² is 0/0. The estimator publishes NaN rather
        than a number: a "perfect fit" to a flat line would be the most
        misleading possible reading, and it is exactly what a naive
        explained-over-total would print.
        """
        returns, level = known_beta_series(beta=0.0)
        result = beta_mod.estimate(returns, level, RULES)

        assert result.latest() == pytest.approx(0.0, abs=1e-9)
        assert result.latest_r_squared() is None or np.isnan(result.latest_r_squared())

    def test_the_intercept_is_recovered_too(self):
        returns, level = known_beta_series(beta=3.0, alpha=0.01)
        result = beta_mod.estimate(returns, level, RULES)

        assert result.latest() == pytest.approx(3.0, abs=1e-9)
        assert result.alpha.dropna().iloc[-1] == pytest.approx(0.01, abs=1e-9)

    def test_it_still_recovers_the_beta_through_noise(self):
        """The honest version: a relationship that is real but not deterministic."""
        returns, level = known_beta_series(beta=5.0, noise=0.02, months=360)
        result = beta_mod.estimate(returns, level, RULES)

        assert result.latest() == pytest.approx(5.0, rel=0.25)
        # And it says how much it explains, which is the number that matters.
        r_squared = result.latest_r_squared()
        assert r_squared is not None and 0.0 < r_squared < 1.0

    def test_pure_noise_explains_nothing_even_when_the_coefficient_is_large(self):
        """The failure mode this whole engine is careful about, reproduced on purpose.

        With beta truly 0 and noisy returns, the *coefficient* is not pinned down
        at all — this fixture routinely produces betas of ±10 — while the R² sits
        near zero. That is not a defect in the estimator; it is arithmetic. The
        regressor here is a monthly money-supply change with a standard deviation
        of about 0.25%, and the regressand is a 20% monthly return, so the slope's
        standard error is enormous and a large coefficient is ordinary sampling
        noise.

        This is the same arithmetic that produces the real snapshot's near-zero
        R², and it is why the engine publishes the R² beside the beta, why the
        page tells the reader to read it first, and why no expected return is
        derived from either. A test that asserted "beta is small" here would be
        asserting a coincidence.
        """
        returns, level = known_beta_series(beta=0.0, noise=0.2, months=360)
        result = beta_mod.estimate(returns, level, RULES)

        assert (result.latest_r_squared() or 0.0) < 0.1
        assert result.latest() is not None  # a number is published; it just means little


class TestItIsExpandingAndNotRolling:
    """§14.1 rule 4, and the property that makes the replay test able to prove anything."""

    def test_a_coefficient_never_changes_once_published(self):
        """The sharp test. Extend the sample and the past estimates must not move.

        A rolling window would fail this on every date; so would any accidental
        full-sample statistic. This is the property that makes a beta charted for
        2017 the beta a run in 2017 would have published.
        """
        returns, level = known_beta_series(beta=4.0, noise=0.03, months=240)

        early = beta_mod.estimate(returns.iloc[:150], level.iloc[:150], RULES)
        late = beta_mod.estimate(returns, level, RULES)

        shared = early.beta.dropna().index.intersection(late.beta.dropna().index)
        assert len(shared) > 100
        pd.testing.assert_series_equal(
            early.beta.loc[shared], late.beta.loc[shared], check_names=False
        )

    def test_the_sample_size_grows_by_one_per_month(self):
        returns, level = known_beta_series(beta=1.0, months=120)
        result = beta_mod.estimate(returns, level, RULES)

        counts = result.n_observations.dropna()
        assert list(np.diff(counts.to_numpy())) == [1.0] * (len(counts) - 1)

    def test_nothing_is_published_below_the_sample_floor(self):
        """A slope through a handful of points is not a coefficient."""
        rules = BetaRules(min_observations=36)
        returns, level = known_beta_series(beta=2.0, months=60)
        result = beta_mod.estimate(returns, level, rules)

        published = result.beta.dropna()
        assert len(published) == 60 - 36 + 1 - 1  # one month lost to the log diff
        assert result.n_observations.dropna().min() >= 36


class TestTheComposite:
    def test_the_two_legs_are_scaled_into_the_same_unit(self):
        """FRED publishes M2SL in billions and WALCL in millions.

        Added unscaled, the Fed's balance sheet would outweigh all of M2 by a
        factor of a thousand — a mistake that changes the beta's sign nowhere and
        its meaning entirely.
        """
        index = pd.date_range("2020-01-31", periods=12, freq="ME")
        frame = pd.DataFrame(
            {M2: 21_000.0, CENTRAL_BANK_ASSETS: 6_600_000.0},
            index=index,
        )
        level, legs = beta_mod.composite_liquidity(
            frame, {"m2": M2, "central_bank_assets": CENTRAL_BANK_ASSETS}, RULES
        )

        assert legs == {"m2": True, "central_bank_assets": True}
        # 21,000 $bn of M2 plus 6,600 $bn of Fed assets.
        assert level.iloc[-1] == pytest.approx(27_600.0)

    def test_a_missing_leg_is_dropped_rather_than_zero_filled(self):
        """A zero would claim the Fed's balance sheet is empty.

        That is a much stronger statement than "we could not read it", and the
        engine takes a confidence penalty for the second rather than publishing
        the first.
        """
        index = pd.date_range("2020-01-31", periods=12, freq="ME")
        frame = pd.DataFrame({M2: 21_000.0}, index=index)
        level, legs = beta_mod.composite_liquidity(
            frame, {"m2": M2, "central_bank_assets": CENTRAL_BANK_ASSETS}, RULES
        )

        assert legs == {"m2": True, "central_bank_assets": False}
        assert level.iloc[-1] == pytest.approx(21_000.0)

    def test_no_legs_at_all_gives_an_empty_series_rather_than_an_exception(self):
        frame = pd.DataFrame(index=pd.date_range("2020-01-31", periods=4, freq="ME"))
        level, legs = beta_mod.composite_liquidity(
            frame, {"m2": M2, "central_bank_assets": CENTRAL_BANK_ASSETS}, RULES
        )
        assert level.empty
        assert legs == {"m2": False, "central_bank_assets": False}


class TestTheImpliedBand:
    def test_the_band_widens_with_the_square_root_of_the_sample(self):
        """The cumulative residual is a random walk, so its spread grows as sqrt(n).

        A constant-width band would claim a precision about 2024 that it earned
        in 2015.
        """
        returns, level = known_beta_series(beta=4.0, noise=0.05, months=240)
        result = beta_mod.estimate(returns, level, RULES)
        width = beta_mod.band_half_width(result, RULES).dropna()

        assert len(width) > 100
        # Not monotone month to month — sigma is re-estimated — but the trend is
        # unmistakable over the sample.
        assert width.iloc[-1] > width.iloc[0] * 1.5

    def test_a_noiseless_fit_publishes_no_excess_rather_than_zero(self):
        """A degenerate band is not a band the price is inside of.

        With no residuals the band has zero width, so "how many half-widths above
        it" is a division by zero. NaN is the honest answer — and it matters,
        because a published 0.0 would flow into the speculation index's geometric
        mean as a real term and zero the index on the strength of a measurement
        that was never made.
        """
        returns, level = known_beta_series(beta=4.0, noise=0.0)
        result = beta_mod.estimate(returns, level, RULES)

        assert beta_mod.excess_over_band(result, RULES).dropna().empty

    def test_a_price_on_the_fitted_line_scores_no_excess(self):
        """The non-degenerate version: real residuals, price not extended."""
        returns, level = known_beta_series(beta=4.0, noise=0.02, months=240)
        result = beta_mod.estimate(returns, level, RULES)
        excess = beta_mod.excess_over_band(result, RULES).dropna()

        assert not excess.empty
        # Symmetric noise leaves the cumulative residual wandering inside a band
        # that widens with it, so it should rarely be far outside.
        assert excess.max() < 3.0

    def test_being_below_the_band_is_not_negative_speculation(self):
        """One-sided on purpose.

        A price below what the money supply would imply is not evidence of
        speculation. It is not evidence of cheapness either, which is why this
        returns a distance and nothing downstream inverts it.
        """
        returns, level = known_beta_series(beta=4.0, noise=0.05, months=240)
        # Drag every return down so the cumulative residual goes firmly negative.
        result = beta_mod.estimate(returns - 0.05, level, RULES)

        deviation = beta_mod.level_deviation(result).dropna()
        assert deviation.iloc[-1] < 0

        excess = beta_mod.excess_over_band(result, RULES).dropna()
        assert (excess >= 0).all()
        assert excess.iloc[-1] == pytest.approx(0.0)


class TestConfigLoading:
    def test_a_non_mapping_block_is_refused_at_load(self):
        with pytest.raises(ValueError, match="must be a mapping"):
            BetaRules.from_params({"liquidity_beta": [1, 2, 3]})

    def test_the_shipped_scales_match_what_fred_publishes(self, config):
        rules = BetaRules.from_params(config.engines["crypto"].params)
        assert rules.m2_scale == 1.0
        assert rules.central_bank_assets_scale == 0.001


def test_the_real_relationship_is_weak_and_that_is_published_not_hidden(
    crypto_observations, config, crypto_engine
):
    """The finding, asserted so it cannot be quietly tuned away.

    On the real snapshot the coefficient is a plausible small positive number and
    the R² is near zero: monthly money-supply changes are tiny and smooth, and
    bitcoin's monthly returns are neither. A coefficient can be well-determined
    in sign and still explain almost none of the variance.

    This test exists because the temptation in a research module is to reach for
    a specification that produces an impressive number. The engine publishes the
    R² beside the beta, the page tells the reader to read it first, and no
    expected return is derived from either — and if someone changes the model
    until the R² clears 0.2, this fails and asks them to say why.
    """
    from tests.engines.crypto.conftest import world_from

    analysis = crypto_engine.analyze(world_from(crypto_observations))
    assert analysis is not None

    beta = analysis.beta.latest()
    r_squared = analysis.beta.latest_r_squared()

    assert beta is not None and r_squared is not None
    assert 0.0 < beta < 20.0, "a plausible magnitude, not a precise claim"
    assert r_squared < 0.2, (
        "the relationship explains very little of bitcoin's monthly variance. "
        "If this now fails, the model changed — say why, and check the page copy "
        "that tells readers the coefficient is mostly describing noise."
    )
