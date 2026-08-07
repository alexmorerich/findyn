"""FinCrypto's state, and the three gates that keep it out of everything else.

Most of this file is about what the engine **refuses** to say. That is unusual
for an engine test suite and it is the point of the phase: a research module's
value is bounded by how reliably it stays labelled as one, so the label is
tested at least as hard as the model.
"""

from __future__ import annotations

import pandas as pd
import pytest

from findynamics.core.contracts.state import AssetState
from findynamics.core.engine import StateUnavailable
from findynamics.core.registry import (
    ENGINES,
    enabled_engines,
    experimental_engines,
    portfolio_engines,
)
from findynamics.engines.crypto.domain import CRYPTO_METRICS, CRYPTO_REGIMES, regime_code
from findynamics.engines.crypto.engine import CryptoEngine
from tests.engines.crypto.conftest import SNAPSHOT_AS_OF, world_from


@pytest.fixture
def state(crypto_engine, crypto_observations) -> AssetState:
    return crypto_engine.predict(world_from(crypto_observations))


class TestTheEngineDeclinesToClaimAnExpectedReturn:
    """The headline of the module docstring, asserted rather than documented."""

    def test_expected_return_is_none(self, state):
        assert state.expected_return is None

    def test_it_is_none_and_not_zero(self, state):
        """A zero would be a claim. `None` is the absence of one.

        The contract carries `float | None` precisely so this distinction can be
        expressed (03-contracts.md §1: "None where meaningless (crypto)"), and
        the portfolio layer's optimiser would read a 0.0 as a forecast of no
        return rather than as no forecast.
        """
        assert state.expected_return is not None or state.expected_return != 0.0
        assert state.expected_return is None

    def test_the_components_say_so_out_loud(self, state):
        """Something rendering the state needs to see the claim being declined."""
        assert (state.components or {})["expected_return_is_deliberately_absent"] == 1.0

    def test_no_regime_conditional_mean_is_computed_anywhere(self):
        """The specific temptation, closed off.

        FinGold publishes a regime-conditional historical mean and labels it
        loudly as the past tense; that is defensible on six hundred months. The
        same construction on four cycles would be a description of four events,
        so this engine does not compute one at all — there is no private method
        to promote and no components key to start reading.
        """
        import inspect

        source = inspect.getsource(CryptoEngine)
        assert "conditional_return" not in source
        assert "expected_return=None" in source


class TestConfidenceIsCappedByConstruction:
    def test_the_published_confidence_is_at_or_under_the_ceiling(self, state):
        assert 0.0 <= state.confidence <= 0.5

    def test_the_ceiling_is_half_of_golds(self, config):
        """Both engines model an asset with no cash flow; only one has 600 months."""
        crypto = float(config.engines["crypto"].params["confidence"]["ceiling"])
        gold = float(config.engines["gold"].params["confidence"]["ceiling"])
        assert crypto == 0.5
        assert crypto < gold

    def test_no_configuration_can_raise_it_above_the_ceiling(self, crypto_observations, artifacts):
        """The clamp is to `ceiling`, not to 1.0 — so a bonus term cannot escape it.

        This edits the config to zero every penalty and then asserts the result
        is still the ceiling rather than something above it. A future edit that
        adds a term which *increases* confidence would fail here, which is the
        whole reason the clamp is written the way it is.
        """
        from copy import deepcopy
        from dataclasses import replace

        from findynamics.core.config import load_series_config

        config = load_series_config()
        params = deepcopy(config.engines["crypto"].params)
        for key in list(params["confidence"]):
            if key != "ceiling":
                params["confidence"][key] = -5.0  # a "bonus" for every penalty

        patched = replace(
            config,
            engines={
                **config.engines,
                "crypto": replace(config.engines["crypto"], params=params),
            },
        )

        engine = CryptoEngine(patched, artifacts)
        published = engine.predict(world_from(crypto_observations))
        assert published.confidence == 0.5

    def test_a_missing_liquidity_beta_costs_confidence(self, crypto_engine, crypto_observations):
        """Degradation is priced, not hidden."""
        full = crypto_engine.predict(world_from(crypto_observations))

        # Strip the macro legs; the beta becomes unavailable.
        stripped = crypto_observations[
            ~crypto_observations["series_id"].isin(["FRED:M2SL", "FRED:WALCL"])
        ]
        crypto_engine._cache = None
        degraded = crypto_engine.predict(world_from(stripped))

        assert degraded.confidence < full.confidence


class TestTheQuarantine:
    def test_the_engine_class_is_marked_experimental(self):
        assert CryptoEngine.experimental is True

    def test_it_is_the_only_experimental_engine(self):
        import findynamics.engines.crypto  # noqa: F401  (registers it)
        import findynamics.engines.gold  # noqa: F401

        assert set(experimental_engines()) == {"crypto"}

    def test_the_portfolio_layer_excludes_it_by_default(self, crypto_only_config):
        """§3 rule 5, as a function rather than as an intention.

        Crypto is *enabled* in this config — the whole point is that being
        enabled is not enough to reach an allocation. It computes, it writes
        back, it has a page, and the portfolio layer still does not see it.
        """
        assert [e.name for e in enabled_engines(crypto_only_config)] == ["crypto"]
        assert portfolio_engines(crypto_only_config) == []

    def test_it_can_be_included_only_by_asking_for_it_explicitly(self, crypto_only_config):
        names = [e.name for e in portfolio_engines(crypto_only_config, include_experimental=True)]
        assert names == ["crypto"]

    def test_it_ships_disabled(self, config):
        """The third gate. Flipping it is a deliberate act, not a merge."""
        assert config.is_enabled("crypto") is False

    def test_the_registry_knows_it_without_the_portfolio_layer_importing_it(self):
        """Discovery by name is what makes the import contract enforceable."""
        import findynamics.engines.crypto  # noqa: F401

        assert "crypto" in ENGINES
        assert ENGINES["crypto"] is CryptoEngine


class TestTheState:
    def test_the_regime_is_from_the_vocabulary(self, state):
        assert state.regime in CRYPTO_REGIMES

    def test_the_risk_score_is_on_the_zero_to_hundred_axis(self, state):
        assert 0.0 <= state.risk_score <= 100.0

    def test_the_signals_include_the_three_the_phase_asks_for(self, state):
        names = {s.name for s in state.signals}
        assert {"speculation_index", "liquidity_beta", "regime"} <= names

    def test_the_experimental_status_is_itself_a_signal(self, state):
        """Anything reading signals and not the envelope must still be told."""
        experimental = next(s for s in state.signals if s.name == "experimental")
        assert experimental.value == 1.0
        assert "no expected return" in (experimental.note or "").lower()

    def test_the_speculation_index_reads_as_a_risk_not_a_return(self, state):
        """A high reading is adverse, which is a statement about risk.

        The engine has no view on where the price goes next; what it will say is
        that a price which is mostly momentum is a worse risk than one that is
        not.
        """
        signal = next(s for s in state.signals if s.name == "speculation_index")
        assert signal.direction in (-1, 0, 1)
        if signal.value >= 60.0:
            assert signal.direction == -1

    def test_the_components_carry_the_supply_schedule(self, state):
        components = state.components or {}
        assert components["issued_supply"] > 19_000_000
        assert 0.0 < components["issuance_rate"] < 2.0
        assert components["stock_to_flow"] > 50.0

    def test_the_shared_factors_are_published_beside_the_engines_own(
        self, crypto_engine, crypto_observations, config
    ):
        """Layer 0's reading, as a cross-check rather than as an input.

        The scores are 0-100 percentiles and the beta is a coefficient with
        units; the page shows them side by side and the disagreement is the
        information.
        """
        from findynamics.core.contracts.state import FactorState

        factors = {
            "global_liquidity": FactorState(
                name="global_liquidity", as_of=SNAPSHOT_AS_OF, score=62.5, components={}
            )
        }
        crypto_engine._cache = None
        published = crypto_engine.predict(world_from(crypto_observations, factors=factors))
        assert (published.components or {})["factor_global_liquidity"] == 62.5


class TestOutputs:
    def test_every_published_metric_is_in_the_vocabulary(self, crypto_engine, crypto_observations):
        rows = crypto_engine.outputs(world_from(crypto_observations))
        assert rows
        assert {row.metric for row in rows} <= set(CRYPTO_METRICS)

    def test_the_regime_travels_as_its_code_with_the_name_in_meta(
        self, crypto_engine, crypto_observations
    ):
        """`engine_output` stores REALs, so the label rides in `meta`."""
        rows = [
            r
            for r in crypto_engine.outputs(world_from(crypto_observations))
            if r.metric == "regime_code"
        ]
        assert rows
        for row in rows[:50]:
            name = (row.meta or {})["regime"]
            assert name in CRYPTO_REGIMES
            assert row.value == float(regime_code(name))

    def test_the_on_chain_series_are_republished_so_a_dead_feed_is_visible(
        self, crypto_engine, crypto_observations
    ):
        metrics = {r.metric for r in crypto_engine.outputs(world_from(crypto_observations))}
        assert {"tx_volume_usd", "active_addresses", "transactions", "hash_rate"} <= metrics


class TestDegradation:
    def test_no_price_at_all_declines_rather_than_crashes(self, crypto_engine):
        """`StateUnavailable` is a correct answer, not a failure (core/engine.py)."""
        empty = pd.DataFrame(
            columns=["series_id", "obs_date", "release_date", "revision_date", "value"]
        )
        with pytest.raises(StateUnavailable, match="no bitcoin price knowable"):
            crypto_engine.predict(world_from(empty))

    def test_too_little_history_declines_rather_than_publishing_a_guess(
        self, crypto_engine, crypto_observations
    ):
        early = crypto_observations[crypto_observations["obs_date"] < "2015-03-01"]
        crypto_engine._cache = None
        with pytest.raises(StateUnavailable, match="shorter than the regime"):
            crypto_engine.predict(world_from(early, as_of=pd.Timestamp("2015-03-01").date()))

    def test_the_fallback_price_source_is_reported_rather_than_hidden(self, state):
        """Stooq is bot-filtered from every automated egress this project has.

        The fixture therefore carries the Yahoo series, the engine says which
        source carried the run, and confidence takes a small penalty. Silently
        substituting one for the other would make the configured role a fiction.
        """
        assert (state.components or {})["price_from_fallback_source"] == 1.0

    def test_absent_inputs_are_listed_rather_than_treated_as_zero(self, state):
        absent = next((s for s in state.signals if s.name == "inputs_absent"), None)
        assert absent is not None
        # The fixture has no STOOQ:BTCUSD, which is the normal state.
        assert "price" in (absent.note or "")


class TestFit:
    def test_fit_records_that_there_was_nothing_to_fit(
        self, crypto_engine, crypto_observations, artifacts
    ):
        """Every estimator here is an expanding closed form recomputed per run.

        The artifact exists so an operator can tell "no fit was needed" from "the
        refit never ran", which are the same silence otherwise.
        """
        crypto_engine.fit(world_from(crypto_observations))
        payload = artifacts.load("crypto")

        assert payload["model_version"] == CryptoEngine.version
        assert "no fitted parameters" in payload["note"]
        assert payload["observed"]["months"] > 100

    def test_predict_does_not_depend_on_fit_having_run(
        self, crypto_engine, crypto_observations, artifacts
    ):
        """The corollary, and worth pinning: there is no stored state to go stale."""
        before = crypto_engine.predict(world_from(crypto_observations))
        crypto_engine.fit(world_from(crypto_observations))
        crypto_engine._cache = None
        after = crypto_engine.predict(world_from(crypto_observations))

        assert before.regime == after.regime
        assert before.risk_score == after.risk_score
        assert before.confidence == after.confidence
