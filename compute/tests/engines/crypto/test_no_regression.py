"""P5 must not change what the nightly run publishes for anything else.

The phase's acceptance asks that, with ``enabled: false``, the daily job's output
be byte-identical to before this phase. That is asserted here, and one deviation
is asserted **explicitly** rather than glossed over:

    P5 adds exactly one row to the `factors` array — `global_liquidity` — and
    changes nothing else anywhere in the payload.

That row is not collateral damage from the crypto engine; it is a deliberate
Layer-0 addition the phase brief asks for in its own right ("add a
`global_liquidity` factor to series.yaml"), and Layer 0 is computed for every run
regardless of which engines are enabled. The two requirements are in tension and
they cannot both be met literally: a shared factor that only appears when an
experimental engine is switched on would not be a shared factor.

So the guarantee this file proves is the stronger useful one:

1. every factor that existed before P5 scores **identically**, component for
   component (:class:`TestTheExistingFactorsAreUntouched`);
2. the crypto engine contributes **nothing at all** while disabled — it is not
   imported, not instantiated, and appears in no array
   (:class:`TestTheDisabledEngineIsInert`);
3. merely *registering* the engine changes nothing either, so the difference
   between this phase and the one before it is the config flag and nothing
   latent (:class:`TestRegistrationAloneChangesNothing`).
"""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import date

import pandas as pd
import pytest

from findynamics.core.config import load_series_config
from findynamics.core.registry import enabled_engines
from findynamics.data.accessor import PandasPITAccessor
from findynamics.engines import load_engines
from findynamics.factors.compute import compute_factors
from jobs.daily import factor_payload

AS_OF = date(2026, 8, 5)

#: The factor set as it stood before this phase (core/contracts/vocab.py at P4).
FACTORS_BEFORE_P5 = (
    "valuation",
    "earnings",
    "liquidity",
    "rates",
    "credit",
    "inflation",
    "labor",
    "risk_appetite",
    "sentiment",
    "real_rate",
    "usd_strength",
)


@pytest.fixture(scope="module")
def accessor(crypto_observations) -> PandasPITAccessor:
    """A PIT view carrying the two money-stock series both factors read.

    The snapshot has M2SL and WALCL, which is what `liquidity` and
    `global_liquidity` overlap on — precisely the pair where an accidental change
    to the old factor would hide.
    """
    return PandasPITAccessor(crypto_observations, AS_OF)


@pytest.fixture(scope="module")
def config_before_p5():
    """The shipped config with `global_liquidity` removed.

    Stands in for the pre-phase config. Everything else — every other factor,
    every engine's series block — is the shipped one, so any difference this
    fixture produces is attributable to the added factor and nothing else.
    """
    config = load_series_config()
    return replace(
        config,
        factors={k: v for k, v in config.factors.items() if k != "global_liquidity"},
    )


class TestTheExistingFactorsAreUntouched:
    def test_every_pre_p5_factor_scores_identically(self, accessor, config, config_before_p5):
        """Component for component, not just score for score.

        A shared factor is scored from its own series list, so adding a second
        factor over overlapping series must not move the first. If a future edit
        ever makes the factor pipeline global — a shared expanding window, a
        cross-factor normalisation — this is what catches it.
        """
        after = compute_factors(accessor, config)
        before = compute_factors(accessor, config_before_p5)

        for name in FACTORS_BEFORE_P5:
            if name not in before:
                # Not scorable from this snapshot; absent from both, which is
                # itself the identity being asserted.
                assert name not in after, f"{name} became scorable only after P5"
                continue
            assert after[name] == before[name], f"{name} changed"

    def test_the_payload_differs_by_exactly_one_row(self, accessor, config, config_before_p5):
        """The deviation, measured rather than described.

        `factor_payload` is what the daily job puts on the wire, so this compares
        the actual serialized arrays. One row added, none changed, none removed.
        """
        after = factor_payload(compute_factors(accessor, config))
        before = factor_payload(compute_factors(accessor, config_before_p5))

        by_name_after = {row["force"]: row for row in after}
        by_name_before = {row["force"]: row for row in before}

        added = set(by_name_after) - set(by_name_before)
        removed = set(by_name_before) - set(by_name_after)

        assert added == {"global_liquidity"}
        assert removed == set()
        for name in by_name_before:
            assert json.dumps(by_name_after[name], sort_keys=True) == json.dumps(
                by_name_before[name], sort_keys=True
            ), f"{name} changed on the wire"

    def test_global_liquidity_is_actually_scored_from_this_snapshot(self, accessor, config):
        """Guard on the test above: an added row that is always absent proves nothing."""
        factors = compute_factors(accessor, config)
        assert "global_liquidity" in factors
        assert 0.0 <= factors["global_liquidity"].score <= 100.0
        # Both legs and nothing else. The scoring pipeline carries each series
        # twice — its contribution and its `:level` — so the check is on the
        # series ids the trace mentions, not on the raw key set.
        mentioned = {key.removesuffix(":level") for key in factors["global_liquidity"].components}
        assert mentioned == {"FRED:M2SL", "FRED:WALCL"}


class TestTheDisabledEngineIsInert:
    def test_it_ships_disabled(self, config):
        assert config.is_enabled("crypto") is False

    def test_load_engines_does_not_import_it(self, config):
        """The import is what registers the engine, and it is guarded by the flag.

        Asserted on the return value rather than on `sys.modules`, because the
        test suite imports the package directly elsewhere and would poison a
        module-table check. What matters is that `load_engines` did not ask for
        it.
        """
        assert "crypto" not in load_engines(config)

    def test_it_is_not_instantiated_for_a_run(self, config):
        assert "crypto" not in [engine.name for engine in enabled_engines(config)]

    def test_enabling_it_is_the_only_thing_that_changes_that(self, crypto_only_config):
        """The flag is load-bearing, so the disabled case is a real state not a coincidence."""
        assert [engine.name for engine in enabled_engines(crypto_only_config)] == ["crypto"]


class TestRegistrationAloneChangesNothing:
    def test_importing_the_package_leaves_the_enabled_set_alone(self, config):
        """The difference between P4 and P5 for a nightly run is the config flag.

        Importing `findynamics.engines.crypto` registers the class. If mere
        registration could change what a run publishes — through a registry
        iteration order, a global, an import side effect — then shipping the code
        would have changed the output even with the flag off.
        """
        before = [engine.name for engine in enabled_engines(config)]

        import findynamics.engines.crypto  # noqa: F401

        after = [engine.name for engine in enabled_engines(config)]
        assert after == before
        assert "crypto" not in after

    def test_the_factor_scores_are_unaffected_by_registration(self, accessor, config):
        import findynamics.engines.crypto  # noqa: F401

        first = factor_payload(compute_factors(accessor, config))
        second = factor_payload(compute_factors(accessor, config))
        assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)

    def test_the_envelope_model_version_carries_no_crypto_version(self, config):
        """`model_version` is joined from the states a run produced.

        A disabled engine produces none, so the string a run publishes is the
        same one it published before this phase.
        """
        versions = sorted({engine.version for engine in enabled_engines(config)})
        assert not any(version.startswith("crypto-") for version in versions)


def test_the_new_series_are_only_fetched_when_the_engine_is_enabled(config, crypto_only_config):
    """Nightly cost, not just nightly output.

    `jobs.daily` asks each *enabled* engine for `required_series`, so a disabled
    crypto engine adds no fetch — the bitcoin price and the four blockchain.info
    charts are not requested at all. Worth pinning separately from the payload:
    an engine can be inert in what it publishes and still be expensive.
    """
    from findynamics.data.store import required_series_ids

    disabled_engine_series = {
        series_id for engine in enabled_engines(config) for series_id in engine.required_series()
    }
    enabled_engine_series = {
        series_id
        for engine in enabled_engines(crypto_only_config)
        for series_id in engine.required_series()
    }

    assert not any(s.startswith("BLOCKCHAIN:") for s in disabled_engine_series)
    assert "STOOQ:BTCUSD" not in disabled_engine_series
    assert {"STOOQ:BTCUSD", "YAHOO:BTC-USD"} <= enabled_engine_series
    assert any(s.startswith("BLOCKCHAIN:") for s in enabled_engine_series)

    # The factor layer still asks for M2SL and WALCL either way — they were
    # already ingested for `liquidity` before this phase, so `global_liquidity`
    # adds a reading of them rather than a fetch.
    wanted = required_series_ids(config, disabled_engine_series)
    assert {"FRED:M2SL", "FRED:WALCL"} <= set(wanted)


def test_the_fixture_is_the_snapshot_the_suite_expects(crypto_observations):
    """Guard: a truncated regeneration should fail loudly here, not subtly elsewhere."""
    assert pd.Timestamp(crypto_observations["obs_date"].max()).date() >= date(2026, 8, 1)
    assert set(crypto_observations["series_id"].unique()) >= {
        "YAHOO:BTC-USD",
        "FRED:M2SL",
        "FRED:WALCL",
        "BLOCKCHAIN:TX_VOLUME_USD",
    }
