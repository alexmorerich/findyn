"""Crypto must not change what the nightly run publishes for anything else.

P5's acceptance originally asked that, with ``enabled: false``, the daily job's
output be byte-identical to before that phase. The flag is now ``true`` — the
engine publishes so the /crypto page reads real numbers instead of a milestone
placeholder — so the question this file answers has changed shape. It is no
longer "does the disabled engine stay inert" but the strictly harder one:

    **crypto is ADDITIVE.** It contributes its own rows and moves nothing that
    was already there.

That is worth more than the original guarantee, because "inert while switched
off" was only ever true by construction. This version has to hold while the
engine actually runs.

One deviation is still asserted **explicitly** rather than glossed over:

    Crypto adds exactly one row to the `factors` array — `global_liquidity` —
    and changes nothing else anywhere in the payload.

That row is not collateral damage from the crypto engine; it is a deliberate
Layer-0 addition the phase brief asks for in its own right ("add a
`global_liquidity` factor to series.yaml"), and Layer 0 is computed for every run
regardless of which engines are enabled. A shared factor that only appeared when
an experimental engine was switched on would not be a shared factor.

What is asserted, in order:

1. every factor that existed before P5 scores **identically**, component for
   component (:class:`TestTheExistingFactorsAreUntouched`) — unchanged, and still
   the load-bearing one, because `global_liquidity` overlaps `liquidity` on both
   of its series;
2. the engine is enabled, imported and instantiated, and is **still excluded from
   the portfolio layer** (:class:`TestTheEnabledEngineIsAdditive`). Publishing and
   allocating are different verbs and only the first one is switched on;
3. registration order and import side effects still move nothing
   (:class:`TestRegistrationAloneChangesNothing`).
"""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import date

import pandas as pd
import pytest

from findynamics.core.config import load_series_config
from findynamics.core.registry import (
    enabled_engines,
    portfolio_asset_names,
    portfolio_engines,
)
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


@pytest.fixture
def crypto_disabled_config(config):
    """The shipped config with crypto switched back off.

    Stands in for the pre-activation state, so the flag can be shown to be
    load-bearing in *both* directions rather than only in the one it now sits in.
    """
    return replace(
        config,
        engines={
            name: replace(entry, enabled=False) if name == "crypto" else entry
            for name, entry in config.engines.items()
        },
    )


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
        #
        # `components` also carries factor-level scoring metadata that is not a
        # series at all (`inputs_used` / `inputs_configured`, which say how many
        # of the configured inputs the score is a mean over). Series ids are
        # namespaced `PROVIDER:SERIES`, so the colon is what separates the two
        # kinds of key; without this filter the metadata reads as two phantom
        # series and the assertion below fails for a reason that has nothing to
        # do with which series were scored.
        mentioned = {
            key.removesuffix(":level")
            for key in factors["global_liquidity"].components
            if ":" in key
        }
        assert mentioned == {"FRED:M2SL", "FRED:WALCL"}


class TestTheEnabledEngineIsAdditive:
    def test_it_ships_enabled(self, config):
        assert config.is_enabled("crypto") is True

    def test_load_engines_imports_it(self, config):
        """The import is what registers the engine, and it follows the flag.

        Asserted on the return value rather than on `sys.modules`, because the
        test suite imports the package directly elsewhere and would poison a
        module-table check. What matters is that `load_engines` asked for it.
        """
        assert "crypto" in load_engines(config)

    def test_it_is_instantiated_for_a_run(self, config):
        assert "crypto" in [engine.name for engine in enabled_engines(config)]

    def test_it_is_still_excluded_from_the_portfolio_layer(self, config):
        """THE assertion in this class, and the reason enabling it was safe.

        `enabled` governs PUBLICATION; `experimental` governs INFLUENCE. The
        engine now computes, writes back and has a page with real numbers on it,
        and it still cannot move a single allocation weight. If a future edit ever
        collapses those two flags into one — or drops `experimental` because "the
        engine is live now, surely it counts" — this is what fails.
        """
        assert "crypto" not in portfolio_asset_names(config)
        assert "crypto" not in [engine.name for engine in portfolio_engines(config)]

    def test_disabling_it_is_still_what_turns_it_off(self, crypto_disabled_config):
        """The flag is load-bearing in both directions, not just the one it sits in."""
        assert "crypto" not in [engine.name for engine in enabled_engines(crypto_disabled_config)]


class TestRegistrationAloneChangesNothing:
    def test_importing_the_package_leaves_the_enabled_set_alone(self, config):
        """What a run publishes follows config, not the import table.

        Importing `findynamics.engines.crypto` registers the class. If mere
        registration could change what a run publishes — through a registry
        iteration order, a global, an import side effect — then which engines ran
        would depend on which modules some earlier caller happened to touch.

        This used to be trivially true while the flag was off. It is a real
        assertion now: crypto is *in* the set, and importing it again must not
        duplicate, reorder or otherwise disturb it.
        """
        before = [engine.name for engine in enabled_engines(config)]

        import findynamics.engines.crypto  # noqa: F401

        after = [engine.name for engine in enabled_engines(config)]
        assert after == before

    def test_the_factor_scores_are_unaffected_by_registration(self, accessor, config):
        import findynamics.engines.crypto  # noqa: F401

        first = factor_payload(compute_factors(accessor, config))
        second = factor_payload(compute_factors(accessor, config))
        assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)

    def test_the_envelope_model_version_now_carries_the_crypto_version(self, config):
        """`model_version` is joined from the states a run produced.

        An enabled engine contributes its version, so the envelope's run label now
        names crypto. Asserted rather than left implicit because this is the one
        field where enabling an engine is visible to *every* consumer, including
        ones that never read a crypto row — a client pinning the exact envelope
        string will see it change, and should find the reason in a test.
        """
        versions = sorted({engine.version for engine in enabled_engines(config)})
        assert any(version.startswith("crypto-") for version in versions)


def test_the_crypto_series_are_fetched_only_while_the_engine_is_enabled(
    config, crypto_disabled_config
):
    """Nightly cost, not just nightly output.

    `jobs.daily` asks each *enabled* engine for `required_series`. Now that crypto
    is on, the bitcoin price and the four blockchain.info charts ARE requested
    every night — that is the running cost of the /crypto page, and it belongs in
    a test rather than being discovered on a bill. Switching the flag back off
    must remove them again, which is what pins the cost to the flag.
    """
    from findynamics.data.store import required_series_ids

    enabled_engine_series = {
        series_id for engine in enabled_engines(config) for series_id in engine.required_series()
    }
    disabled_engine_series = {
        series_id
        for engine in enabled_engines(crypto_disabled_config)
        for series_id in engine.required_series()
    }

    assert {"STOOQ:BTCUSD", "YAHOO:BTC-USD"} <= enabled_engine_series
    assert any(s.startswith("BLOCKCHAIN:") for s in enabled_engine_series)
    assert not any(s.startswith("BLOCKCHAIN:") for s in disabled_engine_series)
    assert "STOOQ:BTCUSD" not in disabled_engine_series

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
