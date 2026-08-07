"""Factor definitions (03-contracts.md §4).

Layer 0 is the set of variables owned by no engine. The invariant worth pinning
in P0 is that the set has not silently changed shape during the restructure, and
that `core.config` and `factors` agree on what a factor is.
"""

from __future__ import annotations

from findynamics.core.config import load_series_config
from findynamics.core.contracts import vocab
from findynamics.factors.definitions import FACTORS, factor_series_ids, factor_specs


def test_the_factor_set_is_the_nine_v1_forces_plus_later_additions():
    """The nine keep their order; later phases append rather than reshuffling.

    Order is load-bearing — serving mirrors this tuple as FORCES and the drift
    test compares them element by element.
    """
    assert FACTORS == (
        "valuation",
        "earnings",
        "liquidity",
        "rates",
        "credit",
        "inflation",
        "labor",
        "risk_appetite",
        "sentiment",
        # P1
        "real_rate",
        "usd_strength",
        # P5 — the money stock, distinct from `liquidity`'s conditions read.
        "global_liquidity",
    )


def test_global_liquidity_is_not_a_second_name_for_liquidity():
    """The two factors are different questions, and must stay different series.

    `liquidity` blends the money stock with NFCI and overnight RRP take-up, which
    makes it a financial-conditions score. `global_liquidity` is the stock alone,
    because FinCrypto publishes a regression coefficient against it and a beta on
    a half-conditions composite has no statable units. If someone ever "tidies
    up" by pointing them at the same series list, this fails.
    """
    from findynamics.factors.definitions import factor_specs

    specs = factor_specs()
    conditions = {s.id for s in specs["liquidity"].series}
    stock = {s.id for s in specs["global_liquidity"].series}

    assert stock == {"FRED:M2SL", "FRED:WALCL"}
    assert stock < conditions, "global_liquidity must be the stock legs only"
    assert conditions - stock, "liquidity must keep its conditions legs"


def test_definitions_re_export_the_single_canonical_tuple():
    """One definition, imported two ways — not two lists to keep in sync."""
    assert FACTORS is vocab.FACTORS


def test_specs_come_from_the_shipped_config():
    specs = factor_specs()
    assert set(specs) == set(FACTORS)
    assert specs["valuation"].series[0].id == "SHILLER:CAPE"
    assert specs["valuation"].weight == 1.0


def test_specs_accept_an_explicit_config():
    config = load_series_config()
    assert factor_specs(config) == dict(config.factors)


def test_series_ids_are_deduplicated_and_sorted():
    ids = factor_series_ids()
    assert ids == tuple(sorted(set(ids)))
    assert "FRED:DGS10" in ids
    # Engine-private price series are not factor inputs.
    assert not any(i.startswith("PRICE:") for i in ids)
