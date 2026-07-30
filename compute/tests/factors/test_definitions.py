"""Factor definitions (03-contracts.md §4).

Layer 0 is the set of variables owned by no engine. The invariant worth pinning
in P0 is that the set has not silently changed shape during the restructure, and
that `core.config` and `factors` agree on what a factor is.
"""

from __future__ import annotations

from findynamics.core.config import load_series_config
from findynamics.core.contracts import vocab
from findynamics.factors.definitions import FACTORS, factor_series_ids, factor_specs


def test_the_factor_set_is_the_nine_v1_forces_plus_the_p1_additions():
    """The nine keep their order; P1 appends rather than reshuffling.

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
        "real_rate",
        "usd_strength",
    )


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
