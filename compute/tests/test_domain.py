"""Cross-plane parity.

The domain vocabulary and the write-back signature exist in both TypeScript
(serving) and Python (compute). Drift between the two copies would surface as a
silently dropped regime or a rejected write-back at 03:00 UTC, so both are
pinned here.

The Python side is now split across three modules — shared vocabulary in
``core.contracts.vocab``, the factor set re-exported by ``factors.definitions``,
and the equity-private regimes in ``engines.equity.domain`` — while serving keeps
one flat ``domain.ts``. This test is the seam between those two shapes.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from findynamics.core.contracts.vocab import ASSETS, DISCOUNT_HORIZONS, HORIZONS, QUANTILES
from findynamics.engines.crypto.domain import CRYPTO_REGIMES
from findynamics.engines.equity.domain import REGIMES
from findynamics.engines.gold.domain import GOLD_REGIMES
from findynamics.engines.money.domain import MONEY_REGIMES
from findynamics.engines.rates.domain import RATE_REGIMES
from findynamics.factors.definitions import FACTORS
from jobs._common import sign_payload

DOMAIN_TS = Path(__file__).resolve().parents[2] / "serving" / "src" / "domain.ts"


def ts_string_array(name: str) -> list[str]:
    """Extract `export const NAME = [ 'a', 'b' ] as const;` from domain.ts."""
    source = DOMAIN_TS.read_text()
    match = re.search(rf"export const {name} = \[(.*?)\] as const;", source, re.DOTALL)
    assert match, f"{name} not found in {DOMAIN_TS}"
    return re.findall(r"'([^']+)'", match.group(1))


@pytest.mark.parametrize(
    ("name", "python_value"),
    [
        # Serving still calls the shared factors FORCES — the D1 table is
        # force_scores and renaming it is not worth the churn (02-migration-map §3).
        ("FORCES", FACTORS),
        ("ASSETS", ASSETS),
        ("REGIMES", REGIMES),
        ("HORIZONS", HORIZONS),
        ("RATE_REGIMES", RATE_REGIMES),
        # P2. Order is the wire order for MONEY_REGIMES: `engine_output` publishes
        # the liquidity state as its index, so a reordering here silently
        # relabels every row the money engine has ever written.
        ("MONEY_REGIMES", MONEY_REGIMES),
        ("DISCOUNT_HORIZONS", DISCOUNT_HORIZONS),
        # P4. Order is the wire order here too: `engine_output` publishes the
        # gold regime as its index (`regime_code`), so a reordering silently
        # relabels every row the gold engine has ever written.
        ("GOLD_REGIMES", GOLD_REGIMES),
        # P5. Wire order again, and ordered by increasing speculation so a chart
        # of `regime_code` reads upwards as more of the price is momentum.
        ("CRYPTO_REGIMES", CRYPTO_REGIMES),
    ],
)
def test_vocabulary_matches_the_serving_plane(name, python_value):
    assert ts_string_array(name) == list(python_value)


def test_the_experimental_flag_agrees_across_the_two_planes():
    """§3 rule 5, and the one piece of vocabulary derived rather than declared.

    On the Python side ``experimental`` is a class attribute the portfolio
    filter reads; on the serving side it is a list, because the API has to
    answer for an engine that has never run and so has no row to read a flag
    off. Two representations of one fact, which is exactly the drift this
    module exists to catch: a serving plane that forgot crypto was experimental
    would serve research output with a production disclaimer.
    """
    from dataclasses import replace

    from findynamics.core.config import load_series_config
    from findynamics.core.registry import experimental_engines
    from findynamics.engines import load_engines

    # Load every engine regardless of its enable flag: the flag says whether an
    # engine runs tonight, not whether it is experimental.
    config = load_series_config()
    load_engines(
        replace(
            config,
            engines={name: replace(e, enabled=True) for name, e in config.engines.items()},
        )
    )

    assert set(experimental_engines()) == set(ts_string_array("EXPERIMENTAL_ASSETS"))
    assert set(experimental_engines()) == {"crypto"}


def test_the_experimental_disclaimer_is_additional_not_alternative():
    """Both claims are true at once, so serving appends rather than substitutes.

    Asserted from Python because the compute plane is where the reason lives:
    the §18 disclaimer is about the whole system and the experimental sentence
    is about one engine, and a consumer diffing the two strings should see
    exactly what the extra claim is.
    """
    source = DOMAIN_TS.read_text()
    assert "EXPERIMENTAL_DISCLAIMER" in source
    assert "`${DISCLAIMER} ${EXPERIMENTAL_DISCLAIMER}`" in (
        (
            Path(__file__).resolve().parents[2] / "serving" / "src" / "lib" / "responses.ts"
        ).read_text()
    )


def test_money_vocabulary_stays_out_of_the_shared_layer():
    """§3 rule 2 — the liquidity states belong to engines/money, not to core.

    The discount horizons *are* shared vocabulary and do live in core: every
    engine that discounts a cash flow has to mean the same thing by "3y".
    """
    from findynamics.core.contracts import vocab

    assert not hasattr(vocab, "MONEY_REGIMES")
    assert hasattr(vocab, "DISCOUNT_HORIZONS")


def test_quantiles_match_the_serving_plane():
    source = DOMAIN_TS.read_text()
    match = re.search(r"export const QUANTILES = \[(.*?)\] as const;", source, re.DOTALL)
    assert match
    ts_quantiles = [float(q) for q in re.findall(r"[0-9.]+", match.group(1))]
    assert ts_quantiles == list(QUANTILES)


def test_educational_horizons_are_excluded_from_evaluation():
    """§10 — 30/50y scenarios are simulation only and must stay flagged."""
    from findynamics.core.contracts.vocab import EDUCATIONAL_HORIZONS

    assert set(EDUCATIONAL_HORIZONS) == {"educational_30y", "educational_50y"}
    assert EDUCATIONAL_HORIZONS.issubset(HORIZONS)


def test_gold_vocabulary_stays_out_of_the_shared_layer():
    """§3 rule 2 — gold's three states belong to engines/gold, not to core."""
    from findynamics.core.contracts import vocab

    for name in ("GOLD_REGIMES", "GOLD_METRICS"):
        assert not hasattr(vocab, name), f"{name} belongs to engines/gold, not core"


def test_equity_vocabulary_stays_out_of_the_shared_layer():
    """§3 rule 2 — one engine's regimes must not become everyone's vocabulary."""
    from findynamics.core.contracts import vocab

    for name in ("REGIMES", "KINEMATIC_FEATURES", "SHOCK_CLASSES"):
        assert not hasattr(vocab, name), f"{name} belongs to engines/equity, not core"


def test_hmac_matches_the_typescript_verifier():
    """Fixed vector, asserted identically in serving/test/hmac.spec.ts.

    If either implementation changes its canonical string, one of the two tests
    goes red instead of the nightly write-back silently 401-ing.
    """
    timestamp, signature = sign_payload("findyn-parity-vector", '{"a":1}', timestamp=1750000000)
    assert timestamp == "1750000000"
    assert signature == "14f1496721f7ac017aa6b6f0ce9edb1bc7f68ef26ca45e434817413526145747"
