"""Cross-plane parity.

The domain vocabulary and the write-back signature exist in both TypeScript
(serving) and Python (compute). Drift between the two copies would surface as a
silently dropped regime or a rejected write-back at 03:00 UTC, so both are
pinned here.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from findyn.domain import FORCES, HORIZONS, QUANTILES, REGIMES
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
    [("FORCES", FORCES), ("REGIMES", REGIMES), ("HORIZONS", HORIZONS)],
)
def test_vocabulary_matches_the_serving_plane(name, python_value):
    assert ts_string_array(name) == list(python_value)


def test_quantiles_match_the_serving_plane():
    source = DOMAIN_TS.read_text()
    match = re.search(r"export const QUANTILES = \[(.*?)\] as const;", source, re.DOTALL)
    assert match
    ts_quantiles = [float(q) for q in re.findall(r"[0-9.]+", match.group(1))]
    assert ts_quantiles == list(QUANTILES)


def test_educational_horizons_are_excluded_from_evaluation():
    """§10 — 30/50y scenarios are simulation only and must stay flagged."""
    from findyn.domain import EDUCATIONAL_HORIZONS

    assert set(EDUCATIONAL_HORIZONS) == {"educational_30y", "educational_50y"}
    assert EDUCATIONAL_HORIZONS.issubset(HORIZONS)


def test_hmac_matches_the_typescript_verifier():
    """Fixed vector, asserted identically in serving/test/hmac.spec.ts.

    If either implementation changes its canonical string, one of the two tests
    goes red instead of the nightly write-back silently 401-ing.
    """
    timestamp, signature = sign_payload("findyn-parity-vector", '{"a":1}', timestamp=1750000000)
    assert timestamp == "1750000000"
    assert signature == "14f1496721f7ac017aa6b6f0ce9edb1bc7f68ef26ca45e434817413526145747"
