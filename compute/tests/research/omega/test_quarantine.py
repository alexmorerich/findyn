"""The research layer is invisible to production, structurally.

Two gates keep KK-Ω out of anything that matters, and this module tests both
ends of the *structural* one — the flag is the second line of defence and is
only checked here to confirm it ships off.

The first gate is the ``Research is quarantined`` import-linter contract in
pyproject.toml, which ``lint-imports`` enforces in CI. That contract proves
nothing *imports* the module. These tests prove the complementary half: that
importing it has no effect on the production registries and configs, so a
developer exploring the research package in a REPL cannot accidentally change
what a daily run would publish.

The specific accident being guarded against is a research module that reaches
for ``@register_engine`` because that is how everything else in the repo becomes
visible. It cannot succeed — ``core/contracts/vocab.py::ASSETS`` is a closed
five-tuple and ``register_engine`` rejects anything outside it — but "it would
raise" is not the same as "nobody tried", and the second test is what tells the
difference at review time.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from findynamics.core.config import load_series_config
from findynamics.core.contracts.vocab import ASSETS
from findynamics.core.engine import AssetEngine
from findynamics.core.registry import RegistryError, register_engine, registered_engines
from findynamics.research.omega import OmegaEngine, is_enabled

COMPUTE_ROOT = Path(__file__).resolve().parents[3]
CONFIG_DIR = COMPUTE_ROOT / "config"


def test_importing_the_research_module_adds_no_name_to_the_registry():
    """In a fresh interpreter, the module's registry footprint is bounded.

    Not "registers nothing" — that was true in KK0 and is not true now, and
    pretending otherwise would make this test a lie that passes. KK1 reuses the
    equity feature stack (``kalman.filter_state``, ``design.realized_vol``,
    ``rii.vol_of_vol``), which is the explicit architecture decision for this
    track, and importing any ``findynamics.engines.equity`` submodule runs that
    package's ``__init__``, which self-registers ``EquityEngine``.

    So the invariant worth asserting is the one that actually protects
    production: a research import can add **no new name** to the registry. The
    set is pinned to exactly ``{"equity"}`` rather than merely checked against
    ``ASSETS``, so that a later phase reaching for, say, ``engines.crypto``
    fails here and has to say why in a diff.

    A subprocess rather than an in-process import: by the time this body runs,
    every engine in the repo has been registered by some earlier test, and
    re-importing would assert nothing at all.
    """
    program = (
        "import findynamics.research.omega as omega;"
        "from findynamics.core.registry import registered_engines;"
        "from findynamics.core.contracts.vocab import ASSETS;"
        "names = registered_engines();"
        "print(sorted(names), set(names) <= set(ASSETS), 'omega' in names, omega.is_enabled())"
    )
    result = subprocess.run(
        [sys.executable, "-c", program],
        cwd=COMPUTE_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == "['equity'] True False False", result.stdout + result.stderr


def test_the_omega_engine_is_not_an_asset_engine():
    """It has the shape of one and is deliberately not one.

    ``core/contracts/vocab.py::ASSETS`` is a closed five-tuple mirrored in
    ``serving/src/domain.ts``. Ω is not a sixth member and this class is not a
    sixth engine: no subclass, no ``name``, no ``version``, and nothing that
    would let ``enabled_engines`` build it.
    """
    assert not issubclass(OmegaEngine, AssetEngine)
    assert not hasattr(OmegaEngine, "name")
    assert not hasattr(OmegaEngine, "version")
    # `model_version` exists and is deliberately not called `version`: the
    # registry reads the latter, and a research module should not have an
    # attribute whose only effect would be to make a mistaken registration work.
    assert hasattr(OmegaEngine, "model_version")


def test_registering_the_omega_engine_is_refused():
    """Two independent refusals, and the test asserts the first one fires.

    ``register_engine`` reads ``name`` before it reads anything else, so the
    missing attribute stops a copy-paste registration before it can reach the
    ASSETS check. The second half of the test shows that check would have
    refused it too — belt and braces, because the first refusal is an absence
    and absences are easy to accidentally fill in.
    """
    with pytest.raises(RegistryError, match="must declare a non-empty class attribute 'name'"):
        register_engine(OmegaEngine)

    class PretendOmegaEngine:
        name = "omega"
        version = "0.1.0"

    with pytest.raises(RegistryError, match="is not one of"):
        register_engine(PretendOmegaEngine)

    assert "omega" not in registered_engines()


def test_the_core_config_loader_does_not_see_config_research():
    """``config/research/`` must be invisible to ``load_series_config``.

    This is the reason the research config lives where it does.
    ``core/config.py::load_engine_configs`` globs ``config/engines/*.yaml`` and
    raises ``ConfigError`` for any stem that is not in ``ASSETS`` — so a file at
    ``config/engines/omega.yaml`` would not merely be rejected, it would break
    every config load in the repo, for every engine, on every run. Loading the
    shipped config with ``config/research/`` on disk is the check that the
    directory chosen sidesteps that entirely.
    """
    assert (CONFIG_DIR / "research" / "omega.yaml").exists()

    config = load_series_config()

    assert "omega" not in config.engines
    assert set(config.engines) <= set(ASSETS)
    assert "omega" not in config.enabled_engine_names()


def test_the_shipped_research_config_is_disabled():
    """The second line of defence, shipped in the state it is supposed to ship in."""
    assert is_enabled() is False


def test_the_asset_vocabulary_is_untouched():
    """Ω is not a sixth asset, and this track must not have made it one.

    Deliberately not a duplicate of ``tests/test_domain.py``, which owns the
    Python/TypeScript parity check. This asserts membership and length only —
    the property this track could plausibly have broken.
    """
    assert len(ASSETS) == 5
    assert set(ASSETS) == {"money", "rates", "equity", "gold", "crypto"}
    assert "omega" not in ASSETS
