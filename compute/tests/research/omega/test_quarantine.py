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

from findynamics.core.config import load_series_config
from findynamics.core.contracts.vocab import ASSETS
from findynamics.research.omega import is_enabled

COMPUTE_ROOT = Path(__file__).resolve().parents[3]
CONFIG_DIR = COMPUTE_ROOT / "config"


def test_importing_the_research_module_registers_no_engine():
    """In a fresh interpreter, the module's whole side effect is nothing.

    A subprocess rather than an in-process import, because by the time this test
    body runs the package is already in ``sys.modules`` and every engine in the
    repo has been registered by some earlier test. Re-importing here would
    assert nothing at all. A clean interpreter that imports *only* the research
    package and finds an empty registry is the statement worth making.
    """
    program = (
        "import findynamics.research.omega as omega;"
        "from findynamics.core.registry import ENGINES, registered_engines;"
        "print(repr(registered_engines()), len(ENGINES), omega.is_enabled())"
    )
    result = subprocess.run(
        [sys.executable, "-c", program],
        cwd=COMPUTE_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == "() 0 False", result.stdout + result.stderr


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
