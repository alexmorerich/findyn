"""Load and validate ``config/research/omega.yaml``.

A research-local loader, deliberately. ``core/config.py::load_engine_configs``
globs ``config/engines/*.yaml`` and raises ``ConfigError`` for any stem that is
not a member of ``ASSETS``, so a file named ``config/engines/omega.yaml`` would
not merely be rejected — it would break **every** config load in the repo, for
every engine, on every run. The research config therefore lives in its own
directory, is resolved here from ``CONFIG_DIR`` directly, and is invisible to
the core loader by construction rather than by exclusion list.

Strict at load time, like :mod:`findynamics.core.config` and
:mod:`findynamics.portfolio.config`: a malformed file fails when it is read, not
when some walk-forward window three hours in reaches the key that was a string
instead of a float.

**Absence and malformation are different.** A missing file means the module is
not configured, which is a legitimate state and the one a fresh clone of a
future commit that deleted this module would be in — :func:`is_enabled` answers
``False`` for it. A file that exists and does not parse is an error, and it
raises. Collapsing the two would let a typo in the filename read as "disabled",
which is the failure mode where nobody notices for a month.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

#: ``compute/config``. Resolved from this file rather than imported from
#: ``core.config`` so that the research module owns its own path and a future
#: refactor of the core loader cannot silently re-point it.
CONFIG_DIR = Path(__file__).resolve().parents[3] / "config"
OMEGA_CONFIG_PATH = CONFIG_DIR / "research" / "omega.yaml"


class OmegaConfigError(ValueError):
    """Raised when omega.yaml does not satisfy its contract."""


@dataclass(frozen=True)
class OmegaConfig:
    """The research module's flags and parameters.

    ``params`` is the free-form remainder — everything except the two flags —
    mirroring ``core.config.EngineConfig.params``. KK1 onward reads its own
    blocks out of it and validates them there, next to the code that has an
    opinion about what a valid window is; hoisting every window and threshold
    into this dataclass would put the estimator's vocabulary in the loader.
    """

    enabled: bool = False
    experimental: bool = True
    params: dict[str, Any] = field(default_factory=dict)

    def block(self, name: str) -> dict[str, Any]:
        """One parameter block, or an empty mapping when it is absent.

        Empty rather than ``None`` so a caller's ``.get(key, default)`` chain
        reads the same whether the block was omitted or shipped empty.
        """
        value = self.params.get(name)
        if value is None:
            return {}
        if not isinstance(value, dict):
            raise OmegaConfigError(f"{name}: expected a mapping, got {type(value).__name__}")
        return dict(value)


def _require_bool(value: object, where: str) -> bool:
    if not isinstance(value, bool):
        raise OmegaConfigError(f"{where}: must be a boolean, got {value!r}")
    return value


def load_omega_config(path: Path | None = None) -> OmegaConfig:
    """Parse and validate omega.yaml. Raises :class:`OmegaConfigError` on any violation.

    A missing file raises here rather than returning a default. The caller that
    wants "absent means off" is :func:`is_enabled`, and it says so explicitly —
    a loader that invents a config for a file it could not find is a loader that
    cannot report a mistyped path.
    """
    config_path = path or OMEGA_CONFIG_PATH
    if not config_path.exists():
        raise OmegaConfigError(f"omega research config not found: {config_path}")

    raw = yaml.safe_load(config_path.read_text())
    if not isinstance(raw, dict):
        raise OmegaConfigError(f"{config_path.name} must contain a mapping at the top level")

    enabled = _require_bool(raw.get("enabled", False), "enabled")
    experimental = _require_bool(raw.get("experimental", True), "experimental")

    # `experimental: false` is refused rather than honoured. There is no state of
    # this module in which the flag would be true: the `Research is quarantined`
    # contract makes production dependence on it a CI failure, so a config that
    # declares the module non-experimental is describing a system that does not
    # exist. Refusing here means the person who flips it finds out immediately
    # instead of building on the claim.
    if not experimental:
        raise OmegaConfigError(
            "experimental must be true — findynamics.research is quarantined by "
            "import-linter and can never be a production dependency; see "
            "docs/research/kk-omega-design.md §10"
        )

    params = {k: v for k, v in raw.items() if k not in ("enabled", "experimental")}
    return OmegaConfig(enabled=enabled, experimental=experimental, params=params)


@lru_cache(maxsize=1)
def get_omega_config() -> OmegaConfig:
    """Cached accessor for the shipped research config."""
    return load_omega_config()


def is_enabled(path: Path | None = None) -> bool:
    """Whether the research module is switched on. ``False`` when unconfigured.

    The *second* line of defence and never the first. The first is the
    ``Research is quarantined`` import-linter contract, which does not care what
    this returns. Anything that treats a ``True`` here as permission to import
    this package from a production path has misread which gate is load-bearing.
    """
    config_path = path or OMEGA_CONFIG_PATH
    if not config_path.exists():
        return False
    return load_omega_config(config_path).enabled


__all__ = [
    "CONFIG_DIR",
    "OMEGA_CONFIG_PATH",
    "OmegaConfig",
    "OmegaConfigError",
    "get_omega_config",
    "is_enabled",
    "load_omega_config",
]
