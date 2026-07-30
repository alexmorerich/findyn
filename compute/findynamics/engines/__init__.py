"""Asset engines — mutually isolated implementations of ``AssetEngine``.

This is the only module allowed to name the engines (``03-contracts.md`` §3).
Everything else discovers them through ``core.registry`` by name, so nothing
outside this file depends on which engines happen to exist. No engine imports
another (CI-enforced).

Importing an engine package is what registers it. That import is guarded by the
engine's config enable flag, so a half-built engine cannot appear in a daily run
by being merged — it appears when someone flips ``enabled: true``.

``load_engines`` is idempotent: registration rejects genuine duplicates but
tolerates re-registering the same class, so calling it from several jobs is safe.
"""

from __future__ import annotations

import logging

from findynamics.core.config import SeriesConfig, get_series_config

log = logging.getLogger("findynamics.engines")


def load_engines(config: SeriesConfig | None = None) -> tuple[str, ...]:
    """Import every enabled engine package. Returns the names loaded.

    An engine enabled in config whose package does not exist is a configuration
    error and is raised as one — silently skipping it would mean the daily run
    quietly publishes nothing for an asset the operator asked for.
    """
    resolved = config or get_series_config()
    loaded: list[str] = []

    for name in resolved.enabled_engine_names():
        if name == "rates":
            from findynamics.engines import rates  # noqa: F401
        elif name == "money":
            from findynamics.engines import money  # noqa: F401
        elif name == "equity":
            from findynamics.engines import equity  # noqa: F401
        elif name == "gold":
            from findynamics.engines import gold  # noqa: F401
        elif name == "crypto":
            from findynamics.engines import crypto  # noqa: F401
        else:  # pragma: no cover - ASSETS is closed, so this is unreachable
            raise ValueError(f"no engine package for {name!r}")
        loaded.append(name)

    log.debug("loaded engines: %s", ", ".join(loaded) or "(none)")
    return tuple(loaded)
