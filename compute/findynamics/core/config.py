"""Load and validate compute/config/ (FINDYN_V1_SPEC.md §5.2).

Series ids, providers and publication lags live in configuration so that adding
a data source never requires a code change. Validation is strict: a malformed
entry must fail at load time, not silently produce a feature with the wrong
release date.

Two files feed one :class:`SeriesConfig`:

* ``config/series.yaml`` — ``meta``, the shared ``factors`` (Layer 0) and each
  engine's private ``series`` block.
* ``config/engines/<name>.yaml`` — per-engine enable flag and parameters, so
  shipping a new engine never edits job code (§8).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from findynamics.core.contracts.vocab import ASSETS, FACTORS, FREQUENCIES

CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"
SERIES_CONFIG_PATH = CONFIG_DIR / "series.yaml"
ENGINES_CONFIG_DIR = CONFIG_DIR / "engines"

# Only providers with a working adapter (data/providers/registry.py), plus the
# two that are never fetched. Naming an unimplemented provider here would defer
# the failure from config load to the first fetch — a typo should not survive
# that long. tests/data/test_providers.py asserts this set and the registry's
# NETWORK_PROVIDERS stay in lock-step.
VALID_PROVIDERS = frozenset(
    {
        "fred",
        "shiller",
        "bls",
        "bea",
        "stooq",
        # The London bullion benchmark, read from the body that sets it. FRED
        # delisted the LBMA gold series it used to carry, so there is no longer a
        # FRED route to the fix (data/providers/lbma.py).
        "lbma",
        # Keyless Bitcoin network metrics (data/providers/blockchain.py). The
        # paid on-chain vendors are deliberately absent — see the TODO in
        # data/providers/registry.py for why an id with no adapter must not be
        # listed here.
        "blockchain",
        # Fallback only, and isolated behind one adapter so it stays deletable
        # (FINDYN_V1_SPEC.md §5.1 source 7).
        "yahoo",
        # Computed downstream from other configured series, never fetched.
        "derived",
        # Not an external source: another engine's published output, read back
        # from the serving plane so a consumer never imports the producer
        # (core/contracts/vocab.py::ENGINE_SERIES_PREFIX).
        "engine_output",
    }
)
VALID_FREQUENCIES = FREQUENCIES


class ConfigError(ValueError):
    """Raised when the config does not satisfy the spec's contract."""


@dataclass(frozen=True)
class SeriesSpec:
    """One configured input series."""

    id: str
    provider: str
    frequency: str
    #: Conservative fallback lag used to synthesise release_date when the source
    #: exposes no vintage (§14.1). Never negative — that would be lookahead.
    publication_lag_days: int
    #: +1 if a higher value is risk-supportive for its factor, -1 if it is a headwind.
    direction: int = 1
    #: Earliest observation worth requesting, ``YYYY-MM-DD``, or ``None`` for
    #: "everything the source holds".
    #:
    #: A **fetch** hint and nothing more: it bounds what backfill asks a provider
    #: for, and no model reads it. That distinction is the whole reason it is
    #: safe — a value that reached the scoring pipeline would silently shorten an
    #: expanding window and move every historical percentile built on it.
    #:
    #: Its purpose is to stop a job asking for history that does not exist.
    #: Bitcoin has no market price before 2010-08-18, so a backfill that requests
    #: 2009 gets either an error or a run of zeros depending on the vendor's
    #: mood, and neither is a price.
    start: str | None = None


@dataclass(frozen=True)
class FactorSpec:
    """One shared risk factor and the series it is built from (Layer 0)."""

    name: str
    weight: float
    series: tuple[SeriesSpec, ...]


@dataclass(frozen=True)
class EngineConfig:
    """One asset engine: its private series, its enable flag and its parameters."""

    name: str
    enabled: bool = False
    #: Engine-private series, keyed by role (``primary``, ``backfill``, …).
    series: dict[str, SeriesSpec] = field(default_factory=dict)
    #: Free-form block from ``config/engines/<name>.yaml`` minus ``enabled``.
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SeriesConfig:
    spec_version: str
    info_set: str
    history_start: str
    factors: dict[str, FactorSpec]
    engines: dict[str, EngineConfig]
    realtime_cache_ttl_seconds: int
    realtime_cache_series: tuple[str, ...]

    def all_series(self) -> tuple[SeriesSpec, ...]:
        return tuple(s for engine in self.engines.values() for s in engine.series.values()) + tuple(
            s for factor in self.factors.values() for s in factor.series
        )

    def engine_series(self, engine: str) -> dict[str, SeriesSpec]:
        """Private series of one engine; empty when the engine declares none."""
        config = self.engines.get(engine)
        return dict(config.series) if config else {}

    def is_enabled(self, engine: str) -> bool:
        config = self.engines.get(engine)
        return bool(config and config.enabled)

    def enabled_engine_names(self) -> tuple[str, ...]:
        """Enabled engines, in ASSETS order so a run is deterministic."""
        return tuple(name for name in ASSETS if self.is_enabled(name))


def _series_from_mapping(raw: Any, where: str) -> SeriesSpec:
    if not isinstance(raw, dict):
        raise ConfigError(f"{where}: expected a mapping, got {type(raw).__name__}")

    series_id = raw.get("id")
    if not series_id:
        raise ConfigError(f"{where}: missing 'id'")

    provider = raw.get("provider")
    if provider not in VALID_PROVIDERS:
        raise ConfigError(f"{where}: unknown provider {provider!r}")

    frequency = raw.get("frequency")
    if frequency not in VALID_FREQUENCIES:
        raise ConfigError(f"{where}: unknown frequency {frequency!r}")

    lag = raw.get("publication_lag_days")
    if not isinstance(lag, int) or isinstance(lag, bool):
        raise ConfigError(f"{where}: publication_lag_days must be an int, got {lag!r}")
    if lag < 0:
        raise ConfigError(f"{where}: publication_lag_days must be >= 0 (negative lag is lookahead)")

    direction = raw.get("direction", 1)
    if direction not in (1, -1):
        raise ConfigError(f"{where}: direction must be 1 or -1, got {direction!r}")

    start = raw.get("start")
    if start is not None:
        # Validated at load rather than at fetch, like everything else here: a
        # typo'd date should fail the config, not the job that used it.
        if not isinstance(start, str):
            raise ConfigError(f"{where}: start must be a YYYY-MM-DD string, got {start!r}")
        try:
            date.fromisoformat(start)
        except ValueError as err:
            raise ConfigError(f"{where}: start {start!r} is not a valid date: {err}") from None

    return SeriesSpec(
        id=str(series_id),
        provider=provider,
        frequency=frequency,
        publication_lag_days=lag,
        direction=direction,
        start=start,
    )


def _parse_factors(raw: Any) -> dict[str, FactorSpec]:
    factors_raw = raw or {}
    if not isinstance(factors_raw, dict):
        raise ConfigError("factors: expected a mapping")

    missing = set(FACTORS) - set(factors_raw)
    if missing:
        raise ConfigError(f"factors missing from config: {sorted(missing)}")
    unexpected = set(factors_raw) - set(FACTORS)
    if unexpected:
        raise ConfigError(f"unknown factors in config: {sorted(unexpected)}")

    factors: dict[str, FactorSpec] = {}
    for name, entry in factors_raw.items():
        if not isinstance(entry, dict):
            raise ConfigError(f"factors.{name}: expected a mapping")
        series = entry.get("series") or []
        if not series:
            raise ConfigError(f"factors.{name}: at least one series is required")
        factors[name] = FactorSpec(
            name=name,
            weight=float(entry.get("weight", 1.0)),
            series=tuple(
                _series_from_mapping(s, f"factors.{name}.series[{i}]") for i, s in enumerate(series)
            ),
        )
    return factors


def _parse_engine_series(name: str, entry: Any) -> dict[str, SeriesSpec]:
    if not isinstance(entry, dict):
        raise ConfigError(f"engines.{name}: expected a mapping, got {type(entry).__name__}")

    if "series" not in entry:
        return {}
    series_raw = entry["series"]
    if not isinstance(series_raw, dict) or not series_raw:
        raise ConfigError(f"engines.{name}.series: expected a non-empty mapping")

    # Every series declares its provider-native id explicitly. The v1 config
    # synthesised ``PRICE:<symbol>`` ids here, but no adapter recognises those:
    # a synthesised id passed validation and then failed at fetch time, which is
    # exactly the class of failure this module exists to catch at load.
    return {
        role: _series_from_mapping(role_entry, f"engines.{name}.series.{role}")
        for role, role_entry in series_raw.items()
    }


def _parse_engines(raw: Any) -> dict[str, dict[str, SeriesSpec]]:
    engines_raw = raw or {}
    if not isinstance(engines_raw, dict):
        raise ConfigError("engines: expected a mapping")
    unknown = set(engines_raw) - set(ASSETS)
    if unknown:
        raise ConfigError(f"unknown engines in config: {sorted(unknown)}")
    return {name: _parse_engine_series(name, entry) for name, entry in engines_raw.items()}


def load_engine_configs(directory: Path) -> dict[str, tuple[bool, dict[str, Any]]]:
    """Read ``config/engines/*.yaml`` into (enabled, params) pairs.

    A missing directory is not an error: an engine that ships no file is simply
    disabled, which is the correct state for an engine that does not exist yet.
    """
    if not directory.is_dir():
        return {}

    result: dict[str, tuple[bool, dict[str, Any]]] = {}
    for path in sorted(directory.glob("*.yaml")):
        name = path.stem
        if name not in ASSETS:
            raise ConfigError(
                f"engines/{path.name}: unknown engine {name!r}; expected one of {list(ASSETS)}"
            )
        body = yaml.safe_load(path.read_text()) or {}
        if not isinstance(body, dict):
            raise ConfigError(f"engines/{path.name}: expected a mapping at the top level")
        enabled = body.get("enabled", False)
        if not isinstance(enabled, bool):
            raise ConfigError(f"engines/{path.name}: 'enabled' must be a boolean, got {enabled!r}")
        result[name] = (enabled, {k: v for k, v in body.items() if k != "enabled"})
    return result


def load_series_config(
    path: Path | None = None, *, engines_dir: Path | None = None
) -> SeriesConfig:
    """Parse and validate the config. Raises ConfigError on any violation.

    ``engines_dir`` defaults to an ``engines/`` directory beside ``path``, so a
    fixture written to tmp_path never picks up the shipped engine flags.
    """
    config_path = path or SERIES_CONFIG_PATH
    if not config_path.exists():
        raise ConfigError(f"series config not found: {config_path}")

    engines_path = engines_dir if engines_dir is not None else config_path.parent / "engines"

    raw = yaml.safe_load(config_path.read_text())
    if not isinstance(raw, dict):
        raise ConfigError("series.yaml must contain a mapping at the top level")

    meta = raw.get("meta") or {}
    for key in ("spec_version", "info_set", "history_start"):
        if key not in meta:
            raise ConfigError(f"meta.{key} is required")

    factors = _parse_factors(raw.get("factors"))
    engine_series = _parse_engines(raw.get("engines"))
    engine_flags = load_engine_configs(engines_path)

    engines: dict[str, EngineConfig] = {}
    for name in ASSETS:
        if name not in engine_series and name not in engine_flags:
            continue
        enabled, params = engine_flags.get(name, (False, {}))
        engines[name] = EngineConfig(
            name=name,
            enabled=enabled,
            series=engine_series.get(name, {}),
            params=params,
        )

    cache = raw.get("realtime_cache") or {}

    return SeriesConfig(
        spec_version=str(meta["spec_version"]),
        info_set=str(meta["info_set"]),
        history_start=str(meta["history_start"]),
        factors=factors,
        engines=engines,
        realtime_cache_ttl_seconds=int(cache.get("ttl_seconds", 900)),
        realtime_cache_series=tuple(cache.get("series", ())),
    )


@lru_cache(maxsize=1)
def get_series_config() -> SeriesConfig:
    """Cached accessor for the default config path."""
    return load_series_config()
