"""Load and validate compute/config/series.yaml (FINDYN_V1_SPEC.md §5.2).

Series ids, providers and publication lags live in configuration so that adding
a data source never requires a code change. Validation is strict: a malformed
entry must fail at load time, not silently produce a feature with the wrong
release date.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from findyn.domain import FORCES

CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"
SERIES_CONFIG_PATH = CONFIG_DIR / "series.yaml"

VALID_PROVIDERS = frozenset(
    {"fred", "shiller", "bls", "bea", "treasury", "alphavantage", "stooq", "yahoo", "derived"}
)
VALID_FREQUENCIES = frozenset({"daily", "weekly", "monthly", "quarterly"})


class ConfigError(ValueError):
    """Raised when series.yaml does not satisfy the spec's contract."""


@dataclass(frozen=True)
class SeriesSpec:
    """One configured input series."""

    id: str
    provider: str
    frequency: str
    #: Conservative fallback lag used to synthesise release_date when the source
    #: exposes no vintage (§14.1). Never negative — that would be lookahead.
    publication_lag_days: int
    #: +1 if a higher value is risk-supportive for its force, -1 if it is a headwind.
    direction: int = 1


@dataclass(frozen=True)
class ForceSpec:
    name: str
    weight: float
    series: tuple[SeriesSpec, ...]


@dataclass(frozen=True)
class SeriesConfig:
    spec_version: str
    info_set: str
    history_start: str
    price: dict[str, SeriesSpec]
    forces: dict[str, ForceSpec]
    realtime_cache_ttl_seconds: int
    realtime_cache_series: tuple[str, ...]

    def all_series(self) -> tuple[SeriesSpec, ...]:
        return tuple(self.price.values()) + tuple(
            s for force in self.forces.values() for s in force.series
        )


def _series_from_mapping(raw: Any, where: str, *, default_id: str | None = None) -> SeriesSpec:
    if not isinstance(raw, dict):
        raise ConfigError(f"{where}: expected a mapping, got {type(raw).__name__}")

    series_id = raw.get("id", default_id)
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

    return SeriesSpec(
        id=str(series_id),
        provider=provider,
        frequency=frequency,
        publication_lag_days=lag,
        direction=direction,
    )


def load_series_config(path: Path | None = None) -> SeriesConfig:
    """Parse and validate the series map. Raises ConfigError on any violation."""
    config_path = path or SERIES_CONFIG_PATH
    if not config_path.exists():
        raise ConfigError(f"series config not found: {config_path}")

    raw = yaml.safe_load(config_path.read_text())
    if not isinstance(raw, dict):
        raise ConfigError("series.yaml must contain a mapping at the top level")

    meta = raw.get("meta") or {}
    for key in ("spec_version", "info_set", "history_start"):
        if key not in meta:
            raise ConfigError(f"meta.{key} is required")

    price_raw = raw.get("price") or {}
    if "primary" not in price_raw:
        raise ConfigError("price.primary is required")
    price = {
        role: _series_from_mapping(
            entry, f"price.{role}", default_id=f"PRICE:{entry.get('symbol', role)}"
        )
        for role, entry in price_raw.items()
    }

    forces_raw = raw.get("forces") or {}
    missing = set(FORCES) - set(forces_raw)
    if missing:
        raise ConfigError(f"forces missing from config: {sorted(missing)}")
    unexpected = set(forces_raw) - set(FORCES)
    if unexpected:
        raise ConfigError(f"unknown forces in config: {sorted(unexpected)}")

    forces: dict[str, ForceSpec] = {}
    for name, entry in forces_raw.items():
        series = entry.get("series") or []
        if not series:
            raise ConfigError(f"forces.{name}: at least one series is required")
        forces[name] = ForceSpec(
            name=name,
            weight=float(entry.get("weight", 1.0)),
            series=tuple(
                _series_from_mapping(s, f"forces.{name}.series[{i}]") for i, s in enumerate(series)
            ),
        )

    cache = raw.get("realtime_cache") or {}

    return SeriesConfig(
        spec_version=str(meta["spec_version"]),
        info_set=str(meta["info_set"]),
        history_start=str(meta["history_start"]),
        price=price,
        forces=forces,
        realtime_cache_ttl_seconds=int(cache.get("ttl_seconds", 900)),
        realtime_cache_series=tuple(cache.get("series", ())),
    )


@lru_cache(maxsize=1)
def get_series_config() -> SeriesConfig:
    """Cached accessor for the default config path."""
    return load_series_config()
