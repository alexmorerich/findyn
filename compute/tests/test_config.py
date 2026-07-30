"""series.yaml contract tests (FINDYN_V1_SPEC.md §5.2)."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from findyn.config import ConfigError, load_series_config
from findyn.domain import FORCES


def test_shipped_config_loads():
    config = load_series_config()
    assert config.spec_version == "1.0"
    assert config.info_set == "t-1"
    assert config.history_start == "1871-01-01"


def test_every_force_in_the_spec_is_configured():
    config = load_series_config()
    assert set(config.forces) == set(FORCES)
    for name, force in config.forces.items():
        assert force.series, f"{name} has no input series"


def test_no_series_declares_a_negative_publication_lag():
    """A negative lag would mean data known before it was published (§14.1)."""
    for series in load_series_config().all_series():
        assert series.publication_lag_days >= 0, series.id


def test_price_has_a_primary_and_an_isolated_fallback():
    price = load_series_config().price
    assert price["primary"].provider == "alphavantage"
    # Yahoo stays a fallback only, so it can be deleted without touching callers (§5.1).
    assert price["fallback"].provider == "yahoo"
    assert price["backfill"].provider == "stooq"


def test_deep_history_reaches_1871():
    config = load_series_config()
    assert config.price["deep_history"].provider == "shiller"
    assert config.history_start.startswith("1871")


def _write(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "series.yaml"
    path.write_text(textwrap.dedent(body))
    return path


MINIMAL_FORCES = "\n".join(
    f"  {f}:\n"
    f"    weight: 1.0\n"
    f"    series:\n"
    f"      - {{id: 'FRED:X_{f}', provider: fred, frequency: daily, publication_lag_days: 1}}"
    for f in FORCES
)

VALID = f"""
meta:
  spec_version: "1.0"
  info_set: "t-1"
  history_start: "1871-01-01"
price:
  primary: {{provider: alphavantage, symbol: SPY, frequency: daily, publication_lag_days: 0}}
forces:
{MINIMAL_FORCES}
"""


def test_minimal_valid_config_parses(tmp_path):
    assert load_series_config(_write(tmp_path, VALID)).info_set == "t-1"


def test_missing_force_is_rejected(tmp_path):
    body = VALID.replace(
        "  sentiment:\n    weight: 1.0\n", "  weight_removed_sentinel:\n    weight: 1.0\n"
    )
    with pytest.raises(ConfigError):
        load_series_config(_write(tmp_path, body))


def test_unknown_provider_is_rejected(tmp_path):
    body = VALID.replace(
        "provider: fred, frequency: daily", "provider: bloomberg, frequency: daily"
    )
    with pytest.raises(ConfigError, match="unknown provider"):
        load_series_config(_write(tmp_path, body))


def test_negative_lag_is_rejected(tmp_path):
    body = VALID.replace("publication_lag_days: 1}", "publication_lag_days: -1}")
    with pytest.raises(ConfigError, match="lookahead"):
        load_series_config(_write(tmp_path, body))


def test_missing_price_primary_is_rejected(tmp_path):
    body = VALID.replace("  primary: {provider", "  secondary: {provider")
    with pytest.raises(ConfigError, match="price.primary"):
        load_series_config(_write(tmp_path, body))
