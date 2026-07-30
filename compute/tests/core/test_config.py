"""series.yaml + config/engines contract tests (FINDYN_V1_SPEC.md §5.2)."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from findynamics.core.config import ConfigError, load_series_config
from findynamics.factors.definitions import FACTORS


def test_shipped_config_loads():
    config = load_series_config()
    assert config.spec_version == "1.0"
    assert config.info_set == "t-1"
    assert config.history_start == "1871-01-01"


def test_every_factor_in_the_spec_is_configured():
    config = load_series_config()
    assert set(config.factors) == set(FACTORS)
    for name, factor in config.factors.items():
        assert factor.series, f"{name} has no input series"


def test_no_series_declares_a_negative_publication_lag():
    """A negative lag would mean data known before it was published (§14.1)."""
    for series in load_series_config().all_series():
        assert series.publication_lag_days >= 0, series.id


def test_equity_price_has_a_primary_and_an_isolated_fallback():
    price = load_series_config().engine_series("equity")
    assert price["primary"].provider == "alphavantage"
    # Yahoo stays a fallback only, so it can be deleted without touching callers (§5.1).
    assert price["fallback"].provider == "yahoo"
    assert price["backfill"].provider == "stooq"


def test_deep_history_reaches_1871():
    config = load_series_config()
    assert config.engine_series("equity")["deep_history"].provider == "shiller"
    assert config.history_start.startswith("1871")


def test_engine_series_ids_survived_the_move_from_the_price_block():
    """The v1 `PRICE:<symbol>` ids are what macro_series rows are keyed by."""
    price = load_series_config().engine_series("equity")
    assert price["primary"].id == "PRICE:SPY"
    assert price["backfill"].id == "PRICE:^spx"
    assert price["deep_history"].id == "PRICE:deep_history"


def test_only_engines_whose_phase_has_landed_are_enabled():
    """An engine enabled before its models exist would publish nothing daily.

    P1 shipped rates, P2 shipped money. Equity, gold and crypto are configured
    but disabled, which is the correct state for an engine that does not exist —
    and the ordering is ASSETS order, not config order, so a run is deterministic.
    """
    config = load_series_config()
    assert config.enabled_engine_names() == ("money", "rates")
    assert set(config.engines) == {"money", "rates", "equity", "gold", "crypto"}
    for unbuilt in ("equity", "gold", "crypto"):
        assert not config.is_enabled(unbuilt)


def test_the_rates_engine_declares_its_curve_in_config_not_code():
    """Adding the 4-month bill must be a yaml edit, not a deploy."""
    config = load_series_config()
    rates = config.engines["rates"]

    assert {spec.id for spec in rates.series.values()} >= {
        "FRED:DGS1MO",
        "FRED:DGS3MO",
        "FRED:DGS10",
        "FRED:DGS30",
        "FRED:T10YIE",
    }
    assert set(rates.params["tenors"]) == {
        spec.id for spec in rates.series.values() if spec.id != rates.params["breakeven_series"]
    }


def _write(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "series.yaml"
    path.write_text(textwrap.dedent(body))
    return path


MINIMAL_FACTORS = "\n".join(
    f"  {f}:\n"
    f"    weight: 1.0\n"
    f"    series:\n"
    f"      - {{id: 'FRED:X_{f}', provider: fred, frequency: daily, publication_lag_days: 1}}"
    for f in FACTORS
)

VALID = f"""
meta:
  spec_version: "1.0"
  info_set: "t-1"
  history_start: "1871-01-01"
engines:
  equity:
    series:
      primary: {{provider: alphavantage, symbol: SPY, frequency: daily, publication_lag_days: 0}}
factors:
{MINIMAL_FACTORS}
"""


def test_minimal_valid_config_parses(tmp_path):
    assert load_series_config(_write(tmp_path, VALID)).info_set == "t-1"


def test_missing_factor_is_rejected(tmp_path):
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


def test_empty_engine_series_block_is_rejected(tmp_path):
    """An engine that declares `series:` and lists none is a truncated edit."""
    body = VALID.replace(
        "      primary: {provider: alphavantage, symbol: SPY, frequency: daily, publication_lag_days: 0}",
        "",
    )
    with pytest.raises(ConfigError, match="engines.equity.series"):
        load_series_config(_write(tmp_path, body))


def test_unknown_engine_is_rejected(tmp_path):
    body = VALID.replace("  equity:\n", "  commodities:\n")
    with pytest.raises(ConfigError, match="unknown engines"):
        load_series_config(_write(tmp_path, body))


def _engines_dir(tmp_path: Path, name: str, body: str) -> Path:
    directory = tmp_path / "engines"
    directory.mkdir(exist_ok=True)
    (directory / f"{name}.yaml").write_text(textwrap.dedent(body))
    return directory


def test_engine_enable_flag_is_read_from_its_own_yaml(tmp_path):
    path = _write(tmp_path, VALID)
    _engines_dir(tmp_path, "equity", "enabled: true\nlookback_years: 30\n")
    config = load_series_config(path)
    assert config.is_enabled("equity")
    assert config.enabled_engine_names() == ("equity",)
    assert config.engines["equity"].params == {"lookback_years": 30}


def test_engines_default_to_disabled_when_no_yaml_exists(tmp_path):
    config = load_series_config(_write(tmp_path, VALID))
    assert config.enabled_engine_names() == ()
    assert config.engines["equity"].enabled is False


def test_non_boolean_enable_flag_is_rejected(tmp_path):
    path = _write(tmp_path, VALID)
    _engines_dir(tmp_path, "equity", "enabled: yes please\n")
    with pytest.raises(ConfigError, match="must be a boolean"):
        load_series_config(path)


def test_engine_yaml_for_an_unknown_engine_is_rejected(tmp_path):
    path = _write(tmp_path, VALID)
    _engines_dir(tmp_path, "commodities", "enabled: false\n")
    with pytest.raises(ConfigError, match="unknown engine"):
        load_series_config(path)
