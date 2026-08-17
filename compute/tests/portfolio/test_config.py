"""portfolio.yaml is validated strictly, like every other config in this repo."""

from __future__ import annotations

import re
import textwrap
from pathlib import Path

import pytest

from findynamics.portfolio.config import (
    PortfolioConfigError,
    get_portfolio_config,
    load_portfolio_config,
)

DOMAIN_TS = Path(__file__).resolve().parents[3] / "serving" / "src" / "domain.ts"


def _write(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "portfolio.yaml"
    path.write_text(textwrap.dedent(body))
    return path


BASE = """
    model_version: portfolio-test
    distribution:
      samples: 16
      seed: 1
      return_sd: 0.04
      quantiles: [0.05, 0.5, 0.95]
    risk:
      sigma_floor: 0.02
      sigma_scale: 0.30
    max_state_age_days: 5
    profiles:
      balanced:
        neutral: { equity: 0.60, rates: 0.30, gold: 0.10 }
        caps:    { equity: 0.75, rates: 0.50, gold: 0.30, crypto: 0.0 }
        tilt_strength: 1.0
    implications:
      bands: { modest: 0.03, pronounced: 0.10 }
      templates:
        neutral: "neutral {over}"
        modest: "modest {over} {under}"
        pronounced: "pronounced {over} {under}"
      degraded_suffix: "degraded"
      disclaimer: "Not investment advice."
"""


class TestShippedConfig:
    def test_the_shipped_config_loads(self):
        config = get_portfolio_config()
        assert set(config.profiles) == {"conservative", "balanced", "growth"}

    def test_balanced_neutral_is_the_backtest_benchmark(self):
        """balanced-at-neutral must be exactly the 60/30/10 static benchmark."""
        balanced = get_portfolio_config().profile("balanced")
        assert balanced.neutral == {"equity": 0.60, "rates": 0.30, "gold": 0.10}

    def test_every_profile_excludes_crypto(self):
        config = get_portfolio_config()
        for profile in config.profiles.values():
            assert "crypto" not in profile.neutral
            assert profile.cap("crypto") == 0.0

    def test_the_profiles_match_the_serving_plane(self):
        """Cross-plane parity: portfolio.yaml's profiles are serving's PROFILES.

        The compute side owns them in yaml and the serving side in domain.ts;
        a mismatch would 400 a profile the compute plane happily allocated, or
        publish one the API refuses. Order matters — it is the switcher order.
        """
        source = DOMAIN_TS.read_text()
        match = re.search(r"export const PROFILES = \[(.*?)\] as const;", source, re.DOTALL)
        assert match, "PROFILES not found in serving/src/domain.ts"
        ts_profiles = re.findall(r"'([^']+)'", match.group(1))
        assert list(get_portfolio_config().profiles) == ts_profiles


class TestValidation:
    def test_the_base_fixture_is_itself_valid(self, tmp_path):
        config = load_portfolio_config(_write(tmp_path, BASE))
        assert config.profile("balanced").tilt_strength == 1.0

    def test_a_neutral_mix_that_does_not_sum_to_one_is_refused(self, tmp_path):
        body = BASE.replace("equity: 0.60", "equity: 0.50")
        with pytest.raises(PortfolioConfigError, match="must sum to 1"):
            load_portfolio_config(_write(tmp_path, body))

    def test_a_cap_below_neutral_is_refused(self, tmp_path):
        """The stale-engine fallback holds an asset at neutral; a lower cap is illegal."""
        body = BASE.replace("equity: 0.75", "equity: 0.55")
        with pytest.raises(PortfolioConfigError, match="below its neutral weight"):
            load_portfolio_config(_write(tmp_path, body))

    def test_a_nonzero_crypto_cap_is_refused(self, tmp_path):
        body = BASE.replace("crypto: 0.0", "crypto: 0.1")
        with pytest.raises(PortfolioConfigError, match="crypto must be 0.0"):
            load_portfolio_config(_write(tmp_path, body))

    def test_unsorted_quantiles_are_refused(self, tmp_path):
        body = BASE.replace("[0.05, 0.5, 0.95]", "[0.95, 0.05, 0.5]")
        with pytest.raises(PortfolioConfigError, match="sorted ascending"):
            load_portfolio_config(_write(tmp_path, body))

    def test_bands_must_be_ordered(self, tmp_path):
        body = BASE.replace("modest: 0.03, pronounced: 0.10", "modest: 0.10, pronounced: 0.03")
        with pytest.raises(PortfolioConfigError, match="modest < pronounced"):
            load_portfolio_config(_write(tmp_path, body))

    def test_a_zero_neutral_weight_is_refused(self, tmp_path):
        """The multiplicative tilt can never lift a zero prior; naming one is a bug."""
        # gold 0.0, the freed 0.10 added to equity so the mix still sums to 1.
        body = BASE.replace("gold: 0.10 }", "gold: 0.0 }").replace(
            "equity: 0.60, rates", "equity: 0.70, rates"
        )
        with pytest.raises(PortfolioConfigError, match="must be > 0"):
            load_portfolio_config(_write(tmp_path, body))

    def test_a_near_one_neutral_mix_is_normalized_to_exactly_one(self, tmp_path):
        """Sub-1e-6 rounding is absorbed so every allocation sums to exactly 1.0.

        The bug this guards: a mix off by 1e-7 loads (the loader tolerates 1e-6)
        and then fails the 1e-9 sum-to-1 guardrail at allocation time, which the
        daily job swallows — silently dropping every profile's portfolio_state.
        """
        body = BASE.replace("gold: 0.10 }", "gold: 0.1000004 }")
        config = load_portfolio_config(_write(tmp_path, body))
        assert sum(config.profile("balanced").neutral.values()) == pytest.approx(1.0, abs=1e-12)

        # And the allocation it feeds must not raise on the sum-to-1 guardrail.
        from datetime import date

        from findynamics.portfolio import compute_allocations
        from tests.portfolio.conftest import full_panel_states, make_world

        as_of = date(2024, 6, 28)
        alloc = compute_allocations(
            full_panel_states(as_of), make_world(as_of), config=config, profiles=["balanced"]
        )["balanced"]
        assert sum(b.mean for b in alloc.weights.values()) == pytest.approx(1.0, abs=1e-9)
