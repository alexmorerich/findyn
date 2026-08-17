"""The hard constraints, tested against inputs the allocator would never make.

A guardrail is only worth having if it holds for weights nobody meant to produce,
so these feed it contrived vectors: over-cap, all-zero, crypto smuggled in.
"""

from __future__ import annotations

import pytest

from findynamics.portfolio import guardrails
from findynamics.portfolio.guardrails import GuardrailError


class TestEnforceCaps:
    def test_uncapped_weights_are_only_normalized(self):
        out = guardrails.enforce_caps({"equity": 0.6, "rates": 0.3, "gold": 0.1}, {})
        assert pytest.approx(sum(out.values())) == 1.0
        assert out["equity"] == pytest.approx(0.6)

    def test_an_over_cap_asset_is_clipped_and_the_excess_redistributed(self):
        # equity wants 0.9 but is capped at 0.75; the 0.15 goes to rates/gold.
        out = guardrails.enforce_caps(
            {"equity": 0.90, "rates": 0.05, "gold": 0.05},
            {"equity": 0.75, "rates": 0.50, "gold": 0.30},
        )
        assert out["equity"] == pytest.approx(0.75)
        assert sum(out.values()) == pytest.approx(1.0)
        # The freed budget went to the two below-cap assets in proportion.
        assert out["rates"] == pytest.approx(out["gold"])

    def test_a_cascade_of_caps_still_sums_to_one(self):
        # Both equity and rates cap; everything piles onto gold.
        out = guardrails.enforce_caps(
            {"equity": 0.70, "rates": 0.25, "gold": 0.05},
            {"equity": 0.40, "rates": 0.20, "gold": 0.60},
        )
        assert out["equity"] == pytest.approx(0.40)
        assert out["rates"] == pytest.approx(0.20)
        assert out["gold"] == pytest.approx(0.40)
        assert sum(out.values()) == pytest.approx(1.0)

    def test_all_zero_weights_are_refused(self):
        with pytest.raises(GuardrailError, match="all-zero"):
            guardrails.enforce_caps({"equity": 0.0, "rates": 0.0}, {})


class TestValidators:
    def test_sum_to_one_catches_a_short_allocation(self):
        with pytest.raises(GuardrailError, match="sum to 1"):
            guardrails.assert_sums_to_one({"equity": 0.6, "rates": 0.3})

    def test_within_caps_catches_an_over_cap_weight(self):
        with pytest.raises(GuardrailError, match="exceeds its cap"):
            guardrails.assert_within_caps({"equity": 0.8}, {"equity": 0.75})

    def test_within_caps_catches_a_negative_weight(self):
        with pytest.raises(GuardrailError, match="negative"):
            guardrails.assert_within_caps({"equity": -0.01}, {"equity": 0.75})

    def test_experimental_exclusion_catches_smuggled_crypto(self):
        with pytest.raises(GuardrailError, match="experimental"):
            guardrails.assert_experimental_excluded({"equity": 0.6, "crypto": 0.4})

    def test_experimental_exclusion_passes_when_crypto_is_absent_or_zero(self):
        guardrails.assert_experimental_excluded({"equity": 1.0})
        guardrails.assert_experimental_excluded({"equity": 1.0, "crypto": 0.0})

    def test_validate_runs_every_check(self):
        # A legal allocation passes; an illegal one raises.
        guardrails.validate({"equity": 0.6, "rates": 0.3, "gold": 0.1}, {"equity": 0.75})
        with pytest.raises(GuardrailError):
            guardrails.validate({"equity": 0.6, "rates": 0.3, "gold": 0.2}, {"equity": 0.75})


class TestMergeWithNeutral:
    def test_a_degraded_asset_is_pinned_at_neutral(self):
        # equity degraded; rates+gold share the remaining 0.4 budget as tilted.
        merged = guardrails.merge_with_neutral(
            tilted={"rates": 0.30, "gold": 0.10},
            neutral={"equity": 0.60, "rates": 0.30, "gold": 0.10},
            degraded=["equity"],
        )
        assert merged["equity"] == 0.60
        assert sum(merged.values()) == pytest.approx(1.0)

    def test_all_degraded_collapses_to_the_whole_neutral_mix(self):
        neutral = {"equity": 0.60, "rates": 0.30, "gold": 0.10}
        merged = guardrails.merge_with_neutral(tilted={}, neutral=neutral, degraded=list(neutral))
        assert merged == neutral
        assert sum(merged.values()) == pytest.approx(1.0)
