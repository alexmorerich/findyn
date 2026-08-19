"""Panel assembly: risk-free from money, excess returns, staleness."""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from findynamics.portfolio.inputs import assemble_panel
from tests.portfolio.conftest import full_panel_states, load_config, make_state, make_world

AS_OF = date(2024, 6, 28)
UNIVERSE = ("money", "rates", "equity", "gold")


def _panel(states, as_of=AS_OF):
    return assemble_panel(states, make_world(as_of), UNIVERSE, load_config())


class TestPanel:
    def test_risk_free_and_discount_come_from_the_money_engine(self):
        panel = _panel(full_panel_states(AS_OF))
        assert panel.risk_free == pytest.approx(0.045)
        assert panel.discount_factors["1y"] == pytest.approx(0.956)

    def test_excess_return_is_net_of_the_risk_free_rate(self):
        panel = _panel(full_panel_states(AS_OF))
        # equity 0.085 expected, cash 0.045 -> excess 0.040.
        assert panel.get("equity").excess_return == pytest.approx(0.040)

    def test_cash_has_zero_excess_by_definition(self):
        panel = _panel(full_panel_states(AS_OF))
        assert panel.get("money").excess_return == 0.0

    def test_a_missing_engine_is_a_stale_input(self):
        states = dict(full_panel_states(AS_OF))
        del states["gold"]
        panel = _panel(states)
        assert panel.get("gold").stale is True
        assert panel.get("gold").usable is False
        assert panel.degraded is True
        assert "gold" in panel.degraded_assets

    def test_an_old_state_is_treated_as_stale(self):
        states = dict(full_panel_states(AS_OF))
        stale_date = AS_OF - timedelta(days=30)
        states["rates"] = make_state(
            "rates",
            stale_date,
            expected_return=0.052,
            risk_score=38.0,
            confidence=0.65,
            regime="flat",
        )
        panel = _panel(states)
        assert panel.get("rates").stale is True
        assert "30d old" in panel.get("rates").reason

    def test_a_stale_money_engine_leaves_the_panel_without_a_risk_free(self):
        """A week-old cash rate is no benchmark; every excess return goes unknown."""
        states = dict(full_panel_states(AS_OF))
        states["money"] = make_state(
            "money",
            AS_OF - timedelta(days=30),
            expected_return=0.045,
            risk_score=2.0,
            confidence=0.8,
            regime="normal",
        )
        panel = _panel(states)
        assert panel.risk_free is None
        # With no risk-free, the risky assets cannot form an excess return.
        assert panel.get("equity").excess_return is None
        assert panel.get("equity").usable is False
